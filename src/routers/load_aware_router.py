import threading
import time
from collections import deque
from routers.router import Router


class ExpertStats:
    """Per-expert latency, error, and rate-limit metrics. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self.latency_ema_ms: float = 0.0
        self.error_count: int = 0
        self.request_count: int = 0
        self._alpha: float = 0.3
        # Sliding window of request timestamps for rate-limit detection
        self._request_timestamps: deque[float] = deque()
        self._rate_limit_rps: float | None = None

    def record_request(self) -> None:
        with self._lock:
            self.request_count += 1
            self._request_timestamps.append(time.monotonic())

    def record_result(self, latency_ms: float, success: bool) -> None:
        with self._lock:
            if self.latency_ema_ms == 0.0:
                self.latency_ema_ms = latency_ms
            else:
                self.latency_ema_ms = (
                    self._alpha * latency_ms
                    + (1 - self._alpha) * self.latency_ema_ms
                )
            if not success:
                self.error_count += 1

    def set_rate_limit(self, rps: float | None) -> None:
        with self._lock:
            self._rate_limit_rps = rps

    @property
    def error_rate(self) -> float:
        with self._lock:
            if self.request_count == 0:
                return 0.0
            return self.error_count / self.request_count

    def is_rate_limited(self) -> bool:
        with self._lock:
            if self._rate_limit_rps is None:
                return False
            now = time.monotonic()
            # Purge timestamps older than 1 second
            while self._request_timestamps and self._request_timestamps[0] < now - 1.0:
                self._request_timestamps.popleft()
            return len(self._request_timestamps) >= self._rate_limit_rps


class ExpertStatsTracker:
    """Holds per-expert stats. Shared between router (reader) and callables (writer)."""

    def __init__(self, experts: list[str]):
        self._stats: dict[str, ExpertStats] = {e: ExpertStats() for e in experts}

    def record_request(self, expert: str) -> None:
        if expert in self._stats:
            self._stats[expert].record_request()

    def record_result(self, expert: str, latency_ms: float, success: bool) -> None:
        if expert in self._stats:
            self._stats[expert].record_result(latency_ms, success)

    def set_rate_limit(self, expert: str, rps: float | None) -> None:
        if expert in self._stats:
            self._stats[expert].set_rate_limit(rps)

    def get(self, expert: str) -> ExpertStats | None:
        return self._stats.get(expert)

    def snapshot(self) -> dict[str, dict]:
        """Return a plain-dict snapshot of all expert stats for CSV logging."""
        out = {}
        for name, s in self._stats.items():
            out[name] = {
                "latency_ema_ms": round(s.latency_ema_ms, 3),
                "error_rate": round(s.error_rate, 4),
                "request_count": s.request_count,
                "is_rate_limited": s.is_rate_limited(),
            }
        return out


class LoadAwareRouter(Router):
    """
    Wraps a base router and uses live expert stats to avoid overloaded experts.

    If the base router's preferred expert is available (not rate-limited,
    error rate below threshold), use it. Otherwise, pick the least-loaded
    alternative. Falls back to random if all experts are degraded.
    """

    def __init__(
        self,
        *,
        experts: list[str],
        base_router: Router,
        stats_tracker: ExpertStatsTracker,
        latency_penalty_threshold_ms: float = 1000.0,
        error_rate_threshold: float = 0.5,
        mapping_expert_to_descriptors: dict | None = None,
        **kwargs,
    ):
        super().__init__(experts=experts)
        self.base_router = base_router
        self.stats = stats_tracker
        self.latency_threshold = latency_penalty_threshold_ms
        self.error_threshold = error_rate_threshold

    def _is_available(self, expert: str) -> bool:
        s = self.stats.get(expert)
        if s is None:
            return True
        if s.is_rate_limited():
            return False
        if s.error_rate >= self.error_threshold:
            return False
        return True

    def _load_score(self, expert: str) -> float:
        """Lower is better. Combines latency EMA with a utilization penalty."""
        s = self.stats.get(expert)
        if s is None:
            return 0.0
        score = s.latency_ema_ms
        if s.is_rate_limited():
            score += 10000.0
        if s.error_rate >= self.error_threshold:
            score += 5000.0
        return score

    def route(self, query: str) -> str:
        preferred = self.base_router.route(query)

        if self._is_available(preferred):
            return preferred

        # Rank alternatives by load score (lower is better)
        alternatives = [
            (e, self._load_score(e))
            for e in self.experts
            if e != preferred and self._is_available(e)
        ]
        if alternatives:
            alternatives.sort(key=lambda x: x[1])
            return alternatives[0][0]

        # All degraded — fallback to random
        return self.fallback()

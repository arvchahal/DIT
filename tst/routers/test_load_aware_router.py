import pytest
from routers.load_aware_router import ExpertStats, ExpertStatsTracker, LoadAwareRouter
from routers.simple_router import SimpleRouter


# --------------- ExpertStats ---------------

class TestExpertStats:
    def test_initial_state(self):
        s = ExpertStats()
        assert s.latency_ema_ms == 0.0
        assert s.error_count == 0
        assert s.request_count == 0
        assert s.error_rate == 0.0
        assert s.is_rate_limited() is False

    def test_ema_first_record(self):
        s = ExpertStats()
        s.record_result(100.0, True)
        assert s.latency_ema_ms == 100.0

    def test_ema_subsequent_records(self):
        s = ExpertStats()
        s.record_result(100.0, True)
        s.record_result(200.0, True)
        # EMA = 0.3 * 200 + 0.7 * 100 = 60 + 70 = 130
        assert abs(s.latency_ema_ms - 130.0) < 0.01

    def test_error_tracking(self):
        s = ExpertStats()
        s.record_request()
        s.record_result(50.0, True)
        s.record_request()
        s.record_result(50.0, False)
        assert s.error_count == 1
        assert s.request_count == 2
        assert abs(s.error_rate - 0.5) < 0.01

    def test_rate_limit_not_set(self):
        s = ExpertStats()
        for _ in range(10):
            s.record_request()
        assert s.is_rate_limited() is False

    def test_rate_limit_detection(self):
        s = ExpertStats()
        s.set_rate_limit(2.0)  # max 2 requests/sec
        s.record_request()
        s.record_request()
        assert s.is_rate_limited() is True

    def test_rate_limit_cleared(self):
        s = ExpertStats()
        s.set_rate_limit(1.0)
        s.record_request()
        assert s.is_rate_limited() is True
        s.set_rate_limit(None)
        assert s.is_rate_limited() is False


# --------------- ExpertStatsTracker ---------------

class TestExpertStatsTracker:
    def test_set_rate_limit(self):
        tracker = ExpertStatsTracker(["A", "B"])
        tracker.set_rate_limit("A", 1.0)
        assert tracker.get("A").is_rate_limited() is False  # no requests yet
        tracker.record_request("A")
        assert tracker.get("A").is_rate_limited() is True

    def test_snapshot(self):
        tracker = ExpertStatsTracker(["X", "Y"])
        tracker.record_request("X")
        tracker.record_result("X", 100.0, True)
        snap = tracker.snapshot()
        assert "X" in snap
        assert "Y" in snap
        assert snap["X"]["request_count"] == 1
        assert snap["X"]["latency_ema_ms"] == 100.0
        assert snap["Y"]["request_count"] == 0

    def test_unknown_expert_ignored(self):
        tracker = ExpertStatsTracker(["A"])
        tracker.record_request("UNKNOWN")  # should not raise
        tracker.record_result("UNKNOWN", 50.0, True)
        assert tracker.get("UNKNOWN") is None


# --------------- LoadAwareRouter ---------------

class TestLoadAwareRouter:
    @pytest.fixture
    def experts(self):
        return ["A", "B", "C"]

    @pytest.fixture
    def tracker(self, experts):
        return ExpertStatsTracker(experts)

    @pytest.fixture
    def base_router(self, experts):
        return SimpleRouter(experts=experts)

    def test_prefers_base_router_when_available(self, experts, tracker):
        """When no expert is degraded, LoadAwareRouter uses the base router's choice."""
        # Use a deterministic base router that always returns "B"
        class FixedRouter(SimpleRouter):
            def route(self, query):
                return "B"

        base = FixedRouter(experts=experts)
        lar = LoadAwareRouter(
            experts=experts, base_router=base, stats_tracker=tracker)
        assert lar.route("test query") == "B"

    def test_avoids_rate_limited_expert(self, experts, tracker):
        """When the preferred expert is rate-limited, pick an alternative."""
        class AlwaysA(SimpleRouter):
            def route(self, query):
                return "A"

        base = AlwaysA(experts=experts)
        tracker.set_rate_limit("A", 1.0)
        tracker.record_request("A")  # now A is rate-limited

        lar = LoadAwareRouter(
            experts=experts, base_router=base, stats_tracker=tracker)
        result = lar.route("test")
        assert result != "A"
        assert result in ("B", "C")

    def test_avoids_high_error_expert(self, experts, tracker):
        """When the preferred expert has high error rate, pick an alternative."""
        class AlwaysA(SimpleRouter):
            def route(self, query):
                return "A"

        base = AlwaysA(experts=experts)
        # Give A 100% error rate
        tracker.record_request("A")
        tracker.record_result("A", 50.0, False)

        lar = LoadAwareRouter(
            experts=experts, base_router=base, stats_tracker=tracker,
            error_rate_threshold=0.5)
        result = lar.route("test")
        assert result != "A"

    def test_fallback_when_all_degraded(self, experts, tracker):
        """When all experts are degraded, fallback returns one of the experts."""
        class AlwaysA(SimpleRouter):
            def route(self, query):
                return "A"

        base = AlwaysA(experts=experts)

        # Rate-limit all experts
        for e in experts:
            tracker.set_rate_limit(e, 1.0)
            tracker.record_request(e)

        lar = LoadAwareRouter(
            experts=experts, base_router=base, stats_tracker=tracker)
        result = lar.route("test")
        assert result in experts  # fallback returns any expert

    def test_picks_lowest_load_alternative(self, experts, tracker):
        """Among available alternatives, picks the one with lowest load score."""
        class AlwaysA(SimpleRouter):
            def route(self, query):
                return "A"

        base = AlwaysA(experts=experts)
        tracker.set_rate_limit("A", 1.0)
        tracker.record_request("A")  # A is rate-limited

        # Give B high latency, C low latency
        tracker.record_result("B", 500.0, True)
        tracker.record_result("C", 50.0, True)

        lar = LoadAwareRouter(
            experts=experts, base_router=base, stats_tracker=tracker)
        assert lar.route("test") == "C"

    def test_accepts_mapping_kwarg(self, experts, tracker, base_router):
        """Constructor accepts mapping_expert_to_descriptors for compatibility."""
        lar = LoadAwareRouter(
            experts=experts,
            base_router=base_router,
            stats_tracker=tracker,
            mapping_expert_to_descriptors={"A": ["finance"]},
        )
        assert lar.route("anything") in experts

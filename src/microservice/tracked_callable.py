"""Tracked remote callable that records latency/status into ExpertStatsTracker."""

import time
from microservice.publisher import Publisher
from protoc import Status as PbStatus


def make_tracked_remote_callable(publisher: Publisher, model_id: str, stats_tracker):
    """
    Like make_remote_callable but records latency and success/failure
    into the shared ExpertStatsTracker after each call.
    """

    def _call(q: str) -> str:
        stats_tracker.record_request(model_id)
        t0 = time.perf_counter()
        try:
            resp = publisher.ask_sync(model_id, q)
            latency_ms = (time.perf_counter() - t0) * 1000
            success = resp.response_status == PbStatus.SUCCESS
            stats_tracker.record_result(model_id, latency_ms, success)
            if not success:
                raise RuntimeError(resp.error_message or "remote expert error")
            return resp.payload
        except RuntimeError:
            raise
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            stats_tracker.record_result(model_id, latency_ms, False)
            raise RuntimeError(str(exc)) from exc

    return _call

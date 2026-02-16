"""
Experiment 3: Dynamic Rate-Limited Routing
-------------------------------------------
Compares static routers (embedding, domain) against a LoadAwareRouter
that dynamically shifts traffic away from throttled/slow experts.

Simulates throttling by imposing an artificial delay on one expert
during a configurable window of the query stream.

Usage (with NATS workers running):
  python pub.py --experts flan-t5 biomedlm tinyllama law-llm

Echo mode (no NATS needed, for smoke testing):
  python pub.py --experts flan-t5 biomedlm tinyllama law-llm --echo --cycles 1
"""

import os, sys, csv, json, time, argparse
from datetime import datetime

THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(SRC_ROOT, "src"))

from microservice.publisher import Publisher, make_remote_callable
from microservice.tracked_callable import make_tracked_remote_callable
from dit_components.dit import DIT
from dit_components.dit_expert import DitExpert
from routers.embedding_router import EmbeddingRouter
from routers.domain_router import DomainRouter
from routers.load_aware_router import LoadAwareRouter, ExpertStatsTracker

DEFAULT_MAPPING = {
    "flan-t5": [
        "travel", "country", "city", "airport", "tourism", "destination",
        "flight", "hotel", "visa", "landmark", "continent", "island",
        "ocean", "border", "capital", "resort", "cruise", "passport",
    ],
    "biomedlm": [
        "sports", "game", "team", "player", "score", "match", "league",
        "championship", "tournament", "coach", "athlete", "stadium",
        "goal", "medal", "race", "olympic", "season", "draft",
    ],
    "tinyllama": [
        "finance", "money", "invest", "stock", "bank", "loan", "credit",
        "debt", "savings", "budget", "interest", "mortgage", "tax",
        "retirement", "portfolio", "dividend", "income", "expense",
    ],
    "law-llm": [
        "literature", "book", "novel", "author", "poem", "story", "writer",
        "fiction", "chapter", "genre", "literary", "publish", "library",
        "narrative", "prose", "character", "essay", "playwright",
    ],
}


def load_queries(csv_path: str) -> list[dict]:
    """Load queries from a CSV with columns: type, question, oracle_response."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            if q:
                rows.append({
                    "type": row.get("type", "").strip(),
                    "question": q,
                    "oracle_response": row.get("oracle_response", "").strip(),
                })
    return rows


class ThrottledCallable:
    """
    Wraps a callable and injects artificial delay when the expert is over
    its rate limit. Applied equally to ALL routers so static routers feel
    the pain while LoadAwareRouter avoids routing to the throttled expert.
    """

    def __init__(self, inner_callable, model_id: str,
                 stats_tracker: ExpertStatsTracker, delay_s: float = 2.0):
        self._inner = inner_callable
        self._model_id = model_id
        self._stats = stats_tracker
        self._delay_s = delay_s

    def __call__(self, query: str) -> str:
        s = self._stats.get(self._model_id)
        if s is not None and s.is_rate_limited():
            time.sleep(self._delay_s)
        return self._inner(query)


def build_strategies(args, experts_list, mapping, stats_tracker, table_tracked):
    """Build the three routing strategies to compare."""

    # 1) Static EmbeddingRouter
    embedding_router = EmbeddingRouter(experts=experts_list)

    # 2) Static DomainRouter
    domain_router = DomainRouter(
        experts=experts_list,
        mapping_expert_to_descriptors=mapping,
    )

    # 3) LoadAwareRouter wrapping EmbeddingRouter
    base_for_load = EmbeddingRouter(experts=experts_list)
    load_aware_router = LoadAwareRouter(
        experts=experts_list,
        base_router=base_for_load,
        stats_tracker=stats_tracker,
    )

    return {
        "embedding": embedding_router,
        "domain": domain_router,
        "load_aware_embedding": load_aware_router,
    }


def main():
    ap = argparse.ArgumentParser(description="Exp3: Rate-Limited Routing Publisher")
    ap.add_argument("--experts", nargs="+", required=True,
                    help="Expert IDs matching subscriber --model-id values")
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--timeout-ms", type=int, default=15000)
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=1,
                    help="Number of times to repeat the query set")
    ap.add_argument("--queries-file",
                    default=os.path.join(SRC_ROOT, "data", "queries", "combined.csv"))
    ap.add_argument("--mapping-file", default=None,
                    help="JSON file mapping expert names to descriptor keywords")
    ap.add_argument("--echo", action="store_true",
                    help="Use local echo experts instead of NATS (for testing)")
    ap.add_argument("--routers", nargs="+",
                    default=["embedding", "domain", "load_aware_embedding"],
                    choices=["embedding", "domain", "load_aware_embedding"],
                    help="Which routers to benchmark (default: all)")
    # Throttle simulation knobs
    ap.add_argument("--throttle-expert", default=None,
                    help="Expert to throttle (default: first expert)")
    ap.add_argument("--throttle-rps", type=float, default=0.5,
                    help="Rate limit in requests/sec for throttled expert")
    ap.add_argument("--throttle-after-pct", type=float, default=0.3,
                    help="Start throttle after this fraction of queries (0-1)")
    ap.add_argument("--restore-after-pct", type=float, default=0.7,
                    help="Restore throttle after this fraction of queries (0-1)")
    args = ap.parse_args()

    # Load descriptor mapping
    if args.mapping_file:
        with open(args.mapping_file) as f:
            mapping = json.load(f)
    else:
        mapping = {k: v for k, v in DEFAULT_MAPPING.items() if k in args.experts}

    # Load queries
    queries = load_queries(args.queries_file)
    total_per_cycle = len(queries)
    total_queries = total_per_cycle * args.cycles
    print(f"[pub] Loaded {total_per_cycle} queries from {args.queries_file}")
    print(f"[pub] {args.cycles} cycle(s) = {total_queries} total queries per router")

    throttle_expert = args.throttle_expert or args.experts[0]
    throttle_start = int(total_queries * args.throttle_after_pct)
    throttle_end = int(total_queries * args.restore_after_pct)
    print(f"[pub] Throttle: {throttle_expert} at {args.throttle_rps} rps "
          f"from query {throttle_start} to {throttle_end}")

    # Stats tracker (shared across load_aware router and tracked callables)
    stats_tracker = ExpertStatsTracker(args.experts)

    # Build expert tables: one with tracked callables, one without
    publisher = Publisher(args.nats_url, timeout_ms=args.timeout_ms,
                          max_retries=args.retries)

    table_static = {}
    table_tracked = {}
    for mid in args.experts:
        if args.echo:
            echo_fn = lambda s, m=mid: f"[ECHO {m}] {s}"
            table_static[mid] = DitExpert(model=echo_fn)
            tracked_fn = lambda s, m=mid: f"[ECHO {m}] {s}"
            table_tracked[mid] = DitExpert(
                model=ThrottledCallable(tracked_fn, mid, stats_tracker))
        else:
            table_static[mid] = DitExpert(
                model=ThrottledCallable(
                    make_remote_callable(publisher, mid), mid, stats_tracker))
            table_tracked[mid] = DitExpert(
                model=ThrottledCallable(
                    make_tracked_remote_callable(publisher, mid, stats_tracker),
                    mid, stats_tracker))

    strategies = build_strategies(
        args, args.experts, mapping, stats_tracker, table_tracked)

    # Output directory
    ts = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    out_dir = os.path.join(SRC_ROOT, "data", "exp3_rate_limited", ts)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.csv")

    fieldnames = [
        "id", "cycle", "router", "query_type", "query", "selected_expert",
        "response", "oracle_response", "routing_latency_ms", "total_latency_ms",
        "throttled_expert", "is_throttle_active",
        "expert_latency_ema_ms", "expert_error_rate", "expert_request_count",
        "expert_is_rate_limited", "load_state_json",
    ]

    total_written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for router_name in args.routers:
            router = strategies[router_name]
            print(f"\n[pub] === Router: {router_name} ===")

            # Pick the right expert table
            if router_name == "load_aware_embedding":
                dit = DIT(experts=table_tracked, router=router)
            else:
                dit = DIT(experts=table_static, router=router)

            # Reset stats between strategies
            stats_tracker = ExpertStatsTracker(args.experts)
            if isinstance(router, LoadAwareRouter):
                router.stats = stats_tracker
            # Rebuild throttled callables with fresh tracker
            for mid in args.experts:
                if args.echo:
                    echo_fn = lambda s, m=mid: f"[ECHO {m}] {s}"
                    if router_name == "load_aware_embedding":
                        table_tracked[mid] = DitExpert(
                            model=ThrottledCallable(echo_fn, mid, stats_tracker))
                    else:
                        table_static[mid] = DitExpert(
                            model=ThrottledCallable(echo_fn, mid, stats_tracker))

            # Re-create DIT with fresh tables
            if router_name == "load_aware_embedding":
                dit = DIT(experts=table_tracked, router=router)
            else:
                dit = DIT(experts=table_static, router=router)

            is_throttle_active = False
            query_idx = 0

            for cycle in range(args.cycles):
                for i, qrow in enumerate(queries):
                    q = qrow["question"]

                    # Throttle window management
                    if query_idx == throttle_start and not is_throttle_active:
                        stats_tracker.set_rate_limit(throttle_expert, args.throttle_rps)
                        is_throttle_active = True
                        print(f"  [THROTTLE ON] {throttle_expert} "
                              f"limited to {args.throttle_rps} rps at query {query_idx}")
                    elif query_idx == throttle_end and is_throttle_active:
                        stats_tracker.set_rate_limit(throttle_expert, None)
                        is_throttle_active = False
                        print(f"  [THROTTLE OFF] {throttle_expert} "
                              f"restored at query {query_idx}")

                    # Route + infer
                    t_route = time.perf_counter()
                    selected = router.route(q)
                    routing_ms = (time.perf_counter() - t_route) * 1000

                    t_total = time.perf_counter()
                    try:
                        res = dit.exec(q)
                        response = res["response"]
                        selected = res["expert"]
                    except Exception as e:
                        response = f"ERROR: {e}"
                    total_ms = (time.perf_counter() - t_total) * 1000

                    # Grab stats for the selected expert
                    snap = stats_tracker.snapshot()
                    expert_snap = snap.get(selected, {})

                    writer.writerow({
                        "id": query_idx,
                        "cycle": cycle,
                        "router": router_name,
                        "query_type": qrow["type"],
                        "query": q,
                        "selected_expert": selected,
                        "response": response,
                        "oracle_response": qrow["oracle_response"],
                        "routing_latency_ms": round(routing_ms, 3),
                        "total_latency_ms": round(total_ms, 3),
                        "throttled_expert": throttle_expert,
                        "is_throttle_active": is_throttle_active,
                        "expert_latency_ema_ms": expert_snap.get("latency_ema_ms", 0),
                        "expert_error_rate": expert_snap.get("error_rate", 0),
                        "expert_request_count": expert_snap.get("request_count", 0),
                        "expert_is_rate_limited": expert_snap.get("is_rate_limited", False),
                        "load_state_json": json.dumps(snap),
                    })
                    total_written += 1
                    query_idx += 1

                    if total_written % 100 == 0:
                        f.flush()
                        print(f"  [{router_name}] {total_written} rows written ...")

                print(f"  [{router_name}] cycle {cycle} done "
                      f"({total_per_cycle} queries)")

    print(f"\n[pub] Results written to {out_path} ({total_written} rows)")


if __name__ == "__main__":
    main()

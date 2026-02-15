"""
Experiment 1: Router Comparison (Distributed)
----------------------------------------------
Sends queries from data/queries/combined.csv through 4 routing strategies
(round_robin, embedding, domain, domain_simplified) against remote DitExpert
workers running via NATS.

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
from dit_components.dit import DIT
from dit_components.dit_expert import DitExpert
from routers.simple_router import SimpleRouter
from routers.embedding_router import EmbeddingRouter
from routers.domain_router import DomainRouter
from routers.domain_simplified_router import DomainSimplifiedRouter

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

ROUTER_CLASSES = {
    "round_robin": SimpleRouter,
    "embedding": EmbeddingRouter,
    "domain": DomainRouter,
    "domain_simplified": DomainSimplifiedRouter,
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


def main():
    ap = argparse.ArgumentParser(description="Exp1: Router Comparison Publisher")
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
    ap.add_argument("--routers", nargs="+", default=list(ROUTER_CLASSES.keys()),
                    choices=list(ROUTER_CLASSES.keys()),
                    help="Which routers to benchmark (default: all)")
    args = ap.parse_args()

    # Load descriptor mapping
    if args.mapping_file:
        with open(args.mapping_file) as f:
            mapping = json.load(f)
    else:
        mapping = {k: v for k, v in DEFAULT_MAPPING.items() if k in args.experts}

    # Load queries
    queries = load_queries(args.queries_file)
    print(f"[pub] Loaded {len(queries)} queries from {args.queries_file}")

    # Build expert table
    publisher = Publisher(args.nats_url, timeout_ms=args.timeout_ms,
                          max_retries=args.retries)
    table = {}
    for mid in args.experts:
        if args.echo:
            table[mid] = DitExpert(model=lambda s, m=mid: f"[ECHO {m}] {s}")
        else:
            table[mid] = DitExpert(model=make_remote_callable(publisher, mid))

    # Initial router (will be swapped per strategy)
    initial_router = SimpleRouter(experts=args.experts)
    dit = DIT(experts=table, router=initial_router)

    # Output directory
    ts = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    out_dir = os.path.join(SRC_ROOT, "data", "exp1_router_comparison", ts)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.csv")

    fieldnames = [
        "id", "cycle", "router", "query_type", "query", "selected_expert",
        "response", "oracle_response", "routing_latency_ms", "total_latency_ms",
    ]

    total_written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for router_name in args.routers:
            RouterCls = ROUTER_CLASSES[router_name]
            print(f"\n[pub] === Router: {router_name} ===")

            # Build router with appropriate kwargs
            router_kwargs = {"experts": args.experts}
            if router_name in ("domain", "domain_simplified"):
                router_kwargs["mapping_expert_to_descriptors"] = mapping
            router = RouterCls(**router_kwargs)
            dit.router = router

            for cycle in range(args.cycles):
                for i, qrow in enumerate(queries):
                    q = qrow["question"]

                    # Measure routing latency
                    t_route = time.perf_counter()
                    selected = router.route(q)
                    routing_ms = (time.perf_counter() - t_route) * 1000

                    # Measure total latency (route + inference)
                    t_total = time.perf_counter()
                    try:
                        res = dit.exec(q)
                        response = res["response"]
                        selected = res["expert"]
                    except Exception as e:
                        response = f"ERROR: {e}"
                    total_ms = (time.perf_counter() - t_total) * 1000

                    row_id = cycle * len(queries) + i
                    writer.writerow({
                        "id": row_id,
                        "cycle": cycle,
                        "router": router_name,
                        "query_type": qrow["type"],
                        "query": q,
                        "selected_expert": selected,
                        "response": response,
                        "oracle_response": qrow["oracle_response"],
                        "routing_latency_ms": round(routing_ms, 3),
                        "total_latency_ms": round(total_ms, 3),
                    })
                    total_written += 1

                    if total_written % 100 == 0:
                        f.flush()
                        print(f"  [{router_name}] {total_written} rows written ...")

                print(f"  [{router_name}] cycle {cycle} done "
                      f"({len(queries)} queries)")

    print(f"\n[pub] Results written to {out_path} ({total_written} rows)")


if __name__ == "__main__":
    main()

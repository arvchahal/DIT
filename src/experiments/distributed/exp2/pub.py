"""
Experiment 2: Monolith Baseline (Distributed)
----------------------------------------------
Sends ALL queries to a single "monolith" model (no routing decision).
Used as a baseline to compare against DIT's multi-expert routing in Exp1.

Usage (with a single NATS worker running):
  python pub.py --model-id monolith

Echo mode (no NATS needed, for smoke testing):
  python pub.py --model-id monolith --echo
"""

import os, sys, csv, time, argparse
from datetime import datetime

THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(SRC_ROOT, "src"))

from microservice.publisher import Publisher, make_remote_callable
from dit_components.dit_expert import DitExpert


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
    ap = argparse.ArgumentParser(description="Exp2: Monolith Baseline Publisher")
    ap.add_argument("--model-id", required=True,
                    help="The single monolith expert's model ID")
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--timeout-ms", type=int, default=15000)
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--queries-file",
                    default=os.path.join(SRC_ROOT, "data", "queries", "combined.csv"))
    ap.add_argument("--echo", action="store_true",
                    help="Use local echo expert instead of NATS (for testing)")
    args = ap.parse_args()

    # Load queries
    queries = load_queries(args.queries_file)
    print(f"[pub] Loaded {len(queries)} queries from {args.queries_file}")

    # Build single expert
    if args.echo:
        expert = DitExpert(model=lambda s: f"[ECHO {args.model_id}] {s}")
    else:
        publisher = Publisher(args.nats_url, timeout_ms=args.timeout_ms,
                              max_retries=args.retries)
        expert = DitExpert(model=make_remote_callable(publisher, args.model_id))

    # Output directory
    ts = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    out_dir = os.path.join(SRC_ROOT, "data", "exp2_monolith", ts)
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

        for i, qrow in enumerate(queries):
            q = qrow["question"]

            t_total = time.perf_counter()
            try:
                response = expert.run_model(q)
            except Exception as e:
                response = f"ERROR: {e}"
            total_ms = (time.perf_counter() - t_total) * 1000

            writer.writerow({
                "id": i,
                "cycle": 0,
                "router": "monolith",
                "query_type": qrow["type"],
                "query": q,
                "selected_expert": args.model_id,
                "response": response,
                "oracle_response": qrow["oracle_response"],
                "routing_latency_ms": 0.0,
                "total_latency_ms": round(total_ms, 3),
            })
            total_written += 1

            if total_written % 100 == 0:
                f.flush()
                print(f"  [monolith] {total_written} rows written ...")

    print(f"\n[pub] Results written to {out_path} ({total_written} rows)")


if __name__ == "__main__":
    main()

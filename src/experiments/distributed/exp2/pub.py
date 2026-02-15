"""
Experiment 2: Monolith Baseline (OpenAI API)
---------------------------------------------
Sends ALL queries to a single large model (GPT-4o-mini or GPT-4o) via
the OpenAI API. No routing, no NATS — just one big model answering
everything. Used as a baseline to compare against DIT's multi-expert
routing in Exp1.

Usage:
  export OPENAI_API_KEY=sk-...
  python pub.py --model gpt-4o-mini
  python pub.py --model gpt-4o
"""

import os, sys, csv, time, argparse
from datetime import datetime

THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(SRC_ROOT, "src"))

from openai import OpenAI


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


def ask_openai(client: OpenAI, model: str, query: str,
               max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def main():
    ap = argparse.ArgumentParser(description="Exp2: Monolith Baseline (OpenAI)")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="OpenAI model to use (gpt-4o-mini, gpt-4o, etc.)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--queries-file",
                    default=os.path.join(SRC_ROOT, "data", "queries", "combined.csv"))
    ap.add_argument("--api-key", default=None,
                    help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY env var or pass --api-key")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load queries
    queries = load_queries(args.queries_file)
    print(f"[pub] Loaded {len(queries)} queries from {args.queries_file}")
    print(f"[pub] Model: {args.model}")

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
                response = ask_openai(client, args.model, q, args.max_tokens)
            except Exception as e:
                response = f"ERROR: {e}"
            total_ms = (time.perf_counter() - t_total) * 1000

            writer.writerow({
                "id": i,
                "cycle": 0,
                "router": "monolith",
                "query_type": qrow["type"],
                "query": q,
                "selected_expert": args.model,
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

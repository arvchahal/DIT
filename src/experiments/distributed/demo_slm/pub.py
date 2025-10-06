# src/experiments/distributed/test/pub_from_bank.py
import os, sys, argparse, csv, time
from datetime import datetime

# allow: python3 pub_from_bank.py (no PYTHONPATH)
THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..",".."))
print(SRC_ROOT)
sys.path.insert(0, SRC_ROOT)

from microservice.publisher import Publisher, make_remote_callable
from dit_components.dit import DIT
from dit_components.dit_expert import DitExpert
from routers.simple_router import SimpleRouter


def load_queries(bank_dir: str, topic: str) -> list[str]:
    """Load queries from data/query_banks/<topic>.csv (expects a 'queries' column)."""
    path = os.path.join(bank_dir, f"{topic}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No CSV found for topic '{topic}' at {path}")

    queries = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "question" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'queries' column")
        for row in reader:
            q = row["questions"].strip()
            if q:
                queries.append(q)
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, help="Which topic to load (e.g., sports, travel, payments)")
    ap.add_argument("--bank-dir", default=os.path.join(SRC_ROOT, "data", "queries"))
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--experts", nargs="+", default=["Payments", "Search", "Support"])
    ap.add_argument("--timeout-ms", type=int, default=15000)
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--echo", action="store_true", help="Use local echo experts instead of NATS")
    args = ap.parse_args()

    # load queries
    print(args.bank_dir)
    queries = load_queries(args.bank_dir, args.topic)

    # publisher + DIT setup
    publisher = Publisher(args.nats_url, timeout_ms=args.timeout_ms, max_retries=args.retries)
    table = {}
    for mid in args.experts:
        if args.echo:
            table[mid] = DitExpert(model=lambda s, m=mid: f"[LOCAL {m}] {s}")
        else:
            table[mid] = DitExpert(model=make_remote_callable(publisher, mid))

    router = SimpleRouter(experts=args.experts)
    dit = DIT(experts=table, router=router)

    # write results
    ts = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    out_dir = os.path.join(SRC_ROOT, "data", "exp_pubsub_bank", ts)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.topic}.csv")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "topic", "input", "expert", "result_or_error"])
        for i, q in enumerate(queries):
            try:
                res = dit.exec(q)
                w.writerow([i, args.topic, q, res["expert"], res["response"]])
                print(f"[PUB] ({args.topic}) '{q}' -> {res['expert']} :: {res['response']}")
            except Exception as e:
                w.writerow([i, args.topic, q, "", f"ERROR: {e}"])
                print(f"[PUB] ERROR ({args.topic}) '{q}': {e}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

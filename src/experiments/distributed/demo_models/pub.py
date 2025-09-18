# src/experiments/distributed/test/pub_simple.py
import os, sys, argparse, csv, time
from datetime import datetime

# allow: python3 pub_simple.py  (no PYTHONPATH)
THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_ROOT)

from microservice.publisher import Publisher, make_remote_callable
from dit_components.dit import DIT
from dit_components.dit_expert import DitExpert
from routers.simple_router import SimpleRouter  # <-- your existing router

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--experts", nargs="+", default=["Payments"])
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--timeout-ms", type=int, default=800)
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--echo", action="store_true", help="use local echo experts (no NATS)")
    args = ap.parse_args()

    # Publisher with single background loop (fixes 'first works, others timeout')
    publisher = Publisher(args.nats_url, timeout_ms=args.timeout_ms, max_retries=args.retries)

    # Build expert table: remote (NATS) unless --echo is set
    table = {}
    for mid in args.experts:
        if args.echo:
            table[mid] = DitExpert(model=lambda s, m=mid: f"[LOCAL {m}] {s}")
        else:
            table[mid] = DitExpert(model=make_remote_callable(publisher, mid))

    router = SimpleRouter(experts=args.experts)  # <-- use your SimpleRouter
    dit = DIT(experts=table, router=router)

    queries = [f"query_{i}" for i in range(args.queries)]

    # output (same style as your exp0)
    ts = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    out_dir = os.path.join(SRC_ROOT, "data", "exp_pubsub_simple", ts)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.csv")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "router", "input", "route_time", "expert", "result_or_error"])
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            expert_id = router.route(q)                # measure routing only
            route_time = time.perf_counter() - t0
            try:
                res = dit.exec(q)                      # calls remote expert via NATS (or local echo)
                w.writerow([i, "simple", q, f"{route_time:.6f}", res["expert"], res["response"]])
                print(f"[PUB] '{q}' -> {res['expert']} :: {res['response']}")
            except Exception as e:
                w.writerow([i, "simple", q, f"{route_time:.6f}", expert_id, f"ERROR: {e}"])
                print(f"[PUB] ERROR '{q}' -> {expert_id}: {e}")

    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()

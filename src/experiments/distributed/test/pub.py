# src/experiments/distributed/test/pub.py
import os, sys, argparse, csv, time
from datetime import datetime

# add src/ so `python3 pub.py` works from this folder
THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_ROOT)

from microservice.publisher import Publisher, make_remote_callable
from dit_components.dit import DIT
from dit_components.dit_expert import DitExpert

# your routers
from routers.simple_router import SimpleRouter
from routers.domain_router import DomainRouter
from routers.domain_simplified_router import DomainSimplifiedRouter
from routers.embedding_router import EmbeddingRouter

def build_router(name: str, experts: list[str]):
    mapping = {"Payments": ["finance"], "Search": ["find"], "Support": ["help"]}
    if name == "basic":
        return SimpleRouter(experts=experts)
    if name == "domain":
        return DomainRouter(experts=experts, mapping_expert_to_descriptors=mapping)
    if name == "domain_simplified":
        return DomainSimplifiedRouter(experts=experts, mapping_expert_to_descriptors=mapping)
    if name == "embedding":
        return EmbeddingRouter(experts=experts)
    raise ValueError(f"unknown router '{name}'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--experts", nargs="+", default=["Payments", "Search", "Support"])
    ap.add_argument("--router", choices=["basic","domain","domain_simplified","embedding"], default="basic")
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--remote", action="store_true", help="send over NATS to workers")
    args = ap.parse_args()

    publisher = Publisher(args.nats_url)

    # Build expert table. If --remote, each expert uses a remote callable over NATS.
    table = {}
    for mid in args.experts:
        if args.remote:
            table[mid] = DitExpert(model=make_remote_callable(publisher, mid))
        else:
            table[mid] = DitExpert(model=lambda s, m=mid: f"[LOCAL {m}] {s}")

    router = build_router(args.router, args.experts)
    dit = DIT(experts=table, router=router)

    # experiment loop like your single-process exp0.py
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    out_dir = os.path.join(SRC_ROOT, "data", "exp_pubsub", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.router}.csv")

    queries = [f"query_{i}" for i in range(args.queries)]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","router","input","route_time","expert","result_or_error"])
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            routed = router.route(q)                 # route timing only
            route_time = time.perf_counter() - t0
            try:
                res = dit.exec(q)                    # local or remote (unchanged DIT)
                w.writerow([i, args.router, q, f"{route_time:.6f}", res["expert"], res["response"]])
            except Exception as e:
                w.writerow([i, args.router, q, f"{route_time:.6f}", routed, f"ERROR: {e}"])

    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()

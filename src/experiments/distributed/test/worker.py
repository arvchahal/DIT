# src/experiments/distributed/test/worker.py
import os, sys, argparse, asyncio

# put repo's src/ on sys.path so plain `python3 worker.py` works
THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_ROOT)

from microservice.subscriber import Subscriber
from dit_components.dit_expert import DitExpert

async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--queue-group", default=None)
    args = ap.parse_args()

    # swap this lambda for a real HF pipeline via DitExpert.load_model(...)
    expert = DitExpert(model=lambda s: f"[{args.model_id}] {s}")
    sub = Subscriber(args.nats_url, args.model_id, expert, queue_group=args.queue_group)
    await sub.run()

if __name__ == "__main__":
    asyncio.run(amain())

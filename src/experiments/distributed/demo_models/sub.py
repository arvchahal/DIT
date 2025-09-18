# src/experiments/distributed/test/worker_hf.py
import os, sys, argparse, asyncio

# allow: python3 worker_hf.py  (no PYTHONPATH)
THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_ROOT)

from microservice.subscriber import Subscriber
from dit_components.dit_expert import DitExpert

async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="Expert id / subject suffix, e.g., Payments")
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--queue-group", default=None)
    ap.add_argument("--task", default="text-classification")
    ap.add_argument("--model-name", default="distilbert-base-uncased-finetuned-sst-2-english")
    args = ap.parse_args()

    expert = DitExpert()
    expert.load_model(task=args.task, model_name=args.model_name)

    sub = Subscriber(args.nats_url, args.model_id, expert, queue_group=args.queue_group)
    await sub.run()

if __name__ == "__main__":
    asyncio.run(amain())

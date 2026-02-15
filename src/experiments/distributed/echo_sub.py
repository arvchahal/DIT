import asyncio, sys, os

THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(SRC_ROOT, "src"))

from microservice.subscriber import Subscriber
from dit_components.dit_expert import DitExpert

expert = DitExpert(model=lambda q: f"ECHO: {q}")
sub = Subscriber("nats://127.0.0.1:4222", "travel", expert)
asyncio.run(sub.run())

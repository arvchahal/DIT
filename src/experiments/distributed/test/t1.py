import asyncio, sys
from nats.aio.client import Client as NATS

async def main(url="nats://127.0.0.1:4222", subject="models.Payments", payload=b"hello"):
    nc = NATS()
    await nc.connect(url)
    print(f"[pub] -> {subject}: {payload!r}")
    try:
        msg = await nc.request(subject, payload, timeout=1.0)
        print(f"[pub] <- {msg.data!r}")
    except Exception as e:
        print("request failed:", e)
    await nc.drain()

if __name__ == "__main__":
    asyncio.run(main())


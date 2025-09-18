import asyncio
from nats.aio.client import Client as NATS

async def main(url="nats://127.0.0.1:4222", subject="models.Payments"):
    nc = NATS()
    await nc.connect(url)

    async def cb(msg):
        print(f"[worker] got: {msg.data!r} on {msg.subject}")
        await msg.respond(b"ok: " + msg.data)

    await nc.subscribe(subject, queue="ditq.Payments", cb=cb)
    print(f"[worker] subscribed {subject} (queue=ditq.Payments)")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())


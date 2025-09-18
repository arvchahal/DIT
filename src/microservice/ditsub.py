# src/bus/nats_pubsub.py
from __future__ import annotations
import asyncio, time, math, random, uuid
from typing import Optional
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout as NatsTimeout
import os 
# your generated protos
from protoc import Request as PbRequest, Response as PbResponse, Status as PbStatus
from dit_components.dit_expert import DitExpert         # adjust if your import path differs
from routers.router import Router                        # your router interface


class RouterPublisher:
    """
    Router-side publisher: sends PbRequest to models.<model_id> and awaits PbResponse.
    Retries with jittered backoff on timeout.
    """
    def __init__(self, broker_addr: str, request_timeout_ms: int = 3000, max_retries: int = 2):
        self._nc = NATS()
        self._addr = broker_addr
        self._connected = False
        self.request_timeout_ms = request_timeout_ms
        self.max_retries = max_retries

    async def connect(self):
        if self._connected:
            return
        await self._nc.connect(
            self._addr,
            allow_reconnect=True,
            max_reconnect_attempts=-1,
            reconnect_time_wait=0.5,
            ping_interval=10,
        )
        self._connected = True

    async def close(self):
        if self._connected:
            await self._nc.drain()
            self._connected = False

    async def ask(self, model_id: str, payload: str, request_id: Optional[str] = None) -> PbResponse:
        """
        Send a single request to models.<model_id> and return PbResponse.
        """
        await self.connect()

        req = PbRequest()
        req.request_id = request_id or str(uuid.uuid4())
        req.model_id = model_id
        req.payload = payload

        subject = f"models.{model_id}"
        data = req.SerializeToString()

        attempt = 0
        while True:
            try:
                msg = await self._nc.request(subject, data, timeout=self.request_timeout_ms / 1000.0)
                resp = PbResponse()
                resp.ParseFromString(msg.data)
                return resp
            except NatsTimeout as e:
                if attempt >= self.max_retries:
                    err = PbResponse()
                    err.request_id = req.request_id
                    err.model_id = model_id
                    err.payload = ""
                    err.response_status = PbStatus.ERROR
                    err.latency_ms = 0
                    err.error_message = f"timeout after {attempt+1} tries"
                    return err
                attempt += 1
                sleep_ms = random.randint(150, int(150 * (2 ** attempt)))
                await asyncio.sleep(sleep_ms / 1000.0)


class DitSubscriber:
    """
    Expert-side subscriber: consumes PbRequest from models.<model_id> (queue group),
    calls DitExpert.run_model(payload), and replies with PbResponse.
    """
    def __init__(self, broker_addr: str, model_id: str, expert: DitExpert, queue_group: Optional[str] = None, max_inflight: int = 64):
        self._nc = NATS()
        self._addr = broker_addr
        self._connected = False
        self.model_id = model_id
        self.expert = expert
        self.queue_group = queue_group or f"ditq.{model_id}"
        self._sem = asyncio.Semaphore(max_inflight)

    async def connect(self):
        if self._connected:
            return
        await self._nc.connect(
            self._addr,
            allow_reconnect=True,
            max_reconnect_attempts=-1,
            reconnect_time_wait=0.5,
            ping_interval=10,
        )
        self._connected = True

    async def close(self):
        if self._connected:
            await self._nc.drain()
            self._connected = False

    async def _on_msg(self, msg):
        async with self._sem:
            start_ns = time.time_ns()

            # Parse PbRequest
            req = PbRequest()
            try:
                req.ParseFromString(msg.data)
            except Exception as e:
                resp = PbResponse()
                resp.request_id = ""
                resp.model_id = self.model_id
                resp.payload = ""
                resp.response_status = PbStatus.ERROR
                resp.latency_ms = 0
                resp.error_message = f"bad request: {e}"
                await msg.respond(resp.SerializeToString())
                return

            # Execute model
            resp = PbResponse()
            resp.request_id = req.request_id
            resp.model_id = self.model_id
            try:
                out = self.expert.run_model(req.payload)  # your DitExpert wrapper
                # normalize to string
                resp.payload = str(out)
                resp.response_status = PbStatus.SUCCESS
                resp.error_message = ""
            except Exception as e:
                resp.payload = ""
                resp.response_status = PbStatus.ERROR
                resp.error_message = f"{type(e).__name__}: {e}"

            resp.latency_ms = int((time.time_ns() - start_ns) / 1_000_000)
            await msg.respond(resp.SerializeToString())

    async def run(self):
        await self.connect()
        subject = f"models.{self.model_id}"
        await self._nc.subscribe(subject, queue=self.queue_group, cb=self._on_msg)
        print(f"[DitSubscriber:{self.model_id}] subscribed to {subject} (queue={self.queue_group})")
        await asyncio.Event().wait()


class RemoteDIT:
    """
    Drop-in remote version of your DIT.exec() that sends queries via NATS
    instead of calling local experts. Keeps the same response shape.
    """
    def __init__(self, router: Router, publisher: RouterPublisher):
        self.router = router
        self.publisher = publisher

    async def exec_async(self, query: str):
        model_id = self.router.route(query)
        resp = await self.publisher.ask(model_id, query)
        return {
            "response": resp.payload,
            "expert": model_id,
            "status": int(resp.response_status),
            "latency_ms": int(resp.latency_ms),
            "error": resp.error_message,
            "request_id": resp.request_id,
        }

    # Synchronous convenience wrapper (use only from non-async code).
    def exec(self, query: str):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.exec_async(query))
        else:
            # If you're already inside an event loop, prefer calling exec_async() directly.
            raise RuntimeError("RemoteDIT.exec() called from an active event loop; use exec_async() instead.")

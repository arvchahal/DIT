# src/microservice/subscriber.py
from __future__ import annotations
import asyncio, time
from typing import Optional
from nats.aio.client import Client as NATS
from protoc import Request as PbRequest, Response as PbResponse, Status as PbStatus
from dit_components.dit_expert import DitExpert


class Subscriber:
    """
    Expert worker:
    - subscribes to models.<model_id> in a queue group (work queue)
    - runs expert.run_model(payload) and replies with PbResponse
    """
    def __init__(self, nats_url: str, model_id: str, expert: DitExpert,
                 queue_group: Optional[str] = None, max_inflight: int = 64):
        self._nc = NATS()
        self._url = nats_url
        self.model_id = model_id
        self.expert = expert
        self.queue_group = queue_group or f"ditq.{model_id}"
        self._connected = False
        self._sem = asyncio.Semaphore(max_inflight)

    async def connect(self):
        if self._connected:
            return
        await self._nc.connect(
            self._url,
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

    async def _handle(self, msg):
        async with self._sem:
            t0 = time.time_ns()

            # 1) Parse request; always reply on failure
            req = PbRequest()
            try:
                req.ParseFromString(msg.data)
                print(f"[SUB {self.model_id}] got req_id={req.request_id} payload='{req.payload}'")
            except Exception as e:
                resp = PbResponse(
                    request_id="",
                    model_id=self.model_id,
                    payload="",
                    response_status=PbStatus.ERROR,
                    latency_ms=0,
                    error_message=f"bad request: {type(e).__name__}: {e}",
                )
                try:
                    await msg.respond(resp.SerializeToString())
                    print(f"[SUB {self.model_id}] replied parse-error")
                except Exception as e2:
                    print(f"[SUB {self.model_id}] respond failed (parse-error): {e2}")
                return

            # 2) Run model; convert to string to keep payload serializable
            resp = PbResponse(request_id=req.request_id, model_id=self.model_id)
            try:
                out = self.expert.run_model(req.payload)
                resp.payload = str(out)
                resp.response_status = PbStatus.SUCCESS
                resp.error_message = ""
            except Exception as e:
                resp.payload = ""
                resp.response_status = PbStatus.ERROR
                resp.error_message = f"{type(e).__name__}: {e}"

            resp.latency_ms = int((time.time_ns() - t0) / 1_000_000)

            # 3) ALWAYS attempt to respond; log failures
            try:
                await msg.respond(resp.SerializeToString())
                print(f"[SUB {self.model_id}] replied status={resp.response_status} latency={resp.latency_ms}ms")
            except Exception as e2:
                print(f"[SUB {self.model_id}] respond failed: {e2}")

    async def run(self):
        await self.connect()
        subject = f"models.{self.model_id}"
        await self._nc.subscribe(subject, queue=self.queue_group, cb=self._handle)
        print(f"[Subscriber {self.model_id}] subscribed to {subject} (queue={self.queue_group})")
        await asyncio.Event().wait()

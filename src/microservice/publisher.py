# src/microservice/publisher.py
from __future__ import annotations
import asyncio, uuid, random
from typing import Optional
from nats.aio.client import Client as NATS
import threading

try:
    from nats.aio.errors import ErrTimeout as NatsTimeout, ErrNoResponders
except Exception:
    try:
        from nats.errors import TimeoutError as NatsTimeout, NoRespondersError as ErrNoResponders
    except Exception:
        NatsTimeout = asyncio.TimeoutError
        class ErrNoResponders(Exception): ...

from protoc import Request as PbRequest, Response as PbResponse, Status as PbStatus


class Publisher:
    def __init__(self, nats_url: str = "nats://127.0.0.1:4222",
                 timeout_ms: int = 3000, max_retries: int = 2):
        self._nc = NATS()
        self._url = nats_url
        self._connected = False
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries

        # background event loop for all async work
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    # ---------- background loop management ----------
    def _ensure_loop(self):
        if self._loop:
            return
        self._loop = asyncio.new_event_loop()
        def run():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def _run_coro(self, coro):
        self._ensure_loop()
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def close_sync(self):
        if not self._loop:
            return
        def _stop():
            self._loop.stop()
        self._loop.call_soon_threadsafe(_stop)
        if self._thread:
            self._thread.join(timeout=2)
        self._loop = None
        self._thread = None

    # ---------- async API (runs inside bg loop) ----------
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

    async def _ask_async(self, model_id: str, payload: str,
                         request_id: Optional[str]) -> PbResponse:
        await self.connect()

        req = PbRequest(
            request_id=request_id or str(uuid.uuid4()),
            model_id=model_id,
            payload=payload,
        )
        subject = f"models.{model_id}"
        data = req.SerializeToString()

        attempt = 0
        while True:
            try:
                msg = await self._nc.request(subject, data, timeout=self.timeout_ms / 1000.0)
                resp = PbResponse(); resp.ParseFromString(msg.data)
                return resp
            except (NatsTimeout, asyncio.TimeoutError):
                if attempt >= self.max_retries:
                    resp = PbResponse(
                        request_id=req.request_id, model_id=model_id, payload="",
                        response_status=PbStatus.ERROR, latency_ms=0,
                        error_message=f"timeout after {attempt+1} tries",
                    )
                    return resp
                attempt += 1
                await asyncio.sleep(random.uniform(0.15, 0.15 * (2 ** attempt)))
            except ErrNoResponders:
                resp = PbResponse(
                    request_id=req.request_id, model_id=model_id, payload="",
                    response_status=PbStatus.ERROR, latency_ms=0,
                    error_message="no responders (no subscriber for this model_id)",
                )
                return resp

    # public sync method for use from DitExpert callable
    def ask_sync(self, model_id: str, payload: str,
                 request_id: Optional[str] = None) -> PbResponse:
        return self._run_coro(self._ask_async(model_id, payload, request_id))


def make_remote_callable(publisher: Publisher, model_id: str):
    """
    Synchronous callable(query:str)->str for DitExpert(model=...),
    backed by Publisher.ask_sync on a dedicated event loop.
    """
    from protoc import Status
    def _call(q: str) -> str:
        resp = publisher.ask_sync(model_id, q)
        if resp.response_status != Status.SUCCESS:
            raise RuntimeError(resp.error_message or "remote expert error")
        return resp.payload
    return _call

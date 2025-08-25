# the publisher will essentially be the router it will publish its requests with a model id
# then the model will be subscriber to the location it publishes to
import asyncio
from nats.aio.client import Client as nats
from src.protoc import Response, Request, Status
from collections import deque


class Publisher:
    def __init__(self, broker_addr: str):
        self.nats_obj = nats()
        self.broker_addr = broker_addr
        self.connection = None
        self.requests = deque()

    def publish(self, request_id, model_id, query):
        req = Request(request_id=request_id, model_id=model_id, payload=query)
        subject = f"models.{model_id}"
        self.nats_obj.publish(subject, req)

    async def connect(self):
        if self.connection is not None:
            return
        self.connection = await self.nats_obj.connect(self.broker_addr)

    async def run(self):
        while True:
            if len(self.requests) > 0:
                r_id, m_id, query = self.requests.popleft()
                self.publish(request_id=r_id, model_id=m_id, query=query)

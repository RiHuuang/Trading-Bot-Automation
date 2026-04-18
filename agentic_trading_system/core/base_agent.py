import asyncio
import json
import os
import time
import uuid
from typing import Any

import redis.asyncio as redis


class BaseAgent:
    def __init__(self, agent_name: str, redis_url: str | None = None):
        self.agent_name = agent_name
        default_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis = redis.from_url(redis_url or default_url, decode_responses=True)
        self.pubsub = self.redis.pubsub()

    @staticmethod
    def utc_timestamp() -> float:
        return time.time()

    @staticmethod
    def new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    async def publish_event(self, channel: str, payload: dict[str, Any]):
        message = json.dumps(payload)
        await self.redis.publish(channel, message)
        print(f"[{self.agent_name}] Published to {channel}")

    async def listen(self, channel: str, callback_function):
        await self.pubsub.subscribe(channel)
        print(f"[{self.agent_name}] Listening on {channel}...")

        async for message in self.pubsub.listen():
            if message["type"] != "message":
                continue

            try:
                data = json.loads(message["data"])
            except json.JSONDecodeError as exc:
                print(f"[{self.agent_name}] Dropping malformed payload on {channel}: {exc}")
                continue

            try:
                await callback_function(data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[{self.agent_name}] Callback error on {channel}: {exc}")

    async def shutdown(self):
        await self.pubsub.close()
        await self.redis.aclose()

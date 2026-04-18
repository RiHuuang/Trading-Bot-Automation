import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import redis.asyncio as redis


class BaseAgent:
    def __init__(self, agent_name: str, redis_url: str | None = None):
        self.agent_name = agent_name
        default_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis = redis.from_url(redis_url or default_url, decode_responses=True)
        self.pubsub = self.redis.pubsub()
        self.heartbeat_interval_seconds = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "30"))
        self.kill_switch_redis_key = os.getenv("KILL_SWITCH_REDIS_KEY", "trading:kill_switch")
        self.kill_switch_file = os.getenv("KILL_SWITCH_FILE", "/tmp/trading.kill")

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

    async def publish_alert(self, level: str, message: str, details: dict[str, Any] | None = None):
        payload = {
            "agent": self.agent_name,
            "level": level,
            "message": message,
            "timestamp": self.utc_timestamp(),
            "details": details or {},
        }
        await self.publish_event("ALERT_EVENT", payload)

    async def is_kill_switch_active(self) -> bool:
        try:
            redis_value = await self.redis.get(self.kill_switch_redis_key)
            if redis_value and redis_value.lower() in {"1", "true", "on", "enabled"}:
                return True
        except Exception:
            pass
        return os.path.exists(self.kill_switch_file)

    async def set_kill_switch(self, active: bool) -> bool:
        try:
            if active:
                await self.redis.set(self.kill_switch_redis_key, "true")
                Path(self.kill_switch_file).touch()
            else:
                await self.redis.delete(self.kill_switch_redis_key)
                try:
                    Path(self.kill_switch_file).unlink()
                except FileNotFoundError:
                    pass
            return await self.is_kill_switch_active()
        except Exception as exc:
            print(f"[{self.agent_name}] Kill switch update failed: {exc}")
            raise

    async def heartbeat_loop(self):
        while True:
            payload = {
                "agent": self.agent_name,
                "timestamp": self.utc_timestamp(),
                "status": "healthy",
            }
            try:
                await self.redis.set(
                    f"heartbeat:{self.agent_name}",
                    json.dumps(payload),
                    ex=max(self.heartbeat_interval_seconds * 3, 60),
                )
                await self.redis.publish("HEARTBEAT_EVENT", json.dumps(payload))
            except Exception as exc:
                print(f"[{self.agent_name}] Heartbeat failed: {exc}")
            await asyncio.sleep(self.heartbeat_interval_seconds)

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

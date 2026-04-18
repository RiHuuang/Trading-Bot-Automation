import asyncio
import json
import redis.asyncio as redis
import os

class BaseAgent:
    def __init__(self, agent_name: str, redis_url: str = None):
        self.agent_name = agent_name
        # Defaults to the Docker network hostname 'redis', falls back to localhost for local testing
        default_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis = redis.from_url(redis_url or default_url, decode_responses=True)
        self.pubsub = self.redis.pubsub()

    async def publish_event(self, channel: str, payload: dict):
        """Standardized way for agents to broadcast data."""
        message = json.dumps(payload)
        await self.redis.publish(channel, message)
        print(f"[{self.agent_name}] Published to {channel}")

    async def listen(self, channel: str, callback_function):
        """Subscribes to a channel and routes messages to a specific function."""
        await self.pubsub.subscribe(channel)
        print(f"[{self.agent_name}] Listening on {channel}...")
        
        async for message in self.pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                # Route the data to the agent's specific logic
                await callback_function(data)

    async def shutdown(self):
        await self.pubsub.close()
        await self.redis.aclose()

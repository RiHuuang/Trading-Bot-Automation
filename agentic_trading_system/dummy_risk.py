import asyncio
from pydantic import ValidationError
from core.base_agent import BaseAgent
from core.schemas import MarketTick

async def process_tick(data):
    try:
        # 1. Intercept and Validate
        valid_tick = MarketTick(**data)
        # 2. Proceed if valid
        print(f"[Risk/Validation] Clean Tick Verified: {valid_tick.ticker} @ ${valid_tick.price:.2f}")
    except ValidationError as e:
        # 3. Kill the action if invalid
        print(f"[FATAL ERROR CAUGHT] Invalid data received. Dropping tick. Reason:\n{e.errors()[0]['msg']}")

async def main():
    agent = BaseAgent("RiskAgent")
    await agent.listen("TICK_EVENT", process_tick)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nRisk Agent shutting down.")
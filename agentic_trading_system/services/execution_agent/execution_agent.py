import asyncio
import random
from core.base_agent import BaseAgent
from core.schemas import ExecutionOrder

class ExecutionAgent(BaseAgent):
    def __init__(self):
        super().__init__("ExecutionAgent")

    async def execute_order(self, data):
        print(f"\n[ExecutionAgent] RECEIVED APPROVED ORDER: {data['action']} {data['quantity']} {data['ticker']}")
        
        try:
            # 1. Validate the order strictly against the schema
            order = ExecutionOrder(**data)
            
            # 2. Simulate Exchange Latency (50ms - 200ms is standard)
            print("[ExecutionAgent] Routing to Exchange API...")
            await asyncio.sleep(random.uniform(0.05, 0.2))
            
            # 3. Simulate Slippage 
            # You NEVER get the exact price you saw on the screen. The market moves.
            # We simulate a random price slip between -0.1% and +0.1%
            quoted_price = 65000.00 # Hardcoded for this simulation
            slippage_multiplier = random.uniform(0.999, 1.001)
            actual_fill_price = quoted_price * slippage_multiplier
            
            # 4. Confirm the Fill
            print(f"[ExecutionAgent] ORDER FILLED: {order.action} {order.quantity} {order.ticker} @ ${actual_fill_price:.2f}")
            
            # 5. Broadcast the Receipt to the Ledger
            receipt = {
                "ticker": order.ticker,
                "action": order.action,
                "quantity": order.quantity,
                "quoted_price": quoted_price,
                "fill_price": round(actual_fill_price, 2),
                "status": "FILLED"
            }
            await self.publish_event("ORDER_FILL_EVENT", receipt)

        except Exception as e:
            print(f"[ExecutionAgent] FATAL - Order Execution Failed: {e}")

async def main():
    agent = ExecutionAgent()
    await agent.listen("APPROVED_TRADE_EVENT", agent.execute_order)

if __name__ == "__main__":
    asyncio.run(main())
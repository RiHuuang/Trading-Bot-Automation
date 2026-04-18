import asyncio
import random

from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.schemas import ExecutionOrder, OrderFill


class ExecutionAgent(BaseAgent):
    def __init__(self):
        super().__init__("ExecutionAgent")
        self.processed_orders: set[str] = set()

    async def execute_order(self, data):
        try:
            order = ExecutionOrder(**data)
        except ValidationError as exc:
            print(f"[ExecutionAgent] Invalid order payload: {exc}")
            return

        if order.order_id in self.processed_orders:
            print(f"[ExecutionAgent] Skipping duplicate order {order.order_id}.")
            return

        self.processed_orders.add(order.order_id)
        print(
            f"[ExecutionAgent] Executing {order.action} {order.quantity} {order.ticker} "
            f"(paper={order.paper_trade})."
        )

        await asyncio.sleep(random.uniform(0.05, 0.2))

        slippage_bps = random.uniform(-4.0, 8.0)
        actual_fill_price = order.quoted_price * (1 + (slippage_bps / 10000.0))
        receipt = OrderFill(
            order_id=order.order_id,
            proposal_id=order.proposal_id,
            decision_id=order.decision_id,
            ticker=order.ticker,
            action=order.action,
            quantity=order.quantity,
            quoted_price=order.quoted_price,
            fill_price=round(actual_fill_price, 2),
            slippage_bps=round(slippage_bps, 2),
            status="FILLED",
            paper_trade=order.paper_trade,
            timestamp=self.utc_timestamp(),
            reasoning=order.reasoning,
        )
        print(
            f"[ExecutionAgent] Order filled at {receipt.fill_price:.2f} "
            f"with {receipt.slippage_bps:.2f} bps slippage."
        )
        await self.publish_event("ORDER_FILL_EVENT", receipt.model_dump())


async def main():
    agent = ExecutionAgent()
    await agent.listen("APPROVED_TRADE_EVENT", agent.execute_order)


if __name__ == "__main__":
    asyncio.run(main())

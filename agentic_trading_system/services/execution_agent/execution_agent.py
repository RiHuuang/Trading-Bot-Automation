import asyncio
import os
import random

from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.binance_client import BinanceRESTClient, load_binance_config
from core.schemas import ExecutionOrder, OrderFill


class ExecutionAgent(BaseAgent):
    def __init__(self):
        super().__init__("ExecutionAgent")
        self.processed_orders: set[str] = set()
        self.binance_config = load_binance_config()
        self.binance = BinanceRESTClient(self.binance_config)
        self.validate_only = os.getenv("BINANCE_VALIDATE_ONLY", "true").lower() == "true"

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

        broker_response: dict | None = None
        if not order.paper_trade:
            try:
                broker_response = await asyncio.to_thread(
                    self.binance.place_market_order,
                    symbol=self.binance_config.symbol,
                    side=order.action,
                    quantity=order.quantity,
                    client_order_id=order.order_id,
                    validate_only=self.validate_only,
                )
            except Exception as exc:
                print(f"[ExecutionAgent] Binance order request failed: {exc}")
                return

        await asyncio.sleep(random.uniform(0.05, 0.2))

        slippage_bps = random.uniform(-4.0, 8.0)
        actual_fill_price = order.quoted_price * (1 + (slippage_bps / 10000.0))
        fill_price = round(actual_fill_price, 2)
        if broker_response and broker_response.get("fills"):
            weighted_fill = sum(float(fill["price"]) * float(fill["qty"]) for fill in broker_response["fills"])
            total_qty = sum(float(fill["qty"]) for fill in broker_response["fills"])
            if total_qty > 0:
                fill_price = round(weighted_fill / total_qty, 2)

        receipt = OrderFill(
            order_id=order.order_id,
            proposal_id=order.proposal_id,
            decision_id=order.decision_id,
            ticker=order.ticker,
            action=order.action,
            quantity=order.quantity,
            quoted_price=order.quoted_price,
            fill_price=fill_price,
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

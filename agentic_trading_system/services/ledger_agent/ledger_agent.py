import asyncio
import os

import psycopg2
from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.schemas import OrderFill


class LedgerAgent(BaseAgent):
    def __init__(self):
        super().__init__("LedgerAgent")
        self.db_conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            database=os.getenv("POSTGRES_DB", "trading_ledger"),
            user=os.getenv("POSTGRES_USER", "trader"),
            password=os.getenv("POSTGRES_PASSWORD", "securepassword"),
        )
        self._init_db()

    def _init_db(self):
        with self.db_conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    order_id VARCHAR(64) UNIQUE,
                    proposal_id VARCHAR(64),
                    decision_id VARCHAR(64),
                    ticker VARCHAR(10),
                    action VARCHAR(10),
                    quantity NUMERIC,
                    quoted_price NUMERIC,
                    fill_price NUMERIC,
                    slippage_bps NUMERIC,
                    status VARCHAR(20),
                    paper_trade BOOLEAN DEFAULT TRUE,
                    reasoning TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self.db_conn.commit()
        print("[LedgerAgent] Database connection verified.")

    async def record_trade(self, data):
        try:
            fill = OrderFill(**data)
        except ValidationError as exc:
            print(f"[LedgerAgent] Invalid fill payload: {exc}")
            return

        try:
            with self.db_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trades (
                        order_id,
                        proposal_id,
                        decision_id,
                        ticker,
                        action,
                        quantity,
                        quoted_price,
                        fill_price,
                        slippage_bps,
                        status,
                        paper_trade,
                        reasoning
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (order_id) DO NOTHING
                    """,
                    (
                        fill.order_id,
                        fill.proposal_id,
                        fill.decision_id,
                        fill.ticker,
                        fill.action,
                        fill.quantity,
                        fill.quoted_price,
                        fill.fill_price,
                        fill.slippage_bps,
                        fill.status,
                        fill.paper_trade,
                        fill.reasoning,
                    ),
                )
                self.db_conn.commit()
            print(f"[LedgerAgent] Recorded fill for order {fill.order_id}.")
        except Exception as exc:
            print(f"[LedgerAgent] DB error: {exc}")
            self.db_conn.rollback()


async def main():
    agent = LedgerAgent()
    await agent.listen("ORDER_FILL_EVENT", agent.record_trade)


if __name__ == "__main__":
    asyncio.run(main())

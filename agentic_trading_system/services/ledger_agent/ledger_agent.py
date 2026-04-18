import asyncio
import os
import psycopg2
from core.base_agent import BaseAgent

class LedgerAgent(BaseAgent):
    def __init__(self):
        super().__init__("LedgerAgent")
        # Connect to the PostgreSQL container using the credentials from docker-compose
        self.db_conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            database=os.getenv("POSTGRES_DB", "trading_ledger"),
            user=os.getenv("POSTGRES_USER", "trader"),
            password=os.getenv("POSTGRES_PASSWORD", "securepassword")
        )
        self._init_db()

    def _init_db(self):
        """Ensures the immutable ledger table exists before we start listening."""
        with self.db_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10),
                    action VARCHAR(10),
                    quantity NUMERIC,
                    quoted_price NUMERIC,
                    fill_price NUMERIC,
                    status VARCHAR(20),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.db_conn.commit()
        print("[LedgerAgent] Database connection verified. Table secured.")

    async def record_trade(self, data):
        print(f"[LedgerAgent] Catching trade receipt for {data['action']} {data['ticker']}...")
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades (ticker, action, quantity, quoted_price, fill_price, status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (data['ticker'], data['action'], data['quantity'], data['quoted_price'], data['fill_price'], data['status']))
                self.db_conn.commit()
            print("[LedgerAgent] SECURED: Trade permanently logged to PostgreSQL.")
        except Exception as e:
            print(f"[LedgerAgent] FATAL DB ERROR: {e}")
            self.db_conn.rollback()

async def main():
    agent = LedgerAgent()
    await agent.listen("ORDER_FILL_EVENT", agent.record_trade)

if __name__ == "__main__":
    asyncio.run(main())
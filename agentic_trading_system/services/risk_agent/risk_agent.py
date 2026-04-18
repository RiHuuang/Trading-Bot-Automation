import asyncio
import os
import psycopg2
from core.base_agent import BaseAgent
from core.schemas import TradeProposal, ExecutionOrder

class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__("RiskAgent")
        # Hardcoded parameters for the prototype
        self.portfolio_balance_usd = 10000.00
        self.max_risk_per_trade_pct = 0.02
        self.min_confidence_threshold = 0.70
        self.current_btc_price = 65000.00 

        # 1. Connect the Shield to the Ledger
        print("[RiskAgent] Booting up and connecting to Ledger...")
        self.db_conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            database=os.getenv("POSTGRES_DB", "trading_ledger"),
            user=os.getenv("POSTGRES_USER", "trader"),
            password=os.getenv("POSTGRES_PASSWORD", "securepassword")
        )
        self._init_db()

    def _init_db(self):
        """Ensures the table exists in case RiskAgent boots faster than LedgerAgent."""
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

    def get_inventory(self, ticker):
        """Calculates exact coin holdings directly from the immutable ledger."""
        try:
            with self.db_conn.cursor() as cur:
                # Sum Buys as positive, Sells as negative
                cur.execute("""
                    SELECT COALESCE(SUM(CASE WHEN action='BUY' THEN quantity ELSE -quantity END), 0)
                    FROM trades
                    WHERE ticker=%s AND status='FILLED'
                """, (ticker,))
                result = cur.fetchone()
                return float(result[0]) if result else 0.0
        except Exception as e:
            print(f"[RiskAgent] FATAL DB Read Error: {e}")
            self.db_conn.rollback()
            return 0.0

    async def evaluate_proposal(self, data):
        print(f"\n[RiskAgent] Received Proposal: {data['action']} {data['ticker']} (Confidence: {data['confidence']})")
        
        try:
            proposal = TradeProposal(**data)
            
            if proposal.confidence < self.min_confidence_threshold:
                print(f"[RiskAgent] REJECTED: Confidence {proposal.confidence} is below threshold.")
                return

            if proposal.action == "HOLD":
                print("[RiskAgent] REJECTED: Action is HOLD.")
                return

            # 2. The Reality Check: Look at the database
            inventory = self.get_inventory(proposal.ticker)
            print(f"[RiskAgent] STATE: Current Inventory for {proposal.ticker} is {inventory} coins.")

            # 3. Sell Logic (Exit)
            if proposal.action == "SELL":
                # We use 0.00001 to account for floating-point database dust
                if inventory <= 0.00001:
                    print("[RiskAgent] REJECTED: We own 0 coins. Shorting is disabled.")
                    return
                else:
                    # We own the asset. Liquidate the entire position.
                    print(f"[RiskAgent] APPROVED EXIT: Liquidating entire position of {inventory} {proposal.ticker}.")
                    approved_order = ExecutionOrder(
                        ticker=proposal.ticker,
                        action="SELL",
                        quantity=inventory,
                        confidence=proposal.confidence,
                        reasoning=proposal.reasoning
                    )
                    await self.publish_event("APPROVED_TRADE_EVENT", approved_order.model_dump())
                    return

            # 4. Buy Logic (Entry)
            if proposal.action == "BUY":
                # Check if we already own it to prevent catching a falling knife
                if inventory > 0.00001:
                    print(f"[RiskAgent] REJECTED: We already hold {inventory} {proposal.ticker}. Waiting for a sell signal to avoid over-exposure.")
                    return

                # Proceed with standard entry sizing
                dollar_amount_to_risk = self.portfolio_balance_usd * self.max_risk_per_trade_pct
                quantity_to_buy = round(dollar_amount_to_risk / self.current_btc_price, 5)

                approved_order = ExecutionOrder(
                    ticker=proposal.ticker,
                    action="BUY",
                    quantity=quantity_to_buy,
                    confidence=proposal.confidence,
                    reasoning=proposal.reasoning
                )
                print(f"[RiskAgent] APPROVED ENTRY: Sizing position to {quantity_to_buy} {proposal.ticker}.")
                await self.publish_event("APPROVED_TRADE_EVENT", approved_order.model_dump())

        except Exception as e:
            print(f"[RiskAgent] FATAL ERROR parsing proposal: {e}")

async def main():
    agent = RiskAgent()
    await agent.listen("PROPOSAL_EVENT", agent.evaluate_proposal)

if __name__ == "__main__":
    asyncio.run(main())
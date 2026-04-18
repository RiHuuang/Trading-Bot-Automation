import asyncio
import os
from collections import defaultdict

import psycopg2
from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.binance_client import BinanceRESTClient, load_binance_config, load_symbol_map
from core.schemas import ExecutionOrder, JudgeDecision


class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__("RiskAgent")
        self.paper_trade = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.binance_config = load_binance_config()
        self.binance = BinanceRESTClient(self.binance_config)
        self.symbol_map = load_symbol_map()
        self.symbol_info_map = {
            base_asset: self.binance.get_symbol_info(venue_symbol)
            for base_asset, venue_symbol in self.symbol_map.items()
        }
        self.portfolio_balance_quote = float(os.getenv("PAPER_QUOTE_BALANCE", os.getenv("PAPER_USD_BALANCE", "10000")))
        self.max_risk_per_trade_pct = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "0.01"))
        self.max_position_notional_quote = float(os.getenv("MAX_POSITION_NOTIONAL", os.getenv("MAX_POSITION_NOTIONAL_USD", "2000")))
        self.min_confidence_threshold = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.70"))
        self.max_signal_age_seconds = int(os.getenv("MAX_SIGNAL_AGE_SECONDS", "180"))
        self.min_atr_fraction = float(os.getenv("MIN_ATR_FRACTION", "0.002"))
        self.max_spread_bps = float(os.getenv("MAX_LIVE_SPREAD_BPS", "12"))
        self.min_depth_notional = float(os.getenv("MIN_LIVE_DEPTH_NOTIONAL", "100000"))
        self.symbol_cooldown_seconds = int(os.getenv("SYMBOL_COOLDOWN_SECONDS", "1800"))
        self.max_daily_loss_pct = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))
        self.max_daily_trades = int(os.getenv("MAX_DAILY_TRADES", "8"))
        self.last_trade_at: dict[str, float] = defaultdict(float)

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

    def get_today_trade_count(self) -> int:
        try:
            with self.db_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM trades
                    WHERE timestamp::date = CURRENT_DATE AND status = 'FILLED'
                    """
                )
                result = cur.fetchone()
                return int(result[0]) if result else 0
        except Exception as exc:
            print(f"[RiskAgent] DB trade count error: {exc}")
            self.db_conn.rollback()
            return 0

    def get_today_realized_pnl(self) -> float:
        try:
            with self.db_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(
                        CASE
                            WHEN action = 'SELL' AND status = 'FILLED' THEN (fill_price - quoted_price) * quantity
                            WHEN action = 'BUY' AND status = 'FILLED' THEN (quoted_price - fill_price) * quantity
                            ELSE 0
                        END
                    ), 0)
                    FROM trades
                    WHERE timestamp::date = CURRENT_DATE
                    """
                )
                result = cur.fetchone()
                return float(result[0]) if result else 0.0
        except Exception as exc:
            print(f"[RiskAgent] DB pnl error: {exc}")
            self.db_conn.rollback()
            return 0.0

    def get_inventory(self, ticker: str) -> float:
        try:
            with self.db_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(
                        SUM(
                            CASE
                                WHEN action = 'BUY' AND status = 'FILLED' THEN quantity
                                WHEN action = 'SELL' AND status = 'FILLED' THEN -quantity
                                ELSE 0
                            END
                        ),
                        0
                    )
                    FROM trades
                    WHERE ticker = %s
                    """,
                    (ticker,),
                )
                result = cur.fetchone()
                return float(result[0]) if result else 0.0
        except Exception as exc:
            print(f"[RiskAgent] DB read error: {exc}")
            self.db_conn.rollback()
            return 0.0

    async def evaluate_decision(self, data):
        try:
            decision = JudgeDecision(**data)
        except ValidationError as exc:
            print(f"[RiskAgent] Invalid judge decision: {exc}")
            return

        print(f"[RiskAgent] Received {decision.verdict} for {decision.proposal.ticker}.")

        if decision.verdict != "APPROVE" or decision.approved_action is None:
            print("[RiskAgent] No executable trade from judge.")
            return

        if decision.confidence < self.min_confidence_threshold:
            print("[RiskAgent] Rejected: judge confidence below threshold.")
            return

        signal_age = self.utc_timestamp() - decision.proposal.market_snapshot.generated_at
        if signal_age > self.max_signal_age_seconds:
            print("[RiskAgent] Rejected: market snapshot is stale.")
            return

        today_trade_count = self.get_today_trade_count()
        if today_trade_count >= self.max_daily_trades:
            print("[RiskAgent] Rejected: max daily trades reached.")
            return

        today_realized_pnl = self.get_today_realized_pnl()
        if today_realized_pnl <= -(self.portfolio_balance_quote * self.max_daily_loss_pct):
            print("[RiskAgent] Rejected: max daily loss reached.")
            return

        if self.utc_timestamp() - self.last_trade_at[decision.proposal.ticker] < self.symbol_cooldown_seconds:
            print("[RiskAgent] Rejected: symbol cooldown active.")
            return

        snapshot = decision.proposal.market_snapshot
        venue_symbol = self.symbol_map.get(decision.proposal.ticker, self.binance_config.symbol)
        symbol_info = self.symbol_info_map.get(decision.proposal.ticker)
        if snapshot.atr_14 / snapshot.close < self.min_atr_fraction:
            print("[RiskAgent] Rejected: ATR too low, move likely not worth trading.")
            return

        try:
            depth = self.binance.get_depth(symbol=venue_symbol, limit=20)
        except Exception as exc:
            print(f"[RiskAgent] Rejected: failed to fetch live depth: {exc}")
            return

        if depth.spread_bps > self.max_spread_bps:
            print("[RiskAgent] Rejected: live spread too wide.")
            return

        if min(depth.bid_notional_top_n, depth.ask_notional_top_n) < self.min_depth_notional:
            print("[RiskAgent] Rejected: live order book too thin.")
            return

        inventory = self.get_inventory(decision.proposal.ticker)
        stop_distance = max(snapshot.atr_14 * 2.0, snapshot.close * 0.003)
        risk_budget = self.portfolio_balance_quote * self.max_risk_per_trade_pct
        size_from_risk = risk_budget / stop_distance
        size_from_notional = self.max_position_notional_quote / snapshot.close
        quantity = min(size_from_risk, size_from_notional)
        if symbol_info and symbol_info.step_size and symbol_info.step_size > 0:
            quantity = max(
                round((quantity // symbol_info.step_size) * symbol_info.step_size, 8),
                0.0,
            )
        else:
            quantity = round(quantity, 6)

        if quantity <= 0:
            print("[RiskAgent] Rejected: quantity computed as zero.")
            return

        if symbol_info and symbol_info.min_notional and (quantity * snapshot.close) < symbol_info.min_notional:
            print("[RiskAgent] Rejected: quantity is below Binance min notional.")
            return

        if decision.approved_action == "BUY" and inventory > 0.000001:
            print("[RiskAgent] Rejected: existing long inventory, no pyramiding.")
            return

        if decision.approved_action == "SELL":
            if inventory <= 0.000001:
                print("[RiskAgent] Rejected: no inventory to sell.")
                return
            quantity = round(inventory, 6)

        stop_loss = snapshot.close - stop_distance if decision.approved_action == "BUY" else snapshot.close + stop_distance
        take_profit = snapshot.close + (stop_distance * 2.0) if decision.approved_action == "BUY" else snapshot.close - (stop_distance * 2.0)

        order = ExecutionOrder(
            order_id=self.new_id("order"),
            proposal_id=decision.proposal_id,
            decision_id=decision.decision_id,
            ticker=decision.proposal.ticker,
            action=decision.approved_action,
            quantity=quantity,
            quoted_price=snapshot.close,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            created_at=self.utc_timestamp(),
            paper_trade=self.paper_trade,
        )
        print(
            f"[RiskAgent] Approved {order.action} {order.quantity} {order.ticker} "
            f"at {order.quoted_price:.2f}."
        )
        self.last_trade_at[order.ticker] = self.utc_timestamp()
        await self.publish_event("APPROVED_TRADE_EVENT", order.model_dump())


async def main():
    agent = RiskAgent()
    await agent.listen("JUDGE_DECISION_EVENT", agent.evaluate_decision)


if __name__ == "__main__":
    asyncio.run(main())

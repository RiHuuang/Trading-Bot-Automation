import asyncio
import json
import os
from collections import defaultdict, deque

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.binance_client import BinanceRESTClient, load_binance_config, load_symbol_map
from core.indicator_engine import build_market_snapshot, evaluate_models, evaluate_snapshot
from core.market_context import fetch_timeframe_context, format_timeframe_context
from core.schemas import MarketSnapshot, MarketTick, TradeProposal, TradeProposalDraft

load_dotenv()


class SignalAgent(BaseAgent):
    def __init__(self):
        super().__init__("SignalAgent")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = os.getenv("SIGNAL_MODEL", "gemini-2.5-flash")
        self.candle_buffers: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=250))
        self.minimum_bars = int(os.getenv("SIGNAL_MIN_BARS", "120"))
        self.symbol_map = load_symbol_map()
        self.target_symbols = set(self.symbol_map.keys())
        self.binance_config = load_binance_config()
        self.binance = BinanceRESTClient(self.binance_config)
        self.venue_symbol_map = dict(self.symbol_map)

    async def process_tick(self, data):
        try:
            tick = MarketTick(**data)
        except ValidationError as exc:
            print(f"[SignalAgent] Dropping invalid tick: {exc}")
            return

        if tick.ticker not in self.target_symbols:
            return

        buffer = self.candle_buffers[tick.ticker]
        buffer.append(tick.model_dump())
        print(f"[SignalAgent] Stored {len(buffer)}/{self.minimum_bars} candles for {tick.ticker}.")

        if len(buffer) < self.minimum_bars:
            return

        snapshot = self.build_snapshot(tick.ticker)
        if snapshot is None:
            return

        await self.analyze_market(snapshot)

    async def update_universe(self, data):
        symbols = data.get("symbols", [])
        ranked_assets = {
            item["symbol"].replace("USDT", "").replace("IDR", "")
            for item in symbols
            if isinstance(item, dict) and "symbol" in item
        }
        if ranked_assets:
            self.target_symbols = ranked_assets.intersection(set(self.symbol_map.keys())) or self.target_symbols
            print(f"[SignalAgent] Updated live universe: {sorted(self.target_symbols)}")

    def build_snapshot(self, ticker: str) -> MarketSnapshot | None:
        df = pd.DataFrame(self.candle_buffers[ticker])
        try:
            snapshot = build_market_snapshot(df)
            if snapshot is None:
                print("[SignalAgent] Indicator set still warming up.")
            return snapshot
        except Exception as exc:
            print(f"[SignalAgent] Failed to build market snapshot: {exc}")
            return None

    def get_higher_timeframe_context(self, ticker: str) -> str:
        symbol = self.venue_symbol_map.get(ticker)
        if not symbol:
            return "No higher timeframe context available."
        contexts = []
        for interval in ("1h", "4h"):
            try:
                context = fetch_timeframe_context(
                    client=self.binance,
                    symbol=symbol,
                    base_asset=ticker,
                    interval=interval,
                )
            except Exception as exc:
                print(f"[SignalAgent] Failed loading {interval} context for {ticker}: {exc}")
                context = None
            if context is not None:
                contexts.append(context)
        return format_timeframe_context(contexts)

    async def analyze_market(self, snapshot: MarketSnapshot):
        model_decisions = evaluate_models(snapshot)
        baseline = evaluate_snapshot(snapshot)
        higher_timeframes = self.get_higher_timeframe_context(snapshot.ticker)
        prompt = f"""
You are the author agent for a systematic crypto trading desk.

Your job is to propose one action: BUY, SELL, or HOLD.
Use the indicator snapshot below and produce a cautious, high-quality JSON TradeProposal.
Only issue BUY or SELL when multiple indicators align. Otherwise return HOLD.

Indicator snapshot:
- Price close: {snapshot.close:.2f}
- ATR(14): {snapshot.atr_14:.4f}
- ADX(14): {snapshot.adx_14:.2f}
- RSI(14): {snapshot.rsi_14:.2f}
- Stochastic %K/%D: {snapshot.stoch_k:.2f} / {snapshot.stoch_d:.2f}
- MACD / Signal / Histogram: {snapshot.macd:.4f} / {snapshot.macd_signal:.4f} / {snapshot.macd_histogram:.4f}
- EMA(20) / EMA(50): {snapshot.ema_20:.2f} / {snapshot.ema_50:.2f}
- Bollinger upper / mid / lower / bandwidth: {snapshot.bollinger_upper:.2f} / {snapshot.bollinger_mid:.2f} / {snapshot.bollinger_lower:.2f} / {snapshot.bollinger_bandwidth:.2f}
- 5-bar return: {snapshot.returns_5:.4%}
- 20-bar return: {snapshot.returns_20:.4%}
- 20-bar realized volatility: {snapshot.volatility_20:.4%}
- Last candle volume: {snapshot.volume:.4f}

Deterministic baseline:
- Suggested action: {baseline.action}
- Baseline confidence: {baseline.confidence:.2f}
- Summary: {baseline.summary}
- Reasons: {", ".join(baseline.reasons)}
- Sub-model opinions: {" | ".join(f"{item.model_name}:{item.action}:{item.confidence:.2f}" for item in model_decisions)}

Higher timeframe context:
{higher_timeframes}

Decision policy:
- Prefer trades where 15m signals align with 1h and ideally 4h context.
- Trend strength is stronger when ADX > 20.
- Uptrend bias exists when EMA20 > EMA50 and MACD histogram is positive.
- Downtrend bias exists when EMA20 < EMA50 and MACD histogram is negative.
- Mean reversion conditions are stronger near Bollinger extremes with RSI/Stochastic confirmation.
- If you disagree with the deterministic baseline, explain exactly why.
- If indicators conflict or confidence is below 0.60, output HOLD.
- Confidence must be calibrated between 0 and 1.
- Thesis and invalidation must be concise and concrete.
        """

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TradeProposalDraft,
                    temperature=0.1,
                ),
            )
            draft = TradeProposalDraft(**json.loads(response.text))
            proposal_data = draft.model_dump()
            proposal_data["proposal_id"] = self.new_id("proposal")
            proposal_data["source_agent"] = self.agent_name
            proposal_data["model_name"] = self.model_name
            proposal_data["generated_at"] = self.utc_timestamp()
            proposal_data["ticker"] = snapshot.ticker
            proposal_data["market_snapshot"] = snapshot.model_dump()
            proposal = TradeProposal(**proposal_data)
            print(
                f"[SignalAgent] Proposed {proposal.action} "
                f"with confidence {proposal.confidence:.2f}."
            )
            await self.publish_event("SIGNAL_PROPOSAL_EVENT", proposal.model_dump())
        except Exception as exc:
            print(f"[SignalAgent] LLM generation failed: {exc}")
            await asyncio.sleep(5)

    async def run(self):
        await self.pubsub.subscribe("TICK_EVENT", "SCOUT_UNIVERSE_EVENT")
        print("[SignalAgent] Listening on TICK_EVENT and SCOUT_UNIVERSE_EVENT...")

        async for message in self.pubsub.listen():
            if message["type"] != "message":
                continue

            channel = message["channel"]
            try:
                data = json.loads(message["data"])
            except json.JSONDecodeError as exc:
                print(f"[SignalAgent] Dropping malformed payload: {exc}")
                continue

            if channel == "TICK_EVENT":
                await self.process_tick(data)
            elif channel == "SCOUT_UNIVERSE_EVENT":
                await self.update_universe(data)


async def main():
    agent = SignalAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

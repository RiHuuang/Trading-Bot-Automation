import asyncio
import json
import os
from collections import deque

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.indicator_engine import build_market_snapshot, evaluate_models, evaluate_snapshot
from core.schemas import MarketSnapshot, MarketTick, TradeProposal, TradeProposalDraft

load_dotenv()


class SignalAgent(BaseAgent):
    def __init__(self):
        super().__init__("SignalAgent")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = os.getenv("SIGNAL_MODEL", "gemini-2.5-flash")
        self.candle_buffer: deque[dict] = deque(maxlen=250)
        self.minimum_bars = int(os.getenv("SIGNAL_MIN_BARS", "120"))

    async def process_tick(self, data):
        try:
            tick = MarketTick(**data)
        except ValidationError as exc:
            print(f"[SignalAgent] Dropping invalid tick: {exc}")
            return

        self.candle_buffer.append(tick.model_dump())
        print(f"[SignalAgent] Stored {len(self.candle_buffer)}/{self.minimum_bars} candles.")

        if len(self.candle_buffer) < self.minimum_bars:
            return

        snapshot = self.build_snapshot()
        if snapshot is None:
            return

        await self.analyze_market(snapshot)

    def build_snapshot(self) -> MarketSnapshot | None:
        df = pd.DataFrame(self.candle_buffer)
        try:
            snapshot = build_market_snapshot(df)
            if snapshot is None:
                print("[SignalAgent] Indicator set still warming up.")
            return snapshot
        except Exception as exc:
            print(f"[SignalAgent] Failed to build market snapshot: {exc}")
            return None

    async def analyze_market(self, snapshot: MarketSnapshot):
        model_decisions = evaluate_models(snapshot)
        baseline = evaluate_snapshot(snapshot)
        prompt = f"""
You are the author agent for a BTC systematic trading desk.

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

Decision policy:
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


async def main():
    agent = SignalAgent()
    await agent.listen("TICK_EVENT", agent.process_tick)


if __name__ == "__main__":
    asyncio.run(main())

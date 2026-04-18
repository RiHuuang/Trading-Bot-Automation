import asyncio
import json
import os
from collections import deque

import pandas as pd
import ta
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.schemas import MarketSnapshot, MarketTick, TradeProposal, TradeProposalDraft

load_dotenv()


class SignalAgent(BaseAgent):
    def __init__(self):
        super().__init__("SignalAgent")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = os.getenv("SIGNAL_MODEL", "gemini-2.5-flash")
        self.candle_buffer: deque[dict] = deque(maxlen=250)
        self.minimum_bars = 60

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
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]

            df["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_histogram"] = macd.macd_diff()
            bollinger = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            df["bb_upper"] = bollinger.bollinger_hband()
            df["bb_lower"] = bollinger.bollinger_lband()
            df["bb_mid"] = bollinger.bollinger_mavg()
            df["bb_width"] = bollinger.bollinger_wband()
            df["atr_14"] = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()
            df["adx_14"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()
            stoch = ta.momentum.StochasticOscillator(
                high=high, low=low, close=close, window=14, smooth_window=3
            )
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()
            df["ema_20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
            df["ema_50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
            df["returns_5"] = close.pct_change(periods=5)
            df["returns_20"] = close.pct_change(periods=20)
            df["volatility_20"] = close.pct_change().rolling(window=20).std()

            latest = df.iloc[-1]
            if latest.isna().any():
                print("[SignalAgent] Indicator set still warming up.")
                return None

            return MarketSnapshot(
                ticker=df.iloc[-1]["ticker"],
                interval=df.iloc[-1]["interval"],
                generated_at=float(df.iloc[-1]["candle_close_timestamp"]),
                close=float(latest["close"]),
                atr_14=float(latest["atr_14"]),
                adx_14=float(latest["adx_14"]),
                rsi_14=float(latest["rsi_14"]),
                stoch_k=float(latest["stoch_k"]),
                stoch_d=float(latest["stoch_d"]),
                macd=float(latest["macd"]),
                macd_signal=float(latest["macd_signal"]),
                macd_histogram=float(latest["macd_histogram"]),
                ema_20=float(latest["ema_20"]),
                ema_50=float(latest["ema_50"]),
                bollinger_upper=float(latest["bb_upper"]),
                bollinger_lower=float(latest["bb_lower"]),
                bollinger_mid=float(latest["bb_mid"]),
                bollinger_bandwidth=float(latest["bb_width"]),
                returns_5=float(latest["returns_5"]),
                returns_20=float(latest["returns_20"]),
                volatility_20=float(latest["volatility_20"]),
                volume=float(latest["volume"]),
            )
        except Exception as exc:
            print(f"[SignalAgent] Failed to build market snapshot: {exc}")
            return None

    async def analyze_market(self, snapshot: MarketSnapshot):
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

Decision policy:
- Trend strength is stronger when ADX > 20.
- Uptrend bias exists when EMA20 > EMA50 and MACD histogram is positive.
- Downtrend bias exists when EMA20 < EMA50 and MACD histogram is negative.
- Mean reversion conditions are stronger near Bollinger extremes with RSI/Stochastic confirmation.
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

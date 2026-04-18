import asyncio
import math
import os

from core.base_agent import BaseAgent
from core.binance_client import BinanceRESTClient, infer_quote_asset, load_binance_config, split_base_quote
from core.indicator_engine import build_market_snapshot, evaluate_snapshot
from core.market_context import build_dataframe_from_klines, fetch_timeframe_context


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class ScoutAgent(BaseAgent):
    def __init__(self):
        super().__init__("ScoutAgent")
        self.binance_config = load_binance_config()
        self.binance = BinanceRESTClient(self.binance_config)
        self.quote_asset = os.getenv("SCOUT_QUOTE_ASSET", infer_quote_asset(self.binance_config.symbol)).upper()
        self.max_symbols = int(os.getenv("SCOUT_MAX_SYMBOLS", "12"))
        self.scan_limit = int(os.getenv("SCOUT_SCAN_LIMIT", "100"))
        self.min_quote_volume = float(os.getenv("SCOUT_MIN_QUOTE_VOLUME", "50000000"))
        self.min_depth_notional = float(os.getenv("SCOUT_MIN_DEPTH_NOTIONAL", "150000"))
        self.max_spread_bps = float(os.getenv("SCOUT_MAX_SPREAD_BPS", "12"))
        self.min_candidate_confidence = float(os.getenv("SCOUT_MIN_CONFIDENCE", "0.58"))
        self.sleep_seconds = int(os.getenv("SCOUT_INTERVAL_SECONDS", "900"))
        self.prefer_long_only = os.getenv("SCOUT_LONG_ONLY", "true").lower() == "true"
        self.primary_interval_limit = int(os.getenv("SCOUT_PRIMARY_INTERVAL_BARS", "260"))
        configured_contexts = os.getenv("SCOUT_CONTEXT_INTERVALS", "1h,4h")
        self.context_intervals = [item.strip() for item in configured_contexts.split(",") if item.strip()]
        self.excluded_symbols = {"USDCUSDT", "FDUSDUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT"}
        self.excluded_suffixes = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")

    def is_candidate_symbol(self, symbol: str) -> bool:
        if not symbol.endswith(self.quote_asset):
            return False
        if symbol in self.excluded_symbols:
            return False
        if any(symbol.endswith(suffix) for suffix in self.excluded_suffixes):
            return False
        return True

    def select_scan_universe(self) -> list[dict]:
        tickers = self.binance.get_24h_tickers()
        candidates = []
        for ticker in tickers:
            symbol = ticker["symbol"].upper()
            if not self.is_candidate_symbol(symbol):
                continue
            quote_volume = float(ticker.get("quoteVolume", 0.0))
            trade_count = int(ticker.get("count", 0))
            if quote_volume < self.min_quote_volume or trade_count < 1000:
                continue
            candidates.append(ticker)
        candidates.sort(key=lambda item: float(item.get("quoteVolume", 0.0)), reverse=True)
        return candidates[: self.scan_limit]

    def compute_profit_score(self, ticker: dict) -> dict | None:
        symbol = ticker["symbol"].upper()
        base_asset, _ = split_base_quote(symbol)
        try:
            depth = self.binance.get_depth(symbol=symbol, limit=10)
        except Exception:
            return None

        depth_score = min(depth.bid_notional_top_n, depth.ask_notional_top_n)
        if depth.spread_bps > self.max_spread_bps or depth_score < self.min_depth_notional:
            return None

        try:
            raw = self.binance.get_klines(
                symbol=symbol,
                interval=self.binance_config.interval,
                limit=self.primary_interval_limit,
            )
        except Exception:
            return None

        snapshot = build_market_snapshot(
            build_dataframe_from_klines(raw=raw, base_asset=base_asset, interval=self.binance_config.interval)
        )
        if snapshot is None:
            return None

        ensemble = evaluate_snapshot(snapshot)
        if ensemble.action == "HOLD" or ensemble.confidence < self.min_candidate_confidence:
            return None
        if self.prefer_long_only and ensemble.action != "BUY":
            return None

        aligned_contexts = 0
        disagreeing_contexts = 0
        context_summaries: list[dict] = []
        for interval in self.context_intervals:
            try:
                context = fetch_timeframe_context(
                    client=self.binance,
                    symbol=symbol,
                    base_asset=base_asset,
                    interval=interval,
                    limit=220,
                )
            except Exception:
                context = None
            if context is None:
                continue
            context_summaries.append(
                {
                    "interval": interval,
                    "action": context.ensemble_action,
                    "confidence": round(context.ensemble_confidence, 3),
                }
            )
            if context.ensemble_action == ensemble.action:
                aligned_contexts += 1
            elif context.ensemble_action != "HOLD":
                disagreeing_contexts += 1

        if aligned_contexts == 0 and self.context_intervals:
            return None

        atr_fraction = snapshot.atr_14 / snapshot.close if snapshot.close > 0 else 0.0
        volatility_component = clamp((atr_fraction - 0.0035) / 0.012, 0.0, 1.0)
        realized_vol_component = clamp(snapshot.volatility_20 / 0.035, 0.0, 1.0)
        trend_component = clamp(snapshot.adx_14 / 35.0, 0.0, 1.0)
        spread_component = clamp(1.0 - (depth.spread_bps / max(self.max_spread_bps, 1.0)), 0.0, 1.0)
        liquidity_component = clamp(math.log10(max(depth_score, 1.0)) / 7.0, 0.0, 1.0)
        participation_component = clamp(abs(snapshot.taker_buy_ratio - 0.5) * 2.2, 0.0, 1.0)
        flow_component = clamp(snapshot.volume_zscore_20 / 3.0, 0.0, 1.0)
        alignment_component = clamp((aligned_contexts - disagreeing_contexts + len(self.context_intervals)) / (2 * max(len(self.context_intervals), 1)), 0.0, 1.0)

        score = (
            ensemble.confidence * 32.0
            + alignment_component * 20.0
            + trend_component * 12.0
            + volatility_component * 10.0
            + realized_vol_component * 8.0
            + spread_component * 8.0
            + liquidity_component * 6.0
            + participation_component * 2.0
            + flow_component * 2.0
        )

        return {
            "symbol": symbol,
            "base_asset": base_asset,
            "direction": ensemble.action,
            "score": round(score, 2),
            "confidence": round(ensemble.confidence, 3),
            "aligned_contexts": aligned_contexts,
            "disagreeing_contexts": disagreeing_contexts,
            "quote_volume": round(float(ticker.get("quoteVolume", 0.0)), 2),
            "trade_count": int(ticker.get("count", 0)),
            "spread_bps": round(depth.spread_bps, 4),
            "depth_notional_top10": round(depth_score, 2),
            "atr_fraction": round(atr_fraction, 5),
            "volatility_20": round(snapshot.volatility_20, 5),
            "adx_14": round(snapshot.adx_14, 2),
            "returns_20": round(snapshot.returns_20, 4),
            "volume_zscore_20": round(snapshot.volume_zscore_20, 3),
            "contexts": context_summaries,
            "summary": ensemble.summary,
        }

    def rank_symbols(self) -> list[dict]:
        ranked = []
        for ticker in self.select_scan_universe():
            candidate = self.compute_profit_score(ticker)
            if candidate is not None:
                ranked.append(candidate)
        ranked.sort(
            key=lambda item: (
                item["score"],
                item["confidence"],
                item["aligned_contexts"],
                item["quote_volume"],
            ),
            reverse=True,
        )
        return ranked[: self.max_symbols]

    async def publish_universe(self):
        ranked = await asyncio.to_thread(self.rank_symbols)
        payload = {
            "generated_at": self.utc_timestamp(),
            "quote_asset": self.quote_asset,
            "scan_limit": self.scan_limit,
            "symbols": ranked,
        }
        print(f"[ScoutAgent] Ranked {len(ranked)} profit-seeking symbols from top {self.scan_limit} candidates.")
        await self.publish_event("SCOUT_UNIVERSE_EVENT", payload)

    async def run(self):
        while True:
            try:
                await self.publish_universe()
            except Exception as exc:
                print(f"[ScoutAgent] Ranking failed: {exc}")
            await asyncio.sleep(self.sleep_seconds)


async def main():
    agent = ScoutAgent()
    await asyncio.gather(agent.heartbeat_loop(), agent.run())


if __name__ == "__main__":
    asyncio.run(main())

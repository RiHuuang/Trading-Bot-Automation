import asyncio
import json
import os

from core.base_agent import BaseAgent
from core.binance_client import BinanceRESTClient, load_binance_config


class ScoutAgent(BaseAgent):
    def __init__(self):
        super().__init__("ScoutAgent")
        self.binance_config = load_binance_config()
        self.binance = BinanceRESTClient(self.binance_config)
        self.quote_asset = os.getenv("SCOUT_QUOTE_ASSET", "USDT").upper()
        self.max_symbols = int(os.getenv("SCOUT_MAX_SYMBOLS", "8"))
        self.min_quote_volume = float(os.getenv("SCOUT_MIN_QUOTE_VOLUME", "50000000"))
        self.max_spread_bps = float(os.getenv("SCOUT_MAX_SPREAD_BPS", "12"))
        self.sleep_seconds = int(os.getenv("SCOUT_INTERVAL_SECONDS", "900"))

    def rank_symbols(self) -> list[dict]:
        tickers = self.binance.get_24h_tickers()
        candidates = []
        for ticker in tickers:
            symbol = ticker["symbol"]
            if not symbol.endswith(self.quote_asset):
                continue
            if ticker.get("symbol") in {"USDCUSDT", "FDUSDUSDT", "BUSDUSDT"}:
                continue
            quote_volume = float(ticker.get("quoteVolume", 0.0))
            trade_count = int(ticker.get("count", 0))
            if quote_volume < self.min_quote_volume or trade_count < 1000:
                continue
            try:
                depth = self.binance.get_depth(symbol=symbol, limit=10)
            except Exception:
                continue
            if depth.spread_bps > self.max_spread_bps:
                continue
            momentum = abs(float(ticker.get("priceChangePercent", 0.0)))
            depth_score = min(depth.bid_notional_top_n, depth.ask_notional_top_n)
            score = (quote_volume / 1_000_000) * 0.45 + trade_count * 0.0002 + depth_score / 1_000_000 * 0.35 + momentum * 3.0 - depth.spread_bps * 4.0
            candidates.append(
                {
                    "symbol": symbol,
                    "quote_volume": round(quote_volume, 2),
                    "trade_count": trade_count,
                    "spread_bps": round(depth.spread_bps, 4),
                    "depth_notional_top10": round(depth_score, 2),
                    "momentum_pct": round(momentum, 3),
                    "score": round(score, 2),
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[: self.max_symbols]

    async def publish_universe(self):
        ranked = await asyncio.to_thread(self.rank_symbols)
        payload = {
            "generated_at": self.utc_timestamp(),
            "quote_asset": self.quote_asset,
            "symbols": ranked,
        }
        print(f"[ScoutAgent] Ranked {len(ranked)} liquid symbols.")
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
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

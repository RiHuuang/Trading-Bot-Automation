import asyncio
import json
import time

import websockets

from core.base_agent import BaseAgent
from core.binance_client import BinanceRESTClient, load_binance_config, load_symbol_map
from core.schemas import MarketTick


class IngestAgent(BaseAgent):
    def __init__(self):
        super().__init__("IngestAgent")
        self.config = load_binance_config()
        self.symbol_map = load_symbol_map()
        self.default_symbols = sorted(set(self.symbol_map.values()))
        self.target_symbols = list(self.default_symbols)
        self.binance = BinanceRESTClient(self.config)
        self.symbol_infos: dict[str, object] = {}

    def build_stream_uri(self, symbols: list[str]) -> str:
        stream_names = [f"{symbol.lower()}@kline_{self.config.interval}" for symbol in symbols]
        if len(stream_names) == 1:
            return f"{self.config.market_ws_base}/{stream_names[0]}"
        combined_base = self.config.market_ws_base[:-3] if self.config.market_ws_base.endswith("/ws") else self.config.market_ws_base
        streams = "/".join(stream_names)
        return f"{combined_base}/stream?streams={streams}"

    def update_target_symbols(self, data: dict):
        symbols = [
            item.get("symbol", "").upper()
            for item in data.get("symbols", [])
            if isinstance(item, dict) and item.get("symbol")
        ]
        next_symbols = sorted(set(symbols)) or self.default_symbols
        if next_symbols != self.target_symbols:
            self.target_symbols = next_symbols
            print(f"[IngestAgent] Updated live universe: {', '.join(self.target_symbols)}")

    def ensure_symbol_metadata(self, symbols: list[str]):
        for venue_symbol in symbols:
            key = venue_symbol.lower()
            if key in self.symbol_infos:
                continue
            try:
                self.symbol_infos[key] = self.binance.get_symbol_info(venue_symbol)
            except Exception as exc:
                print(f"[IngestAgent] Could not load symbol metadata for {venue_symbol}: {exc}")

    async def listen_for_universe(self):
        await self.pubsub.subscribe("SCOUT_UNIVERSE_EVENT")
        async for message in self.pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                data = json.loads(message["data"])
            except json.JSONDecodeError as exc:
                print(f"[IngestAgent] Dropping malformed universe payload: {exc}")
                continue
            self.update_target_symbols(data)

    async def stream_loop(self):
        while True:
            active_symbols = list(self.target_symbols or self.default_symbols)
            self.ensure_symbol_metadata(active_symbols)
            uri = self.build_stream_uri(active_symbols)
            print(f"[IngestAgent] Connecting to Binance WebSocket: {uri}")
            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                    print("[IngestAgent] Connected to live OHLCV market data.")
                    while True:
                        if active_symbols != list(self.target_symbols or self.default_symbols):
                            print("[IngestAgent] Universe changed, reconnecting market data stream.")
                            break
                        try:
                            raw_message = await asyncio.wait_for(websocket.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            continue
                        envelope = json.loads(raw_message)
                        payload = envelope.get("data", envelope)
                        candle = payload["k"]
                        symbol = candle["s"].lower()
                        symbol_info = self.symbol_infos.get(symbol)

                        tick = MarketTick(
                            ticker=symbol_info.base_asset if symbol_info else candle["s"],
                            interval=candle["i"],
                            source=f"binance_{self.config.env}",
                            open=float(candle["o"]),
                            high=float(candle["h"]),
                            low=float(candle["l"]),
                            close=float(candle["c"]),
                            volume=float(candle["v"]),
                            quote_volume=float(candle["q"]),
                            trade_count=int(candle["n"]),
                            taker_buy_base_volume=float(candle["V"]),
                            taker_buy_quote_volume=float(candle["Q"]),
                            event_timestamp=time.time(),
                            candle_close_timestamp=float(candle["T"]) / 1000.0,
                            is_closed=bool(candle["x"]),
                        )

                        if tick.is_closed:
                            await self.publish_event("TICK_EVENT", tick.model_dump())
            except Exception as exc:
                print(f"[IngestAgent] WebSocket disconnected: {exc}. Reconnecting in 5s...")
                await asyncio.sleep(5)


async def main():
    agent = IngestAgent()
    await asyncio.gather(agent.heartbeat_loop(), agent.listen_for_universe(), agent.stream_loop())


if __name__ == "__main__":
    asyncio.run(main())

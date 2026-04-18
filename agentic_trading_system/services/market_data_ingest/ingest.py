import asyncio
import json
import time

import websockets

from core.base_agent import BaseAgent
from core.binance_client import BinanceRESTClient, load_binance_config, load_symbol_map
from core.schemas import MarketTick


async def binance_stream():
    agent = BaseAgent("IngestAgent")
    config = load_binance_config()
    symbol_map = load_symbol_map()
    binance = BinanceRESTClient(config)
    symbol_infos = {}
    for venue_symbol in symbol_map.values():
        try:
            info = binance.get_symbol_info(venue_symbol)
            symbol_infos[venue_symbol.lower()] = info
        except Exception as exc:
            print(f"[IngestAgent] Could not load symbol metadata for {venue_symbol}: {exc}")

    streams = "/".join(f"{symbol.lower()}@kline_{config.interval}" for symbol in symbol_map.values())
    uri = f"{config.market_ws_base}/stream?streams={streams}"

    print(f"[IngestAgent] Connecting to Binance WebSocket: {uri}")

    async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
        print("[IngestAgent] Connected to live OHLCV market data.")
        while True:
            envelope = json.loads(await websocket.recv())
            payload = envelope.get("data", envelope)
            candle = payload["k"]
            symbol = candle["s"].lower()
            symbol_info = symbol_infos.get(symbol)

            tick = MarketTick(
                ticker=symbol_info.base_asset if symbol_info else candle["s"],
                interval=candle["i"],
                source=f"binance_{config.env}",
                open=float(candle["o"]),
                high=float(candle["h"]),
                low=float(candle["l"]),
                close=float(candle["c"]),
                volume=float(candle["v"]),
                event_timestamp=time.time(),
                candle_close_timestamp=float(candle["T"]) / 1000.0,
                is_closed=bool(candle["x"]),
            )

            if tick.is_closed:
                await agent.publish_event("TICK_EVENT", tick.model_dump())


async def main():
    while True:
        try:
            await binance_stream()
        except Exception as exc:
            print(f"[IngestAgent] WebSocket disconnected: {exc}. Reconnecting in 5s...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())

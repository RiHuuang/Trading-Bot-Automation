import asyncio
import json
import time

import websockets

from core.base_agent import BaseAgent
from core.binance_client import load_binance_config
from core.schemas import MarketTick


async def binance_stream():
    agent = BaseAgent("IngestAgent")
    config = load_binance_config()
    symbol_info = None
    try:
        from core.binance_client import BinanceRESTClient

        symbol_info = BinanceRESTClient(config).get_symbol_info(config.symbol)
    except Exception as exc:
        print(f"[IngestAgent] Could not load symbol metadata: {exc}")
    stream_name = f"{config.symbol.lower()}@kline_{config.interval}"
    uri = f"{config.market_ws_base}/{stream_name}"

    print(f"[IngestAgent] Connecting to Binance WebSocket: {uri}")

    async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
        print("[IngestAgent] Connected to live OHLCV market data.")
        while True:
            payload = json.loads(await websocket.recv())
            candle = payload["k"]

            tick = MarketTick(
                ticker=symbol_info.base_asset if symbol_info else config.symbol,
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

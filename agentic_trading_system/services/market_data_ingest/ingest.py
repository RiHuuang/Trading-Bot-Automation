import asyncio
import json
import time
import websockets
from core.base_agent import BaseAgent

async def binance_stream():
    agent = BaseAgent("IngestAgent")
    # We connect to the 1-second ticker stream for BTC/USDT
    uri = "wss://stream.binance.com:9443/ws/btcusdt@ticker"

    print(f"[IngestAgent] Connecting to live Binance WebSocket: {uri}")

    async with websockets.connect(uri) as websocket:
        print("[IngestAgent] Connected to LIVE market data.")
        while True:
            # 1. Wait for Binance to push live data
            message = await websocket.recv()
            data = json.loads(message)

            # 2. Extract the 'c' (current close price) from the Binance payload
            live_price = round(float(data['c']), 2)

            raw_data = {
                "ticker": "BTC",
                "price": live_price,
                "timestamp": time.time()
            }

            # 3. Publish to your internal message broker
            await agent.publish_event("TICK_EVENT", raw_data)
            
            # Binance updates this stream every 1000ms, so we don't need a manual sleep here.
            # The speed of the loop is governed by the speed of the global market.

async def main():
    while True:
        try:
            await binance_stream()
        except Exception as e:
            print(f"[IngestAgent] WebSocket Disconnected: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
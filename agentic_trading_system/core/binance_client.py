from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from urllib.parse import urlencode

import requests


@dataclass
class BinanceConfig:
    env: str
    symbol: str
    interval: str
    api_key: str | None
    api_secret: str | None
    market_ws_base: str
    rest_base: str


@dataclass
class SymbolInfo:
    symbol: str
    base_asset: str
    quote_asset: str
    tick_size: float | None = None
    step_size: float | None = None
    min_notional: float | None = None


@dataclass
class OrderBookSnapshot:
    best_bid: float
    best_ask: float
    bid_notional_top_n: float
    ask_notional_top_n: float
    spread_bps: float


def load_binance_config() -> BinanceConfig:
    env = os.getenv("BINANCE_ENV", "testnet").lower()
    symbol = os.getenv("BINANCE_SYMBOL", "BTCUSDT").upper()
    interval = os.getenv("BINANCE_STREAM_INTERVAL", "15m")

    if env == "live":
        market_ws_base = "wss://stream.binance.com:9443/ws"
        rest_base = "https://api.binance.com"
    else:
        market_ws_base = "wss://stream.testnet.binance.vision/ws"
        rest_base = "https://testnet.binance.vision"

    return BinanceConfig(
        env=env,
        symbol=symbol,
        interval=interval,
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
        market_ws_base=market_ws_base,
        rest_base=rest_base,
    )


def load_symbol_map() -> dict[str, str]:
    configured = os.getenv("BINANCE_SYMBOLS", os.getenv("BINANCE_SYMBOL", "BTCUSDT"))
    mapping: dict[str, str] = {}
    for raw_symbol in [item.strip().upper() for item in configured.split(",") if item.strip()]:
        if raw_symbol.endswith("USDT"):
            base_asset = raw_symbol[: -4]
        elif raw_symbol.endswith("IDR"):
            base_asset = raw_symbol[: -3]
        else:
            base_asset = raw_symbol
        mapping[base_asset] = raw_symbol
    return mapping


class BinanceRESTClient:
    def __init__(self, config: BinanceConfig):
        self.config = config
        self.session = requests.Session()
        if config.api_key:
            self.session.headers.update({"X-MBX-APIKEY": config.api_key})

    def _sign_params(self, params: dict) -> str:
        if not self.config.api_secret:
            raise ValueError("BINANCE_API_SECRET is required for signed requests.")
        payload = urlencode(params, doseq=True)
        signature = hmac.new(
            self.config.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{payload}&signature={signature}"

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[list]:
        params: dict[str, int | str] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        response = self.session.get(f"{self.config.rest_base}/api/v3/klines", params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    def get_symbol_info(self, symbol: str | None = None) -> SymbolInfo:
        response = self.session.get(
            f"{self.config.rest_base}/api/v3/exchangeInfo",
            params={"symbol": symbol or self.config.symbol},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        raw_symbol = payload["symbols"][0]
        filters = {item["filterType"]: item for item in raw_symbol.get("filters", [])}
        return SymbolInfo(
            symbol=raw_symbol["symbol"],
            base_asset=raw_symbol["baseAsset"],
            quote_asset=raw_symbol["quoteAsset"],
            tick_size=float(filters["PRICE_FILTER"]["tickSize"]) if "PRICE_FILTER" in filters else None,
            step_size=float(filters["LOT_SIZE"]["stepSize"]) if "LOT_SIZE" in filters else None,
            min_notional=float(filters["NOTIONAL"]["minNotional"]) if "NOTIONAL" in filters else None,
        )

    def get_last_price(self, symbol: str) -> float:
        response = self.session.get(
            f"{self.config.rest_base}/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=15,
        )
        response.raise_for_status()
        return float(response.json()["price"])

    def get_depth(self, symbol: str | None = None, limit: int = 20) -> OrderBookSnapshot:
        response = self.session.get(
            f"{self.config.rest_base}/api/v3/depth",
            params={"symbol": symbol or self.config.symbol, "limit": limit},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        best_bid = float(payload["bids"][0][0])
        best_ask = float(payload["asks"][0][0])
        bid_notional_top_n = sum(float(price) * float(qty) for price, qty in payload["bids"])
        ask_notional_top_n = sum(float(price) * float(qty) for price, qty in payload["asks"])
        mid = (best_bid + best_ask) / 2
        spread_bps = ((best_ask - best_bid) / mid) * 10000 if mid > 0 else 0.0
        return OrderBookSnapshot(
            best_bid=best_bid,
            best_ask=best_ask,
            bid_notional_top_n=bid_notional_top_n,
            ask_notional_top_n=ask_notional_top_n,
            spread_bps=spread_bps,
        )

    def get_24h_tickers(self, symbols: list[str] | None = None) -> list[dict]:
        params = {}
        if symbols:
            if len(symbols) == 1:
                params["symbol"] = symbols[0]
            else:
                params["symbols"] = str(symbols).replace("'", '"')
        response = self.session.get(
            f"{self.config.rest_base}/api/v3/ticker/24hr",
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else [payload]

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str,
        validate_only: bool = True,
    ) -> dict:
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": f"{quantity:.6f}",
            "newClientOrderId": client_order_id,
            "newOrderRespType": "FULL",
            "recvWindow": 5000,
            "timestamp": int(time.time() * 1000),
        }
        path = "/api/v3/order/test" if validate_only else "/api/v3/order"
        signed_query = self._sign_params(params)
        response = self.session.post(
            f"{self.config.rest_base}{path}?{signed_query}",
            timeout=15,
        )
        response.raise_for_status()
        if response.text.strip():
            return response.json()
        return {"status": "VALIDATED"}

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.binance_client import BinanceRESTClient
from core.indicator_engine import build_market_snapshot, evaluate_models, evaluate_snapshot


@dataclass
class TimeframeContext:
    interval: str
    close: float
    rsi_14: float
    adx_14: float
    ema_20: float
    ema_50: float
    macd_histogram: float
    returns_20: float
    ensemble_action: str
    ensemble_confidence: float


def build_dataframe_from_klines(raw: list[list], base_asset: str, interval: str) -> pd.DataFrame:
    records = []
    for row in raw:
        records.append(
            {
                "ticker": base_asset,
                "interval": interval,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "event_timestamp": float(row[6]) / 1000.0,
                "candle_close_timestamp": float(row[6]) / 1000.0,
                "is_closed": True,
            }
        )
    return pd.DataFrame(records)


def fetch_timeframe_context(
    client: BinanceRESTClient,
    symbol: str,
    base_asset: str,
    interval: str,
    limit: int = 200,
) -> TimeframeContext | None:
    raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = build_dataframe_from_klines(raw=raw, base_asset=base_asset, interval=interval)
    snapshot = build_market_snapshot(df)
    if snapshot is None:
        return None
    ensemble = evaluate_snapshot(snapshot)
    return TimeframeContext(
        interval=interval,
        close=snapshot.close,
        rsi_14=snapshot.rsi_14,
        adx_14=snapshot.adx_14,
        ema_20=snapshot.ema_20,
        ema_50=snapshot.ema_50,
        macd_histogram=snapshot.macd_histogram,
        returns_20=snapshot.returns_20,
        ensemble_action=ensemble.action,
        ensemble_confidence=ensemble.confidence,
    )


def format_timeframe_context(contexts: list[TimeframeContext]) -> str:
    if not contexts:
        return "No higher timeframe context available."
    lines = []
    for item in contexts:
        lines.append(
            f"- {item.interval}: close={item.close:.4f}, RSI={item.rsi_14:.2f}, ADX={item.adx_14:.2f}, "
            f"EMA20/EMA50={item.ema_20:.4f}/{item.ema_50:.4f}, MACD_hist={item.macd_histogram:.4f}, "
            f"20-bar return={item.returns_20:.2%}, ensemble={item.ensemble_action} ({item.ensemble_confidence:.2f})"
        )
    return "\n".join(lines)

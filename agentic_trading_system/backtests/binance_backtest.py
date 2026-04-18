from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import pandas as pd

from core.binance_client import BinanceConfig, BinanceRESTClient
from core.indicator_engine import build_market_snapshot, evaluate_models, evaluate_snapshot


@dataclass
class Position:
    side: str | None = None
    entry_price: float = 0.0
    quantity: float = 0.0


def infer_ticker(symbol: str, quote_asset: str) -> str:
    if symbol.endswith(quote_asset):
        return symbol[: -len(quote_asset)]
    return symbol


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    config = BinanceConfig(
        env="live",
        symbol=symbol,
        interval=interval,
        api_key=None,
        api_secret=None,
        market_ws_base="wss://stream.binance.com:9443/ws",
        rest_base="https://api.binance.com",
    )
    client = BinanceRESTClient(config)
    symbol_info = client.get_symbol_info(symbol)
    raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    records = []
    for row in raw:
        records.append(
            {
                "ticker": infer_ticker(symbol, symbol_info.quote_asset),
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


def convert_reporting_value(value: float, quote_asset: str, reporting_currency: str) -> tuple[float, float]:
    if quote_asset == reporting_currency:
        return value, 1.0
    pair = f"{quote_asset}{reporting_currency}"
    client = BinanceRESTClient(
        BinanceConfig(
            env="live",
            symbol=pair,
            interval="1m",
            api_key=None,
            api_secret=None,
            market_ws_base="wss://stream.binance.com:9443/ws",
            rest_base="https://api.binance.com",
        )
    )
    rate = client.get_last_price(pair)
    return value * rate, rate


def run_backtest(
    df: pd.DataFrame,
    initial_cash: float,
    risk_pct: float,
    entry_threshold: float,
    stop_atr_multiple: float,
    take_profit_multiple: float,
    minimum_agreeing_models: int,
    quote_asset: str,
    reporting_currency: str,
) -> dict:
    cash = initial_cash
    equity_curve: list[float] = []
    trades = 0
    wins = 0
    position = Position()

    for idx in range(120, len(df)):
        window = df.iloc[: idx + 1]
        snapshot = build_market_snapshot(window)
        if snapshot is None:
            continue

        submodels = evaluate_models(snapshot)
        decision = evaluate_snapshot(snapshot)
        price = snapshot.close

        buy_votes = sum(1 for item in submodels if item.action == "BUY")
        sell_votes = sum(1 for item in submodels if item.action == "SELL")

        if (
            position.side is None
            and decision.action == "BUY"
            and decision.confidence >= entry_threshold
            and buy_votes >= minimum_agreeing_models
            and snapshot.adx_14 >= 18
        ):
            risk_budget = cash * risk_pct
            stop_distance = max(snapshot.atr_14 * stop_atr_multiple, price * 0.003)
            quantity = risk_budget / stop_distance
            notional_cap_qty = min(quantity, (cash * 0.2) / price)
            quantity = max(round(notional_cap_qty, 6), 0.0)
            if quantity > 0:
                cost = quantity * price
                cash -= cost
                position = Position(side="LONG", entry_price=price, quantity=quantity)
                trades += 1

        elif position.side == "LONG":
            stop_price = position.entry_price - max(snapshot.atr_14 * stop_atr_multiple, position.entry_price * 0.003)
            take_profit = position.entry_price + (position.entry_price - stop_price) * take_profit_multiple
            exit_due_to_signal = (
                decision.action == "SELL"
                and decision.confidence >= entry_threshold
                and sell_votes >= minimum_agreeing_models
            )
            exit_due_to_stop = price <= stop_price
            exit_due_to_tp = price >= take_profit

            if exit_due_to_signal or exit_due_to_stop or exit_due_to_tp:
                proceeds = position.quantity * price
                pnl = proceeds - (position.quantity * position.entry_price)
                cash += proceeds
                if pnl > 0:
                    wins += 1
                position = Position()

        mark_to_market = cash
        if position.side == "LONG":
            mark_to_market += position.quantity * price
        equity_curve.append(mark_to_market)

    if position.side == "LONG":
        final_price = float(df.iloc[-1]["close"])
        cash += position.quantity * final_price
        position = Position()

    if not equity_curve:
        raise RuntimeError("Not enough candles to run the backtest.")

    peak = -math.inf
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)

    total_return = (cash - initial_cash) / initial_cash
    final_equity_reporting, fx_rate = convert_reporting_value(cash, quote_asset=quote_asset, reporting_currency=reporting_currency)
    return {
        "initial_cash": initial_cash,
        "quote_asset": quote_asset,
        "reporting_currency": reporting_currency,
        "fx_rate_to_reporting": round(fx_rate, 4),
        "final_equity": round(cash, 2),
        "final_equity_reporting": round(final_equity_reporting, 2),
        "total_return_pct": round(total_return * 100, 2),
        "trades": trades,
        "win_rate_pct": round((wins / trades) * 100, 2) if trades else 0.0,
        "max_drawdown_pct": round(max_drawdown * 100, 2),
    }


def optimize_parameters(df: pd.DataFrame, initial_cash: float, risk_pct: float, quote_asset: str, reporting_currency: str) -> dict:
    candidates: list[tuple[float, float, float, int]] = [
        (0.55, 1.8, 1.8, 1),
        (0.58, 2.0, 2.0, 1),
        (0.60, 2.2, 2.5, 1),
        (0.62, 2.2, 2.5, 2),
    ]
    best_result: dict | None = None
    best_score = float("-inf")
    for entry_threshold, stop_atr_multiple, take_profit_multiple, minimum_agreeing_models in candidates:
        result = run_backtest(
            df=df,
            initial_cash=initial_cash,
            risk_pct=risk_pct,
            entry_threshold=entry_threshold,
            stop_atr_multiple=stop_atr_multiple,
            take_profit_multiple=take_profit_multiple,
            minimum_agreeing_models=minimum_agreeing_models,
            quote_asset=quote_asset,
            reporting_currency=reporting_currency,
        )
        trade_bonus = min(result["trades"], 20) * 0.03
        inactivity_penalty = 1.0 if result["trades"] == 0 else 0.0
        score = result["total_return_pct"] - (result["max_drawdown_pct"] * 0.75) + trade_bonus - inactivity_penalty
        if best_result is None or score > best_score:
            best_score = score
            best_result = {
                "entry_threshold": entry_threshold,
                "stop_atr_multiple": stop_atr_multiple,
                "take_profit_multiple": take_profit_multiple,
                "minimum_agreeing_models": minimum_agreeing_models,
                "result": result,
                "score": round(score, 2),
            }
    return best_result or {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--risk-pct", type=float, default=0.01)
    parser.add_argument("--entry-threshold", type=float, default=0.65)
    parser.add_argument("--stop-atr-multiple", type=float, default=2.2)
    parser.add_argument("--take-profit-multiple", type=float, default=2.5)
    parser.add_argument("--minimum-agreeing-models", type=int, default=1)
    parser.add_argument("--reporting-currency", default="USD")
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    df = fetch_klines(symbol=args.symbol, interval=args.interval, limit=args.limit)
    quote_asset = "IDR" if args.symbol.endswith("IDR") else "USDT"
    reporting_currency = args.reporting_currency.upper()
    if reporting_currency == "USD":
        reporting_currency = quote_asset

    if args.optimize:
        result = optimize_parameters(
            df=df,
            initial_cash=args.initial_cash,
            risk_pct=args.risk_pct,
            quote_asset=quote_asset,
            reporting_currency=reporting_currency,
        )
    else:
        result = run_backtest(
            df=df,
            initial_cash=args.initial_cash,
            risk_pct=args.risk_pct,
            entry_threshold=args.entry_threshold,
            stop_atr_multiple=args.stop_atr_multiple,
            take_profit_multiple=args.take_profit_multiple,
            minimum_agreeing_models=args.minimum_agreeing_models,
            quote_asset=quote_asset,
            reporting_currency=reporting_currency,
        )
    print(result)


if __name__ == "__main__":
    main()

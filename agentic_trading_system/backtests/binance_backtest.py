from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import pandas as pd

from core.binance_client import BinanceConfig, BinanceRESTClient, OrderBookSnapshot
from core.indicator_engine import build_market_snapshot, evaluate_models, evaluate_snapshot


@dataclass
class Position:
    side: str | None = None
    entry_price: float = 0.0
    quantity: float = 0.0
    stop_price: float = 0.0
    take_profit_price: float = 0.0


def infer_ticker(symbol: str, quote_asset: str) -> str:
    if symbol.endswith(quote_asset):
        return symbol[: -len(quote_asset)]
    return symbol


def build_public_client(symbol: str, interval: str) -> BinanceRESTClient:
    return BinanceRESTClient(
        BinanceConfig(
            env="live",
            symbol=symbol,
            interval=interval,
            api_key=None,
            api_secret=None,
            market_ws_base="wss://stream.binance.com:9443/ws",
            rest_base="https://api.binance.com",
        )
    )


def fetch_klines(symbol: str, interval: str, limit: int) -> tuple[pd.DataFrame, str, OrderBookSnapshot]:
    client = build_public_client(symbol=symbol, interval=interval)
    symbol_info = client.get_symbol_info(symbol)
    depth = client.get_depth(symbol=symbol, limit=20)
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
                "quote_volume": float(row[7]),
                "trade_count": int(row[8]),
                "taker_buy_base_volume": float(row[9]),
                "taker_buy_quote_volume": float(row[10]),
                "event_timestamp": float(row[6]) / 1000.0,
                "candle_close_timestamp": float(row[6]) / 1000.0,
                "is_closed": True,
            }
        )
    return pd.DataFrame(records), symbol_info.quote_asset, depth


def convert_reporting_value(value: float, quote_asset: str, reporting_currency: str) -> tuple[float, float]:
    if quote_asset == reporting_currency:
        return value, 1.0
    pair = f"{quote_asset}{reporting_currency}"
    rate = build_public_client(symbol=pair, interval="1m").get_last_price(pair)
    return value * rate, rate


def estimate_execution_cost_bps(
    bar_quote_volume: float,
    trade_notional: float,
    baseline_spread_bps: float,
    fee_bps: float,
    latency_bps: float,
    orderbook_notional: float,
) -> float:
    if trade_notional <= 0:
        return fee_bps
    volume_participation = trade_notional / max(bar_quote_volume, trade_notional)
    depth_participation = trade_notional / max(orderbook_notional, trade_notional)
    market_impact_bps = min(120.0, (volume_participation ** 0.65) * 35.0 + (depth_participation ** 0.75) * 45.0)
    spread_cost_bps = baseline_spread_bps * 0.5
    return fee_bps + spread_cost_bps + latency_bps + market_impact_bps


def apply_fill_price(price: float, total_cost_bps: float, side: str) -> float:
    multiplier = 1 + (total_cost_bps / 10000.0)
    return price * multiplier if side == "BUY" else price / multiplier


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
    fee_bps: float,
    latency_bps: float,
    slippage_buffer_bps: float,
    baseline_spread_bps: float,
    orderbook_notional: float,
    min_trade_count: int,
) -> dict:
    cash = initial_cash
    equity_curve: list[float] = []
    trades = 0
    wins = 0
    skipped_liquidity = 0
    position = Position()

    for idx in range(120, len(df)):
        window = df.iloc[: idx + 1]
        current_bar = df.iloc[idx]
        snapshot = build_market_snapshot(window)
        if snapshot is None:
            continue

        submodels = evaluate_models(snapshot)
        decision = evaluate_snapshot(snapshot)
        mark_price = snapshot.close
        buy_votes = sum(1 for item in submodels if item.action == "BUY")
        sell_votes = sum(1 for item in submodels if item.action == "SELL")

        if current_bar["trade_count"] < min_trade_count:
            skipped_liquidity += 1
            if position.side == "LONG":
                equity_curve.append(cash + (position.quantity * mark_price))
            else:
                equity_curve.append(cash)
            continue

        if (
            position.side is None
            and decision.action == "BUY"
            and decision.confidence >= entry_threshold
            and buy_votes >= minimum_agreeing_models
            and snapshot.adx_14 >= 18
        ):
            risk_budget = cash * risk_pct
            stop_distance = max(snapshot.atr_14 * stop_atr_multiple, mark_price * 0.003)
            quantity = risk_budget / stop_distance
            notional_cap_qty = min(quantity, (cash * 0.2) / mark_price)
            quantity = max(round(notional_cap_qty, 6), 0.0)
            trade_notional = quantity * mark_price
            if quantity > 0 and trade_notional > 0:
                total_cost_bps = estimate_execution_cost_bps(
                    bar_quote_volume=float(current_bar["quote_volume"]),
                    trade_notional=trade_notional,
                    baseline_spread_bps=baseline_spread_bps + slippage_buffer_bps,
                    fee_bps=fee_bps,
                    latency_bps=latency_bps,
                    orderbook_notional=orderbook_notional,
                )
                fill_price = apply_fill_price(mark_price, total_cost_bps, "BUY")
                total_cost = quantity * fill_price
                if total_cost <= cash:
                    cash -= total_cost
                    position = Position(
                        side="LONG",
                        entry_price=fill_price,
                        quantity=quantity,
                        stop_price=fill_price - stop_distance,
                        take_profit_price=fill_price + (stop_distance * take_profit_multiple),
                    )
                    trades += 1

        elif position.side == "LONG":
            exit_due_to_signal = (
                decision.action == "SELL"
                and decision.confidence >= entry_threshold
                and sell_votes >= minimum_agreeing_models
            )
            exit_due_to_stop = float(current_bar["low"]) <= position.stop_price
            exit_due_to_tp = float(current_bar["high"]) >= position.take_profit_price

            if exit_due_to_signal or exit_due_to_stop or exit_due_to_tp:
                if exit_due_to_stop:
                    raw_exit_price = min(position.stop_price, mark_price)
                elif exit_due_to_tp:
                    raw_exit_price = max(position.take_profit_price, mark_price)
                else:
                    raw_exit_price = mark_price

                trade_notional = position.quantity * raw_exit_price
                total_cost_bps = estimate_execution_cost_bps(
                    bar_quote_volume=float(current_bar["quote_volume"]),
                    trade_notional=trade_notional,
                    baseline_spread_bps=baseline_spread_bps + slippage_buffer_bps,
                    fee_bps=fee_bps,
                    latency_bps=latency_bps,
                    orderbook_notional=orderbook_notional,
                )
                fill_price = apply_fill_price(raw_exit_price, total_cost_bps, "SELL")
                proceeds = position.quantity * fill_price
                pnl = proceeds - (position.quantity * position.entry_price)
                cash += proceeds
                if pnl > 0:
                    wins += 1
                position = Position()

        mark_to_market = cash
        if position.side == "LONG":
            unrealized_exit = apply_fill_price(
                mark_price,
                estimate_execution_cost_bps(
                    bar_quote_volume=float(current_bar["quote_volume"]),
                    trade_notional=position.quantity * mark_price,
                    baseline_spread_bps=baseline_spread_bps + slippage_buffer_bps,
                    fee_bps=fee_bps,
                    latency_bps=latency_bps,
                    orderbook_notional=orderbook_notional,
                ),
                "SELL",
            )
            mark_to_market += position.quantity * unrealized_exit
        equity_curve.append(mark_to_market)

    if position.side == "LONG":
        final_bar = df.iloc[-1]
        final_price = apply_fill_price(
            float(final_bar["close"]),
            estimate_execution_cost_bps(
                bar_quote_volume=float(final_bar["quote_volume"]),
                trade_notional=position.quantity * float(final_bar["close"]),
                baseline_spread_bps=baseline_spread_bps + slippage_buffer_bps,
                fee_bps=fee_bps,
                latency_bps=latency_bps,
                orderbook_notional=orderbook_notional,
            ),
            "SELL",
        )
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
        "skipped_liquidity_bars": skipped_liquidity,
        "fee_bps": fee_bps,
        "baseline_spread_bps": round(baseline_spread_bps, 4),
    }


def optimize_parameters(
    df: pd.DataFrame,
    initial_cash: float,
    risk_pct: float,
    quote_asset: str,
    reporting_currency: str,
    fee_bps: float,
    latency_bps: float,
    slippage_buffer_bps: float,
    baseline_spread_bps: float,
    orderbook_notional: float,
    min_trade_count: int,
) -> dict:
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
            fee_bps=fee_bps,
            latency_bps=latency_bps,
            slippage_buffer_bps=slippage_buffer_bps,
            baseline_spread_bps=baseline_spread_bps,
            orderbook_notional=orderbook_notional,
            min_trade_count=min_trade_count,
        )
        trade_bonus = min(result["trades"], 20) * 0.03
        inactivity_penalty = 1.0 if result["trades"] == 0 else 0.0
        cost_penalty = (fee_bps + baseline_spread_bps + slippage_buffer_bps) * 0.01
        score = result["total_return_pct"] - (result["max_drawdown_pct"] * 0.75) + trade_bonus - inactivity_penalty - cost_penalty
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
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--latency-bps", type=float, default=2.0)
    parser.add_argument("--slippage-buffer-bps", type=float, default=3.0)
    parser.add_argument("--min-trade-count", type=int, default=200)
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    df, quote_asset, depth = fetch_klines(symbol=args.symbol, interval=args.interval, limit=args.limit)
    reporting_currency = args.reporting_currency.upper()
    if reporting_currency == "USD":
        reporting_currency = quote_asset

    baseline_spread_bps = depth.spread_bps
    orderbook_notional = min(depth.bid_notional_top_n, depth.ask_notional_top_n)

    if args.optimize:
        result = optimize_parameters(
            df=df,
            initial_cash=args.initial_cash,
            risk_pct=args.risk_pct,
            quote_asset=quote_asset,
            reporting_currency=reporting_currency,
            fee_bps=args.fee_bps,
            latency_bps=args.latency_bps,
            slippage_buffer_bps=args.slippage_buffer_bps,
            baseline_spread_bps=baseline_spread_bps,
            orderbook_notional=orderbook_notional,
            min_trade_count=args.min_trade_count,
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
            fee_bps=args.fee_bps,
            latency_bps=args.latency_bps,
            slippage_buffer_bps=args.slippage_buffer_bps,
            baseline_spread_bps=baseline_spread_bps,
            orderbook_notional=orderbook_notional,
            min_trade_count=args.min_trade_count,
        )
    print(result)


if __name__ == "__main__":
    main()

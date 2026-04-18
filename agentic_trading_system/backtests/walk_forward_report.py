from __future__ import annotations

import argparse
import json

from backtests.binance_backtest import fetch_klines, optimize_parameters, run_backtest


def walk_forward(symbol: str, interval: str, total_limit: int, train_size: int, test_size: int, initial_cash: float, risk_pct: float, reporting_currency: str, fee_bps: float, latency_bps: float, slippage_buffer_bps: float, min_trade_count: int) -> dict:
    df, quote_asset, depth = fetch_klines(symbol=symbol, interval=interval, limit=total_limit)
    windows = []
    start = 0
    while start + train_size + test_size <= len(df):
        train_df = df.iloc[start : start + train_size]
        test_df = df.iloc[start + train_size : start + train_size + test_size]
        tuned = optimize_parameters(
            df=train_df,
            initial_cash=initial_cash,
            risk_pct=risk_pct,
            quote_asset=quote_asset,
            reporting_currency=reporting_currency,
            fee_bps=fee_bps,
            latency_bps=latency_bps,
            slippage_buffer_bps=slippage_buffer_bps,
            baseline_spread_bps=depth.spread_bps,
            orderbook_notional=min(depth.bid_notional_top_n, depth.ask_notional_top_n),
            min_trade_count=min_trade_count,
        )
        test_result = run_backtest(
            df=test_df,
            initial_cash=initial_cash,
            risk_pct=risk_pct,
            entry_threshold=tuned["entry_threshold"],
            stop_atr_multiple=tuned["stop_atr_multiple"],
            take_profit_multiple=tuned["take_profit_multiple"],
            minimum_agreeing_models=tuned["minimum_agreeing_models"],
            quote_asset=quote_asset,
            reporting_currency=reporting_currency,
            fee_bps=fee_bps,
            latency_bps=latency_bps,
            slippage_buffer_bps=slippage_buffer_bps,
            baseline_spread_bps=depth.spread_bps,
            orderbook_notional=min(depth.bid_notional_top_n, depth.ask_notional_top_n),
            min_trade_count=min_trade_count,
        )
        windows.append(
            {
                "train_start": start,
                "train_end": start + train_size - 1,
                "test_start": start + train_size,
                "test_end": start + train_size + test_size - 1,
                "tuned": tuned,
                "test_result": test_result,
            }
        )
        start += test_size
    avg_return = sum(item["test_result"]["total_return_pct"] for item in windows) / len(windows) if windows else 0.0
    avg_drawdown = sum(item["test_result"]["max_drawdown_pct"] for item in windows) / len(windows) if windows else 0.0
    profitable_windows = sum(1 for item in windows if item["test_result"]["total_return_pct"] > 0)
    return {
        "symbol": symbol,
        "interval": interval,
        "windows": windows,
        "summary": {
            "window_count": len(windows),
            "avg_test_return_pct": round(avg_return, 2),
            "avg_test_drawdown_pct": round(avg_drawdown, 2),
            "profitable_windows": profitable_windows,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--train-size", type=int, default=700)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--risk-pct", type=float, default=0.01)
    parser.add_argument("--reporting-currency", default="IDR")
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--latency-bps", type=float, default=2.0)
    parser.add_argument("--slippage-buffer-bps", type=float, default=3.0)
    parser.add_argument("--min-trade-count", type=int, default=200)
    args = parser.parse_args()

    reports = []
    for symbol in [item.strip().upper() for item in args.symbols.split(",") if item.strip()]:
        reports.append(
            walk_forward(
                symbol=symbol,
                interval=args.interval,
                total_limit=args.limit,
                train_size=args.train_size,
                test_size=args.test_size,
                initial_cash=args.initial_cash,
                risk_pct=args.risk_pct,
                reporting_currency=args.reporting_currency,
                fee_bps=args.fee_bps,
                latency_bps=args.latency_bps,
                slippage_buffer_bps=args.slippage_buffer_bps,
                min_trade_count=args.min_trade_count,
            )
        )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import ta

from core.schemas import MarketSnapshot


@dataclass
class TechnicalDecision:
    model_name: str
    action: str
    confidence: float
    summary: str
    reasons: list[str]


def build_market_snapshot(df: pd.DataFrame) -> MarketSnapshot | None:
    working = df.copy()
    close = working["close"]
    high = working["high"]
    low = working["low"]

    working["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    working["macd"] = macd.macd()
    working["macd_signal"] = macd.macd_signal()
    working["macd_histogram"] = macd.macd_diff()
    bollinger = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    working["bb_upper"] = bollinger.bollinger_hband()
    working["bb_lower"] = bollinger.bollinger_lband()
    working["bb_mid"] = bollinger.bollinger_mavg()
    working["bb_width"] = bollinger.bollinger_wband()
    working["atr_14"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    working["adx_14"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()
    stoch = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close, window=14, smooth_window=3
    )
    working["stoch_k"] = stoch.stoch()
    working["stoch_d"] = stoch.stoch_signal()
    working["ema_20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    working["ema_50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    working["returns_5"] = close.pct_change(periods=5)
    working["returns_20"] = close.pct_change(periods=20)
    working["volatility_20"] = close.pct_change().rolling(window=20).std()

    latest = working.iloc[-1]
    if latest.isna().any():
        return None

    return MarketSnapshot(
        ticker=working.iloc[-1]["ticker"],
        interval=working.iloc[-1]["interval"],
        generated_at=float(working.iloc[-1]["candle_close_timestamp"]),
        close=float(latest["close"]),
        atr_14=float(latest["atr_14"]),
        adx_14=float(latest["adx_14"]),
        rsi_14=float(latest["rsi_14"]),
        stoch_k=float(latest["stoch_k"]),
        stoch_d=float(latest["stoch_d"]),
        macd=float(latest["macd"]),
        macd_signal=float(latest["macd_signal"]),
        macd_histogram=float(latest["macd_histogram"]),
        ema_20=float(latest["ema_20"]),
        ema_50=float(latest["ema_50"]),
        bollinger_upper=float(latest["bb_upper"]),
        bollinger_lower=float(latest["bb_lower"]),
        bollinger_mid=float(latest["bb_mid"]),
        bollinger_bandwidth=float(latest["bb_width"]),
        returns_5=float(latest["returns_5"]),
        returns_20=float(latest["returns_20"]),
        volatility_20=float(latest["volatility_20"]),
        volume=float(latest["volume"]),
    )


def evaluate_trend_model(snapshot: MarketSnapshot) -> TechnicalDecision:
    bullish_score = 0
    bearish_score = 0
    reasons: list[str] = []

    if snapshot.adx_14 >= 20:
        reasons.append("trend strength confirmed by ADX")
        if snapshot.ema_20 > snapshot.ema_50 and snapshot.macd_histogram > 0:
            bullish_score += 2
            reasons.append("EMA20 above EMA50 with positive MACD histogram")
        elif snapshot.ema_20 < snapshot.ema_50 and snapshot.macd_histogram < 0:
            bearish_score += 2
            reasons.append("EMA20 below EMA50 with negative MACD histogram")

    if snapshot.rsi_14 <= 35 and snapshot.stoch_k <= 25 and snapshot.close <= snapshot.bollinger_lower * 1.01:
        bullish_score += 1
        reasons.append("oversold conditions near lower Bollinger band")

    if snapshot.rsi_14 >= 65 and snapshot.stoch_k >= 75 and snapshot.close >= snapshot.bollinger_upper * 0.99:
        bearish_score += 1
        reasons.append("overbought conditions near upper Bollinger band")

    if snapshot.returns_5 > 0 and snapshot.returns_20 > 0:
        bullish_score += 1
        reasons.append("positive short and medium momentum")
    elif snapshot.returns_5 < 0 and snapshot.returns_20 < 0:
        bearish_score += 1
        reasons.append("negative short and medium momentum")

    if snapshot.volatility_20 > 0.03:
        reasons.append("elevated realized volatility")

    score_gap = bullish_score - bearish_score
    raw_confidence = min(0.55 + (abs(score_gap) * 0.1), 0.9)

    if score_gap >= 2:
        return TechnicalDecision(
            model_name="trend",
            action="BUY",
            confidence=raw_confidence,
            summary="Bullish confluence from trend and momentum indicators.",
            reasons=reasons,
        )
    if score_gap <= -2:
        return TechnicalDecision(
            model_name="trend",
            action="SELL",
            confidence=raw_confidence,
            summary="Bearish confluence from trend and momentum indicators.",
            reasons=reasons,
        )
    return TechnicalDecision(
        model_name="trend",
        action="HOLD",
        confidence=min(raw_confidence, 0.6),
        summary="Indicators are mixed or insufficiently aligned.",
        reasons=reasons or ["no strong multi-indicator alignment"],
    )


def evaluate_mean_reversion_model(snapshot: MarketSnapshot) -> TechnicalDecision:
    reasons: list[str] = []
    bullish_score = 0
    bearish_score = 0

    if snapshot.close <= snapshot.bollinger_lower * 1.005:
        bullish_score += 1
        reasons.append("price is pressing the lower Bollinger band")
    if snapshot.close >= snapshot.bollinger_upper * 0.995:
        bearish_score += 1
        reasons.append("price is pressing the upper Bollinger band")

    if snapshot.rsi_14 <= 33 and snapshot.stoch_k <= 20 and snapshot.stoch_d <= 25:
        bullish_score += 2
        reasons.append("oscillators show oversold exhaustion")
    if snapshot.rsi_14 >= 67 and snapshot.stoch_k >= 80 and snapshot.stoch_d >= 75:
        bearish_score += 2
        reasons.append("oscillators show overbought exhaustion")

    if snapshot.adx_14 >= 25:
        reasons.append("strong trend reduces mean-reversion conviction")
        bullish_score -= 1
        bearish_score -= 1

    score_gap = bullish_score - bearish_score
    raw_confidence = min(0.52 + (abs(score_gap) * 0.11), 0.85)
    if score_gap >= 2:
        return TechnicalDecision("mean_reversion", "BUY", raw_confidence, "Oversold mean-reversion setup.", reasons)
    if score_gap <= -2:
        return TechnicalDecision("mean_reversion", "SELL", raw_confidence, "Overbought mean-reversion setup.", reasons)
    return TechnicalDecision("mean_reversion", "HOLD", min(raw_confidence, 0.58), "No clean reversal setup.", reasons or ["no reversal edge"])


def evaluate_breakout_model(snapshot: MarketSnapshot) -> TechnicalDecision:
    reasons: list[str] = []
    bullish_score = 0
    bearish_score = 0

    if snapshot.bollinger_bandwidth >= 6:
        reasons.append("volatility expansion supports breakout follow-through")
        if snapshot.macd_histogram > 0 and snapshot.returns_5 > 0:
            bullish_score += 2
            reasons.append("positive momentum during expansion")
        elif snapshot.macd_histogram < 0 and snapshot.returns_5 < 0:
            bearish_score += 2
            reasons.append("negative momentum during expansion")

    if snapshot.close > snapshot.bollinger_mid and snapshot.ema_20 > snapshot.ema_50:
        bullish_score += 1
        reasons.append("price is above mid-band and fast trend is positive")
    if snapshot.close < snapshot.bollinger_mid and snapshot.ema_20 < snapshot.ema_50:
        bearish_score += 1
        reasons.append("price is below mid-band and fast trend is negative")

    score_gap = bullish_score - bearish_score
    raw_confidence = min(0.5 + (abs(score_gap) * 0.12), 0.84)
    if score_gap >= 2:
        return TechnicalDecision("breakout", "BUY", raw_confidence, "Bullish breakout continuation setup.", reasons)
    if score_gap <= -2:
        return TechnicalDecision("breakout", "SELL", raw_confidence, "Bearish breakout continuation setup.", reasons)
    return TechnicalDecision("breakout", "HOLD", min(raw_confidence, 0.58), "No breakout continuation edge.", reasons or ["no breakout edge"])


def evaluate_models(snapshot: MarketSnapshot) -> list[TechnicalDecision]:
    return [
        evaluate_trend_model(snapshot),
        evaluate_mean_reversion_model(snapshot),
        evaluate_breakout_model(snapshot),
    ]


def evaluate_snapshot(snapshot: MarketSnapshot) -> TechnicalDecision:
    decisions = evaluate_models(snapshot)
    weighted_votes = {"BUY": 0.0, "SELL": 0.0}
    reasons: list[str] = []
    hold_confidence = 0.0
    for decision in decisions:
        if decision.action == "HOLD":
            hold_confidence += decision.confidence
        else:
            weighted_votes[decision.action] += decision.confidence
        reasons.append(f"{decision.model_name}:{decision.action}:{decision.confidence:.2f}")

    ranked = sorted(weighted_votes.items(), key=lambda item: item[1], reverse=True)
    winner, winner_score = ranked[0]
    runner_up_score = ranked[1][1]
    consensus_gap = winner_score - runner_up_score
    confidence = min(0.52 + consensus_gap * 0.22, 0.92)

    if winner_score < 0.6 or consensus_gap < 0.10:
        return TechnicalDecision(
            model_name="ensemble",
            action="HOLD",
            confidence=min(max(hold_confidence / max(len(decisions), 1), 0.5), 0.62),
            summary="Directional consensus is too weak.",
            reasons=reasons,
        )

    return TechnicalDecision(
        model_name="ensemble",
        action=winner,
        confidence=confidence,
        summary=f"Ensemble voted {winner} with weighted consensus.",
        reasons=reasons,
    )

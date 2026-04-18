from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


TradeAction = Literal["BUY", "SELL", "HOLD"]
ReviewVerdict = Literal["SUPPORT", "CHALLENGE", "ABSTAIN"]
DecisionVerdict = Literal["APPROVE", "REJECT", "NEEDS_MORE_DATA"]


class MarketTick(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(..., min_length=1, max_length=10)
    interval: str = Field(default="1m")
    source: str = Field(default="binance")
    open: float = Field(..., gt=0.0)
    high: float = Field(..., gt=0.0)
    low: float = Field(..., gt=0.0)
    close: float = Field(..., gt=0.0)
    volume: float = Field(..., ge=0.0)
    quote_volume: float = Field(default=0.0, ge=0.0)
    trade_count: int = Field(default=0, ge=0)
    taker_buy_base_volume: float = Field(default=0.0, ge=0.0)
    taker_buy_quote_volume: float = Field(default=0.0, ge=0.0)
    event_timestamp: float = Field(..., gt=0.0)
    candle_close_timestamp: float = Field(..., gt=0.0)
    is_closed: bool = True


class MarketSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    interval: str
    generated_at: float = Field(..., gt=0.0)
    close: float = Field(..., gt=0.0)
    atr_14: float = Field(..., ge=0.0)
    adx_14: float = Field(..., ge=0.0)
    rsi_14: float = Field(..., ge=0.0, le=100.0)
    stoch_k: float = Field(..., ge=0.0, le=100.0)
    stoch_d: float = Field(..., ge=0.0, le=100.0)
    macd: float
    macd_signal: float
    macd_histogram: float
    ema_20: float = Field(..., gt=0.0)
    ema_50: float = Field(..., gt=0.0)
    bollinger_upper: float = Field(..., gt=0.0)
    bollinger_lower: float = Field(..., gt=0.0)
    bollinger_mid: float = Field(..., gt=0.0)
    bollinger_bandwidth: float = Field(..., ge=0.0)
    returns_5: float
    returns_20: float
    roc_12: float
    volatility_20: float = Field(..., ge=0.0)
    sma_20: float = Field(..., gt=0.0)
    sma_50: float = Field(..., gt=0.0)
    cci_20: float
    williams_r_14: float
    mfi_14: float = Field(..., ge=0.0, le=100.0)
    obv: float
    obv_slope_5: float
    vwap_14: float = Field(..., gt=0.0)
    aroon_up_25: float = Field(..., ge=0.0, le=100.0)
    aroon_down_25: float = Field(..., ge=0.0, le=100.0)
    ichimoku_conversion: float = Field(..., gt=0.0)
    ichimoku_base: float = Field(..., gt=0.0)
    ichimoku_a: float = Field(..., gt=0.0)
    ichimoku_b: float = Field(..., gt=0.0)
    fib_236: float = Field(..., gt=0.0)
    fib_382: float = Field(..., gt=0.0)
    fib_500: float = Field(..., gt=0.0)
    fib_618: float = Field(..., gt=0.0)
    volume: float = Field(..., ge=0.0)
    quote_volume: float = Field(..., ge=0.0)
    trade_count: int = Field(..., ge=0)
    taker_buy_ratio: float = Field(..., ge=0.0)
    volume_zscore_20: float


class TradeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    proposal_id: str
    source_agent: str
    model_name: str
    generated_at: float = Field(..., gt=0.0)
    ticker: str
    action: TradeAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    thesis: str = Field(..., min_length=10, max_length=800)
    reasoning: str = Field(..., min_length=10, max_length=1600)
    invalidation: str = Field(..., min_length=10, max_length=400)
    market_snapshot: MarketSnapshot


class TradeProposalDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    action: TradeAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    thesis: str = Field(..., min_length=10, max_length=800)
    reasoning: str = Field(..., min_length=10, max_length=1600)
    invalidation: str = Field(..., min_length=10, max_length=400)


class ProposalReview(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    review_id: str
    proposal_id: str
    reviewer_agent: str
    model_name: str
    generated_at: float = Field(..., gt=0.0)
    verdict: ReviewVerdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    blocking: bool = False
    concerns: list[str] = Field(default_factory=list, max_length=8)
    reasoning: str = Field(..., min_length=10, max_length=1600)
    recommended_action: TradeAction | None = None


class ProposalReviewDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    verdict: ReviewVerdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    blocking: bool = False
    concerns: list[str] = Field(default_factory=list, max_length=8)
    reasoning: str = Field(..., min_length=10, max_length=1600)
    recommended_action: TradeAction | None = None


class JudgeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    decision_id: str
    proposal_id: str
    source_agent: str
    model_name: str
    generated_at: float = Field(..., gt=0.0)
    verdict: DecisionVerdict
    approved_action: Literal["BUY", "SELL"] | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=10, max_length=1600)
    blockers: list[str] = Field(default_factory=list, max_length=8)
    proposal: TradeProposal
    review: ProposalReview


class JudgeDecisionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    verdict: DecisionVerdict
    approved_action: Literal["BUY", "SELL"] | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=10, max_length=1600)
    blockers: list[str] = Field(default_factory=list, max_length=8)


class ExecutionOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    proposal_id: str
    decision_id: str
    ticker: str
    action: Literal["BUY", "SELL"]
    quantity: float = Field(..., gt=0.0)
    quoted_price: float = Field(..., gt=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=10, max_length=1600)
    stop_loss: float = Field(..., gt=0.0)
    take_profit: float = Field(..., gt=0.0)
    created_at: float = Field(..., gt=0.0)
    paper_trade: bool = True


class OrderFill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    proposal_id: str
    decision_id: str
    ticker: str
    action: Literal["BUY", "SELL"]
    quantity: float = Field(..., gt=0.0)
    quoted_price: float = Field(..., gt=0.0)
    fill_price: float = Field(..., gt=0.0)
    slippage_bps: float
    status: Literal["FILLED", "REJECTED"]
    paper_trade: bool = True
    timestamp: float = Field(..., gt=0.0)
    reasoning: str = Field(..., min_length=5, max_length=1600)

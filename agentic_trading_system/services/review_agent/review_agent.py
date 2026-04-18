import asyncio
import json
import os

from dotenv import load_dotenv
from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.binance_client import BinanceRESTClient, infer_quote_asset, load_binance_config, load_symbol_map, resolve_venue_symbol
from core.llm_client import LLMClient
from core.market_context import fetch_timeframe_context, format_timeframe_context, get_configured_higher_timeframes
from core.schemas import ProposalReview, ProposalReviewDraft, TradeProposal

load_dotenv()


class ReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__("ReviewAgent")
        self.provider = os.getenv("REVIEW_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "gemini"))
        self.model_name = os.getenv(
            "REVIEW_MODEL",
            "gpt-5.4-mini" if self.provider == "openai" else "gemini-2.5-flash",
        )
        self.client = LLMClient(provider=self.provider, model_name=self.model_name)
        self.binance_config = load_binance_config()
        self.binance = BinanceRESTClient(self.binance_config)
        self.symbol_map = load_symbol_map()
        self.quote_asset = infer_quote_asset(self.binance_config.symbol)
        self.higher_timeframes = get_configured_higher_timeframes()

    def get_higher_timeframe_context(self, proposal: TradeProposal) -> str:
        symbol = resolve_venue_symbol(
            proposal.ticker,
            symbol_map=self.symbol_map,
            fallback_symbol=self.binance_config.symbol,
            quote_asset=self.quote_asset,
        )
        contexts = []
        for interval in self.higher_timeframes:
            try:
                context = fetch_timeframe_context(
                    client=self.binance,
                    symbol=symbol,
                    base_asset=proposal.ticker,
                    interval=interval,
                )
            except Exception as exc:
                print(f"[ReviewAgent] Failed loading {interval} context for {proposal.ticker}: {exc}")
                context = None
            if context is not None:
                contexts.append(context)
        return format_timeframe_context(contexts)

    async def review_proposal(self, data):
        try:
            proposal = TradeProposal(**data)
        except ValidationError as exc:
            print(f"[ReviewAgent] Invalid proposal payload: {exc}")
            return

        snapshot = proposal.market_snapshot
        higher_timeframes = self.get_higher_timeframe_context(proposal)
        prompt = f"""
You are the critic agent on a systematic crypto trading desk.
Your job is to challenge weak trade ideas and block trades that are under-evidenced.

Proposal under review:
- Action: {proposal.action}
- Confidence: {proposal.confidence:.2f}
- Thesis: {proposal.thesis}
- Reasoning: {proposal.reasoning}
- Invalidation: {proposal.invalidation}

Market snapshot:
- Price: {snapshot.close:.2f}
- ATR(14): {snapshot.atr_14:.4f}
- ADX(14): {snapshot.adx_14:.2f}
- RSI(14): {snapshot.rsi_14:.2f}
- Stochastic %K/%D: {snapshot.stoch_k:.2f} / {snapshot.stoch_d:.2f}
- MACD / Signal / Histogram: {snapshot.macd:.4f} / {snapshot.macd_signal:.4f} / {snapshot.macd_histogram:.4f}
- EMA(20) / EMA(50): {snapshot.ema_20:.2f} / {snapshot.ema_50:.2f}
- Bollinger upper / lower / bandwidth: {snapshot.bollinger_upper:.2f} / {snapshot.bollinger_lower:.2f} / {snapshot.bollinger_bandwidth:.2f}
- 5-bar return: {snapshot.returns_5:.4%}
- 20-bar return: {snapshot.returns_20:.4%}
- 20-bar volatility: {snapshot.volatility_20:.4%}
- ROC(12): {snapshot.roc_12:.2f}
- CCI(20): {snapshot.cci_20:.2f}
- Williams %R(14): {snapshot.williams_r_14:.2f}
- MFI(14): {snapshot.mfi_14:.2f}
- Aroon up/down: {snapshot.aroon_up_25:.2f} / {snapshot.aroon_down_25:.2f}
- OBV slope(5): {snapshot.obv_slope_5:.2f}
- Taker buy ratio: {snapshot.taker_buy_ratio:.2f}
- VWAP(14): {snapshot.vwap_14:.2f}
- Ichimoku A/B: {snapshot.ichimoku_a:.2f} / {snapshot.ichimoku_b:.2f}
- Fibonacci 38.2 / 61.8: {snapshot.fib_382:.2f} / {snapshot.fib_618:.2f}

Higher timeframe context:
{higher_timeframes}

Review policy:
- Prefer CHALLENGE when trend and momentum disagree.
- Prefer CHALLENGE when 15m trade direction fights the 1h or 4h context.
- Prefer CHALLENGE when confidence appears overstated.
- Prefer CHALLENGE when action is BUY or SELL but the evidence is mixed.
- Use SUPPORT only when the proposal is coherent, calibrated, and aligned with the snapshot.
- Set blocking=true when the proposal should not be allowed downstream.
        """

        try:
            review_data = await asyncio.to_thread(
                self.client.generate_json,
                prompt,
                ProposalReviewDraft,
                0.1,
            )
            review_data["review_id"] = self.new_id("review")
            review_data["proposal_id"] = proposal.proposal_id
            review_data["reviewer_agent"] = self.agent_name
            review_data["model_name"] = self.model_name
            review_data["generated_at"] = self.utc_timestamp()
            review = ProposalReview(**review_data)
            print(f"[ReviewAgent] {review.verdict} for proposal {proposal.proposal_id}.")
            await self.publish_event("SIGNAL_REVIEW_EVENT", review.model_dump())
        except Exception as exc:
            print(f"[ReviewAgent] Review generation failed: {exc}")
            await asyncio.sleep(5)


async def main():
    agent = ReviewAgent()
    await asyncio.gather(agent.heartbeat_loop(), agent.listen("SIGNAL_PROPOSAL_EVENT", agent.review_proposal))


if __name__ == "__main__":
    asyncio.run(main())

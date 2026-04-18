import asyncio
import json
import os

from dotenv import load_dotenv
from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.binance_client import BinanceRESTClient, infer_quote_asset, load_binance_config, load_symbol_map, resolve_venue_symbol
from core.llm_client import LLMClient
from core.market_context import fetch_timeframe_context, format_timeframe_context, get_configured_higher_timeframes
from core.schemas import JudgeDecision, JudgeDecisionDraft, ProposalReview, TradeProposal

load_dotenv()


class JudgeAgent(BaseAgent):
    def __init__(self):
        super().__init__("JudgeAgent")
        self.provider = os.getenv("JUDGE_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "gemini"))
        self.model_name = os.getenv(
            "JUDGE_MODEL",
            "gpt-5.4" if self.provider == "openai" else "gemini-2.5-flash",
        )
        self.client = LLMClient(provider=self.provider, model_name=self.model_name)
        self.proposals: dict[str, TradeProposal] = {}
        self.reviews: dict[str, ProposalReview] = {}
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
                print(f"[JudgeAgent] Failed loading {interval} context for {proposal.ticker}: {exc}")
                context = None
            if context is not None:
                contexts.append(context)
        return format_timeframe_context(contexts)

    async def cache_proposal(self, data):
        try:
            proposal = TradeProposal(**data)
        except ValidationError as exc:
            print(f"[JudgeAgent] Invalid proposal payload: {exc}")
            return

        self.proposals[proposal.proposal_id] = proposal
        print(f"[JudgeAgent] Cached proposal {proposal.proposal_id}.")
        pending_review = self.reviews.get(proposal.proposal_id)
        if pending_review is not None:
            await self.issue_decision(proposal, pending_review)

    async def evaluate_review(self, data):
        try:
            review = ProposalReview(**data)
        except ValidationError as exc:
            print(f"[JudgeAgent] Invalid review payload: {exc}")
            return

        proposal = self.proposals.get(review.proposal_id)
        if proposal is None:
            self.reviews[review.proposal_id] = review
            print(f"[JudgeAgent] Review arrived before proposal for {review.proposal_id}.")
            return

        self.reviews[review.proposal_id] = review
        await self.issue_decision(proposal, review)

    async def issue_decision(self, proposal: TradeProposal, review: ProposalReview):
        snapshot = proposal.market_snapshot
        higher_timeframes = self.get_higher_timeframe_context(proposal)
        prompt = f"""
You are the judge agent on a systematic crypto trading desk.
Decide whether to APPROVE, REJECT, or request NEEDS_MORE_DATA.
Be conservative. Reject trades when the critic found valid blocking concerns.

Author proposal:
- Action: {proposal.action}
- Confidence: {proposal.confidence:.2f}
- Thesis: {proposal.thesis}
- Reasoning: {proposal.reasoning}
- Invalidation: {proposal.invalidation}

Critic review:
- Verdict: {review.verdict}
- Confidence: {review.confidence:.2f}
- Blocking: {review.blocking}
- Concerns: {review.concerns}
- Reasoning: {review.reasoning}
- Recommended action: {review.recommended_action}

Market snapshot:
- Price: {snapshot.close:.2f}
- ATR(14): {snapshot.atr_14:.4f}
- ADX(14): {snapshot.adx_14:.2f}
- RSI(14): {snapshot.rsi_14:.2f}
- Stochastic %K/%D: {snapshot.stoch_k:.2f} / {snapshot.stoch_d:.2f}
- MACD histogram: {snapshot.macd_histogram:.4f}
- EMA(20) / EMA(50): {snapshot.ema_20:.2f} / {snapshot.ema_50:.2f}
- Bollinger bandwidth: {snapshot.bollinger_bandwidth:.2f}
- 20-bar volatility: {snapshot.volatility_20:.4%}
- CCI(20): {snapshot.cci_20:.2f}
- MFI(14): {snapshot.mfi_14:.2f}
- Aroon up/down: {snapshot.aroon_up_25:.2f} / {snapshot.aroon_down_25:.2f}
- OBV slope(5): {snapshot.obv_slope_5:.2f}
- Taker buy ratio: {snapshot.taker_buy_ratio:.2f}
- Fibonacci 38.2 / 61.8: {snapshot.fib_382:.2f} / {snapshot.fib_618:.2f}

Higher timeframe context:
{higher_timeframes}

Decision policy:
- REJECT if the review is blocking.
- REJECT if proposal confidence is below 0.60.
- REJECT if 15m direction materially conflicts with both 1h and 4h context.
- NEEDS_MORE_DATA if signals are mixed but not clearly wrong.
- APPROVE only BUY or SELL actions, never HOLD.
        """

        try:
            decision_data = await asyncio.to_thread(
                self.client.generate_json,
                prompt,
                JudgeDecisionDraft,
                0.0,
            )
            decision_data["decision_id"] = self.new_id("decision")
            decision_data["proposal_id"] = proposal.proposal_id
            decision_data["source_agent"] = self.agent_name
            decision_data["model_name"] = self.model_name
            decision_data["generated_at"] = self.utc_timestamp()
            decision_data["proposal"] = proposal.model_dump()
            decision_data["review"] = review.model_dump()

            if proposal.action == "HOLD":
                decision_data["verdict"] = "REJECT"
                decision_data["approved_action"] = None
                decision_data["blockers"] = ["Author returned HOLD, so there is no trade to execute."]

            decision = JudgeDecision(**decision_data)
            print(f"[JudgeAgent] {decision.verdict} for proposal {proposal.proposal_id}.")
            await self.publish_event("JUDGE_DECISION_EVENT", decision.model_dump())
            self.proposals.pop(proposal.proposal_id, None)
            self.reviews.pop(proposal.proposal_id, None)
        except Exception as exc:
            print(f"[JudgeAgent] Decision generation failed: {exc}")
            await asyncio.sleep(5)

    async def run(self):
        await self.pubsub.subscribe("SIGNAL_PROPOSAL_EVENT", "SIGNAL_REVIEW_EVENT")
        print("[JudgeAgent] Listening on SIGNAL_PROPOSAL_EVENT and SIGNAL_REVIEW_EVENT...")

        async for message in self.pubsub.listen():
            if message["type"] != "message":
                continue

            channel = message["channel"]
            try:
                data = json.loads(message["data"])
            except json.JSONDecodeError as exc:
                print(f"[JudgeAgent] Dropping malformed payload: {exc}")
                continue

            if channel == "SIGNAL_PROPOSAL_EVENT":
                await self.cache_proposal(data)
            elif channel == "SIGNAL_REVIEW_EVENT":
                await self.evaluate_review(data)


async def main():
    agent = JudgeAgent()
    await asyncio.gather(agent.heartbeat_loop(), agent.run())


if __name__ == "__main__":
    asyncio.run(main())

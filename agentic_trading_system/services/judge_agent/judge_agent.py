import asyncio
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.binance_client import BinanceRESTClient, load_binance_config, load_symbol_map
from core.market_context import fetch_timeframe_context, format_timeframe_context
from core.schemas import JudgeDecision, JudgeDecisionDraft, ProposalReview, TradeProposal

load_dotenv()


class JudgeAgent(BaseAgent):
    def __init__(self):
        super().__init__("JudgeAgent")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
        self.proposals: dict[str, TradeProposal] = {}
        self.reviews: dict[str, ProposalReview] = {}
        self.binance_config = load_binance_config()
        self.binance = BinanceRESTClient(self.binance_config)
        self.symbol_map = load_symbol_map()

    def get_higher_timeframe_context(self, proposal: TradeProposal) -> str:
        symbol = self.symbol_map.get(proposal.ticker, self.binance_config.symbol)
        contexts = []
        for interval in ("1h", "4h"):
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
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=JudgeDecisionDraft,
                    temperature=0.0,
                ),
            )
            draft = JudgeDecisionDraft(**json.loads(response.text))
            decision_data = draft.model_dump()
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
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

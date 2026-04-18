import asyncio
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.schemas import ProposalReview, ProposalReviewDraft, TradeProposal

load_dotenv()


class ReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__("ReviewAgent")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = os.getenv("REVIEW_MODEL", "gemini-2.5-flash")

    async def review_proposal(self, data):
        try:
            proposal = TradeProposal(**data)
        except ValidationError as exc:
            print(f"[ReviewAgent] Invalid proposal payload: {exc}")
            return

        snapshot = proposal.market_snapshot
        prompt = f"""
You are the critic agent on a BTC trading desk.
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

Review policy:
- Prefer CHALLENGE when trend and momentum disagree.
- Prefer CHALLENGE when confidence appears overstated.
- Prefer CHALLENGE when action is BUY or SELL but the evidence is mixed.
- Use SUPPORT only when the proposal is coherent, calibrated, and aligned with the snapshot.
- Set blocking=true when the proposal should not be allowed downstream.
        """

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ProposalReviewDraft,
                    temperature=0.1,
                ),
            )
            draft = ProposalReviewDraft(**json.loads(response.text))
            review_data = draft.model_dump()
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
    await agent.listen("SIGNAL_PROPOSAL_EVENT", agent.review_proposal)


if __name__ == "__main__":
    asyncio.run(main())

from pydantic import BaseModel, Field, ValidationError
from typing import Literal

class MarketTick(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    price: float = Field(..., description="Price must be strictly positive")
    timestamp: float

class TradeProposal(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    reasoning: str

class ExecutionOrder(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL"]
    quantity: float = Field(..., gt=0.0, description="The exact amount of the asset to trade")
    confidence: float
    reasoning: str
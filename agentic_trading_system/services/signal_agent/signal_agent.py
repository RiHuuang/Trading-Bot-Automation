import asyncio
import os
import json
import pandas as pd
import ta
from google import genai
from google.genai import types
from dotenv import load_dotenv
from core.base_agent import BaseAgent
from core.schemas import TradeProposal

load_dotenv()

class SignalAgent(BaseAgent):
    def __init__(self):
        super().__init__("SignalAgent")
        self.tick_buffer = []
        # We need at least 15 ticks to calculate a 14-period RSI
        self.buffer_limit = 15 
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    async def process_tick(self, data):
        self.tick_buffer.append(data)
        print(f"[SignalAgent] Aggregating Market Data {len(self.tick_buffer)}/{self.buffer_limit}")

        if len(self.tick_buffer) >= self.buffer_limit:
            await self.analyze_market()

    async def analyze_market(self):
        # 1. Convert raw buffer into a Pandas DataFrame
        df = pd.DataFrame(self.tick_buffer)
        
        # We only keep the last 15 ticks and drop the oldest to maintain a rolling window
        self.tick_buffer = self.tick_buffer[1:] 

        # 2. Calculate Technical Indicators
        try:
            # 14-period Relative Strength Index (RSI)
            df['rsi'] = ta.momentum.RSIIndicator(close=df['price'], window=14).rsi()
            # 10-period Simple Moving Average (SMA)
            df['sma'] = ta.trend.SMAIndicator(close=df['price'], window=10).sma_indicator()
            
            # Extract the latest values
            latest = df.iloc[-1]
            current_price = latest['price']
            current_rsi = latest['rsi']
            current_sma = latest['sma']
            
            # If indicators are still calculating (NaN), abort and wait for more data
            if pd.isna(current_rsi) or pd.isna(current_sma):
                return

        except Exception as e:
            print(f"[SignalAgent] Math Error during TA calculation: {e}")
            return

        # 3. Construct the Quant Prompt
        prompt = f"""
        You are a quantitative trading AI analyzing Bitcoin (BTC).
        
        CURRENT MARKET STATE:
        - Current Price: ${current_price:.2f}
        - 10-Period SMA: ${current_sma:.2f}
        - 14-Period RSI: {current_rsi:.2f}
        
        RULES:
        - If RSI is above 70, the asset is overbought (Look for SELL).
        - If RSI is below 30, the asset is oversold (Look for BUY).
        - If the Current Price is above the SMA, it is a short-term uptrend.
        - If the Current Price is below the SMA, it is a short-term downtrend.
        - If RSI is between 40 and 60, the market is chopping. Output HOLD.
        
        Analyze this data and output a valid JSON TradeProposal.
        """
        
        print(f"[SignalAgent] Formatted TA -> Price: ${current_price:.2f} | RSI: {current_rsi:.2f} | SMA: ${current_sma:.2f}")
        print("[SignalAgent] Querying Gemini AI with Context...")
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TradeProposal,
                    temperature=0.1, # Extremely low temperature. We want logic, not creativity.
                )
            )
            
            proposal_data = json.loads(response.text)
            print(f"[SignalAgent] LLM Decision: {proposal_data['action']} | Reason: {proposal_data['reasoning']}")
            
            await self.publish_event("PROPOSAL_EVENT", proposal_data)
            
        except Exception as e:
            print(f"[SignalAgent] FATAL - LLM Generation Failed: {e}")
            print("[SignalAgent] CIRCUIT BREAKER TRIPPED. Sleeping 60s...")
            await asyncio.sleep(60)

async def main():
    agent = SignalAgent()
    await agent.listen("TICK_EVENT", agent.process_tick)

if __name__ == "__main__":
    asyncio.run(main())
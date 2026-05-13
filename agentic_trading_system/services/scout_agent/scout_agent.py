import asyncio
import math
import os
import time
from typing import Literal

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from requests import HTTPError

from core.base_agent import BaseAgent
from core.binance_client import BinanceRESTClient, infer_quote_asset, load_binance_config, split_base_quote
from core.indicator_engine import build_market_snapshot, evaluate_snapshot
from core.llm_client import LLMClient
from core.market_context import build_dataframe_from_klines, fetch_timeframe_context

load_dotenv()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class NarrativeAssessmentDraft(BaseModel):
    catalyst_summary: str = Field(..., min_length=10, max_length=400)
    narrative_verdict: Literal["JUSTIFIED", "QUESTIONABLE", "INCONCLUSIVE"]
    invalidation_level: str = Field(..., min_length=5, max_length=200)
    reasoning: str = Field(..., min_length=10, max_length=1000)
    notable_risks: list[str] = Field(default_factory=list, max_length=6)


class ScoutAgent(BaseAgent):
    def __init__(self):
        super().__init__("ScoutAgent")
        self.binance_config = load_binance_config()
        self.binance = BinanceRESTClient(self.binance_config)
        self.provider = os.getenv("SCOUT_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "gemini"))
        self.model_name = os.getenv(
            "SCOUT_MODEL",
            "gpt-5.4-mini" if self.provider == "openai" else "gemini-2.5-flash",
        )
        self.client = LLMClient(provider=self.provider, model_name=self.model_name)
        self.quote_asset = os.getenv("SCOUT_QUOTE_ASSET", infer_quote_asset(self.binance_config.symbol)).upper()
        self.market_cap_limit = int(os.getenv("SCOUT_MARKET_CAP_LIMIT", "500"))
        self.max_symbols = int(os.getenv("SCOUT_MAX_SYMBOLS", "12"))
        self.scan_limit = int(os.getenv("SCOUT_SCAN_LIMIT", str(self.market_cap_limit)))
        self.min_quote_volume = float(os.getenv("SCOUT_MIN_QUOTE_VOLUME", "10000000"))
        self.min_volume_market_cap_ratio = float(os.getenv("SCOUT_MIN_VOLUME_MARKET_CAP_RATIO", "0.05"))
        self.min_depth_notional = float(os.getenv("SCOUT_MIN_DEPTH_NOTIONAL", "150000"))
        self.required_depth_multiple = float(os.getenv("SCOUT_REQUIRED_DEPTH_MULTIPLE", "8"))
        self.depth_band_pct = float(os.getenv("SCOUT_DEPTH_BAND_PCT", "0.02"))
        self.max_spread_bps = float(os.getenv("SCOUT_MAX_SPREAD_BPS", "15"))
        self.min_candidate_confidence = float(os.getenv("SCOUT_MIN_CONFIDENCE", "0.58"))
        self.min_volume_delta_ratio = float(os.getenv("SCOUT_MIN_VOLUME_DELTA_RATIO", "3.0"))
        self.min_social_velocity = float(os.getenv("SCOUT_MIN_SOCIAL_VELOCITY", "1.1"))
        self.min_dev_activity = int(os.getenv("SCOUT_MIN_DEV_ACTIVITY", "5"))
        self.min_onchain_velocity = float(os.getenv("SCOUT_MIN_ONCHAIN_VELOCITY", "1.1"))
        self.short_squeeze_funding_threshold = float(os.getenv("SCOUT_SHORT_SQUEEZE_FUNDING_THRESHOLD", "-0.0005"))
        self.sleep_seconds = int(os.getenv("SCOUT_INTERVAL_SECONDS", "900"))
        self.prefer_long_only = os.getenv("SCOUT_LONG_ONLY", "true").lower() == "true"
        self.primary_interval_limit = int(os.getenv("SCOUT_PRIMARY_INTERVAL_BARS", "260"))
        self.enable_narrative_llm = os.getenv("SCOUT_ENABLE_NARRATIVE_LLM", "true").lower() == "true"
        configured_contexts = os.getenv("SCOUT_CONTEXT_INTERVALS", "1h,4h")
        self.context_intervals = [item.strip() for item in configured_contexts.split(",") if item.strip()]
        self.excluded_symbols = {"USDCUSDT", "FDUSDUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT"}
        self.excluded_suffixes = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
        self.coingecko_base = os.getenv("COINGECKO_API_BASE", "https://api.coingecko.com/api/v3").rstrip("/")
        self.coingecko_key = os.getenv("COINGECKO_API_KEY", "").strip()
        self.coingecko_session = requests.Session()
        if self.coingecko_key:
            self.coingecko_session.headers.update({"x-cg-demo-api-key": self.coingecko_key})
        self.market_cache_ttl_seconds = int(os.getenv("SCOUT_MARKET_CACHE_TTL_SECONDS", "300"))
        self.detail_cache_ttl_seconds = int(os.getenv("SCOUT_DETAIL_CACHE_TTL_SECONDS", "1800"))
        self.market_cache_stale_ttl_seconds = int(os.getenv("SCOUT_MARKET_CACHE_STALE_TTL_SECONDS", "3600"))
        self.detail_request_min_interval_seconds = float(os.getenv("SCOUT_DETAIL_REQUEST_MIN_INTERVAL_SECONDS", "1.25"))
        self.market_cache: tuple[float, list[dict]] | None = None
        self.detail_cache: dict[str, tuple[float, dict]] = {}
        self.last_scan_summary: dict = {}
        self.next_detail_request_at = 0.0
        self.coingecko_backoff_until = 0.0

    def is_candidate_symbol(self, symbol: str) -> bool:
        if not symbol.endswith(self.quote_asset):
            return False
        if symbol in self.excluded_symbols:
            return False
        if any(symbol.endswith(suffix) for suffix in self.excluded_suffixes):
            return False
        return True

    def fetch_top_market_cap_coins(self) -> list[dict]:
        now = time.time()
        if self.market_cache and now - self.market_cache[0] < self.market_cache_ttl_seconds:
            return self.market_cache[1]
        if now < self.coingecko_backoff_until:
            if self.market_cache and now - self.market_cache[0] < self.market_cache_stale_ttl_seconds:
                return self.market_cache[1]
            return []

        per_page = min(250, max(50, self.market_cap_limit))
        pages = math.ceil(self.market_cap_limit / per_page)
        rows: list[dict] = []
        for page in range(1, pages + 1):
            try:
                response = self.coingecko_session.get(
                    f"{self.coingecko_base}/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "order": "market_cap_desc",
                        "per_page": per_page,
                        "page": page,
                        "sparkline": "false",
                        "price_change_percentage": "24h",
                    },
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json()
            except HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code == 429:
                    retry_after = 60.0
                    if exc.response is not None:
                        retry_after = float(exc.response.headers.get("Retry-After", retry_after))
                    self.coingecko_backoff_until = time.time() + max(retry_after, 30.0)
                    if self.market_cache and now - self.market_cache[0] < self.market_cache_stale_ttl_seconds:
                        print(
                            f"[ScoutAgent] CoinGecko market-cap fetch rate-limited; using stale cache for "
                            f"{round(max(retry_after, 30.0), 1)}s."
                        )
                        return self.market_cache[1]
                    print(
                        f"[ScoutAgent] CoinGecko market-cap fetch rate-limited and no cache available; "
                        f"backing off for {round(max(retry_after, 30.0), 1)}s."
                    )
                    return []
                raise
            if not isinstance(payload, list):
                continue
            rows.extend(payload)
            if len(rows) >= self.market_cap_limit:
                break

        self.market_cache = (now, rows[: self.market_cap_limit])
        return self.market_cache[1]

    def fetch_coin_detail(self, coin_id: str) -> dict:
        now = time.time()
        cached = self.detail_cache.get(coin_id)
        if cached and now - cached[0] < self.detail_cache_ttl_seconds:
            return cached[1]

        if now < self.coingecko_backoff_until:
            raise RuntimeError(
                f"CoinGecko detail backoff active for another {round(self.coingecko_backoff_until - now, 1)}s"
            )

        wait_seconds = max(0.0, self.next_detail_request_at - now)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        try:
            response = self.coingecko_session.get(
                f"{self.coingecko_base}/coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "market_data": "true",
                    "community_data": "true",
                    "developer_data": "true",
                    "sparkline": "false",
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 429:
                retry_after = 60.0
                if exc.response is not None:
                    retry_after = float(exc.response.headers.get("Retry-After", retry_after))
                self.coingecko_backoff_until = time.time() + max(retry_after, 30.0)
                raise RuntimeError(
                    f"CoinGecko rate limit hit; backing off for {round(max(retry_after, 30.0), 1)}s"
                ) from exc
            raise
        finally:
            self.next_detail_request_at = time.time() + self.detail_request_min_interval_seconds

        self.detail_cache[coin_id] = (now, payload if isinstance(payload, dict) else {})
        return self.detail_cache[coin_id][1]

    def fetch_depth_metrics(self, symbol: str, limit: int = 100) -> dict | None:
        try:
            response = self.binance.session.get(
                f"{self.binance_config.rest_base}/api/v3/depth",
                params={"symbol": symbol, "limit": limit},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None

        bids = payload.get("bids", [])
        asks = payload.get("asks", [])
        if not bids or not asks:
            return None

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2
        if mid <= 0:
            return None

        lower_bound = mid * (1 - self.depth_band_pct)
        upper_bound = mid * (1 + self.depth_band_pct)
        bid_depth = sum(float(price) * float(qty) for price, qty in bids if float(price) >= lower_bound)
        ask_depth = sum(float(price) * float(qty) for price, qty in asks if float(price) <= upper_bound)
        spread_bps = ((best_ask - best_bid) / mid) * 10000
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_bps": spread_bps,
            "bid_depth_notional_band": bid_depth,
            "ask_depth_notional_band": ask_depth,
        }

    def fetch_funding_signal(self, symbol: str) -> dict:
        futures_symbol = symbol.upper()
        try:
            response = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": futures_symbol, "limit": 1},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload:
                return {"available": False, "funding_rate": None, "short_squeeze_flag": False}
            latest = payload[-1]
            funding_rate = float(latest.get("fundingRate", 0.0))
            return {
                "available": True,
                "funding_rate": funding_rate,
                "short_squeeze_flag": funding_rate <= self.short_squeeze_funding_threshold,
                "funding_time": int(latest.get("fundingTime", 0)) / 1000.0 if latest.get("fundingTime") else None,
            }
        except Exception:
            return {"available": False, "funding_rate": None, "short_squeeze_flag": False}

    def compute_social_velocity(self, detail: dict) -> dict:
        community = detail.get("community_data") or {}
        followers = float(community.get("twitter_followers") or 0.0)
        reddit = float(community.get("reddit_subscribers") or 0.0)
        telegram = float(community.get("telegram_channel_user_count") or 0.0)
        market_data = detail.get("market_data") or {}
        market_cap = float((market_data.get("market_cap") or {}).get("usd") or 0.0)
        raw_social = followers + reddit + telegram
        baseline = max(market_cap / 1_000_000.0, 1.0)
        velocity = raw_social / baseline
        return {
            "twitter_followers": int(followers),
            "reddit_subscribers": int(reddit),
            "telegram_users": int(telegram),
            "social_velocity": round(velocity, 3),
            "social_spike_flag": velocity >= self.min_social_velocity,
        }

    def compute_dev_activity(self, detail: dict) -> dict:
        developer = detail.get("developer_data") or {}
        commit_count = int(developer.get("commit_count_4_weeks") or 0)
        merged_prs = int(developer.get("pull_requests_merged") or 0)
        stars = int(developer.get("stars") or 0)
        return {
            "commit_count_4_weeks": commit_count,
            "pull_requests_merged": merged_prs,
            "stars": stars,
            "dev_activity_flag": commit_count >= self.min_dev_activity or merged_prs >= self.min_dev_activity,
        }

    def compute_onchain_proxy(self, detail: dict) -> dict:
        market_data = detail.get("market_data") or {}
        current_volume = float((market_data.get("total_volume") or {}).get("usd") or 0.0)
        market_cap = float((market_data.get("market_cap") or {}).get("usd") or 0.0)
        raw_ratio = current_volume / max(market_cap, 1.0)
        velocity = raw_ratio / max(self.min_volume_market_cap_ratio, 1e-6)
        platforms = detail.get("platforms") or {}
        return {
            "token_platforms": [name for name, address in platforms.items() if address][:5],
            "onchain_velocity": round(velocity, 3),
            "onchain_velocity_flag": velocity >= self.min_onchain_velocity,
        }

    def build_detail_fallback(self, candidate: dict, reason: str) -> dict:
        coin = candidate["coingecko"]
        market_cap = float(coin.get("market_cap") or 0.0)
        total_volume = float(coin.get("total_volume") or 0.0)
        return {
            "id": coin.get("id"),
            "symbol": coin.get("symbol"),
            "name": coin.get("name"),
            "categories": [],
            "links": {"homepage": [], "repos_url": {"github": []}},
            "platforms": {},
            "market_data": {
                "market_cap": {"usd": market_cap},
                "total_volume": {"usd": total_volume},
            },
            "community_data": {},
            "developer_data": {},
            "_fallback_reason": reason,
        }

    def build_binance_symbol_map(self) -> dict[str, dict]:
        tickers = self.binance.get_24h_tickers()
        candidates: dict[str, dict] = {}
        for ticker in tickers:
            symbol = ticker["symbol"].upper()
            if not self.is_candidate_symbol(symbol):
                continue
            base_asset, _ = split_base_quote(symbol)
            quote_volume = float(ticker.get("quoteVolume", 0.0))
            trade_count = int(ticker.get("count", 0))
            if quote_volume < self.min_quote_volume or trade_count < 1000:
                continue
            existing = candidates.get(base_asset)
            if existing is None or quote_volume > float(existing.get("quoteVolume", 0.0)):
                candidates[base_asset] = ticker
        return candidates

    def select_scan_universe(self) -> tuple[list[dict], dict]:
        top_market_cap = self.fetch_top_market_cap_coins()
        tradable_symbols = self.build_binance_symbol_map()
        selected = []
        summary = {
            "market_cap_candidates": len(top_market_cap),
            "binance_tradable_matches": 0,
            "liquidity_survivors": 0,
            "catalyst_survivors": 0,
            "narrative_survivors": 0,
            "rejections": {
                "not_tradeable_on_binance": 0,
                "low_quote_volume": 0,
                "weak_volume_market_cap_ratio": 0,
                "wide_spread": 0,
                "thin_depth": 0,
                "weak_catalyst": 0,
                "weak_technicals": 0,
                "narrative_failed": 0,
                "non_buy_when_long_only": 0,
            },
        }

        for coin in top_market_cap:
            base_asset = str(coin.get("symbol", "")).upper()
            ticker = tradable_symbols.get(base_asset)
            if ticker is None:
                summary["rejections"]["not_tradeable_on_binance"] += 1
                continue
            quote_volume = float(ticker.get("quoteVolume", 0.0))
            if quote_volume < self.min_quote_volume:
                summary["rejections"]["low_quote_volume"] += 1
                continue
            selected.append(
                {
                    "coingecko": coin,
                    "ticker_24h": ticker,
                    "venue_symbol": ticker["symbol"].upper(),
                    "base_asset": base_asset,
                }
            )
            if len(selected) >= self.scan_limit:
                break

        summary["binance_tradable_matches"] = len(selected)
        return selected, summary

    def run_liquidity_gate(self, candidate: dict) -> dict | None:
        coin = candidate["coingecko"]
        ticker = candidate["ticker_24h"]
        market_cap = float(coin.get("market_cap") or 0.0)
        total_volume = float(coin.get("total_volume") or 0.0)
        volume_market_cap_ratio = total_volume / max(market_cap, 1.0)
        if volume_market_cap_ratio < self.min_volume_market_cap_ratio:
            return None

        depth = self.fetch_depth_metrics(candidate["venue_symbol"])
        if depth is None:
            return None

        required_depth = max(self.min_depth_notional, float(os.getenv("MAX_POSITION_NOTIONAL", "2000")) * self.required_depth_multiple)
        bid_depth = float(depth["bid_depth_notional_band"])
        ask_depth = float(depth["ask_depth_notional_band"])
        if depth["spread_bps"] > self.max_spread_bps:
            return None
        if min(bid_depth, ask_depth) < required_depth:
            return None

        return {
            **candidate,
            "liquidity": {
                "market_cap_usd": round(market_cap, 2),
                "total_volume_usd": round(total_volume, 2),
                "volume_market_cap_ratio": round(volume_market_cap_ratio, 4),
                "spread_bps": round(float(depth["spread_bps"]), 4),
                "depth_band_pct": self.depth_band_pct,
                "bid_depth_notional_band": round(bid_depth, 2),
                "ask_depth_notional_band": round(ask_depth, 2),
                "required_depth_notional": round(required_depth, 2),
            },
        }

    def run_catalyst_engine(self, candidate: dict) -> dict | None:
        symbol = candidate["venue_symbol"]
        base_asset = candidate["base_asset"]
        ticker = candidate["ticker_24h"]

        try:
            raw = self.binance.get_klines(
                symbol=symbol,
                interval=self.binance_config.interval,
                limit=self.primary_interval_limit,
            )
        except Exception:
            return None

        snapshot = build_market_snapshot(
            build_dataframe_from_klines(raw=raw, base_asset=base_asset, interval=self.binance_config.interval)
        )
        if snapshot is None:
            return None

        hourly = self.binance.get_klines(symbol=symbol, interval="1h", limit=30)
        if len(hourly) < 25:
            return None
        closed_hourly = hourly[:-1] if len(hourly) > 1 else hourly
        recent_bar = closed_hourly[-1]
        last_hour_quote_volume = float(recent_bar[7])
        rolling_avg_quote_volume = sum(float(row[7]) for row in closed_hourly[-24:]) / min(24, len(closed_hourly))
        volume_delta_ratio = last_hour_quote_volume / max(rolling_avg_quote_volume, 1.0)

        try:
            detail = self.fetch_coin_detail(candidate["coingecko"]["id"])
        except Exception as exc:
            detail = self.build_detail_fallback(candidate, str(exc))
        social = self.compute_social_velocity(detail)
        dev = self.compute_dev_activity(detail)
        onchain_proxy = self.compute_onchain_proxy(detail)
        funding = self.fetch_funding_signal(symbol)

        ensemble = evaluate_snapshot(snapshot)
        catalyst_flags = [
            volume_delta_ratio >= self.min_volume_delta_ratio,
            funding["short_squeeze_flag"],
            social["social_spike_flag"],
            dev["dev_activity_flag"],
            onchain_proxy["onchain_velocity_flag"],
        ]
        if not any(catalyst_flags):
            return None

        aligned_contexts = 0
        disagreeing_contexts = 0
        context_summaries: list[dict] = []
        for interval in self.context_intervals:
            try:
                context = fetch_timeframe_context(
                    client=self.binance,
                    symbol=symbol,
                    base_asset=base_asset,
                    interval=interval,
                    limit=220,
                )
            except Exception:
                context = None
            if context is None:
                continue
            context_summaries.append(
                {
                    "interval": interval,
                    "action": context.ensemble_action,
                    "confidence": round(context.ensemble_confidence, 3),
                }
            )
            if context.ensemble_action == ensemble.action:
                aligned_contexts += 1
            elif context.ensemble_action != "HOLD":
                disagreeing_contexts += 1

        if aligned_contexts == 0 and self.context_intervals:
            return None

        if ensemble.action == "HOLD" or ensemble.confidence < self.min_candidate_confidence:
            return None
        if self.prefer_long_only and ensemble.action != "BUY":
            return None

        atr_fraction = snapshot.atr_14 / snapshot.close if snapshot.close > 0 else 0.0
        volatility_component = clamp((atr_fraction - 0.0035) / 0.012, 0.0, 1.0)
        realized_vol_component = clamp(snapshot.volatility_20 / 0.035, 0.0, 1.0)
        trend_component = clamp(snapshot.adx_14 / 35.0, 0.0, 1.0)
        spread_component = clamp(1.0 - (candidate["liquidity"]["spread_bps"] / max(self.max_spread_bps, 1.0)), 0.0, 1.0)
        depth_score = min(candidate["liquidity"]["bid_depth_notional_band"], candidate["liquidity"]["ask_depth_notional_band"])
        liquidity_component = clamp(math.log10(max(depth_score, 1.0)) / 7.0, 0.0, 1.0)
        participation_component = clamp(abs(snapshot.taker_buy_ratio - 0.5) * 2.2, 0.0, 1.0)
        flow_component = clamp(snapshot.volume_zscore_20 / 3.0, 0.0, 1.0)
        alignment_component = clamp((aligned_contexts - disagreeing_contexts + len(self.context_intervals)) / (2 * max(len(self.context_intervals), 1)), 0.0, 1.0)
        catalyst_component = clamp(sum(1 for flag in catalyst_flags if flag) / len(catalyst_flags), 0.0, 1.0)

        score = (
            ensemble.confidence * 26.0
            + alignment_component * 18.0
            + catalyst_component * 18.0
            + trend_component * 12.0
            + volatility_component * 6.0
            + realized_vol_component * 6.0
            + spread_component * 6.0
            + liquidity_component * 4.0
            + participation_component * 2.0
            + flow_component * 2.0
        )

        return {
            **candidate,
            "symbol": symbol,
            "coin_id": candidate["coingecko"]["id"],
            "name": candidate["coingecko"].get("name", base_asset),
            "market_cap_rank": candidate["coingecko"].get("market_cap_rank"),
            "direction": ensemble.action,
            "score": round(score, 2),
            "confidence": round(ensemble.confidence, 3),
            "aligned_contexts": aligned_contexts,
            "disagreeing_contexts": disagreeing_contexts,
            "quote_volume": round(float(ticker.get("quoteVolume", 0.0)), 2),
            "trade_count": int(ticker.get("count", 0)),
            "spread_bps": round(candidate["liquidity"]["spread_bps"], 4),
            "depth_notional_top10": round(depth_score, 2),
            "atr_fraction": round(atr_fraction, 5),
            "volatility_20": round(snapshot.volatility_20, 5),
            "adx_14": round(snapshot.adx_14, 2),
            "returns_20": round(snapshot.returns_20, 4),
            "volume_zscore_20": round(snapshot.volume_zscore_20, 3),
            "contexts": context_summaries,
            "summary": ensemble.summary,
            "catalyst": {
                "last_hour_quote_volume": round(last_hour_quote_volume, 2),
                "avg_hourly_quote_volume_24h": round(rolling_avg_quote_volume, 2),
                "volume_delta_ratio": round(volume_delta_ratio, 3),
                "volume_delta_flag": volume_delta_ratio >= self.min_volume_delta_ratio,
                "detail_fallback_reason": detail.get("_fallback_reason"),
                "funding": funding,
                "social": social,
                "developer": dev,
                "onchain_proxy": onchain_proxy,
            },
            "market_snapshot": {
                "close": round(snapshot.close, 6),
                "atr_14": round(snapshot.atr_14, 6),
                "adx_14": round(snapshot.adx_14, 2),
                "volatility_20": round(snapshot.volatility_20, 5),
                "taker_buy_ratio": round(snapshot.taker_buy_ratio, 3),
                "ema_20": round(snapshot.ema_20, 6),
                "ema_50": round(snapshot.ema_50, 6),
                "macd_histogram": round(snapshot.macd_histogram, 6),
            },
            "detail": detail,
        }

    def run_narrative_layer(self, candidate: dict) -> dict | None:
        detail = candidate.get("detail") or {}
        links = detail.get("links") or {}
        repos = [repo for repo in (links.get("repos_url") or {}).get("github", []) if repo][:3]
        community = candidate["catalyst"]["social"]
        developer = candidate["catalyst"]["developer"]
        funding = candidate["catalyst"]["funding"]
        onchain_proxy = candidate["catalyst"]["onchain_proxy"]
        prompt = f"""
You are the narrative scout on a systematic crypto trading desk.
Decide whether the current catalyst stack justifies the momentum or looks manipulative.

Coin:
- Name: {candidate['name']}
- Symbol: {candidate['base_asset']}
- Rank: {candidate.get('market_cap_rank')}
- Direction: {candidate['direction']}
- Confidence: {candidate['confidence']:.2f}

Liquidity gate:
- Volume/Market Cap Ratio: {candidate['liquidity']['volume_market_cap_ratio']:.4f}
- Spread (bps): {candidate['liquidity']['spread_bps']:.2f}
- Depth within +/- {candidate['liquidity']['depth_band_pct']:.0%}: bid {candidate['liquidity']['bid_depth_notional_band']:.2f}, ask {candidate['liquidity']['ask_depth_notional_band']:.2f}

Catalyst and momentum:
- Last 1h volume delta vs 24h hourly average: {candidate['catalyst']['volume_delta_ratio']:.2f}x
- Funding rate: {funding.get('funding_rate')}
- Social velocity proxy: {community['social_velocity']:.2f}
- Dev commits (4 weeks): {developer['commit_count_4_weeks']}
- PRs merged: {developer['pull_requests_merged']}
- On-chain proxy velocity: {onchain_proxy['onchain_velocity']:.2f}
- Contexts: {candidate['contexts']}
- Technical summary: {candidate['summary']}

Project metadata:
- Homepage: {(links.get('homepage') or [''])[0]}
- GitHub repos: {repos}
- Categories: {detail.get('categories', [])[:6]}

Return whether the current news/catalyst context appears justified, questionable, or inconclusive.
Flag manipulative-looking setups when the catalyst is weak relative to the volume spike.
Output rules:
- narrative_verdict must be exactly one of: JUSTIFIED, QUESTIONABLE, INCONCLUSIVE.
- invalidation_level must be a short sentence of at least 5 characters, not a single label.
- notable_risks must be a JSON array of short strings. Use [] when there are no notable risks.
        """

        if self.enable_narrative_llm:
            try:
                narrative = self.client.generate_json(prompt, NarrativeAssessmentDraft, 0.1)
                return {
                    **candidate,
                    "narrative": narrative,
                }
            except Exception as exc:
                print(
                    f"[ScoutAgent] Narrative output invalid for {candidate['symbol']}; "
                    "using rule-based fallback."
                )

        fallback_verdict = "JUSTIFIED" if (
            candidate["catalyst"]["volume_delta_flag"]
            and (community["social_spike_flag"] or developer["dev_activity_flag"] or funding["short_squeeze_flag"])
        ) else "INCONCLUSIVE"
        return {
            **candidate,
            "narrative": {
                "catalyst_summary": "Rule-based narrative fallback used because the scout LLM was unavailable.",
                "narrative_verdict": fallback_verdict,
                "invalidation_level": "Narrative weakens if the volume anomaly fades within the next few candles.",
                "reasoning": "Fallback accepts setups with confirmed volume plus at least one supporting social, developer, or funding catalyst.",
                "notable_risks": ["Narrative LLM unavailable; review manually if position size is large."],
            },
        }

    def rank_symbols(self) -> list[dict]:
        universe, summary = self.select_scan_universe()
        liquidity_survivors = []
        ranked = []
        for candidate in universe:
            liquidity_candidate = self.run_liquidity_gate(candidate)
            if liquidity_candidate is None:
                coin = candidate["coingecko"]
                volume_market_cap_ratio = float(coin.get("total_volume") or 0.0) / max(float(coin.get("market_cap") or 0.0), 1.0)
                if volume_market_cap_ratio < self.min_volume_market_cap_ratio:
                    summary["rejections"]["weak_volume_market_cap_ratio"] += 1
                else:
                    depth = self.fetch_depth_metrics(candidate["venue_symbol"])
                    if depth is None or float(depth["spread_bps"]) > self.max_spread_bps:
                        summary["rejections"]["wide_spread"] += 1
                    else:
                        summary["rejections"]["thin_depth"] += 1
                continue
            liquidity_survivors.append(liquidity_candidate)

        summary["liquidity_survivors"] = len(liquidity_survivors)
        catalyst_survivors = []
        for candidate in liquidity_survivors:
            enriched = self.run_catalyst_engine(candidate)
            if enriched is None:
                summary["rejections"]["weak_catalyst"] += 1
                continue
            catalyst_survivors.append(enriched)
        summary["catalyst_survivors"] = len(catalyst_survivors)

        for candidate in catalyst_survivors:
            if candidate["direction"] == "HOLD" or candidate["confidence"] < self.min_candidate_confidence:
                summary["rejections"]["weak_technicals"] += 1
                continue
            if self.prefer_long_only and candidate["direction"] != "BUY":
                summary["rejections"]["non_buy_when_long_only"] += 1
                continue
            enriched = self.run_narrative_layer(candidate)
            if enriched is None:
                summary["rejections"]["narrative_failed"] += 1
                continue
            if enriched["narrative"]["narrative_verdict"] == "QUESTIONABLE":
                summary["rejections"]["narrative_failed"] += 1
                continue
            enriched.pop("detail", None)
            ranked.append(enriched)

        ranked.sort(
            key=lambda item: (
                item["score"],
                item["confidence"],
                item["aligned_contexts"],
                -1 * (item.get("market_cap_rank") or 10_000),
                item["quote_volume"],
            ),
            reverse=True,
        )
        ranked = ranked[: self.max_symbols]
        summary["narrative_survivors"] = len(ranked)
        self.last_scan_summary = summary
        return ranked

    async def publish_universe(self):
        ranked = await asyncio.to_thread(self.rank_symbols)
        payload = {
            "generated_at": self.utc_timestamp(),
            "quote_asset": self.quote_asset,
            "market_cap_limit": self.market_cap_limit,
            "scan_limit": self.scan_limit,
            "filters": {
                "min_quote_volume": self.min_quote_volume,
                "min_volume_market_cap_ratio": self.min_volume_market_cap_ratio,
                "max_spread_bps": self.max_spread_bps,
                "depth_band_pct": self.depth_band_pct,
                "min_depth_notional": self.min_depth_notional,
                "min_volume_delta_ratio": self.min_volume_delta_ratio,
                "short_squeeze_funding_threshold": self.short_squeeze_funding_threshold,
            },
            "provider_status": {
                "coingecko_backoff_active": time.time() < self.coingecko_backoff_until,
                "coingecko_backoff_remaining_seconds": round(max(0.0, self.coingecko_backoff_until - time.time()), 1),
            },
            "summary": self.last_scan_summary,
            "symbols": ranked,
        }
        print(
            f"[ScoutAgent] Ranked {len(ranked)} symbols after "
            f"{self.last_scan_summary.get('liquidity_survivors', 0)} liquidity and "
            f"{self.last_scan_summary.get('catalyst_survivors', 0)} catalyst survivors."
        )
        await self.publish_event("SCOUT_UNIVERSE_EVENT", payload)

    async def run(self):
        while True:
            try:
                await self.publish_universe()
            except Exception as exc:
                print(f"[ScoutAgent] Ranking failed: {exc}")
            await asyncio.sleep(self.sleep_seconds)


async def main():
    agent = ScoutAgent()
    await asyncio.gather(agent.heartbeat_loop(), agent.run())


if __name__ == "__main__":
    asyncio.run(main())

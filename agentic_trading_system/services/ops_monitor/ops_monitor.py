import asyncio
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psycopg2

from core.base_agent import BaseAgent
from core.binance_client import BinanceRESTClient, infer_quote_asset, load_binance_config
from core.db import ensure_trades_table


class MonitorState:
    def __init__(self):
        self.started_at = time.time()
        self.last_heartbeats: dict[str, float] = {}
        self.event_counts: dict[str, int] = {
            "alerts": 0,
            "fills": 0,
            "decisions": 0,
            "heartbeats": 0,
            "proposals": 0,
            "reviews": 0,
            "universe_updates": 0,
        }
        self.latest_alerts: list[dict] = []
        self.last_fill_at: float | None = None
        self.last_decision_at: float | None = None
        self.latest_heartbeats: dict[str, dict] = {}
        self.latest_proposals: list[dict] = []
        self.latest_reviews: list[dict] = []
        self.latest_decisions: list[dict] = []
        self.latest_fills: list[dict] = []
        self.latest_universe: dict | None = None
        self.recent_events: list[dict] = []
        self.recent_trades: list[dict] = []
        self.kill_switch_active: bool = False
        self.last_updated_at: float = self.started_at
        self.kill_switch_message: str | None = None
        self.account_snapshot: dict = {
            "available": False,
            "environment": None,
            "quote_asset": None,
            "wallet_estimate_quote": None,
            "quote_balance_free": None,
            "quote_balance_locked": None,
            "open_orders": None,
            "non_zero_assets": [],
            "recent_exchange_trades": [],
            "last_refreshed_at": None,
            "error": None,
        }

    def push_alert(self, alert: dict):
        self.event_counts["alerts"] += 1
        self.latest_alerts.append(alert)
        self.latest_alerts = self.latest_alerts[-50:]
        self.last_updated_at = time.time()

    def push_recent_event(self, channel: str, summary: dict):
        self.recent_events.append(
            {
                "channel": channel,
                "timestamp": time.time(),
                "summary": summary,
            }
        )
        self.recent_events = self.recent_events[-60:]
        self.last_updated_at = time.time()

    def mark_updated(self):
        self.last_updated_at = time.time()


def build_payload(state: MonitorState, stale_seconds: int) -> dict:
    now = time.time()
    stale_agents = [
        agent
        for agent, ts in state.last_heartbeats.items()
        if now - ts > stale_seconds
    ]
    healthy = len(stale_agents) == 0
    return {
        "healthy": healthy,
        "uptime_seconds": round(now - state.started_at, 2),
        "stale_agents": stale_agents,
        "event_counts": state.event_counts,
        "last_fill_at": state.last_fill_at,
        "last_decision_at": state.last_decision_at,
        "kill_switch_active": state.kill_switch_active,
        "kill_switch_message": state.kill_switch_message,
        "account_snapshot": state.account_snapshot,
        "agents": sorted(state.latest_heartbeats.values(), key=lambda item: item["agent"]),
        "alerts": state.latest_alerts,
        "recent_events": state.recent_events,
        "recent_trades": state.recent_trades,
        "latest_universe": state.latest_universe,
        "latest_proposals": state.latest_proposals,
        "latest_reviews": state.latest_reviews,
        "latest_decisions": state.latest_decisions,
        "latest_fills": state.latest_fills,
        "updated_at": state.last_updated_at,
    }


def make_handler(state: MonitorState, stale_seconds: int, dashboard_html: str):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status_code: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/api/kill-switch":
                self.send_response(404)
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON body"}, 400)
                return

            active = data.get("active")
            if not isinstance(active, bool):
                self._send_json({"error": "Field 'active' must be boolean"}, 400)
                return

            try:
                result = state.toggle_kill_switch(active)
            except Exception as exc:
                self._send_json({"error": f"Kill switch update failed: {exc}"}, 500)
                return

            payload = build_payload(state, stale_seconds)
            payload["kill_switch_active"] = result
            self._send_json(payload)

        def do_GET(self):
            payload = build_payload(state, stale_seconds)
            if self.path == "/" or self.path == "/dashboard":
                body = dashboard_html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                last_sent_at = None
                try:
                    while True:
                        payload = build_payload(state, stale_seconds)
                        if payload["updated_at"] != last_sent_at:
                            body = json.dumps(payload)
                            self.wfile.write(f"data: {body}\n\n".encode("utf-8"))
                            self.wfile.flush()
                            last_sent_at = payload["updated_at"]
                        time.sleep(1)
                except (BrokenPipeError, ConnectionResetError):
                    return
            if self.path == "/alerts":
                payload = {"alerts": state.latest_alerts}
            elif self.path == "/metrics":
                payload = {
                    "heartbeats": state.event_counts["heartbeats"],
                    "alerts": state.event_counts["alerts"],
                    "fills": state.event_counts["fills"],
                    "decisions": state.event_counts["decisions"],
                    "proposals": state.event_counts["proposals"],
                    "reviews": state.event_counts["reviews"],
                    "universe_updates": state.event_counts["universe_updates"],
                    "known_agents": sorted(state.last_heartbeats.keys()),
                }
            elif self.path == "/api/snapshot":
                pass
            elif self.path != "/health":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"not found")
                return

            self._send_json(payload, 200 if payload.get("healthy", True) else 503)

        def log_message(self, format, *args):
            return

    return Handler


class OpsMonitor(BaseAgent):
    def __init__(self):
        super().__init__("OpsMonitor")
        self.state = MonitorState()
        self.state.toggle_kill_switch = self.toggle_kill_switch
        self.stale_heartbeat_seconds = int(os.getenv("STALE_HEARTBEAT_SECONDS", "120"))
        self.max_fill_slippage_bps = float(os.getenv("ALERT_MAX_FILL_SLIPPAGE_BPS", "15"))
        self.http_port = int(os.getenv("OPS_MONITOR_PORT", "8080"))
        self.dashboard_html = Path(__file__).with_name("dashboard.html").read_text()
        self.binance_config = load_binance_config()
        self.binance = BinanceRESTClient(self.binance_config)
        self.quote_asset = infer_quote_asset(self.binance_config.symbol)
        self.db_config = {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "database": os.getenv("POSTGRES_DB", "trading_ledger"),
            "user": os.getenv("POSTGRES_USER", "trader"),
            "password": os.getenv("POSTGRES_PASSWORD", "securepassword"),
        }
        self.db_conn = None
        self.loop = None
        self.account_refresh_seconds = int(os.getenv("ACCOUNT_REFRESH_SECONDS", "20"))

    def toggle_kill_switch(self, active: bool) -> bool:
        if self.loop is None:
            raise RuntimeError("OpsMonitor event loop is not ready")
        future = asyncio.run_coroutine_threadsafe(self.set_kill_switch(active), self.loop)
        is_active = future.result(timeout=10)
        self.state.kill_switch_active = is_active
        self.state.kill_switch_message = (
            "Trading approvals and execution are blocked."
            if is_active
            else "Trading approvals and execution are enabled."
        )
        self.state.mark_updated()
        return is_active

    def ensure_db_conn(self):
        if self.db_conn and not self.db_conn.closed:
            return self.db_conn
        self.db_conn = psycopg2.connect(**self.db_config)
        self.db_conn.autocommit = False
        ensure_trades_table(self.db_conn)
        return self.db_conn

    def start_http_server(self):
        server = ThreadingHTTPServer(
            ("0.0.0.0", self.http_port),
            make_handler(self.state, self.stale_heartbeat_seconds, self.dashboard_html),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[OpsMonitor] HTTP dashboard listening on :{self.http_port}")

    def load_recent_trades(self):
        try:
            conn = self.ensure_db_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ticker, action, quantity, fill_price, slippage_bps, paper_trade, timestamp
                    FROM trades
                    ORDER BY timestamp DESC
                    LIMIT 25
                    """
                )
                rows = cur.fetchall()
        except Exception as exc:
            print(f"[OpsMonitor] Failed loading trades: {exc}")
            if self.db_conn and not self.db_conn.closed:
                self.db_conn.rollback()
                self.db_conn.close()
            return

        self.state.recent_trades = [
            {
                "ticker": row[0],
                "action": row[1],
                "quantity": float(row[2]),
                "fill_price": float(row[3]),
                "slippage_bps": float(row[4]) if row[4] is not None else None,
                "paper_trade": bool(row[5]) if row[5] is not None else True,
                "timestamp": row[6].isoformat(sep=" ", timespec="seconds"),
            }
            for row in rows
        ]
        self.state.mark_updated()

    def refresh_account_snapshot(self):
        snapshot = {
            "available": False,
            "environment": self.binance_config.env,
            "quote_asset": self.quote_asset,
            "wallet_estimate_quote": None,
            "quote_balance_free": None,
            "quote_balance_locked": None,
            "open_orders": None,
            "non_zero_assets": [],
            "recent_exchange_trades": [],
            "last_refreshed_at": time.time(),
            "error": None,
        }
        try:
            account = self.binance.get_account()
            balances = account.get("balances", [])
            non_zero_assets = []
            wallet_estimate = 0.0
            price_symbols: list[str] = []
            for row in balances:
                free = float(row.get("free", 0.0))
                locked = float(row.get("locked", 0.0))
                total = free + locked
                if total <= 0:
                    continue
                asset = row.get("asset", "")
                non_zero_assets.append(
                    {
                        "asset": asset,
                        "free": free,
                        "locked": locked,
                        "total": total,
                        "value_in_quote": None,
                    }
                )
                if asset == self.quote_asset:
                    wallet_estimate += total
                else:
                    price_symbols.append(f"{asset}{self.quote_asset}")

            price_map = {}
            for venue_symbol in price_symbols[:100]:
                try:
                    price_map[venue_symbol] = self.binance.get_last_price(venue_symbol)
                except Exception:
                    continue

            for item in non_zero_assets:
                if item["asset"] == self.quote_asset:
                    item["value_in_quote"] = round(item["total"], 4)
                else:
                    venue_symbol = f"{item['asset']}{self.quote_asset}"
                    last_price = price_map.get(venue_symbol)
                    if last_price:
                        item["value_in_quote"] = round(item["total"] * last_price, 4)
                        wallet_estimate += item["total"] * last_price

            non_zero_assets.sort(
                key=lambda item: item["value_in_quote"] if item["value_in_quote"] is not None else 0.0,
                reverse=True,
            )
            quote_balance = next((item for item in non_zero_assets if item["asset"] == self.quote_asset), None)
            open_orders = self.binance.get_open_orders()

            candidate_symbols = {
                trade.get("ticker")
                for trade in self.state.recent_trades
                if trade.get("ticker")
            }
            latest_universe = self.state.latest_universe or {}
            for item in latest_universe.get("symbols", [])[:12]:
                if isinstance(item, dict) and item.get("symbol"):
                    candidate_symbols.add(item["symbol"].replace(self.quote_asset, ""))
            candidate_symbols.update({"BTC", "ETH"})

            exchange_trades: list[dict] = []
            for ticker in sorted(candidate_symbols):
                venue_symbol = f"{ticker}{self.quote_asset}"
                try:
                    trades = self.binance.get_my_trades(venue_symbol, limit=5)
                except Exception:
                    continue
                for trade in trades:
                    exchange_trades.append(
                        {
                            "symbol": venue_symbol,
                            "side": "BUY" if trade.get("isBuyer") else "SELL",
                            "price": float(trade.get("price", 0.0)),
                            "qty": float(trade.get("qty", 0.0)),
                            "quote_qty": float(trade.get("quoteQty", 0.0)),
                            "commission": f"{trade.get('commission', '0')} {trade.get('commissionAsset', '')}".strip(),
                            "time": int(trade.get("time", 0)) / 1000.0 if trade.get("time") else None,
                        }
                    )
            exchange_trades.sort(key=lambda item: item["time"] or 0.0, reverse=True)

            snapshot.update(
                {
                    "available": True,
                    "wallet_estimate_quote": round(wallet_estimate, 4),
                    "quote_balance_free": round(quote_balance["free"], 4) if quote_balance else 0.0,
                    "quote_balance_locked": round(quote_balance["locked"], 4) if quote_balance else 0.0,
                    "open_orders": len(open_orders),
                    "permissions": account.get("permissions", []),
                    "can_trade": bool(account.get("canTrade", False)),
                    "maker_commission_bps": float(account.get("makerCommission", 0)) / 100.0,
                    "taker_commission_bps": float(account.get("takerCommission", 0)) / 100.0,
                    "non_zero_assets": non_zero_assets[:10],
                    "recent_exchange_trades": exchange_trades[:10],
                }
            )
        except Exception as exc:
            snapshot["error"] = str(exc)

        self.state.account_snapshot = snapshot
        self.state.mark_updated()

    async def monitor_loop(self):
        last_account_refresh = 0.0
        while True:
            self.state.kill_switch_active = await self.is_kill_switch_active()
            self.state.kill_switch_message = (
                "Trading approvals and execution are blocked."
                if self.state.kill_switch_active
                else "Trading approvals and execution are enabled."
            )
            await asyncio.to_thread(self.load_recent_trades)
            if time.time() - last_account_refresh >= self.account_refresh_seconds:
                await asyncio.to_thread(self.refresh_account_snapshot)
                last_account_refresh = time.time()
            await asyncio.sleep(5)

    async def handle_event(self, channel: str, data: dict):
        now = self.utc_timestamp()
        if channel == "HEARTBEAT_EVENT":
            agent = data.get("agent")
            if agent:
                self.state.last_heartbeats[agent] = float(data.get("timestamp", now))
                self.state.latest_heartbeats[agent] = {
                    "agent": agent,
                    "timestamp": float(data.get("timestamp", now)),
                    "status": data.get("status", "healthy"),
                }
                self.state.event_counts["heartbeats"] += 1
                self.state.mark_updated()
        elif channel == "ORDER_FILL_EVENT":
            self.state.event_counts["fills"] += 1
            self.state.last_fill_at = now
            self.state.latest_fills.append(data)
            self.state.latest_fills = self.state.latest_fills[-25:]
            self.state.push_recent_event(
                channel,
                {
                    "ticker": data.get("ticker"),
                    "action": data.get("action"),
                    "fill_price": data.get("fill_price"),
                    "slippage_bps": data.get("slippage_bps"),
                },
            )
            slippage = float(data.get("slippage_bps", 0.0))
            if abs(slippage) > self.max_fill_slippage_bps:
                self.state.push_alert(
                    {
                        "agent": "ExecutionAgent",
                        "level": "warning",
                        "message": "High slippage detected",
                        "timestamp": now,
                        "details": data,
                    }
                )
        elif channel == "JUDGE_DECISION_EVENT":
            self.state.event_counts["decisions"] += 1
            self.state.last_decision_at = now
            self.state.latest_decisions.append(data)
            self.state.latest_decisions = self.state.latest_decisions[-25:]
            self.state.push_recent_event(
                channel,
                {
                    "ticker": data.get("proposal", {}).get("ticker"),
                    "verdict": data.get("verdict"),
                    "approved_action": data.get("approved_action"),
                    "confidence": data.get("confidence"),
                },
            )
        elif channel == "SIGNAL_PROPOSAL_EVENT":
            self.state.event_counts["proposals"] += 1
            self.state.latest_proposals.append(data)
            self.state.latest_proposals = self.state.latest_proposals[-25:]
            self.state.push_recent_event(
                channel,
                {
                    "ticker": data.get("ticker"),
                    "action": data.get("action"),
                    "confidence": data.get("confidence"),
                },
            )
        elif channel == "SIGNAL_REVIEW_EVENT":
            self.state.event_counts["reviews"] += 1
            self.state.latest_reviews.append(data)
            self.state.latest_reviews = self.state.latest_reviews[-25:]
            self.state.push_recent_event(
                channel,
                {
                    "ticker": data.get("proposal", {}).get("ticker"),
                    "verdict": data.get("verdict"),
                    "blocking": data.get("blocking"),
                    "confidence": data.get("confidence"),
                },
            )
        elif channel == "SCOUT_UNIVERSE_EVENT":
            self.state.event_counts["universe_updates"] += 1
            self.state.latest_universe = data
            top_symbols = [item.get("symbol") for item in data.get("symbols", [])[:5]]
            self.state.push_recent_event(channel, {"top_symbols": top_symbols})
        elif channel == "ALERT_EVENT":
            self.state.push_alert(data)

    async def run(self):
        self.start_http_server()
        await self.pubsub.subscribe(
            "HEARTBEAT_EVENT",
            "ORDER_FILL_EVENT",
            "JUDGE_DECISION_EVENT",
            "SIGNAL_PROPOSAL_EVENT",
            "SIGNAL_REVIEW_EVENT",
            "SCOUT_UNIVERSE_EVENT",
            "ALERT_EVENT",
        )
        print("[OpsMonitor] Listening on heartbeat, workflow, scout, and alert channels...")

        async for message in self.pubsub.listen():
            if message["type"] != "message":
                continue
            channel = message["channel"]
            try:
                data = json.loads(message["data"])
            except json.JSONDecodeError as exc:
                print(f"[OpsMonitor] Dropping malformed payload: {exc}")
                continue
            await self.handle_event(channel, data)


async def main():
    agent = OpsMonitor()
    agent.loop = asyncio.get_running_loop()
    await asyncio.gather(agent.heartbeat_loop(), agent.monitor_loop(), agent.run())


if __name__ == "__main__":
    asyncio.run(main())

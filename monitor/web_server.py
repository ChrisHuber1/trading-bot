"""
Web dashboard server using aiohttp.

Serves a live HTML dashboard on the configured port and pushes
real-time state updates to connected browsers via WebSocket.
Runs as an asyncio task inside the main trading loop.
"""

import asyncio
import json
import weakref
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path

from aiohttp import web
from loguru import logger

import config

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard.html"


class _DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return str(o)
        return super().default(o)


class WebDashboard:
    """aiohttp-based web dashboard with WebSocket push."""

    def __init__(self, host: str = config.WEB_DASHBOARD_HOST,
                 port: int = config.WEB_DASHBOARD_PORT) -> None:
        self._host = host
        self._port = port
        self._clients: weakref.WeakSet = weakref.WeakSet()
        self._html = _TEMPLATE_PATH.read_text(encoding="utf-8")
        self._last_state: str = "{}"

    async def start(self) -> None:
        """Start the web server as a background asyncio task."""
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/ws", self._handle_ws)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        logger.info(
            "Web dashboard running at http://{host}:{port}",
            host=self._host, port=self._port,
        )

    async def _handle_index(self, request: web.Request) -> web.Response:
        return web.Response(text=self._html, content_type="text/html")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        logger.debug("WebSocket client connected ({n} total)", n=len(self._clients))

        # Send current state immediately on connect
        try:
            await ws.send_str(self._last_state)
        except Exception:
            pass

        try:
            async for msg in ws:
                pass  # We don't expect messages from the client
        finally:
            self._clients.discard(ws)
            logger.debug("WebSocket client disconnected ({n} remaining)", n=len(self._clients))

        return ws

    async def broadcast(self, state: dict) -> None:
        """Push a JSON state snapshot to all connected clients."""
        payload = json.dumps(state, cls=_DecimalEncoder)
        self._last_state = payload

        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_str(payload)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self._clients.discard(ws)

    @staticmethod
    def _read_trade_history(hours: int = 48) -> list[dict]:
        """Read trades.jsonl and return entries from the last N hours."""
        if not config.TRADE_LOG.exists():
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        trades = []
        try:
            with open(config.TRADE_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts_str = record.get("timestamp", "")
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        continue
                    if ts < cutoff:
                        continue
                    event = record.get("event", "")
                    if event not in ("order_filled", "position_closed"):
                        continue
                    trades.append(record)
        except OSError:
            pass
        return trades

    @staticmethod
    def build_snapshot(
        mode: str,
        uptime_seconds: int,
        last_signal: str,
        portfolio_value: Decimal,
        current_prices: dict,
        feeds: dict,
        order_mgr,
        risk_mgr,
    ) -> dict:
        """Collect all dashboard state into a JSON-serializable dict."""
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        market = []
        for pair in config.TRADING_PAIRS:
            price = current_prices.get(pair, Decimal("0"))
            feed = feeds.get(pair)
            latest = feed.latest if feed else None

            macd_hist = float(latest.get("macd_hist", 0) or 0) if latest is not None else 0.0
            adx_val = float(latest.get("adx", 0) or 0) if latest is not None else 0.0
            stoch_k = float(latest.get("stochrsi_k", 0) or 0) if latest is not None else 0.0
            stoch_d = float(latest.get("stochrsi_d", 0) or 0) if latest is not None else 0.0

            trend = "BULL" if macd_hist > 0 and adx_val >= config.ADX_THRESHOLD else "BEAR"

            market.append({
                "pair": pair,
                "price": str(price),
                "macd_hist": f"{macd_hist:+.8f}",
                "adx": f"{adx_val:.1f}",
                "stochrsi_k": f"{stoch_k:.4f}",
                "stochrsi_d": f"{stoch_d:.4f}",
                "trend": trend,
            })

        open_positions = []
        for p in order_mgr.open_positions:
            p_price = current_prices.get(p.pair, p.entry_price)
            if p.side == "long":
                unrealized = (p_price - p.entry_price) * p.amount
            else:
                unrealized = (p.entry_price - p_price) * p.amount

            open_positions.append({
                "id": p.entry_order_id,
                "pair": p.pair,
                "side": p.side,
                "entry_price": str(p.entry_price),
                "amount": str(p.amount),
                "stop_loss": str(p.stop_loss),
                "take_profit": str(p.take_profit),
                "unrealized_pnl": str(unrealized),
            })

        realized = order_mgr.total_realized_pnl
        unrealized_total = sum(
            Decimal(p["unrealized_pnl"]) for p in open_positions
        )
        total_pnl = realized + unrealized_total

        return {
            "mode": mode.upper(),
            "uptime": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "last_signal": last_signal,
            "portfolio_value": str(portfolio_value),
            "market": market,
            "open_positions": open_positions,
            "closed_trades": len(order_mgr.closed_positions),
            "risk": {
                "halted": risk_mgr.is_halted,
                "halt_reason": risk_mgr.halt_reason if risk_mgr.is_halted else None,
                "capital_floor": str(config.CAPITAL_FLOOR),
                "portfolio_pct": float(portfolio_value / config.STARTING_CAPITAL * 100),
                "open_count": len(order_mgr.open_positions),
                "max_positions": config.MAX_OPEN_POSITIONS,
                "per_trade_risk_pct": str(config.PER_TRADE_RISK_PCT),
                "stop_loss_pct": str(config.STOP_LOSS_PCT),
                "take_profit_pct": str(config.TAKE_PROFIT_PCT),
                "tp_mode": config.TP_MODE,
            },
            "pnl": {
                "realized": str(realized),
                "unrealized": str(unrealized_total),
                "total": str(total_pnl),
            },
            "trade_history": WebDashboard._read_trade_history(hours=48),
        }

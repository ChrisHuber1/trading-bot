"""
Rich terminal dashboard for the trading bot.

Displays real-time information: mode (PAPER/LIVE), current price,
indicators, open positions, P&L, risk status, and recent trade log.
Refreshes every N seconds (configured in config.py).
"""

from decimal import Decimal
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config
from execution.order_manager import OrderManager, Position
from risk.manager import RiskManager


class Dashboard:
    """Rich-based terminal dashboard for real-time bot monitoring."""

    def __init__(
        self,
        order_manager: OrderManager,
        risk_manager: RiskManager,
        paper_mode: bool = True,
    ) -> None:
        """
        Initialize the dashboard.

        Args:
            order_manager: OrderManager for position and trade data.
            risk_manager: RiskManager for halt status and risk state.
            paper_mode: Whether the bot is in paper trading mode.
        """
        self._om = order_manager
        self._rm = risk_manager
        self._paper_mode = paper_mode
        self._console = Console()
        self._live: Optional[Live] = None

        # State updated externally before each refresh
        self.current_prices: dict[str, Decimal] = {}
        self.portfolio_value: Decimal = config.STARTING_CAPITAL
        self.feeds: dict = {}
        self.last_signal: str = "HOLD"
        self.uptime_seconds: int = 0

    def start(self) -> Live:
        """
        Start the Live display context. Returns the Live object so the
        main loop can call live.update() with generate_layout().
        """
        self._live = Live(
            self.generate_layout(),
            console=self._console,
            refresh_per_second=1,
            screen=True,
        )
        return self._live

    def generate_layout(self) -> Layout:
        """
        Generate the full dashboard layout.

        Returns:
            Rich Layout with all panels arranged.
        """
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        layout["header"].update(self._header_panel())
        layout["left"].split_column(
            Layout(self._market_panel(), name="market", ratio=1),
            Layout(self._positions_panel(), name="positions", ratio=1),
        )
        layout["right"].split_column(
            Layout(self._risk_panel(), name="risk", ratio=1),
            Layout(self._pnl_panel(), name="pnl", ratio=1),
        )
        layout["footer"].update(self._status_bar())

        return layout

    # ── Panels ───────────────────────────────────────────────────────────

    def _header_panel(self) -> Panel:
        """Top banner showing mode and pairs."""
        mode_text = "[bold white on red] LIVE MODE [/]" if not self._paper_mode else "[bold white on blue] PAPER MODE [/]"
        pairs_str = ", ".join(config.TRADING_PAIRS)
        title = Text.from_markup(
            f" {mode_text}  {pairs_str}  |  {config.TIMEFRAME_ENTRY} candles"
        )
        return Panel(title, style="bold")

    def _market_panel(self) -> Panel:
        """Current market data and indicators for all pairs."""
        table = Table(expand=True)
        table.add_column("Pair", style="bold", ratio=1)
        table.add_column("Price", justify="right", ratio=1)
        table.add_column("MACD", justify="right", ratio=1)
        table.add_column("ADX", justify="right", ratio=1)
        table.add_column("StochRSI", justify="right", ratio=1)
        table.add_column("Trend", justify="center", ratio=1)

        for pair in config.TRADING_PAIRS:
            price = self.current_prices.get(pair, Decimal("0"))
            feed = self.feeds.get(pair)
            latest = feed.latest if feed else None

            macd_hist = float(latest.get("macd_hist", 0) or 0) if latest is not None else 0.0
            adx = float(latest.get("adx", 0) or 0) if latest is not None else 0.0
            stoch_k = float(latest.get("stochrsi_k", 0) or 0) if latest is not None else 0.0

            macd_style = "green" if macd_hist > 0 else "red"
            adx_style = "green" if adx >= config.ADX_THRESHOLD else "dim"

            if stoch_k >= float(config.STOCHRSI_OVERBOUGHT):
                stoch_style = "red"
            elif stoch_k <= float(config.STOCHRSI_OVERSOLD):
                stoch_style = "green"
            else:
                stoch_style = "yellow"

            trend = "[green]BULL[/]" if macd_hist > 0 and adx >= config.ADX_THRESHOLD else "[red]BEAR[/]"

            table.add_row(
                pair,
                f"${price:,.4f}",
                f"[{macd_style}]{macd_hist:+.6f}[/]",
                f"[{adx_style}]{adx:.1f}[/]",
                f"[{stoch_style}]{stoch_k:.3f}[/]",
                trend,
            )

        table.add_row("", "", "", "", "", "")
        table.add_row("[dim]Last Signal[/]", f"[bold]{self.last_signal}[/]", "", "", "", "")

        return Panel(table, title="Market Data", border_style="cyan")

    def _positions_panel(self) -> Panel:
        """Open positions table."""
        table = Table(expand=True)
        table.add_column("ID", style="dim", max_width=14)
        table.add_column("Pair", style="bold")
        table.add_column("Side", justify="center")
        table.add_column("Entry", justify="right")
        table.add_column("Amount", justify="right")
        table.add_column("SL", justify="right")
        table.add_column("TP", justify="right")
        table.add_column("Unrealized", justify="right")

        for pos in self._om.open_positions:
            p_price = self.current_prices.get(pos.pair, pos.entry_price)
            if pos.side == "long":
                unrealized = (p_price - pos.entry_price) * pos.amount
                side_display = "[green]LONG[/]"
            else:
                unrealized = (pos.entry_price - p_price) * pos.amount
                side_display = "[red]SHORT[/]"
            pnl_style = "green" if unrealized >= 0 else "red"
            table.add_row(
                pos.entry_order_id[-10:],
                pos.pair,
                side_display,
                f"${pos.entry_price:,.4f}",
                f"{pos.amount:.8f}",
                f"${pos.stop_loss:,.4f}",
                f"${pos.take_profit:,.4f}",
                f"[{pnl_style}]${unrealized:,.4f}[/]",
            )

        if not self._om.open_positions:
            table.add_row("--", "--", "--", "--", "--", "--", "--", "--")

        return Panel(table, title=f"Open Positions ({len(self._om.open_positions)}/{config.MAX_OPEN_POSITIONS})", border_style="green")

    def _risk_panel(self) -> Panel:
        """Risk manager status."""
        table = Table(show_header=False, expand=True, box=None)
        table.add_column("Rule", style="dim", ratio=2)
        table.add_column("Status", ratio=1)

        halt_status = "[bold red]HALTED[/]" if self._rm.is_halted else "[green]OK[/]"
        table.add_row("Trading Status", halt_status)

        if self._rm.is_halted:
            table.add_row("Halt Reason", f"[red]{self._rm.halt_reason}[/]")

        floor_pct = (self.portfolio_value / config.STARTING_CAPITAL * 100)
        floor_style = "green" if self.portfolio_value > config.CAPITAL_FLOOR else "red"
        table.add_row("Capital Floor", f"[{floor_style}]${config.CAPITAL_FLOOR} (at ${self.portfolio_value:,.2f} = {floor_pct:.1f}%)[/]")

        table.add_row("Max Positions", f"{len(self._om.open_positions)}/{config.MAX_OPEN_POSITIONS}")
        table.add_row("Per-Trade Risk", f"{config.PER_TRADE_RISK_PCT}%")
        table.add_row("Stop-Loss", f"{config.STOP_LOSS_PCT}%")
        table.add_row("Take-Profit", f"{config.TAKE_PROFIT_PCT}%")

        return Panel(table, title="Risk Manager", border_style="yellow")

    def _pnl_panel(self) -> Panel:
        """Profit & Loss summary."""
        table = Table(show_header=False, expand=True, box=None)
        table.add_column("Metric", style="dim", ratio=2)
        table.add_column("Value", ratio=1)

        realized = self._om.total_realized_pnl
        realized_style = "green" if realized >= 0 else "red"
        table.add_row("Realized P&L", f"[{realized_style}]${realized:,.2f}[/]")

        # Unrealized P&L from open positions (directional, multi-pair)
        unrealized = Decimal("0")
        for p in self._om.open_positions:
            p_price = self.current_prices.get(p.pair, p.entry_price)
            if p.side == "long":
                unrealized += (p_price - p.entry_price) * p.amount
            else:
                unrealized += (p.entry_price - p_price) * p.amount
        unrealized_style = "green" if unrealized >= 0 else "red"
        table.add_row("Unrealized P&L", f"[{unrealized_style}]${unrealized:,.4f}[/]")

        total = realized + unrealized
        total_style = "green" if total >= 0 else "red"
        table.add_row("Total P&L", f"[bold {total_style}]${total:,.2f}[/]")

        table.add_row("Portfolio Value", f"[bold]${self.portfolio_value:,.2f}[/]")
        table.add_row("Closed Trades", str(len(self._om.closed_positions)))

        return Panel(table, title="Performance", border_style="magenta")

    def _status_bar(self) -> Panel:
        """Bottom status bar with uptime and operational info."""
        hours, remainder = divmod(self.uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        mode = "PAPER" if self._paper_mode else "LIVE"
        pairs_str = ", ".join(config.TRADING_PAIRS)
        status = Text.from_markup(
            f" Mode: [bold]{mode}[/]  |  "
            f"Uptime: {uptime_str}  |  "
            f"Pairs: {pairs_str}  |  "
            f"Timeframe: {config.TIMEFRAME_ENTRY}  |  "
            f"Press [bold]Ctrl+C[/] to stop"
        )
        return Panel(status, style="dim")

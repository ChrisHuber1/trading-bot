"""
Trading bot entry point.

Usage:
    python main.py --mode paper     # Paper trading (default)
    python main.py --mode backtest  # Run backtest on historical data
    python main.py --mode live      # Live trading (requires 14-day paper gate)

The bot connects to Kraken, streams real-time candle data, applies the
EMA/RSI strategy, routes orders through the risk manager, and displays
everything on a Rich terminal dashboard.
"""

import argparse
import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import os

import config
from exchange.kraken_client import KrakenClient, KrakenClientError
from data.feed import DataFeed
from strategy.signals import Signal, TradeSignal
from strategy.macd_stochrsi import MacdStochRsiStrategy
from risk.manager import RiskManager
from execution.order_manager import OrderManager
from monitor.dashboard import Dashboard
from monitor.web_server import WebDashboard
from backtest.runner import BacktestRunner


# ── Logging Setup ────────────────────────────────────────────────────────

def setup_logging() -> None:
    """Configure loguru: structured JSON to file, human-readable to stderr."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    logger.add(
        config.LOG_DIR / "bot_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        serialize=True,  # JSON format
    )


# ── Paper Trading Gate ───────────────────────────────────────────────────

def check_paper_trading_gate() -> None:
    """
    Enforce the 14-day paper trading requirement before live trading.

    Reads paper_trading_start_date from state.json. If the bot hasn't
    been paper trading for at least 14 days, live mode is refused.
    This cannot be bypassed by editing config.py -- it reads state.json.

    Raises:
        SystemExit: If the gate is not satisfied.
    """
    if not config.STATE_FILE.exists():
        logger.error(
            "state.json not found. Run in paper mode first to initialize it."
        )
        sys.exit(1)

    try:
        with open(config.STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read state.json: {err}", err=exc)
        sys.exit(1)

    start_date_str = state.get("paper_trading_start_date")
    if not start_date_str:
        logger.error(
            "No paper_trading_start_date in state.json. "
            "Run in paper mode first."
        )
        sys.exit(1)

    start_date = datetime.fromisoformat(start_date_str).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    elapsed_days = (now - start_date).days

    if elapsed_days < config.PAPER_TRADING_DAYS_REQUIRED:
        remaining = config.PAPER_TRADING_DAYS_REQUIRED - elapsed_days
        logger.error(
            "Paper trading gate: {elapsed} of {required} days completed. "
            "{remaining} days remaining. Live trading not yet allowed.",
            elapsed=elapsed_days,
            required=config.PAPER_TRADING_DAYS_REQUIRED,
            remaining=remaining,
        )
        sys.exit(1)

    logger.info(
        "Paper trading gate passed: {days} days of paper trading completed",
        days=elapsed_days,
    )


def initialize_state() -> None:
    """
    Create or update state.json with paper trading start date.
    Only sets the start date if it doesn't already exist.
    """
    state = {}
    if config.STATE_FILE.exists():
        try:
            with open(config.STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            state = {}

    if "paper_trading_start_date" not in state:
        state["paper_trading_start_date"] = datetime.now(timezone.utc).isoformat()
        logger.info("Paper trading started -- 14-day gate begins now")

    with open(config.STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ── Main Trading Loop ───────────────────────────────────────────────────

async def run_trading_loop(mode: str) -> None:
    """
    Main async trading loop for multiple pairs.

    Connects to Kraken, loads history for each pair, then enters a loop that:
    1. Refreshes candle data via REST for all pairs
    2. Fetches current prices for all pairs
    3. Checks exit conditions on open positions
    4. Evaluates the strategy per pair on new candles
    5. Routes new orders through the shared risk manager
    6. Updates the terminal dashboard

    Args:
        mode: 'paper' or 'live'.
    """
    load_dotenv()

    api_key = os.environ.get("KRAKEN_API_KEY", "")
    api_secret = os.environ.get("KRAKEN_API_SECRET", "")

    if not api_key or not api_secret:
        logger.error(
            "KRAKEN_API_KEY and KRAKEN_API_SECRET must be set in .env file. "
            "See .env.example for the template."
        )
        sys.exit(1)

    paper_mode = mode == "paper"
    pairs = config.TRADING_PAIRS

    # ── Initialize shared components ─────────────────────────────────
    client = KrakenClient(
        api_key=api_key,
        api_secret=api_secret,
        sandbox=paper_mode,
    )

    risk_mgr = RiskManager(starting_capital=config.STARTING_CAPITAL)
    order_mgr = OrderManager(client, risk_mgr, paper_mode=paper_mode)
    dashboard = Dashboard(order_mgr, risk_mgr, paper_mode=paper_mode)
    web_dash = WebDashboard()

    # ── Per-pair components ──────────────────────────────────────────
    feeds: dict[str, DataFeed] = {}
    strategies: dict[str, MacdStochRsiStrategy] = {}
    first_candle_eval: dict[str, bool] = {}

    for pair in pairs:
        feeds[pair] = DataFeed(client, pair=pair)
        strategies[pair] = MacdStochRsiStrategy(portfolio_value=config.STARTING_CAPITAL)
        first_candle_eval[pair] = True

    # ── Load historical data ─────────────────────────────────────────
    logger.info("Loading historical candle data for {n} pairs...", n=len(pairs))
    for pair in pairs:
        logger.info("Loading history for {pair}...", pair=pair)
        feeds[pair].load_history(limit=500)

    portfolio_value = config.STARTING_CAPITAL
    start_time = time.monotonic()

    # ── Start dashboards ─────────────────────────────────────────────
    live = dashboard.start()
    await web_dash.start()

    logger.info(
        "Trading bot started | mode={mode} | pairs={pairs}",
        mode=mode.upper(),
        pairs=pairs,
    )

    poll_interval = 30
    last_candle_refresh = 0.0

    try:
        with live:
            while True:
                try:
                    now = time.monotonic()
                    current_prices: dict[str, Decimal] = {}

                    # ── Refresh candle data via REST (all pairs) ─────
                    if now - last_candle_refresh >= 60:
                        for pair in pairs:
                            try:
                                raw_entry = client.fetch_ohlcv(
                                    pair, limit=100, timeframe=config.TIMEFRAME_ENTRY
                                )
                                if raw_entry:
                                    for candle in raw_entry[-5:]:
                                        feeds[pair].update_entry_candle(candle)

                                raw_trend = client.fetch_ohlcv(
                                    pair, limit=100, timeframe=config.TIMEFRAME_TREND
                                )
                                if raw_trend:
                                    for candle in raw_trend[-5:]:
                                        feeds[pair].update_trend_candle(candle)
                            except KrakenClientError as exc:
                                logger.warning("Candle refresh failed for {pair}: {err}", pair=pair, err=exc)

                        last_candle_refresh = now

                    # ── Get current prices via ticker (all pairs) ────
                    skip_cycle = False
                    for pair in pairs:
                        try:
                            ticker = client.fetch_ticker(pair)
                            current_prices[pair] = Decimal(str(ticker["last"]))
                        except KrakenClientError as exc:
                            logger.warning("Ticker fetch failed for {pair}: {err}", pair=pair, err=exc)

                    if not current_prices:
                        await asyncio.sleep(poll_interval)
                        continue

                    # ── Update risk manager state ────────────────────
                    risk_mgr.update_state(
                        portfolio_value=portfolio_value,
                        open_positions=len(order_mgr.open_positions),
                    )

                    # ── Check exit conditions on open positions ──────
                    positions_before = len(order_mgr.open_positions)
                    order_mgr.check_exit_conditions(current_prices)
                    positions_closed = positions_before - len(order_mgr.open_positions)

                    # ── Evaluate strategy per pair ───────────────────
                    # If a TP/SL just freed a slot, force one immediate
                    # re-evaluation of all pairs to catch new entries
                    # without waiting for the next hourly candle.
                    force_eval = positions_closed > 0

                    last_signal_text = "HOLD"
                    for pair in pairs:
                        if pair not in current_prices:
                            continue

                        feed = feeds[pair]
                        df_entry = feed.entry_dataframe
                        df_trend = feed.trend_dataframe
                        min_periods = config.MACD_SLOW + config.MACD_SIGNAL
                        if df_entry is None or len(df_entry) < min_periods:
                            continue
                        if df_trend is None or len(df_trend) < config.ADX_PERIOD:
                            continue

                        has_new_candle = feed.consume_new_candle_flag()
                        if has_new_candle or first_candle_eval[pair] or force_eval:
                            first_candle_eval[pair] = False
                            strategies[pair].update_portfolio_value(portfolio_value)
                            trade_signal = strategies[pair].evaluate(df_entry, df_trend)

                            if trade_signal.signal != Signal.HOLD:
                                last_signal_text = f"{pair} {trade_signal.signal.value}"
                                order_mgr.process_signal(trade_signal, portfolio_value, pair)

                    if force_eval:
                        logger.info(
                            "Immediate re-scan after {n} position(s) closed -- looking for new entries",
                            n=positions_closed,
                        )

                    # ── Update portfolio value ───────────────────────
                    unrealized = Decimal("0")
                    for p in order_mgr.open_positions:
                        p_price = current_prices.get(p.pair)
                        if p_price is None:
                            continue
                        if p.side == "long":
                            unrealized += (p_price - p.entry_price) * p.amount
                        else:
                            unrealized += (p.entry_price - p_price) * p.amount

                    portfolio_value = (
                        config.STARTING_CAPITAL
                        + order_mgr.total_realized_pnl
                        + unrealized
                    )

                    # ── Update dashboard ─────────────────────────────
                    dashboard.current_prices = current_prices
                    dashboard.portfolio_value = portfolio_value
                    dashboard.last_signal = last_signal_text
                    dashboard.uptime_seconds = int(time.monotonic() - start_time)
                    dashboard.feeds = feeds

                    live.update(dashboard.generate_layout())

                    # ── Push to web dashboard ────────────────────────
                    snapshot = WebDashboard.build_snapshot(
                        mode=mode,
                        uptime_seconds=int(time.monotonic() - start_time),
                        last_signal=last_signal_text,
                        portfolio_value=portfolio_value,
                        current_prices=current_prices,
                        feeds=feeds,
                        order_mgr=order_mgr,
                        risk_mgr=risk_mgr,
                    )
                    await web_dash.broadcast(snapshot)

                    # ── Wait before next poll ────────────────────────
                    await asyncio.sleep(poll_interval)

                except KrakenClientError as exc:
                    logger.error("Exchange error: {err}", err=exc)
                    await asyncio.sleep(poll_interval)

                except Exception as exc:
                    logger.exception("Unexpected error in trading loop: {err}", err=exc)
                    await asyncio.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    finally:
        logger.info("Trading bot stopped")


# ── Backtest Mode ────────────────────────────────────────────────────────

def run_backtest() -> None:
    """Run the backtesting engine with current config parameters."""
    runner = BacktestRunner()
    results = runner.run(days=config.BACKTEST_DEFAULT_DAYS)
    runner._print_report(results)


def run_optimize() -> None:
    """Run the parameter optimizer to find the best strategy settings."""
    runner = BacktestRunner()
    runner.optimize(days=config.BACKTEST_DEFAULT_DAYS)


# ── Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments and dispatch to the correct mode."""
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot -- BTC/USD Momentum Breakout Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "backtest", "optimize"],
        default="paper",
        help="Trading mode: paper (default), live, backtest, or optimize",
    )
    args = parser.parse_args()

    setup_logging()

    logger.info("=" * 60)
    logger.info("Trading Bot starting | mode={mode}", mode=args.mode.upper())
    logger.info("=" * 60)

    if args.mode == "backtest":
        run_backtest()
        return

    if args.mode == "optimize":
        run_optimize()
        return

    if args.mode == "live":
        check_paper_trading_gate()
        logger.warning("=" * 60)
        logger.warning("  LIVE TRADING MODE -- REAL MONEY AT RISK")
        logger.warning("=" * 60)

    if args.mode == "paper":
        initialize_state()

    asyncio.run(run_trading_loop(args.mode))


if __name__ == "__main__":
    main()

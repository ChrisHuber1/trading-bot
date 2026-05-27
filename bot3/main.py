"""Bot3 entry point -- ML-driven trading bot with learning loop.

Usage:
    python -m bot3.main --mode paper
    python -m bot3.main --mode live
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import os

from bot3 import config

# Shared modules (risk, execution, monitor) import the root `config` module.
# Override its values with Bot3's settings so they use our paths and limits.
import config as _root_config
_root_config.TRADE_LOG = config.TRADE_LOG
_root_config.POSITIONS_FILE = config.POSITIONS_FILE
_root_config.MAX_OPEN_POSITIONS = config.MAX_OPEN_POSITIONS
_root_config.CAPITAL_FLOOR = config.CAPITAL_FLOOR
_root_config.PER_TRADE_RISK_PCT = config.PER_TRADE_RISK_PCT
_root_config.MAX_DAILY_LOSS_PCT = config.MAX_DAILY_LOSS_PCT
_root_config.FEE_GUARD_PCT = config.FEE_GUARD_PCT
_root_config.STOP_LOSS_PCT = config.FALLBACK_SL_PCT
_root_config.TAKE_PROFIT_PCT = config.FALLBACK_TP_PCT
_root_config.WEB_DASHBOARD_HOST = config.WEB_DASHBOARD_HOST
_root_config.WEB_DASHBOARD_PORT = config.WEB_DASHBOARD_PORT
_root_config.STARTING_CAPITAL = config.STARTING_CAPITAL
_root_config.TP_MODE = config.TP_MODE

from exchange.kraken_client import KrakenClient, KrakenClientError
from data.feed import DataFeed
from strategy.signals import Signal
from bot3.strategy.ml_strategy import MLStrategy
from ml_pipeline.inference import MLSignalFilter
from risk.manager import RiskManager
from execution.order_manager import OrderManager
from monitor.dashboard import Dashboard
from monitor.web_server import WebDashboard
from shared.pair_selector import get_active_pairs


def compute_regime(feeds: dict[str, "DataFeed"]) -> str:
    """Check BTC and ETH vs their EMA to determine market regime."""
    above = 0
    below = 0
    for rp in config.REGIME_PAIRS:
        feed = feeds.get(rp)
        if feed is None:
            continue
        df = feed.entry_dataframe
        if df is None or len(df) < config.REGIME_EMA_PERIOD:
            continue
        ema = df["close"].ewm(span=config.REGIME_EMA_PERIOD, adjust=False).mean().iloc[-1]
        close = df["close"].iloc[-1]
        if close > ema:
            above += 1
        else:
            below += 1

    if above == len(config.REGIME_PAIRS):
        return "bullish"
    if below == len(config.REGIME_PAIRS):
        return "bearish"
    return "neutral"


def setup_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr, level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>[bot3]</cyan> {message}",
    )
    logger.add(
        config.LOG_DIR / "bot3_{time:YYYY-MM-DD}.log",
        level="DEBUG", rotation="1 day", retention="30 days", serialize=True,
    )


def check_paper_trading_gate() -> None:
    if not config.STATE_FILE.exists():
        logger.error("bot3_state.json not found. Run in paper mode first.")
        sys.exit(1)

    with open(config.STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    start_str = state.get("paper_trading_start_date")
    if not start_str:
        logger.error("No paper_trading_start_date in bot3_state.json.")
        sys.exit(1)

    start = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    elapsed = (datetime.now(timezone.utc) - start).days

    if elapsed < config.PAPER_TRADING_DAYS_REQUIRED:
        remaining = config.PAPER_TRADING_DAYS_REQUIRED - elapsed
        logger.error("Paper gate: {e}/{r} days. {rem} remaining.", e=elapsed, r=config.PAPER_TRADING_DAYS_REQUIRED, rem=remaining)
        sys.exit(1)

    logger.info("Paper trading gate passed: {d} days", d=elapsed)


def initialize_state() -> None:
    state = {}
    if config.STATE_FILE.exists():
        try:
            with open(config.STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            state = {}

    if "paper_trading_start_date" not in state:
        state["paper_trading_start_date"] = datetime.now(timezone.utc).isoformat()
        logger.info("Bot3 paper trading started -- 14-day gate begins now")

    with open(config.STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


async def run_trading_loop(mode: str) -> None:
    load_dotenv()

    api_key = os.environ.get("KRAKEN_API_KEY", "")
    api_secret = os.environ.get("KRAKEN_API_SECRET", "")
    if not api_key or not api_secret:
        logger.error("KRAKEN_API_KEY and KRAKEN_API_SECRET must be set in .env")
        sys.exit(1)

    paper_mode = mode == "paper"
    pairs = get_active_pairs(count=config.ACTIVE_PAIRS_COUNT)

    # Patch root config values that shared modules (Dashboard, RiskManager) read
    _root_config.TRADING_PAIRS = pairs
    _root_config.TIMEFRAME_ENTRY = config.TIMEFRAME
    _root_config.ADX_THRESHOLD = 20
    _root_config.STOCHRSI_OVERBOUGHT = Decimal("0.80")
    _root_config.STOCHRSI_OVERSOLD = Decimal("0.20")
    if not hasattr(_root_config, "RSI_PERIOD"):
        _root_config.RSI_PERIOD = 14

    # ── Initialize components ────────────────────────────────────────
    client = KrakenClient(api_key=api_key, api_secret=api_secret, sandbox=paper_mode)
    risk_mgr = RiskManager(starting_capital=config.STARTING_CAPITAL)
    order_mgr = OrderManager(
        client, risk_mgr, paper_mode=paper_mode,
        positions_file=config.POSITIONS_FILE,
    )
    dashboard = Dashboard(order_mgr, risk_mgr, paper_mode=paper_mode)
    web_dash = WebDashboard(port=config.WEB_DASHBOARD_PORT)

    pinned = config.PINNED_MODEL_VERSION
    ml_filter = MLSignalFilter(
        confidence_threshold=config.ML_CONFIDENCE_ENTRY,
        version=pinned,
    )
    ml_strategy = MLStrategy(portfolio_value=config.STARTING_CAPITAL, ml_filter=ml_filter)
    logger.info("ML strategy loaded | model={v} | pinned={p}", v=ml_strategy.model_version, p=pinned or "auto")

    # ── Learning components (import here to keep startup fast if unavailable) ──
    decision_logger = None
    outcome_tracker = None
    post_mortem = None
    try:
        from bot3.learning.decision_logger import DecisionLogger
        from bot3.learning.outcome_tracker import OutcomeTracker
        from bot3.learning.post_mortem import PostMortem

        decision_logger = DecisionLogger(model_version=ml_strategy.model_version)
        outcome_tracker = OutcomeTracker()
        post_mortem = PostMortem()
        logger.info("Learning loop active")
    except Exception as exc:
        logger.warning("Learning loop unavailable: {err}", err=exc)

    # ── Model watcher (champion/challenger) ─────────────────────────
    model_watcher_task = None
    watcher = None
    try:
        from bot3.learning.model_watcher import ModelWatcher

        def on_new_model(new_filter):
            ml_strategy.update_ml_filter(new_filter)
            if decision_logger:
                decision_logger.update_model_version(new_filter.version)

        watcher = ModelWatcher(
            model_dir=config.MODEL_DIR,
            current_version=ml_strategy.model_version,
            callback=on_new_model,
            check_interval=config.MODEL_CHECK_INTERVAL,
            confidence_threshold=config.ML_CONFIDENCE_ENTRY,
            shadow_period_hours=config.SHADOW_PERIOD_HOURS,
            promotion_threshold_pct=config.PROMOTION_THRESHOLD_PCT,
        )
        model_watcher_task = asyncio.create_task(watcher.run())
        logger.info("Model watcher started | check every {s}s", s=config.MODEL_CHECK_INTERVAL)
    except Exception as exc:
        logger.warning("Model watcher unavailable: {err}", err=exc)

    # ── Per-pair data feeds ──────────────────────────────────────────
    feeds: dict[str, DataFeed] = {}
    first_eval: dict[str, bool] = {}

    for pair in pairs:
        feeds[pair] = DataFeed(client, pair=pair)
        first_eval[pair] = True

    # Ensure regime pairs (BTC/ETH) have feeds even if not in active pairs
    for rp in config.REGIME_PAIRS:
        if rp not in feeds:
            feeds[rp] = DataFeed(client, pair=rp)

    # ── Load history ─────────────────────────────────────────────────
    all_feed_pairs = set(pairs) | set(config.REGIME_PAIRS)
    logger.info("Loading history for {n} pairs...", n=len(all_feed_pairs))
    for pair in all_feed_pairs:
        try:
            feeds[pair].load_history(limit=500)
        except Exception as exc:
            logger.warning("Failed to load history for {pair}: {err}", pair=pair, err=exc)

    # ── Reconcile positions from prior session ─────────────────────
    if order_mgr.open_positions:
        logger.info("Fetching prices for {n} restored positions...", n=len(order_mgr.open_positions))
        reconcile_prices: dict[str, Decimal] = {}
        for pos in order_mgr.open_positions:
            try:
                ticker = client.fetch_ticker(pos.pair)
                if ticker.get("last") is not None:
                    reconcile_prices[pos.pair] = Decimal(str(ticker["last"]))
            except KrakenClientError as exc:
                logger.warning("Price fetch failed for {pair}: {err}", pair=pos.pair, err=exc)
        order_mgr.reconcile_after_restart(reconcile_prices)

    portfolio_value = config.STARTING_CAPITAL + order_mgr.total_realized_pnl
    start_time = time.monotonic()

    live = dashboard.start()
    await web_dash.start()

    logger.info(
        "Bot3 started | mode={mode} | pairs={n} | model={v} | portfolio=${pv}",
        mode=mode.upper(), n=len(pairs), v=ml_strategy.model_version,
        pv=portfolio_value,
    )

    last_candle_refresh = 0.0

    try:
        with live:
            while True:
                try:
                    now = time.monotonic()
                    current_prices: dict[str, Decimal] = {}

                    # ── Refresh candles ───────────────────────────────
                    if now - last_candle_refresh >= config.CANDLE_REFRESH_INTERVAL:
                        for pair in all_feed_pairs:
                            try:
                                raw = client.fetch_ohlcv(pair, limit=100, timeframe=config.TIMEFRAME)
                                if raw:
                                    for candle in raw[-5:]:
                                        feeds[pair].update_entry_candle(candle)
                                        feeds[pair].update_trend_candle(candle)
                            except KrakenClientError as exc:
                                logger.warning("Candle refresh failed {pair}: {err}", pair=pair, err=exc)
                        last_candle_refresh = now

                    # ── Fetch prices ──────────────────────────────────
                    for pair in pairs:
                        try:
                            ticker = client.fetch_ticker(pair)
                            if ticker.get("last") is not None:
                                current_prices[pair] = Decimal(str(ticker["last"]))
                        except KrakenClientError:
                            pass

                    if not current_prices:
                        await asyncio.sleep(config.POLL_INTERVAL)
                        continue

                    # ── Risk manager ──────────────────────────────────
                    risk_mgr.update_state(
                        portfolio_value=portfolio_value,
                        open_positions=len(order_mgr.open_positions),
                    )

                    # ── Check exits (SL/TP) ───────────────────────────
                    positions_before = len(order_mgr.open_positions)
                    order_mgr.check_exit_conditions(current_prices)
                    positions_closed = positions_before - len(order_mgr.open_positions)

                    # ── Record outcomes for closed positions ──────────
                    if positions_closed > 0 and outcome_tracker:
                        for pos in order_mgr.closed_positions[-positions_closed:]:
                            try:
                                outcome = outcome_tracker.record_outcome(pos)
                                if post_mortem and pos.pnl and pos.pnl < 0:
                                    post_mortem.analyze_loss(pos, outcome, decision_logger)
                            except Exception as exc:
                                logger.warning("Outcome tracking failed: {err}", err=exc)

                    # ── ML exit signals for open positions ────────────
                    for pos in list(order_mgr.open_positions):
                        if pos.pair not in current_prices:
                            continue
                        df = feeds[pos.pair].entry_dataframe
                        if df is None or len(df) < 200:
                            continue
                        has_new = feeds[pos.pair].consume_new_candle_flag()
                        if not has_new and not first_eval.get(pos.pair, False):
                            continue

                        exit_signal, exit_score = ml_strategy.get_exit_signal(
                            pos.pair, pos.side, df,
                        )
                        if exit_signal:
                            order_mgr.process_signal(exit_signal, portfolio_value, pos.pair)
                            if decision_logger:
                                try:
                                    decision_logger.log_decision(
                                        pair=pos.pair, action=f"exit_{exit_signal.signal.value.lower()}",
                                        ml_score=exit_score, trade_signal=exit_signal,
                                        df=df,
                                    )
                                except Exception:
                                    pass

                    force_eval = positions_closed > 0

                    # ── Market regime ─────────────────────────────────
                    regime = compute_regime(feeds)
                    ml_strategy.update_regime(regime)

                    # ── Evaluate strategy per pair ────────────────────
                    last_signal_text = "HOLD"
                    for pair in pairs:
                        if pair not in current_prices:
                            continue

                        feed = feeds[pair]
                        df_entry = feed.entry_dataframe
                        if df_entry is None or len(df_entry) < 200:
                            continue

                        has_new = feed.consume_new_candle_flag()
                        if not (has_new or first_eval.get(pair, False) or force_eval):
                            continue
                        first_eval[pair] = False

                        ml_strategy.update_portfolio_value(portfolio_value)
                        trade_signal, score = ml_strategy.evaluate(df_entry, pair)

                        # Shadow score candidate model if one is active
                        if watcher and watcher.shadow_scorer:
                            try:
                                watcher.shadow_scorer.score(
                                    pair=pair, df=df_entry,
                                    current_price=float(current_prices[pair]),
                                )
                            except Exception:
                                pass

                        # Log every decision (entries, rejections, holds)
                        if decision_logger and score:
                            action = "hold"
                            if trade_signal.signal == Signal.BUY:
                                action = "entry_long"
                            elif trade_signal.signal == Signal.SHORT:
                                action = "entry_short"
                            elif trade_signal.signal == Signal.HOLD and score.get("class_name"):
                                action = f"reject_{score['class_name']}"
                            try:
                                decision_logger.log_decision(
                                    pair=pair, action=action,
                                    ml_score=score, trade_signal=trade_signal,
                                    df=df_entry,
                                )
                            except Exception:
                                pass

                        if trade_signal.signal != Signal.HOLD:
                            last_signal_text = f"{pair} {trade_signal.signal.value}"
                            order_mgr.process_signal(trade_signal, portfolio_value, pair)

                    if force_eval:
                        logger.info("Re-scan after {n} close(s)", n=positions_closed)

                    # ── Portfolio value ────────────────────────────────
                    unrealized = Decimal("0")
                    for p in order_mgr.open_positions:
                        p_price = current_prices.get(p.pair)
                        if p_price is None:
                            continue
                        if p.side == "long":
                            unrealized += (p_price - p.entry_price) * p.amount
                        else:
                            unrealized += (p.entry_price - p_price) * p.amount

                    portfolio_value = config.STARTING_CAPITAL + order_mgr.total_realized_pnl + unrealized

                    # ── Dashboards ────────────────────────────────────
                    dashboard.current_prices = current_prices
                    dashboard.portfolio_value = portfolio_value
                    dashboard.last_signal = last_signal_text
                    dashboard.uptime_seconds = int(time.monotonic() - start_time)
                    dashboard.feeds = feeds
                    live.update(dashboard.generate_layout())

                    snapshot = WebDashboard.build_snapshot(
                        mode=mode, uptime_seconds=int(time.monotonic() - start_time),
                        last_signal=last_signal_text, portfolio_value=portfolio_value,
                        current_prices=current_prices, feeds=feeds,
                        order_mgr=order_mgr, risk_mgr=risk_mgr,
                    )
                    await web_dash.broadcast(snapshot)

                    await asyncio.sleep(config.POLL_INTERVAL)

                except KrakenClientError as exc:
                    logger.error("Exchange error: {err}", err=exc)
                    await asyncio.sleep(config.POLL_INTERVAL)
                except Exception as exc:
                    logger.exception("Unexpected error: {err}", err=exc)
                    await asyncio.sleep(config.POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Bot3 shutdown requested")
    finally:
        if model_watcher_task:
            model_watcher_task.cancel()
        logger.info("Bot3 stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bot3 -- ML-Driven Trading Bot")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    args = parser.parse_args()

    setup_logging()
    logger.info("=" * 60)
    logger.info("Bot3 starting | mode={mode}", mode=args.mode.upper())
    logger.info("=" * 60)

    if args.mode == "live":
        check_paper_trading_gate()
        logger.warning("  LIVE TRADING MODE -- REAL MONEY AT RISK")

    if args.mode == "paper":
        initialize_state()

    asyncio.run(run_trading_loop(args.mode))


if __name__ == "__main__":
    main()

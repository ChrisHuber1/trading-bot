"""
Central configuration for the trading bot.
All tunable parameters live here -- nothing magic-numbered in strategy files.
"""

from decimal import Decimal
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
STATE_FILE = PROJECT_ROOT / "state.json"
TRADE_LOG = PROJECT_ROOT / "logs" / "trades.jsonl"
LOG_DIR = PROJECT_ROOT / "logs"

# ── Exchange ─────────────────────────────────────────────────────────────
EXCHANGE_ID = "kraken"
TRADING_PAIRS = [
    "XRP/USD",
    "PENGU/USD",
    "SOL/USD",
    "DOGE/USD",
    "PEPE/USD",
    "AVAX/USD",
    "LINK/USD",
    "NEAR/USD",
    "SHIB/USD",
    "RENDER/USD",
    "WIF/USD",
    "BONK/USD",
]
TIMEFRAME_ENTRY = "1h"                  # Entry + indicator timeframe
TIMEFRAME_TREND = "1h"                  # Trend filter timeframe (same as entry)
ALLOWED_DOMAINS = ["api.kraken.com", "ws.kraken.com"]

# ── Capital & Sizing ─────────────────────────────────────────────────────
STARTING_CAPITAL = Decimal("1000.00")   # USD
DECIMAL_PRECISION = 8                   # Decimal places for monetary values

# ── Strategy: MACD + ADX (Primary) + Stochastic RSI (Secondary) ─────────
MACD_FAST = 12                          # MACD fast EMA period
MACD_SLOW = 26                          # MACD slow EMA period
MACD_SIGNAL = 9                         # MACD signal line period
ADX_PERIOD = 14                         # ADX lookback period
ADX_THRESHOLD = 20                      # Minimum ADX for trend strength
STOCHRSI_PERIOD = 14                    # Stochastic RSI lookback
STOCHRSI_SMOOTH_K = 3                   # %K smoothing
STOCHRSI_SMOOTH_D = 3                   # %D smoothing
STOCHRSI_OVERSOLD = Decimal("0.20")     # Below this = oversold zone
STOCHRSI_OVERBOUGHT = Decimal("0.80")   # Above this = overbought zone
RSI_PERIOD = 14                         # RSI for StochRSI base

# ── Risk Management ──────────────────────────────────────────────────────
CAPITAL_FLOOR = Decimal("700.00")       # 30% drawdown from $1000 → halt
PER_TRADE_RISK_PCT = Decimal("100")     # Paper testing: no per-trade cap (revert to 1.5 for live)
STOP_LOSS_PCT = Decimal("4.0")          # 4% from entry
TAKE_PROFIT_PCT = Decimal("1.5")        # 1.5% from entry (fixed mode)
MAX_OPEN_POSITIONS = 5

# ── Take-Profit Mode ────────────────────────────────────────────────────
TP_MODE = "fixed"                       # "fixed" = % from entry, "atr" = ATR multiple
ATR_TP_MULTIPLIER = Decimal("1.5")      # TP at 1.5x ATR from entry (used when TP_MODE="atr")
MAX_DAILY_LOSS_PCT = Decimal("5.0")     # -5% daily → halt for the day
FEE_GUARD_PCT = Decimal("20.0")         # Skip if fee > 20% of expected profit

# ── Paper Trading Gate ───────────────────────────────────────────────────
PAPER_TRADING_DAYS_REQUIRED = 14        # Minimum days before live allowed

# ── Dashboard ────────────────────────────────────────────────────────────
DASHBOARD_REFRESH_SECONDS = 2           # Terminal dashboard refresh rate

# ── Web Dashboard ───────────────────────────────────────────────────────
WEB_DASHBOARD_HOST = "0.0.0.0"
WEB_DASHBOARD_PORT = 8080

# ── Backtest ─────────────────────────────────────────────────────────────
BACKTEST_DEFAULT_DAYS = 90              # Days of historical data for backtest

# ── Optimizer ────────────────────────────────────────────────────────────
OPTIMIZER_TOP_N = 10                    # Number of top results to display

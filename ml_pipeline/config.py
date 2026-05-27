"""ML feature pipeline configuration."""

import os

# ── Database ────────────────────────────────────────────────────────────────
DB_HOST = os.environ.get("TRADING_DB_HOST", "YOUR_DB_HOST")
DB_PORT = int(os.environ.get("TRADING_DB_PORT", "5432"))
DB_NAME = os.environ.get("TRADING_DB_NAME", "trading_data")
DB_USER = os.environ.get("TRADING_DB_USER", "YOUR_DB_USER")
DB_PASS = os.environ.get("TRADING_DB_PASS", "")

# ── Feature computation ─────────────────────────────────────────────────────
FEATURE_TIMEFRAMES = ["1h"]
MAX_LOOKBACK = 200  # bars needed for EMA_200 warmup

# ── Label generation ────────────────────────────────────────────────────────
LABEL_TP_PCT = 0.015   # 1.5% take-profit threshold
LABEL_SL_PCT = 0.04    # 4% stop-loss threshold
LABEL_HORIZONS = [12, 24, 48]  # bars forward for TP/SL outcome labels
FWD_RETURN_HORIZONS = [1, 4, 12, 24]  # bars forward for raw return labels

# ── Universe filter thresholds ──────────────────────────────────────────────
MIN_DAILY_VOLUME_USD = 50_000
MAX_SPREAD_PCT = 2.0  # estimated from 1h (high-low)/close, not actual bid-ask spread
MIN_ATR_PCT = 0.005
MIN_CANDLE_COVERAGE = 0.90
MAX_GAP_HOURS = 6

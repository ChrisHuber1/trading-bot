"""Bot3 configuration.

Thresholds, model paths, and trading parameters.
Sensitive values withheld -- this is a live trading system.
"""

import os

MODEL_VERSION = os.getenv("BOT3_MODEL_VERSION", "latest")
CONFIDENCE_THRESHOLD = float(os.getenv("BOT3_CONFIDENCE", "0.65"))
MAX_PAIRS = int(os.getenv("BOT3_MAX_PAIRS", "40"))

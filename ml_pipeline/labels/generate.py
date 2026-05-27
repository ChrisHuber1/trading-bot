"""Label generation for ML training.

Defines what constitutes a profitable trade signal based on
forward-looking price movement. Labels are computed from historical
OHLCV data and stored in TimescaleDB.

Implementation details withheld -- this is a live trading system.
"""


def generate_labels(pair: str, timeframe: str = "1h") -> None:
    """Generate training labels for a given pair."""
    raise NotImplementedError("Label generation withheld")

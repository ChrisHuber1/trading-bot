"""
Shared signal types used by all strategy implementations.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"
    HOLD = "HOLD"


@dataclass(frozen=True)
class TradeSignal:
    """Represents a trading signal with sizing and risk levels."""
    signal: Signal
    price: Decimal
    size: Decimal               # Position size in base currency (BTC)
    stop_loss: Decimal          # Stop-loss price
    take_profit: Decimal        # Take-profit price
    reason: str                 # Human-readable reason for the signal

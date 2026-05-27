"""
Risk manager -- the non-negotiable safety layer.

ALL risk enforcement lives in this module. No other module may bypass,
override, or weaken these rules. Every order must pass through
RiskManager.validate_order() before being sent to the exchange.
"""

import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from loguru import logger

import config


class RiskViolation(Exception):
    """Raised when an order or state violates a risk rule."""


class RiskManager:
    """
    Enforces all risk rules defined in the project spec.

    Rules:
        1. Capital floor: Halt if portfolio drops to $70 (30% drawdown).
        2. Per-trade risk: Reject if order risks > 1.5% of portfolio.
        3. Stop-loss: Every order must have a stop-loss set.
        4. Take-profit: Every order must have a take-profit set.
        5. Max open positions: Reject if already at limit (2).
        6. Max daily loss: Halt for the day if daily loss exceeds -5%.
        7. Order type: Only limit (maker) orders allowed.
        8. Fee guard: Skip if estimated fee > 20% of expected profit.
    """

    def __init__(self, starting_capital: Decimal) -> None:
        """
        Initialize the risk manager.

        Args:
            starting_capital: The initial capital amount for drawdown calculations.
        """
        self._starting_capital = starting_capital
        self._daily_start_value: Optional[Decimal] = None
        self._current_date: Optional[date] = None
        self._halted = False
        self._halt_reason: Optional[str] = None
        self._open_position_count = 0

    @property
    def is_halted(self) -> bool:
        """Whether trading has been halted by a risk rule."""
        return self._halted

    @property
    def halt_reason(self) -> Optional[str]:
        """Human-readable reason for the halt, or None if not halted."""
        return self._halt_reason

    def update_state(
        self,
        portfolio_value: Decimal,
        open_positions: int,
    ) -> None:
        """
        Update the risk manager with the current portfolio state.
        Must be called before validate_order() on each tick.

        Args:
            portfolio_value: Current total portfolio value in USD.
            open_positions: Number of currently open positions.
        """
        today = datetime.now(timezone.utc).date()

        # Reset daily tracking at the start of a new day
        if self._current_date != today:
            self._current_date = today
            self._daily_start_value = portfolio_value
            # Clear daily halt if it was a daily-loss halt
            if self._halted and self._halt_reason and "daily loss" in self._halt_reason.lower():
                self._halted = False
                self._halt_reason = None
                logger.info("Daily loss halt cleared -- new trading day")

        self._open_position_count = open_positions

        # ── Rule 1: Capital floor ────────────────────────────────────────
        if portfolio_value <= config.CAPITAL_FLOOR:
            self._halted = True
            self._halt_reason = (
                f"CAPITAL FLOOR BREACHED: portfolio ${portfolio_value} "
                f"<= floor ${config.CAPITAL_FLOOR}"
            )
            logger.critical(self._halt_reason)
            return

        # ── Rule 6: Max daily loss ───────────────────────────────────────
        if self._daily_start_value is not None and self._daily_start_value > 0:
            daily_loss_pct = (
                (self._daily_start_value - portfolio_value)
                / self._daily_start_value
                * 100
            )
            if daily_loss_pct >= config.MAX_DAILY_LOSS_PCT:
                self._halted = True
                self._halt_reason = (
                    f"MAX DAILY LOSS: {daily_loss_pct:.2f}% "
                    f">= limit {config.MAX_DAILY_LOSS_PCT}%"
                )
                logger.critical(self._halt_reason)

    def validate_order(
        self,
        side: str,
        amount: Decimal,
        price: Decimal,
        stop_loss: Decimal,
        take_profit: Decimal,
        portfolio_value: Decimal,
        order_type: str = "limit",
        estimated_fee: Decimal = Decimal("0"),
    ) -> None:
        """
        Validate a proposed order against ALL risk rules.
        Raises RiskViolation if any rule is breached.

        Args:
            side: 'buy' or 'sell'.
            amount: Order size in base currency.
            price: Limit price in quote currency.
            stop_loss: Stop-loss price.
            take_profit: Take-profit price.
            portfolio_value: Current portfolio value in USD.
            order_type: Must be 'limit'. Market orders are rejected.
            estimated_fee: Estimated exchange fee for the order.

        Raises:
            RiskViolation: If any risk rule is breached.
        """
        # ── Check halt state ─────────────────────────────────────────────
        if self._halted:
            raise RiskViolation(f"Trading is halted: {self._halt_reason}")

        # ── Rule 7: Order type -- limit only ──────────────────────────────
        if order_type != "limit":
            raise RiskViolation(
                f"Only limit (maker) orders allowed. Got: {order_type!r}"
            )

        # ── Rule 5: Max open positions ───────────────────────────────────
        if side in ("buy", "short") and self._open_position_count >= config.MAX_OPEN_POSITIONS:
            raise RiskViolation(
                f"Max open positions ({config.MAX_OPEN_POSITIONS}) reached. "
                f"Currently open: {self._open_position_count}"
            )

        # ── Rule 3 & 4: Stop-loss and take-profit must be set ───────────
        if side in ("buy", "short"):
            if stop_loss <= Decimal("0"):
                raise RiskViolation(f"Stop-loss must be set (> 0) for {side} orders")
            if take_profit <= Decimal("0"):
                raise RiskViolation(f"Take-profit must be set (> 0) for {side} orders")

        # ── Rule 2: Per-trade risk ───────────────────────────────────────
        if side in ("buy", "short"):
            if side == "buy":
                dollar_risk = amount * (price - stop_loss)
            else:
                # Short: risk is stop_loss - price (stop is above entry)
                dollar_risk = amount * (stop_loss - price)

            max_allowed_risk = portfolio_value * config.PER_TRADE_RISK_PCT / 100

            if dollar_risk > max_allowed_risk:
                raise RiskViolation(
                    f"Per-trade risk ${dollar_risk:.2f} exceeds max "
                    f"${max_allowed_risk:.2f} ({config.PER_TRADE_RISK_PCT}% "
                    f"of ${portfolio_value})"
                )

        # ── Rule 8: Fee guard ────────────────────────────────────────────
        if side in ("buy", "short") and estimated_fee > 0:
            if side == "buy":
                expected_profit = amount * (take_profit - price)
            else:
                expected_profit = amount * (price - take_profit)

            if expected_profit > 0:
                fee_pct_of_profit = (estimated_fee / expected_profit) * 100
                if fee_pct_of_profit > config.FEE_GUARD_PCT:
                    raise RiskViolation(
                        f"Fee guard: estimated fee ${estimated_fee:.4f} is "
                        f"{fee_pct_of_profit:.1f}% of expected profit "
                        f"${expected_profit:.4f} (limit: {config.FEE_GUARD_PCT}%)"
                    )

        logger.debug(
            "Order validated | side={side} | amount={amount} | price={price}",
            side=side,
            amount=amount,
            price=price,
        )

    def check_stop_loss(self, entry_price: Decimal, current_price: Decimal) -> bool:
        """
        Check if the current price has breached the stop-loss level.

        Args:
            entry_price: The original entry price.
            current_price: The current market price.

        Returns:
            True if stop-loss is triggered (price dropped below threshold).
        """
        stop_price = entry_price * (1 - config.STOP_LOSS_PCT / 100)
        return current_price <= stop_price

    def check_take_profit(self, entry_price: Decimal, current_price: Decimal) -> bool:
        """
        Check if the current price has reached the take-profit level.

        Args:
            entry_price: The original entry price.
            current_price: The current market price.

        Returns:
            True if take-profit is triggered (price rose above threshold).
        """
        tp_price = entry_price * (1 + config.TAKE_PROFIT_PCT / 100)
        return current_price >= tp_price

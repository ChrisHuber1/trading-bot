"""
Order placement, fill tracking, and position management.

Bridges the strategy signals with the exchange client, routing every
order through the risk manager first. Tracks open positions, handles
stop-loss/take-profit exits, and logs every order event to trades.jsonl.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger

import config
from exchange.kraken_client import KrakenClient, KrakenClientError
from risk.manager import RiskManager, RiskViolation
from strategy.signals import TradeSignal, Signal


class OrderStatus(Enum):
    """Lifecycle states for an order."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Position:
    """Represents an open position (long or short)."""
    entry_order_id: str
    pair: str                   # Trading pair (e.g. 'XRP/USD')
    side: str                   # 'long' or 'short'
    entry_price: Decimal
    amount: Decimal             # Base currency quantity
    stop_loss: Decimal
    take_profit: Decimal
    entry_time: str             # ISO 8601
    exit_order_id: Optional[str] = None
    exit_price: Optional[Decimal] = None
    exit_time: Optional[str] = None
    pnl: Optional[Decimal] = None


class OrderManager:
    """Manages the full order lifecycle: signal → risk check → place → track → close."""

    def __init__(
        self,
        client: KrakenClient,
        risk_manager: RiskManager,
        paper_mode: bool = True,
        positions_file: Optional[Path] = None,
    ) -> None:
        """
        Initialize the order manager.

        Args:
            client: KrakenClient for exchange communication.
            risk_manager: RiskManager instance for pre-trade validation.
            paper_mode: If True, simulate orders locally instead of hitting the exchange.
            positions_file: Path to persist open positions. Enables crash recovery.
        """
        self._client = client
        self._risk = risk_manager
        self._paper_mode = paper_mode
        self._positions: list[Position] = []
        self._order_counter = 0  # Paper mode order ID generator
        self._positions_file = positions_file
        self._accumulated_pnl = Decimal("0")

        # Ensure log directory exists
        config.TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)

        if positions_file:
            self._load_state()

    @property
    def open_positions(self) -> list[Position]:
        """Return list of currently open positions."""
        return [p for p in self._positions if p.exit_order_id is None]

    @property
    def closed_positions(self) -> list[Position]:
        """Return list of closed positions with realized P&L."""
        return [p for p in self._positions if p.exit_order_id is not None]

    @property
    def total_realized_pnl(self) -> Decimal:
        """Sum of all realized P&L including prior sessions."""
        session_pnl = sum(
            (p.pnl for p in self.closed_positions if p.pnl is not None),
            Decimal("0"),
        )
        return self._accumulated_pnl + session_pnl

    def has_open_position(self, pair: str) -> bool:
        """Check if there is already an open position for this pair."""
        return any(p.pair == pair for p in self.open_positions)

    def process_signal(
        self, signal: TradeSignal, portfolio_value: Decimal, pair: str
    ) -> Optional[str]:
        """
        Process a trade signal: validate through risk manager and execute.

        Args:
            signal: The TradeSignal from the strategy.
            portfolio_value: Current portfolio value for risk calculations.
            pair: Trading pair symbol (e.g. 'XRP/USD').

        Returns:
            Order ID if an order was placed, None otherwise.
        """
        if signal.signal == Signal.HOLD:
            return None

        # One position per pair -- prevent duplicate entries
        if signal.signal in (Signal.BUY, Signal.SHORT) and self.has_open_position(pair):
            return None

        # Real-time max position check (risk manager's count is only updated once per cycle)
        if signal.signal in (Signal.BUY, Signal.SHORT):
            current_open = len(self.open_positions)
            if current_open >= config.MAX_OPEN_POSITIONS:
                self._log_trade_event("order_rejected", {
                    "pair": pair,
                    "side": signal.signal.value.lower(),
                    "amount": str(signal.size),
                    "price": str(signal.price),
                    "reason": f"Max open positions ({config.MAX_OPEN_POSITIONS}) reached. Currently open: {current_open}",
                })
                logger.warning(
                    "Order rejected: max positions ({max}) reached, currently {n} open",
                    max=config.MAX_OPEN_POSITIONS, n=current_open,
                )
                return None

        if signal.signal == Signal.BUY:
            return self._execute_buy(signal, portfolio_value, pair)

        if signal.signal == Signal.SELL:
            return self._execute_sell(signal, pair)

        if signal.signal == Signal.SHORT:
            return self._execute_short(signal, portfolio_value, pair)

        if signal.signal == Signal.COVER:
            return self._execute_cover(signal, pair)

        return None

    def check_exit_conditions(self, prices: dict[str, Decimal]) -> list[str]:
        """
        Check all open positions for stop-loss or take-profit triggers.

        Handles both long and short positions with correct directional logic:
        - Long: SL triggers when price drops below stop, TP when price rises above target
        - Short: SL triggers when price rises above stop, TP when price drops below target

        Args:
            prices: Dict mapping pair symbol to current price.

        Returns:
            List of order IDs for exit orders that were placed.
        """
        exit_orders = []

        for position in self.open_positions:
            current_price = prices.get(position.pair)
            if current_price is None:
                continue

            sl_triggered = False
            tp_triggered = False

            if position.side == "long":
                sl_triggered = current_price <= position.stop_loss
                tp_triggered = current_price >= position.take_profit
            elif position.side == "short":
                sl_triggered = current_price >= position.stop_loss
                tp_triggered = current_price <= position.take_profit

            if sl_triggered:
                # Paper mode: fill at the stop price, not the gapped market price.
                # Simulates a stop-limit order resting on the exchange.
                sl_fill = position.stop_loss if self._paper_mode else current_price
                logger.warning(
                    "STOP-LOSS triggered | pair={pair} | side={side} | position={id} | "
                    "entry={entry} | current={current} | stop={stop}",
                    pair=position.pair,
                    side=position.side,
                    id=position.entry_order_id,
                    entry=position.entry_price,
                    current=current_price,
                    stop=position.stop_loss,
                )
                order_id = self._close_position(position, sl_fill, "stop_loss")
                if order_id:
                    exit_orders.append(order_id)

            elif tp_triggered:
                tp_fill = position.take_profit if self._paper_mode else current_price
                logger.info(
                    "TAKE-PROFIT triggered | pair={pair} | side={side} | position={id} | "
                    "entry={entry} | current={current} | tp={tp}",
                    pair=position.pair,
                    side=position.side,
                    id=position.entry_order_id,
                    entry=position.entry_price,
                    current=current_price,
                    tp=position.take_profit,
                )
                order_id = self._close_position(position, tp_fill, "take_profit")
                if order_id:
                    exit_orders.append(order_id)

        return exit_orders

    # ── Internal ─────────────────────────────────────────────────────────

    def _execute_buy(
        self, signal: TradeSignal, portfolio_value: Decimal, pair: str
    ) -> Optional[str]:
        """
        Validate and execute a BUY order.

        Routes through risk manager, then places the order on the exchange
        (or simulates it in paper mode).
        """
        try:
            self._risk.validate_order(
                side="buy",
                amount=signal.size,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                portfolio_value=portfolio_value,
                order_type="limit",
            )
        except RiskViolation as exc:
            self._log_trade_event("order_rejected", {
                "pair": pair,
                "side": "buy",
                "amount": str(signal.size),
                "price": str(signal.price),
                "reason": str(exc),
            })
            logger.warning("Order rejected by risk manager: {reason}", reason=exc)
            return None

        if self._paper_mode:
            order_id = self._paper_order_id()
            now = datetime.now(timezone.utc).isoformat()

            position = Position(
                entry_order_id=order_id,
                pair=pair,
                side="long",
                entry_price=signal.price,
                amount=signal.size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                entry_time=now,
            )
            self._positions.append(position)

            self._log_trade_event("order_filled", {
                "order_id": order_id,
                "pair": pair,
                "side": "buy",
                "amount": str(signal.size),
                "price": str(signal.price),
                "stop_loss": str(signal.stop_loss),
                "take_profit": str(signal.take_profit),
                "mode": "paper",
                "reason": signal.reason,
            })
            logger.info("PAPER BUY filled | pair={pair} | id={id} | {reason}", pair=pair, id=order_id, reason=signal.reason)
            self._save_state()
            return order_id

        # Live mode
        try:
            order = self._client.place_limit_order(
                pair=pair,
                side="buy",
                amount=signal.size,
                price=signal.price,
            )
            order_id = order["id"]

            position = Position(
                entry_order_id=order_id,
                pair=pair,
                side="long",
                entry_price=signal.price,
                amount=signal.size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                entry_time=datetime.now(timezone.utc).isoformat(),
            )
            self._positions.append(position)

            self._log_trade_event("order_placed", {
                "order_id": order_id,
                "pair": pair,
                "side": "buy",
                "amount": str(signal.size),
                "price": str(signal.price),
                "stop_loss": str(signal.stop_loss),
                "take_profit": str(signal.take_profit),
                "mode": "live",
                "reason": signal.reason,
            })
            self._save_state()
            return order_id

        except KrakenClientError as exc:
            self._log_trade_event("order_error", {
                "pair": pair,
                "side": "buy",
                "error": str(exc),
            })
            logger.error("Failed to place buy order on {pair}: {err}", pair=pair, err=exc)
            return None

    def _execute_sell(self, signal: TradeSignal, pair: str) -> Optional[str]:
        """
        Execute a SELL signal -- close open long positions for the given pair.
        """
        long_positions = [p for p in self.open_positions if p.side == "long" and p.pair == pair]
        if not long_positions:
            logger.debug("SELL signal for {pair} but no open long positions to close", pair=pair)
            return None

        order_ids = []
        for position in list(long_positions):
            order_id = self._close_position(position, signal.price, "strategy_sell")
            if order_id:
                order_ids.append(order_id)

        return order_ids[0] if order_ids else None

    def _execute_short(
        self, signal: TradeSignal, portfolio_value: Decimal, pair: str
    ) -> Optional[str]:
        """
        Validate and execute a SHORT order.

        Routes through risk manager, then opens a short position on the exchange
        (or simulates it in paper mode). Stop-loss is above entry, take-profit below.
        """
        try:
            self._risk.validate_order(
                side="short",
                amount=signal.size,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                portfolio_value=portfolio_value,
                order_type="limit",
            )
        except RiskViolation as exc:
            self._log_trade_event("order_rejected", {
                "pair": pair,
                "side": "short",
                "amount": str(signal.size),
                "price": str(signal.price),
                "reason": str(exc),
            })
            logger.warning("Order rejected by risk manager: {reason}", reason=exc)
            return None

        if self._paper_mode:
            order_id = self._paper_order_id()
            now = datetime.now(timezone.utc).isoformat()

            position = Position(
                entry_order_id=order_id,
                pair=pair,
                side="short",
                entry_price=signal.price,
                amount=signal.size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                entry_time=now,
            )
            self._positions.append(position)

            self._log_trade_event("order_filled", {
                "order_id": order_id,
                "pair": pair,
                "side": "short",
                "amount": str(signal.size),
                "price": str(signal.price),
                "stop_loss": str(signal.stop_loss),
                "take_profit": str(signal.take_profit),
                "mode": "paper",
                "reason": signal.reason,
            })
            logger.info("PAPER SHORT filled | pair={pair} | id={id} | {reason}", pair=pair, id=order_id, reason=signal.reason)
            self._save_state()
            return order_id

        # Live mode
        try:
            order = self._client.place_limit_order(
                pair=pair,
                side="sell",
                amount=signal.size,
                price=signal.price,
            )
            order_id = order["id"]

            position = Position(
                entry_order_id=order_id,
                pair=pair,
                side="short",
                entry_price=signal.price,
                amount=signal.size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                entry_time=datetime.now(timezone.utc).isoformat(),
            )
            self._positions.append(position)

            self._log_trade_event("order_placed", {
                "order_id": order_id,
                "pair": pair,
                "side": "short",
                "amount": str(signal.size),
                "price": str(signal.price),
                "stop_loss": str(signal.stop_loss),
                "take_profit": str(signal.take_profit),
                "mode": "live",
                "reason": signal.reason,
            })
            self._save_state()
            return order_id

        except KrakenClientError as exc:
            self._log_trade_event("order_error", {
                "pair": pair,
                "side": "short",
                "error": str(exc),
            })
            logger.error("Failed to place short order on {pair}: {err}", pair=pair, err=exc)
            return None

    def _execute_cover(self, signal: TradeSignal, pair: str) -> Optional[str]:
        """
        Execute a COVER signal -- close open short positions for the given pair.
        """
        short_positions = [p for p in self.open_positions if p.side == "short" and p.pair == pair]
        if not short_positions:
            logger.debug("COVER signal for {pair} but no open short positions to close", pair=pair)
            return None

        order_ids = []
        for position in list(short_positions):
            order_id = self._close_position(position, signal.price, "strategy_cover")
            if order_id:
                order_ids.append(order_id)

        return order_ids[0] if order_ids else None

    def _close_position(
        self, position: Position, exit_price: Decimal, reason: str
    ) -> Optional[str]:
        """
        Close an open position at the given price.

        Args:
            position: The Position to close.
            exit_price: The price at which to exit.
            reason: Why the position is being closed (stop_loss, take_profit, strategy_sell).

        Returns:
            Exit order ID, or None if failed.
        """
        # Close side is opposite of position side
        close_side = "sell" if position.side == "long" else "buy"

        if self._paper_mode:
            order_id = self._paper_order_id()
        else:
            try:
                order = self._client.place_limit_order(
                    pair=position.pair,
                    side=close_side,
                    amount=position.amount,
                    price=exit_price,
                )
                order_id = order["id"]
            except KrakenClientError as exc:
                self._log_trade_event("order_error", {
                    "pair": position.pair,
                    "side": close_side,
                    "error": str(exc),
                    "position_id": position.entry_order_id,
                })
                logger.error("Failed to close position on {pair}: {err}", pair=position.pair, err=exc)
                return None

        # Calculate P&L: long profits when price rises, short profits when price falls
        if position.side == "long":
            pnl = (exit_price - position.entry_price) * position.amount
        else:
            pnl = (position.entry_price - exit_price) * position.amount

        position.exit_order_id = order_id
        position.exit_price = exit_price
        position.exit_time = datetime.now(timezone.utc).isoformat()
        position.pnl = pnl

        self._log_trade_event("position_closed", {
            "entry_order_id": position.entry_order_id,
            "exit_order_id": order_id,
            "entry_price": str(position.entry_price),
            "exit_price": str(exit_price),
            "amount": str(position.amount),
            "pnl": str(pnl),
            "reason": reason,
            "mode": "paper" if self._paper_mode else "live",
        })
        logger.info(
            "Position closed | entry={entry} | exit={exit} | pnl={pnl} | reason={reason}",
            entry=position.entry_price,
            exit=exit_price,
            pnl=pnl,
            reason=reason,
        )
        self._save_state()
        return order_id

    def _paper_order_id(self) -> str:
        """Generate a unique order ID for paper trading."""
        self._order_counter += 1
        return f"PAPER-{self._order_counter:06d}"

    def reconcile_after_restart(self, prices: dict[str, Decimal]) -> None:
        """Check restored positions against current prices and emergency-close any past SL/TP.

        Must be called after __init__ restores positions and before the trading loop starts.
        In paper mode, SL fills at current price (gap loss is real), TP fills at TP price
        (limit order would have filled). In live mode, both fill at current market price.
        """
        if not self.open_positions:
            return

        logger.info(
            "Reconciling {n} restored positions against current prices",
            n=len(self.open_positions),
        )

        for position in list(self.open_positions):
            current_price = prices.get(position.pair)
            if current_price is None:
                logger.warning(
                    "No price for restored position {pair} ({id}) -- cannot reconcile, will monitor normally",
                    pair=position.pair, id=position.entry_order_id,
                )
                continue

            sl_blown = False
            tp_blown = False

            if position.side == "long":
                sl_blown = current_price <= position.stop_loss
                tp_blown = current_price >= position.take_profit
            elif position.side == "short":
                sl_blown = current_price >= position.stop_loss
                tp_blown = current_price <= position.take_profit

            if sl_blown:
                # SL was missed during downtime -- close at current price (the real loss)
                logger.warning(
                    "EMERGENCY SL | {pair} {side} gapped past stop during downtime | "
                    "entry={entry} | SL={sl} | current={cur} | closing at market",
                    pair=position.pair, side=position.side,
                    entry=position.entry_price, sl=position.stop_loss, cur=current_price,
                )
                self._close_position(position, current_price, "stop_loss_gap")
            elif tp_blown:
                fill_price = position.take_profit if self._paper_mode else current_price
                logger.info(
                    "MISSED TP | {pair} {side} passed take-profit during downtime | "
                    "entry={entry} | TP={tp} | current={cur} | closing at {fill}",
                    pair=position.pair, side=position.side,
                    entry=position.entry_price, tp=position.take_profit,
                    cur=current_price, fill=fill_price,
                )
                self._close_position(position, fill_price, "take_profit_gap")
            else:
                logger.info(
                    "Restored {pair} {side} | entry={entry} | SL={sl} | TP={tp} | current={cur} | OK",
                    pair=position.pair, side=position.side,
                    entry=position.entry_price, sl=position.stop_loss,
                    tp=position.take_profit, cur=current_price,
                )

    def _save_state(self) -> None:
        """Persist open positions, order counter, and realized P&L atomically."""
        if not self._positions_file:
            return

        open_pos = []
        for p in self.open_positions:
            open_pos.append({
                "entry_order_id": p.entry_order_id,
                "pair": p.pair,
                "side": p.side,
                "entry_price": str(p.entry_price),
                "amount": str(p.amount),
                "stop_loss": str(p.stop_loss),
                "take_profit": str(p.take_profit),
                "entry_time": p.entry_time,
            })

        state = {
            "order_counter": self._order_counter,
            "realized_pnl": str(self.total_realized_pnl),
            "open_positions": open_pos,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            tmp = self._positions_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            # Atomic replace -- prevents corruption if bot crashes mid-write
            tmp.replace(self._positions_file)
        except OSError as exc:
            logger.error("Failed to save position state: {err}", err=exc)

    def _load_state(self) -> None:
        """Restore open positions, order counter, and realized P&L from disk."""
        if not self._positions_file or not self._positions_file.exists():
            return

        try:
            with open(self._positions_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load position state (starting fresh): {err}", err=exc)
            return

        self._order_counter = state.get("order_counter", 0)
        self._accumulated_pnl = Decimal(state.get("realized_pnl", "0"))

        for p_data in state.get("open_positions", []):
            pos = Position(
                entry_order_id=p_data["entry_order_id"],
                pair=p_data["pair"],
                side=p_data["side"],
                entry_price=Decimal(p_data["entry_price"]),
                amount=Decimal(p_data["amount"]),
                stop_loss=Decimal(p_data["stop_loss"]),
                take_profit=Decimal(p_data["take_profit"]),
                entry_time=p_data["entry_time"],
            )
            self._positions.append(pos)

        logger.info(
            "Restored state | {n} open positions | counter={c} | realized_pnl=${pnl}",
            n=len(self.open_positions), c=self._order_counter, pnl=self._accumulated_pnl,
        )

    def _log_trade_event(self, event_type: str, data: dict) -> None:
        """
        Append a structured trade event to trades.jsonl.

        Every order attempt, fill, cancellation, and error is logged here.

        Args:
            event_type: Type of event (order_placed, order_filled, etc.).
            data: Event-specific data dict.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data,
        }
        try:
            with open(config.TRADE_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.error("Failed to write trade log: {err}", err=exc)

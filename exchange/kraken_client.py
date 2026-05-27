"""
CCXT-based Kraken exchange client with WebSocket support.

Handles all communication with the Kraken API: authentication,
fetching balances/OHLCV, placing/cancelling limit orders, and
streaming real-time ticker data over WebSocket.
"""

import asyncio
from decimal import Decimal
from typing import Optional

import ccxt
import ccxt.pro as ccxtpro
from loguru import logger

import config


class KrakenClientError(Exception):
    """Raised when an exchange operation fails."""


class KrakenClient:
    """Wrapper around CCXT Kraken for REST and WebSocket operations."""

    def __init__(self, api_key: str, api_secret: str, sandbox: bool = True) -> None:
        """
        Initialize the Kraken client.

        Args:
            api_key: Kraken API key.
            api_secret: Kraken API secret.
            sandbox: If True, paper trading mode. Kraken doesn't support
                     sandbox URLs, so orders are simulated locally by
                     OrderManager instead.
        """
        common_opts = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }

        self._rest = ccxt.kraken(common_opts)
        self._ws = ccxtpro.kraken(common_opts)

        # Kraken has no sandbox URL -- paper trading is handled at the
        # OrderManager level, not the exchange level.

        self._rest.load_markets()
        # Share loaded markets with the WS client to avoid a redundant REST call
        self._ws.markets = self._rest.markets
        self._ws.markets_by_id = self._rest.markets_by_id
        self._ws.symbols = self._rest.symbols
        self._ws.currencies = self._rest.currencies
        self._ws.currencies_by_id = self._rest.currencies_by_id
        logger.info(
            "KrakenClient initialized | sandbox={sandbox} | pairs={pairs}",
            sandbox=sandbox,
            pairs=config.TRADING_PAIRS,
        )

    # ── Market Data (REST) ───────────────────────────────────────────────

    def fetch_ohlcv(
        self, pair: str, since: Optional[int] = None, limit: int = 500,
        timeframe: Optional[str] = None,
    ) -> list[list]:
        """
        Fetch historical OHLCV candles for the given pair.

        Args:
            pair: Trading pair symbol (e.g. 'XRP/USD').
            since: Start timestamp in milliseconds. None fetches most recent.
            limit: Maximum number of candles to return.
            timeframe: Candle timeframe (e.g. '5m', '1h'). Defaults to TIMEFRAME_ENTRY.

        Returns:
            List of [timestamp, open, high, low, close, volume] lists.
        """
        tf = timeframe or config.TIMEFRAME_ENTRY
        try:
            candles = self._rest.fetch_ohlcv(
                pair,
                timeframe=tf,
                since=since,
                limit=limit,
            )
            logger.debug("Fetched {n} candles for {pair}", n=len(candles), pair=pair)
            return candles
        except ccxt.BaseError as exc:
            raise KrakenClientError(f"Failed to fetch OHLCV for {pair}: {exc}") from exc

    def fetch_ticker(self, pair: str) -> dict:
        """
        Fetch the current ticker for the given trading pair.

        Args:
            pair: Trading pair symbol (e.g. 'XRP/USD').

        Returns:
            Ticker dict with bid, ask, last, etc.
        """
        try:
            return self._rest.fetch_ticker(pair)
        except ccxt.BaseError as exc:
            raise KrakenClientError(f"Failed to fetch ticker for {pair}: {exc}") from exc

    # ── Account ──────────────────────────────────────────────────────────

    def fetch_balance(self) -> dict[str, Decimal]:
        """
        Fetch account balances, returning only non-zero assets as Decimal.

        Returns:
            Dict mapping currency symbol to Decimal balance.
        """
        try:
            raw = self._rest.fetch_balance()
            balances = {}
            for currency, amount in raw.get("total", {}).items():
                if amount and float(amount) > 0:
                    balances[currency] = Decimal(str(amount))
            return balances
        except ccxt.BaseError as exc:
            raise KrakenClientError(f"Failed to fetch balance: {exc}") from exc

    # ── Orders ───────────────────────────────────────────────────────────

    def place_limit_order(
        self, pair: str, side: str, amount: Decimal, price: Decimal
    ) -> dict:
        """
        Place a limit (maker) order on Kraken.

        Args:
            pair: Trading pair symbol (e.g. 'XRP/USD').
            side: 'buy' or 'sell'.
            amount: Order quantity in base currency.
            price: Limit price in quote currency (USD).

        Returns:
            CCXT order structure with id, status, etc.

        Raises:
            KrakenClientError: If order placement fails.
            ValueError: If side is not 'buy' or 'sell'.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid order side: {side!r}")

        try:
            order = self._rest.create_order(
                symbol=pair,
                type="limit",
                side=side,
                amount=float(amount),
                price=float(price),
                params={"postOnly": True},  # Ensure maker-only
            )
            logger.info(
                "Order placed | pair={pair} | id={id} | side={side} | amount={amount} | price={price}",
                pair=pair,
                id=order["id"],
                side=side,
                amount=amount,
                price=price,
            )
            return order
        except ccxt.BaseError as exc:
            raise KrakenClientError(f"Failed to place {side} order on {pair}: {exc}") from exc

    def cancel_order(self, order_id: str, pair: str) -> dict:
        """
        Cancel an open order by ID.

        Args:
            order_id: The exchange-assigned order ID.
            pair: Trading pair symbol.

        Returns:
            CCXT order structure reflecting the cancelled state.
        """
        try:
            result = self._rest.cancel_order(order_id, pair)
            logger.info("Order cancelled | id={id}", id=order_id)
            return result
        except ccxt.BaseError as exc:
            raise KrakenClientError(
                f"Failed to cancel order {order_id}: {exc}"
            ) from exc

    def fetch_order(self, order_id: str, pair: str) -> dict:
        """
        Fetch the current status of an order.

        Args:
            order_id: The exchange-assigned order ID.
            pair: Trading pair symbol.

        Returns:
            CCXT order structure with current status and fill info.
        """
        try:
            return self._rest.fetch_order(order_id, pair)
        except ccxt.BaseError as exc:
            raise KrakenClientError(
                f"Failed to fetch order {order_id}: {exc}"
            ) from exc

    def fetch_open_orders(self, pair: str) -> list[dict]:
        """
        Fetch all currently open orders for the given pair.

        Args:
            pair: Trading pair symbol.

        Returns:
            List of CCXT order structures.
        """
        try:
            return self._rest.fetch_open_orders(pair)
        except ccxt.BaseError as exc:
            raise KrakenClientError(f"Failed to fetch open orders for {pair}: {exc}") from exc

    # ── WebSocket ────────────────────────────────────────────────────────

    async def watch_ohlcv(self, pair: str, timeframe: Optional[str] = None) -> list[list]:
        """
        Stream real-time OHLCV candle updates via WebSocket.

        Args:
            pair: Trading pair symbol.
            timeframe: Candle timeframe (e.g. '5m', '1h'). Defaults to TIMEFRAME_ENTRY.

        Returns:
            Latest OHLCV data from the WebSocket stream.
        """
        tf = timeframe or config.TIMEFRAME_ENTRY
        try:
            return await self._ws.watch_ohlcv(pair, tf)
        except ccxt.BaseError as exc:
            raise KrakenClientError(f"WebSocket OHLCV error for {pair}: {exc}") from exc

    async def watch_ticker(self, pair: str) -> dict:
        """
        Stream real-time ticker updates via WebSocket.

        Args:
            pair: Trading pair symbol.

        Returns:
            Latest ticker dict from the WebSocket stream.
        """
        try:
            return await self._ws.watch_ticker(pair)
        except ccxt.BaseError as exc:
            raise KrakenClientError(f"WebSocket ticker error for {pair}: {exc}") from exc

    async def close_ws(self) -> None:
        """Gracefully close the WebSocket connection."""
        try:
            await self._ws.close()
            logger.info("WebSocket connection closed")
        except ccxt.BaseError as exc:
            logger.warning("Error closing WebSocket: {exc}", exc=exc)

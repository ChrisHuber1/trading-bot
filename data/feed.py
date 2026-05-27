"""
Real-time OHLCV data aggregation and indicator computation.

Maintains dual-timeframe DataFrames (5m entry + 1h trend), computes
Bollinger Bands and volume SMA on the entry timeframe, and trend EMA
on the higher timeframe.
"""

import asyncio
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import AverageTrueRange

import config
from exchange.kraken_client import KrakenClient


class DataFeed:
    """Aggregates dual-timeframe OHLCV data and computes technical indicators."""

    def __init__(self, client: KrakenClient, pair: str = "BTC/USD") -> None:
        """
        Initialize the data feed.

        Args:
            client: An initialized KrakenClient instance for fetching candles.
            pair: Trading pair symbol (e.g. 'XRP/USD').
        """
        self._client = client
        self._pair = pair
        self._df_entry: Optional[pd.DataFrame] = None
        self._df_trend: Optional[pd.DataFrame] = None
        self._last_candle_ts_entry: int = 0
        self._last_candle_ts_trend: int = 0
        self._new_candle_appended: bool = False

    def load_history(self, limit: int = 500) -> pd.DataFrame:
        """
        Fetch historical candles for both timeframes and build DataFrames.

        Args:
            limit: Number of historical candles to fetch per timeframe.

        Returns:
            Entry timeframe DataFrame with OHLCV data and computed indicators.
        """
        # Fetch entry timeframe
        raw_entry = self._client.fetch_ohlcv(
            self._pair, limit=limit, timeframe=config.TIMEFRAME_ENTRY
        )
        self._df_entry = self._build_dataframe(raw_entry)
        self._compute_entry_indicators()

        # Fetch trend timeframe
        raw_trend = self._client.fetch_ohlcv(
            self._pair, limit=limit, timeframe=config.TIMEFRAME_TREND
        )
        self._df_trend = self._build_dataframe(raw_trend)
        self._compute_trend_indicators()

        if len(self._df_entry) > 0:
            self._last_candle_ts_entry = int(
                self._df_entry.index[-1].timestamp() * 1000
            )

        if len(self._df_trend) > 0:
            self._last_candle_ts_trend = int(
                self._df_trend.index[-1].timestamp() * 1000
            )

        logger.info(
            "History loaded | entry_candles={ne} | trend_candles={nt}",
            ne=len(self._df_entry),
            nt=len(self._df_trend),
        )
        return self._df_entry

    def update_entry_candle(self, candle: list) -> pd.DataFrame:
        """
        Update or append a single 5m candle and recompute entry indicators.

        Args:
            candle: [timestamp_ms, open, high, low, close, volume] list.

        Returns:
            Updated entry DataFrame with recalculated indicators.
        """
        self._df_entry = self._update_df(self._df_entry, candle, "entry")
        self._last_candle_ts_entry = candle[0]
        self._compute_entry_indicators()
        return self._df_entry

    def update_trend_candle(self, candle: list) -> pd.DataFrame:
        """
        Update or append a single 1h candle and recompute trend indicators.

        Args:
            candle: [timestamp_ms, open, high, low, close, volume] list.

        Returns:
            Updated trend DataFrame with recalculated indicators.
        """
        self._df_trend = self._update_df(self._df_trend, candle, "trend")
        self._last_candle_ts_trend = candle[0]
        self._compute_trend_indicators()
        return self._df_trend

    @property
    def entry_dataframe(self) -> Optional[pd.DataFrame]:
        """Return the 5m entry candle DataFrame."""
        return self._df_entry

    @property
    def trend_dataframe(self) -> Optional[pd.DataFrame]:
        """Return the 1h trend candle DataFrame."""
        return self._df_trend

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """Return the entry candle DataFrame (backward compatibility)."""
        return self._df_entry

    @property
    def latest(self) -> Optional[pd.Series]:
        """Return the most recent entry candle row with all indicators."""
        if self._df_entry is not None and len(self._df_entry) > 0:
            return self._df_entry.iloc[-1]
        return None

    def consume_new_candle_flag(self) -> bool:
        """Return True if a new entry candle was appended since the last check, then reset."""
        flag = self._new_candle_appended
        self._new_candle_appended = False
        return flag

    @property
    def latest_price(self) -> Optional[Decimal]:
        """Return the most recent close price as Decimal."""
        if self._df_entry is not None and len(self._df_entry) > 0:
            return Decimal(str(self._df_entry.iloc[-1]["close"]))
        return None

    # ── Internal ─────────────────────────────────────────────────────────

    def _update_df(
        self, df: Optional[pd.DataFrame], candle: list, label: str
    ) -> pd.DataFrame:
        """
        Update or append a candle to a DataFrame.

        Args:
            df: Existing DataFrame to update.
            candle: [timestamp_ms, open, high, low, close, volume] list.
            label: Label for logging ('entry' or 'trend').

        Returns:
            Updated DataFrame.
        """
        ts = pd.Timestamp(candle[0], unit="ms", tz="UTC")
        row = {
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
        }

        if df is not None and len(df) > 0 and ts in df.index:
            for col, val in row.items():
                df.at[ts, col] = val
        else:
            new_row = pd.DataFrame(row, index=[ts])
            new_row.index.name = "timestamp"
            df = pd.concat([df, new_row])
            self._new_candle_appended = True

            # Keep a rolling window to bound memory
            max_rows = max(config.MACD_SLOW, config.ADX_PERIOD, config.STOCHRSI_PERIOD, config.RSI_PERIOD) * 20
            if len(df) > max_rows:
                df = df.iloc[-max_rows:]

        return df

    @staticmethod
    def _build_dataframe(raw_candles: list[list]) -> pd.DataFrame:
        """
        Convert raw OHLCV lists to a pandas DataFrame indexed by timestamp.

        Args:
            raw_candles: List of [timestamp_ms, open, high, low, close, volume].

        Returns:
            DataFrame with columns: open, high, low, close, volume.
        """
        df = pd.DataFrame(
            raw_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def _compute_entry_indicators(self) -> None:
        """
        Compute MACD, ADX, StochRSI, and ATR on the entry DataFrame.

        Adds columns: macd_hist, adx, plus_di, minus_di, stochrsi_k,
        stochrsi_d, atr, rsi.
        """
        min_period = max(config.MACD_SLOW + config.MACD_SIGNAL, config.ADX_PERIOD,
                         config.STOCHRSI_PERIOD + config.STOCHRSI_SMOOTH_K)
        if self._df_entry is None or len(self._df_entry) < min_period:
            return

        close = self._df_entry["close"]
        high = self._df_entry["high"]
        low = self._df_entry["low"]

        macd = MACD(
            close=close,
            window_slow=config.MACD_SLOW,
            window_fast=config.MACD_FAST,
            window_sign=config.MACD_SIGNAL,
        )
        self._df_entry["macd_hist"] = macd.macd_diff()
        self._df_entry["macd_line"] = macd.macd()
        self._df_entry["macd_signal"] = macd.macd_signal()

        adx = ADXIndicator(
            high=high, low=low, close=close, window=config.ADX_PERIOD,
        )
        self._df_entry["adx"] = adx.adx()
        self._df_entry["plus_di"] = adx.adx_pos()
        self._df_entry["minus_di"] = adx.adx_neg()

        stochrsi = StochRSIIndicator(
            close=close,
            window=config.STOCHRSI_PERIOD,
            smooth1=config.STOCHRSI_SMOOTH_K,
            smooth2=config.STOCHRSI_SMOOTH_D,
        )
        self._df_entry["stochrsi_k"] = stochrsi.stochrsi_k()
        self._df_entry["stochrsi_d"] = stochrsi.stochrsi_d()

        self._df_entry["atr"] = AverageTrueRange(
            high=high, low=low, close=close, window=config.ADX_PERIOD
        ).average_true_range()

        self._df_entry["rsi"] = RSIIndicator(
            close=close, window=config.RSI_PERIOD
        ).rsi()

    def _compute_trend_indicators(self) -> None:
        """
        Compute ADX on the trend DataFrame for trend strength confirmation.

        Adds columns: adx, plus_di, minus_di.
        """
        if self._df_trend is None or len(self._df_trend) < config.ADX_PERIOD:
            return

        close = self._df_trend["close"]
        high = self._df_trend["high"]
        low = self._df_trend["low"]

        adx = ADXIndicator(
            high=high, low=low, close=close, window=config.ADX_PERIOD,
        )
        self._df_trend["adx"] = adx.adx()
        self._df_trend["plus_di"] = adx.adx_pos()
        self._df_trend["minus_di"] = adx.adx_neg()

"""
Data ingestion pipeline for the trading ML system.

Fetches OHLCV candles from Kraken for ALL USD-quoted pairs and stores
them in TimescaleDB. Supports initial backfill (1 year) and incremental
updates. Rate-limit aware with Kraken's 720-candle cap per request.

Usage:
    python data/ingestion.py backfill          # Full 1-year backfill, all pairs
    python data/ingestion.py update            # Incremental update, recent candles
    python data/ingestion.py update --pairs XRP/USD,SOL/USD  # Specific pairs
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import ccxt
import psycopg2
from psycopg2.extras import execute_values
from loguru import logger

DB_HOST = os.environ.get("TRADING_DB_HOST", "YOUR_DB_HOST")
DB_PORT = os.environ.get("TRADING_DB_PORT", "5432")
DB_NAME = os.environ.get("TRADING_DB_NAME", "trading_data")
DB_USER = os.environ.get("TRADING_DB_USER", "YOUR_DB_USER")
DB_PASS = os.environ.get("TRADING_DB_PASS", "")

TIMEFRAMES = ["1m", "5m", "1h", "1d"]

TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

BACKFILL_DAYS = {
    "1m": 14,
    "5m": 90,
    "1h": 365,
    "1d": 365,
}


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )


def discover_usd_pairs(exchange: ccxt.kraken) -> list[str]:
    """Find all active USD-quoted spot pairs on Kraken."""
    exchange.load_markets()
    pairs = []
    for symbol, market in exchange.markets.items():
        if (
            market.get("quote") == "USD"
            and market.get("spot", False)
            and market.get("active", True)
            and "/" in symbol
        ):
            pairs.append(symbol)
    pairs.sort()
    logger.info("Discovered {n} USD pairs on Kraken", n=len(pairs))
    return pairs


def store_pair_metadata(conn, exchange: ccxt.kraken, pairs: list[str]):
    """Store/update metadata for each pair."""
    cur = conn.cursor()
    for pair in pairs:
        market = exchange.markets.get(pair, {})
        base = market.get("base", pair.split("/")[0])
        quote = market.get("quote", "USD")
        precision = market.get("precision", {})
        limits = market.get("limits", {}).get("amount", {})

        cur.execute("""
            INSERT INTO pair_metadata (pair, base_currency, quote_currency,
                min_order_size, price_decimals, volume_decimals, last_updated, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, NOW(), TRUE)
            ON CONFLICT (pair) DO UPDATE SET
                min_order_size = EXCLUDED.min_order_size,
                price_decimals = EXCLUDED.price_decimals,
                volume_decimals = EXCLUDED.volume_decimals,
                last_updated = NOW(),
                is_active = TRUE
        """, (
            pair, base, quote,
            limits.get("min"),
            precision.get("price"),
            precision.get("amount"),
        ))
    conn.commit()
    logger.info("Updated metadata for {n} pairs", n=len(pairs))


def get_last_timestamp(conn, pair: str, timeframe: str) -> int | None:
    """Get the most recent candle timestamp for a pair/timeframe."""
    cur = conn.cursor()
    cur.execute(
        "SELECT extract(epoch from max(time))::bigint * 1000 FROM ohlcv "
        "WHERE pair = %s AND timeframe = %s",
        (pair, timeframe),
    )
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def fetch_and_store_candles(
    exchange: ccxt.kraken,
    conn,
    pair: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> int:
    """Fetch candles from Kraken and insert into TimescaleDB. Returns count."""
    tf_ms = TIMEFRAME_MS[timeframe]
    cursor_ms = since_ms
    total = 0
    batch_num = 0

    while cursor_ms < until_ms:
        try:
            batch = exchange.fetch_ohlcv(
                pair, timeframe=timeframe, since=cursor_ms, limit=720,
            )
        except ccxt.BadSymbol:
            logger.warning("Pair not available for {tf}: {pair}", tf=timeframe, pair=pair)
            return total
        except ccxt.BaseError as exc:
            logger.error("Fetch error {pair} {tf}: {err}", pair=pair, tf=timeframe, err=exc)
            time.sleep(2)
            continue

        if not batch:
            break

        rows = [
            (
                datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
                pair,
                timeframe,
                c[1], c[2], c[3], c[4], c[5],
            )
            for c in batch
        ]

        cur = conn.cursor()
        execute_values(
            cur,
            """INSERT INTO ohlcv (time, pair, timeframe, open, high, low, close, volume)
               VALUES %s
               ON CONFLICT (pair, timeframe, time) DO UPDATE SET
                   open = EXCLUDED.open, high = EXCLUDED.high,
                   low = EXCLUDED.low, close = EXCLUDED.close,
                   volume = EXCLUDED.volume""",
            rows,
        )
        conn.commit()

        total += len(batch)
        batch_num += 1
        cursor_ms = batch[-1][0] + tf_ms

        if batch_num % 10 == 0:
            logger.info(
                "  {pair} {tf}: {n} candles so far...",
                pair=pair, tf=timeframe, n=total,
            )

        time.sleep(0.5)

    return total


def backfill(pairs: list[str] | None = None):
    """Full backfill: 1 year of 1h/1d, 90 days of 5m, 14 days of 1m."""
    exchange = ccxt.kraken({"enableRateLimit": True})
    all_pairs = discover_usd_pairs(exchange)

    if pairs:
        all_pairs = [p for p in all_pairs if p in pairs]

    conn = get_db_connection()
    store_pair_metadata(conn, exchange, all_pairs)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    total_pairs = len(all_pairs)

    for tf in TIMEFRAMES:
        days = BACKFILL_DAYS[tf]
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        logger.info(
            "=== Backfilling {tf} ({days}d) for {n} pairs ===",
            tf=tf, days=days, n=total_pairs,
        )

        for i, pair in enumerate(all_pairs, 1):
            last_ts = get_last_timestamp(conn, pair, tf)
            start_ms = (last_ts + TIMEFRAME_MS[tf]) if last_ts else since_ms

            if start_ms >= now_ms:
                logger.debug("Up to date: {pair} {tf}", pair=pair, tf=tf)
                continue

            count = fetch_and_store_candles(exchange, conn, pair, tf, start_ms, now_ms)
            if count > 0:
                logger.info(
                    "[{i}/{total}] {pair} {tf}: {n} candles",
                    i=i, total=total_pairs, pair=pair, tf=tf, n=count,
                )

    conn.close()
    logger.info("Backfill complete")


def update(pairs: list[str] | None = None):
    """Incremental update: fetch only new candles since last stored."""
    exchange = ccxt.kraken({"enableRateLimit": True})
    all_pairs = discover_usd_pairs(exchange)

    if pairs:
        all_pairs = [p for p in all_pairs if p in pairs]

    conn = get_db_connection()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    for tf in TIMEFRAMES:
        for pair in all_pairs:
            last_ts = get_last_timestamp(conn, pair, tf)
            if not last_ts:
                days = BACKFILL_DAYS[tf]
                start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
            else:
                start_ms = last_ts + TIMEFRAME_MS[tf]

            if start_ms >= now_ms:
                continue

            count = fetch_and_store_candles(exchange, conn, pair, tf, start_ms, now_ms)
            if count > 0:
                logger.info("{pair} {tf}: +{n} new candles", pair=pair, tf=tf, n=count)

    conn.close()
    logger.info("Update complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading data ingestion pipeline")
    parser.add_argument("command", choices=["backfill", "update"],
                        help="backfill = full historical load, update = incremental")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Comma-separated pairs to process (default: all USD)")
    parser.add_argument("--timeframes", type=str, default=None,
                        help="Comma-separated timeframes (default: 1m,5m,1h,1d)")

    args = parser.parse_args()

    pair_list = args.pairs.split(",") if args.pairs else None
    if args.timeframes:
        TIMEFRAMES = args.timeframes.split(",")

    logger.add(
        "logs/ingestion_{time:YYYY-MM-DD}.log",
        rotation="1 day", retention="30 days",
    )

    if args.command == "backfill":
        backfill(pair_list)
    else:
        update(pair_list)

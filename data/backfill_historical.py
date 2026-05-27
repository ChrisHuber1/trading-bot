"""
Backfill OLDER historical OHLCV data from Kraken into TimescaleDB.

The existing ingestion.py backfill() only fills FORWARD from the latest
existing candle. This script goes BACKWARDS from the earliest existing
candle to fill in older data needed for ML training (bearish periods, etc.).

Only backfills the 1h timeframe (used by the ML pipeline).
Target: 365 days of history per pair.

Usage:
    cd trading-bot
    python data/backfill_historical.py
"""

import sys
import time
from datetime import datetime, timedelta, timezone

import ccxt
from loguru import logger

# Import shared utilities from the existing ingestion module
from data.ingestion import (
    get_db_connection,
    discover_usd_pairs,
    fetch_and_store_candles,
    TIMEFRAME_MS,
)

TIMEFRAME = "1h"
TARGET_DAYS = 365


def get_existing_pairs_and_min_time(conn):
    """Get all distinct pairs from existing 1h OHLCV data with their earliest candle."""
    cur = conn.cursor()
    cur.execute(
        "SELECT pair, extract(epoch from min(time))::bigint * 1000 "
        "FROM ohlcv WHERE timeframe = %s GROUP BY pair ORDER BY pair",
        (TIMEFRAME,),
    )
    return cur.fetchall()


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        "logs/backfill_historical_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
    )

    logger.info("=== Historical Backfill (backwards) -- 1h timeframe ===")

    conn = get_db_connection()
    exchange = ccxt.kraken({"enableRateLimit": True})

    # Get pairs already in the database with their earliest timestamp
    pairs_with_min = get_existing_pairs_and_min_time(conn)

    if not pairs_with_min:
        logger.warning("No existing 1h data found in ohlcv table. Nothing to backfill from.")
        logger.info("Run 'python data/ingestion.py backfill' first to seed initial data.")
        conn.close()
        return

    logger.info("Found {n} pairs with existing 1h data", n=len(pairs_with_min))

    now = datetime.now(timezone.utc)
    target_start_ms = int((now - timedelta(days=TARGET_DAYS)).timestamp() * 1000)

    total_pairs = len(pairs_with_min)
    total_candles = 0
    skipped = 0
    failed = []

    for i, (pair, min_time_ms) in enumerate(pairs_with_min, 1):
        min_time_dt = datetime.fromtimestamp(min_time_ms / 1000, tz=timezone.utc)

        if min_time_ms <= target_start_ms:
            logger.debug(
                "  {pair}: already has data back to {dt}, skipping",
                pair=pair, dt=min_time_dt.strftime("%Y-%m-%d"),
            )
            skipped += 1
            if i % 10 == 0:
                logger.info(
                    "Progress: {i}/{total} pairs processed ({candles} candles inserted so far)",
                    i=i, total=total_pairs, candles=total_candles,
                )
            continue

        gap_days = (min_time_ms - target_start_ms) / 86_400_000
        logger.info(
            "[{i}/{total}] {pair}: earliest={dt}, need ~{days:.0f} more days back",
            i=i, total=total_pairs, pair=pair,
            dt=min_time_dt.strftime("%Y-%m-%d %H:%M"),
            days=gap_days,
        )

        try:
            count = fetch_and_store_candles(
                exchange, conn, pair, TIMEFRAME, target_start_ms, min_time_ms,
            )
            total_candles += count
            if count > 0:
                logger.info(
                    "  {pair}: inserted {n} historical candles",
                    pair=pair, n=count,
                )
            else:
                logger.info("  {pair}: no older data available on Kraken", pair=pair)
        except Exception as exc:
            logger.error("  {pair}: FAILED -- {err}", pair=pair, err=exc)
            failed.append(pair)

        if i % 10 == 0:
            logger.info(
                "Progress: {i}/{total} pairs processed ({candles} candles inserted so far)",
                i=i, total=total_pairs, candles=total_candles,
            )

    conn.close()

    logger.info("=" * 60)
    logger.info("Historical backfill complete")
    logger.info("  Pairs processed: {n}", n=total_pairs)
    logger.info("  Already had full history: {n}", n=skipped)
    logger.info("  Total candles inserted: {n}", n=total_candles)
    if failed:
        logger.warning("  Failed pairs ({n}): {pairs}", n=len(failed), pairs=", ".join(failed))
    else:
        logger.info("  Failed pairs: 0")


if __name__ == "__main__":
    main()

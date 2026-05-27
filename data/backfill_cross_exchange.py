"""
Backfill historical OHLCV from BinanceUS and OKX into TimescaleDB.

Kraken's API only returns ~720 recent candles, so we pull older data from
alternate exchanges where the same assets trade (as USDT pairs). Data is
stored under the Kraken pair name (BASE/USD) so the ML pipeline sees a
seamless history.

Only backfills the 1h timeframe. Fills the gap between 365 days ago and
each pair's earliest existing candle.

Usage:
    cd trading-bot
    python data/backfill_cross_exchange.py
    python data/backfill_cross_exchange.py --dry-run
"""

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone

import ccxt
import psycopg2
from psycopg2.extras import execute_values
from loguru import logger

from data.ingestion import get_db_connection, TIMEFRAME_MS

TIMEFRAME = "1h"
TF_MS = TIMEFRAME_MS[TIMEFRAME]
TARGET_DAYS = 365


def get_pairs_needing_backfill(conn, target_start_ms):
    cur = conn.cursor()
    cur.execute(
        "SELECT pair, extract(epoch from min(time))::bigint * 1000 "
        "FROM ohlcv WHERE timeframe = %s GROUP BY pair ORDER BY pair",
        (TIMEFRAME,),
    )
    pairs = []
    for pair, min_time_ms in cur.fetchall():
        if min_time_ms > target_start_ms:
            pairs.append((pair, min_time_ms))
    return pairs


def build_pair_map(kraken_pairs, exchange):
    """Map Kraken BASE/USD pairs to exchange BASE/USDT (or BASE/USD) symbols."""
    mapping = {}
    for kp, min_ms in kraken_pairs:
        base = kp.split("/")[0]
        for quote in ("USDT", "USD"):
            candidate = f"{base}/{quote}"
            if candidate in exchange.markets:
                mapping[kp] = (candidate, min_ms)
                break
    return mapping


def fetch_and_store(exchange, conn, kraken_pair, exchange_pair, since_ms, until_ms):
    """Fetch candles from exchange and store under the Kraken pair name."""
    cursor_ms = since_ms
    total = 0

    while cursor_ms < until_ms:
        try:
            batch = exchange.fetch_ohlcv(
                exchange_pair, timeframe=TIMEFRAME, since=cursor_ms, limit=720,
            )
        except ccxt.BadSymbol:
            logger.warning("  {pair} not available on {ex}", pair=exchange_pair, ex=exchange.id)
            return total
        except (ccxt.RateLimitExceeded, ccxt.RequestTimeout) as exc:
            logger.debug("  {err_type}, retrying in 5s...", err_type=type(exc).__name__)
            time.sleep(5)
            continue
        except ccxt.BaseError as exc:
            logger.error("  Fetch error {pair}: {err}", pair=exchange_pair, err=exc)
            time.sleep(2)
            continue

        if not batch:
            break

        filtered = [c for c in batch if c[0] < until_ms]
        if not filtered:
            break

        rows = [
            (
                datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
                kraken_pair,
                TIMEFRAME,
                c[1], c[2], c[3], c[4], c[5],
            )
            for c in filtered
        ]

        cur = conn.cursor()
        execute_values(
            cur,
            """INSERT INTO ohlcv (time, pair, timeframe, open, high, low, close, volume)
               VALUES %s
               ON CONFLICT (pair, timeframe, time) DO NOTHING""",
            rows,
        )
        conn.commit()

        total += len(filtered)
        cursor_ms = batch[-1][0] + TF_MS

        if total > 0 and total % 5000 == 0:
            logger.info("    {pair}: {n} candles so far...", pair=kraken_pair, n=total)

        time.sleep(0.3)

    return total


def main():
    parser = argparse.ArgumentParser(description="Cross-exchange historical backfill")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        "logs/backfill_cross_exchange_{time:YYYY-MM-DD}.log",
        rotation="1 day", retention="30 days", level="DEBUG",
    )

    logger.info("=== Cross-Exchange Historical Backfill (1h) ===")

    conn = get_db_connection()
    now = datetime.now(timezone.utc)
    target_start_ms = int((now - timedelta(days=TARGET_DAYS)).timestamp() * 1000)

    pairs_needing = get_pairs_needing_backfill(conn, target_start_ms)
    logger.info("{n} pairs need older data (target: {d} days ago)",
                n=len(pairs_needing), d=TARGET_DAYS)

    if not pairs_needing:
        logger.info("All pairs already have sufficient history.")
        conn.close()
        return

    exchanges = []
    for name in ("binanceus", "okx"):
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            ex.load_markets()
            exchanges.append(ex)
            logger.info("Loaded {name}: {n} markets", name=name, n=len(ex.markets))
        except Exception as exc:
            logger.warning("Could not load {name}: {err}", name=name, err=exc)

    if not exchanges:
        logger.error("No alternate exchanges available")
        conn.close()
        return

    total_candles = 0
    filled = 0
    skipped = 0

    for i, (pair, min_time_ms) in enumerate(pairs_needing, 1):
        gap_days = (min_time_ms - target_start_ms) / 86_400_000
        min_dt = datetime.fromtimestamp(min_time_ms / 1000, tz=timezone.utc)

        source_ex = None
        exchange_pair = None
        for ex in exchanges:
            mapping = build_pair_map([(pair, min_time_ms)], ex)
            if pair in mapping:
                source_ex = ex
                exchange_pair = mapping[pair][0]
                break

        if not source_ex:
            if i <= 20 or i % 50 == 0:
                logger.debug("[{i}/{n}] {pair}: no alternate source, skipping",
                             i=i, n=len(pairs_needing), pair=pair)
            skipped += 1
            continue

        logger.info(
            "[{i}/{n}] {pair}: need ~{days:.0f}d back (earliest={dt}), source={ex}:{ep}",
            i=i, n=len(pairs_needing), pair=pair, days=gap_days,
            dt=min_dt.strftime("%Y-%m-%d"), ex=source_ex.id, ep=exchange_pair,
        )

        if args.dry_run:
            filled += 1
            continue

        count = fetch_and_store(
            source_ex, conn, pair, exchange_pair, target_start_ms, min_time_ms,
        )
        total_candles += count
        if count > 0:
            filled += 1
            logger.info("  {pair}: inserted {n} candles from {ex}",
                        pair=pair, n=count, ex=source_ex.id)
        else:
            logger.info("  {pair}: no older data on {ex}", pair=pair, ex=source_ex.id)

    conn.close()

    logger.info("=" * 60)
    logger.info("Cross-exchange backfill complete")
    logger.info("  Pairs needing data: {n}", n=len(pairs_needing))
    logger.info("  Filled from alt exchange: {n}", n=filled)
    logger.info("  No alt source available: {n}", n=skipped)
    logger.info("  Total candles inserted: {n}", n=total_candles)


if __name__ == "__main__":
    main()

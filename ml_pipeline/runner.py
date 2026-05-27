"""CLI entry point for the ML feature pipeline.

Usage:
    python -m ml_pipeline.runner universe       # Score and filter tradeable pairs
    python -m ml_pipeline.runner backfill       # Full feature + label computation
    python -m ml_pipeline.runner update         # Incremental features + labels
    python -m ml_pipeline.runner status         # Show computation status
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from loguru import logger

from ml_pipeline import config
from ml_pipeline.db import (
    get_connection, fetch_ohlcv, get_high_water_mark, get_all_pairs,
    get_tradeable_pairs, upsert_features, upsert_labels,
    upsert_universe_scores, log_computation,
)
from ml_pipeline.features.compute import (
    compute_features, compute_cross_timeframe_features, compute_cross_pair_features,
)
from ml_pipeline.labels.generate import compute_labels
from ml_pipeline.universe.filter import compute_universe_scores


def cmd_universe():
    """Score all pairs and update universe_scores table."""
    df = compute_universe_scores()
    if df.empty:
        logger.warning("No pairs scored")
        return
    conn = get_connection()
    upsert_universe_scores(conn, df)
    conn.close()
    logger.info("Universe scores updated: {n} pairs", n=len(df))


def cmd_backfill():
    """Full feature and label computation for all tradeable pairs."""
    conn = get_connection()
    pairs = get_tradeable_pairs(conn)
    if not pairs:
        logger.warning("No tradeable pairs found -- run 'universe' first")
        pairs = get_all_pairs(conn)
        logger.info("Falling back to all {n} pairs", n=len(pairs))

    # Phase 1: compute single-pair features for all pairs + BTC
    btc_features = {}
    market_returns = {}

    for tf in config.FEATURE_TIMEFRAMES:
        logger.info("=== Features [{tf}] for {n} pairs ===", tf=tf, n=len(pairs))
        all_pairs_for_tf = list(set(pairs) | {"BTC/USD"})

        for i, pair in enumerate(all_pairs_for_tf, 1):
            t0 = time.time()
            try:
                df = fetch_ohlcv(conn, pair, tf)
                if df.empty or len(df) < config.MAX_LOOKBACK:
                    logger.debug("Skipping {pair} {tf}: insufficient data ({n} rows)", pair=pair, tf=tf, n=len(df))
                    continue

                feat_df = compute_features(df)

                if pair == "BTC/USD":
                    btc_features[tf] = feat_df
                market_returns[pair] = feat_df["return_1"] if "return_1" in feat_df.columns else None

                # Cross-timeframe: merge daily context onto hourly
                if tf == "1h":
                    df_1d = fetch_ohlcv(conn, pair, "1d")
                    if not df_1d.empty and len(df_1d) >= 50:
                        daily_feat = compute_features(df_1d)
                        feat_df = compute_cross_timeframe_features(feat_df, daily_feat)

                # Strip OHLCV columns before upserting (keep only feature columns)
                feature_cols = [c for c in feat_df.columns if c not in ("open", "high", "low", "close", "volume")]
                upsert_features(conn, feat_df[feature_cols], pair, tf)

                elapsed = time.time() - t0
                last_ts = feat_df.index.max()
                log_computation(conn, pair, tf, last_ts, len(feat_df), elapsed, "success")

                if i % 50 == 0 or i == len(all_pairs_for_tf):
                    logger.info("[{i}/{n}] {pair} {tf}: {rows} rows in {t:.1f}s",
                                i=i, n=len(all_pairs_for_tf), pair=pair, tf=tf, rows=len(feat_df), t=elapsed)

            except Exception as exc:
                conn.rollback()
                elapsed = time.time() - t0
                logger.error("Failed {pair} {tf}: {err}", pair=pair, tf=tf, err=exc)
                try:
                    log_computation(conn, pair, tf, datetime.now(timezone.utc), 0, elapsed, "error", str(exc))
                except Exception:
                    conn.rollback()

        # Phase 2: cross-pair features
        if btc_features.get(tf) is not None:
            mr_df = pd.DataFrame({p: r for p, r in market_returns.items() if r is not None})
            if not mr_df.empty:
                logger.info("Computing cross-pair features for {tf}...", tf=tf)
                for pair in pairs:
                    try:
                        df = fetch_ohlcv(conn, pair, tf)
                        if df.empty or len(df) < config.MAX_LOOKBACK:
                            continue
                        feat_df = compute_features(df)
                        feat_df = compute_cross_pair_features(feat_df, btc_features[tf], mr_df)
                        xp_cols = [c for c in feat_df.columns if c.startswith("xp_")]
                        if xp_cols:
                            upsert_features(conn, feat_df[xp_cols], pair, tf)
                    except Exception as exc:
                        logger.error("Cross-pair failed {pair}: {err}", pair=pair, err=exc)

    # Phase 3: labels
    for tf in config.FEATURE_TIMEFRAMES:
        logger.info("=== Labels [{tf}] for {n} pairs ===", tf=tf, n=len(pairs))
        for i, pair in enumerate(pairs, 1):
            try:
                df = fetch_ohlcv(conn, pair, tf)
                if df.empty or len(df) < 50:
                    continue
                label_df = compute_labels(df)
                upsert_labels(conn, label_df, pair, tf)

                if i % 50 == 0 or i == len(pairs):
                    logger.info("[{i}/{n}] Labels for {pair} {tf}: {rows} rows",
                                i=i, n=len(pairs), pair=pair, tf=tf, rows=len(label_df))
            except Exception as exc:
                conn.rollback()
                logger.error("Labels failed {pair} {tf}: {err}", pair=pair, tf=tf, err=exc)

    conn.close()
    logger.info("Backfill complete")


def cmd_update():
    """Incremental feature and label computation for new data only."""
    conn = get_connection()
    pairs = get_tradeable_pairs(conn)
    if not pairs:
        pairs = get_all_pairs(conn)

    updated = 0
    for tf in config.FEATURE_TIMEFRAMES:
        for pair in pairs:
            hwm = get_high_water_mark(conn, pair, tf)
            if hwm is None:
                since = None
            else:
                since = hwm - timedelta(hours=config.MAX_LOOKBACK)

            df = fetch_ohlcv(conn, pair, tf, since=since)
            if df.empty or len(df) < config.MAX_LOOKBACK:
                continue

            new_rows = len(df[df.index > hwm]) if hwm else len(df)
            if new_rows == 0:
                continue

            t0 = time.time()
            try:
                feat_df = compute_features(df)

                if tf == "1h":
                    df_1d = fetch_ohlcv(conn, pair, "1d")
                    if not df_1d.empty and len(df_1d) >= 50:
                        daily_feat = compute_features(df_1d)
                        feat_df = compute_cross_timeframe_features(feat_df, daily_feat)

                # Only upsert rows after the high water mark
                if hwm:
                    feat_df = feat_df[feat_df.index > hwm]

                feature_cols = [c for c in feat_df.columns if c not in ("open", "high", "low", "close", "volume")]
                upsert_features(conn, feat_df[feature_cols], pair, tf)

                # Labels for the new window
                label_df = compute_labels(df)
                if hwm:
                    label_df = label_df[label_df.index > hwm]
                upsert_labels(conn, label_df, pair, tf)

                elapsed = time.time() - t0
                last_ts = feat_df.index.max() if not feat_df.empty else datetime.now(timezone.utc)
                log_computation(conn, pair, tf, last_ts, len(feat_df), elapsed, "success")
                updated += 1

            except Exception as exc:
                logger.error("Update failed {pair} {tf}: {err}", pair=pair, tf=tf, err=exc)

    conn.close()
    logger.info("Update complete: {n} pairs updated", n=updated)


def cmd_status():
    """Show computation status summary."""
    conn = get_connection()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT pair) FROM features")
    feat_pairs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM features")
    feat_rows = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT pair) FROM labels")
    label_pairs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM labels")
    label_rows = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM universe_scores WHERE is_tradeable = TRUE AND time = (SELECT MAX(time) FROM universe_scores)")
    tradeable = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT pair) FROM ohlcv")
    total_pairs = cur.fetchone()[0]

    conn.close()

    print(f"OHLCV pairs:     {total_pairs}")
    print(f"Tradeable pairs: {tradeable}")
    print(f"Feature rows:    {feat_rows:,} across {feat_pairs} pairs")
    print(f"Label rows:      {label_rows:,} across {label_pairs} pairs")


def cmd_labels():
    """Compute labels only (features must already exist)."""
    conn = get_connection()
    pairs = get_tradeable_pairs(conn)
    if not pairs:
        pairs = get_all_pairs(conn)

    for tf in config.FEATURE_TIMEFRAMES:
        logger.info("=== Labels [{tf}] for {n} pairs ===", tf=tf, n=len(pairs))
        for i, pair in enumerate(pairs, 1):
            try:
                df = fetch_ohlcv(conn, pair, tf)
                if df.empty or len(df) < 50:
                    continue
                label_df = compute_labels(df)
                upsert_labels(conn, label_df, pair, tf)

                if i % 50 == 0 or i == len(pairs):
                    logger.info("[{i}/{n}] Labels for {pair} {tf}: {rows} rows",
                                i=i, n=len(pairs), pair=pair, tf=tf, rows=len(label_df))
            except Exception as exc:
                conn.rollback()
                logger.error("Labels failed {pair} {tf}: {err}", pair=pair, tf=tf, err=exc)

    conn.close()
    logger.info("Labels complete")


def main():
    parser = argparse.ArgumentParser(description="ML feature pipeline")
    parser.add_argument("command", choices=["universe", "backfill", "update", "labels", "status"])
    args = parser.parse_args()

    logger.add("logs/ml_pipeline_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")

    if args.command == "universe":
        cmd_universe()
    elif args.command == "backfill":
        cmd_backfill()
    elif args.command == "update":
        cmd_update()
    elif args.command == "labels":
        cmd_labels()
    elif args.command == "status":
        cmd_status()


if __name__ == "__main__":
    main()

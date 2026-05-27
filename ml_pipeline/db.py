"""Database helpers for the ML feature pipeline."""

from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from ml_pipeline.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS


def get_connection():
    """Return a new psycopg2 connection to TimescaleDB."""
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )


def fetch_ohlcv(
    conn,
    pair: str,
    timeframe: str,
    since: datetime | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data as a DataFrame indexed by time."""
    query = (
        "SELECT time, open, high, low, close, volume FROM ohlcv "
        "WHERE pair = %s AND timeframe = %s"
    )
    params: list = [pair, timeframe]

    if since is not None:
        query += " AND time >= %s"
        params.append(since)

    query += " ORDER BY time ASC"

    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    df = pd.read_sql_query(query, conn, params=params, parse_dates=["time"])
    if not df.empty:
        df = df.set_index("time")
    return df


def get_high_water_mark(conn, pair: str, timeframe: str) -> datetime | None:
    """Get the last successfully computed timestamp from the computation log."""
    cur = conn.cursor()
    cur.execute(
        "SELECT last_computed_at FROM feature_computation_log "
        "WHERE pair = %s AND timeframe = %s AND status = 'success' "
        "ORDER BY last_computed_at DESC LIMIT 1",
        (pair, timeframe),
    )
    row = cur.fetchone()
    return row[0] if row else None


def upsert_features(conn, df: pd.DataFrame, pair: str, timeframe: str) -> None:
    """Bulk upsert feature rows into the features table."""
    if df.empty:
        return

    # Build column list from DataFrame, excluding metadata columns
    feature_cols = [c for c in df.columns if c not in ("time", "pair", "timeframe")]
    all_cols = ["time", "pair", "timeframe"] + feature_cols

    rows = []
    for ts, row in df.iterrows():
        rows.append((ts, pair, timeframe) + tuple(
            None if pd.isna(row[c]) else float(row[c]) for c in feature_cols
        ))

    col_list = ", ".join(all_cols)
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in feature_cols)

    sql = (
        f"INSERT INTO features ({col_list}) VALUES %s "
        f"ON CONFLICT (pair, timeframe, time) DO UPDATE SET {update_set}"
    )

    cur = conn.cursor()
    execute_values(cur, sql, rows, page_size=1000)
    conn.commit()


def _coerce_label_value(val, col_name: str):
    """Convert label values to the correct Python type for PostgreSQL."""
    if pd.isna(val):
        return None
    if "tp_hit" in col_name or "sl_hit" in col_name:
        return bool(val)
    if "outcome" in col_name or "rr_class" in col_name:
        return int(val)
    return float(val)


def upsert_labels(conn, df: pd.DataFrame, pair: str, timeframe: str) -> None:
    """Bulk upsert label rows into the labels table."""
    if df.empty:
        return

    label_cols = [c for c in df.columns if c not in ("time", "pair", "timeframe")]
    all_cols = ["time", "pair", "timeframe"] + label_cols

    rows = []
    for ts, row in df.iterrows():
        rows.append((ts, pair, timeframe) + tuple(
            _coerce_label_value(row[c], c) for c in label_cols
        ))

    col_list = ", ".join(all_cols)
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in label_cols)

    sql = (
        f"INSERT INTO labels ({col_list}) VALUES %s "
        f"ON CONFLICT (pair, timeframe, time) DO UPDATE SET {update_set}"
    )

    cur = conn.cursor()
    execute_values(cur, sql, rows, page_size=1000)
    conn.commit()


def upsert_universe_scores(conn, df: pd.DataFrame) -> None:
    """Bulk upsert universe score rows."""
    if df.empty:
        return

    score_cols = [c for c in df.columns if c not in ("time", "pair")]
    all_cols = ["time", "pair"] + score_cols

    rows = []
    for _, row in df.iterrows():
        rows.append(tuple(
            None if pd.isna(row[c]) else row[c] for c in all_cols
        ))

    col_list = ", ".join(all_cols)
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in score_cols)

    sql = (
        f"INSERT INTO universe_scores ({col_list}) VALUES %s "
        f"ON CONFLICT (pair, time) DO UPDATE SET {update_set}"
    )

    cur = conn.cursor()
    execute_values(cur, sql, rows, page_size=1000)
    conn.commit()


def log_computation(
    conn,
    pair: str,
    timeframe: str,
    last_ts: datetime,
    rows: int,
    duration: float,
    status: str,
    error: str | None = None,
) -> None:
    """Insert a row into the feature computation log."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feature_computation_log "
        "(pair, timeframe, last_computed_at, rows_computed, duration_seconds, status, error_message) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (pair, timeframe, last_ts, rows, duration, status, error),
    )
    conn.commit()


def get_all_pairs(conn) -> list[str]:
    """Return all distinct pairs present in the ohlcv table."""
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT pair FROM ohlcv ORDER BY pair")
    return [row[0] for row in cur.fetchall()]


def get_tradeable_pairs(conn) -> list[str]:
    """Return pairs marked as tradeable in the latest universe scores."""
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT pair FROM universe_scores "
        "WHERE is_tradeable = TRUE "
        "AND time = (SELECT MAX(time) FROM universe_scores) "
        "ORDER BY pair"
    )
    return [row[0] for row in cur.fetchall()]

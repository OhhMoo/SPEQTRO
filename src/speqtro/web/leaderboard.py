"""SQLite-backed analysis history and leaderboard for speqtro web UI."""

import sqlite3
import time
from pathlib import Path


_DB_PATH = Path.home() / ".speqtro" / "leaderboard.db"


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,
                mode        TEXT    NOT NULL,
                query       TEXT,
                compound    TEXT,
                confidence  REAL,
                verdict     TEXT,
                duration_s  REAL,
                tool_calls  INTEGER,
                cost_usd    REAL
            )
        """)
        conn.commit()


def log_analysis(
    mode: str,
    query: str | None = None,
    compound: str | None = None,
    confidence: float | None = None,
    verdict: str | None = None,
    duration_s: float | None = None,
    tool_calls: int | None = None,
    cost_usd: float | None = None,
) -> int:
    """Insert a new analysis row. Returns the new row id."""
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO analyses
               (ts, mode, query, compound, confidence, verdict, duration_s, tool_calls, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), mode, query, compound, confidence, verdict, duration_s, tool_calls, cost_usd),
        )
        conn.commit()
        return cur.lastrowid


def get_history(limit: int = 50) -> list[dict]:
    """Return the most recent analyses, newest first."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    """Return aggregate statistics across all analyses."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            """SELECT
                COUNT(*)                        AS total,
                AVG(confidence)                 AS avg_confidence,
                SUM(cost_usd)                   AS total_cost_usd
               FROM analyses"""
        ).fetchone()

        mode_row = conn.execute(
            """SELECT mode, COUNT(*) AS cnt FROM analyses
               GROUP BY mode ORDER BY cnt DESC LIMIT 1"""
        ).fetchone()

        # streak: count consecutive days up to today with at least one analysis
        days = conn.execute(
            """SELECT DISTINCT DATE(ts, 'unixepoch') AS d
               FROM analyses ORDER BY d DESC"""
        ).fetchall()

    total = row["total"] or 0
    avg_confidence = round(row["avg_confidence"] or 0.0, 1)
    total_cost_usd = round(row["total_cost_usd"] or 0.0, 4)
    top_mode = mode_row["mode"] if mode_row else None

    streak_days = 0
    if days:
        from datetime import date, timedelta
        today = date.today()
        for i, d in enumerate(days):
            expected = today - timedelta(days=i)
            if str(d["d"]) == str(expected):
                streak_days += 1
            else:
                break

    return {
        "total": total,
        "avg_confidence": avg_confidence,
        "top_mode": top_mode,
        "streak_days": streak_days,
        "total_cost_usd": total_cost_usd,
    }

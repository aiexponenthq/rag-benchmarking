from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from harness.schemas import BenchmarkReport

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    n_samples   INTEGER NOT NULL,
    metrics     TEXT NOT NULL,
    config      TEXT NOT NULL,
    skipped     TEXT NOT NULL,
    skip_reasons TEXT NOT NULL
);
"""


class ResultStore:
    """SQLite-backed store for BenchmarkReport persistence and comparison."""

    def __init__(self, db_path: str = "data/benchmark_results.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with sqlite3.connect(db_path) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def save_run(self, report: BenchmarkReport) -> None:
        """Persist a BenchmarkReport. Overwrites existing run with same run_id."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO benchmark_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    report.run_id,
                    report.created_at.isoformat(),
                    report.n_samples,
                    json.dumps(report.metrics),
                    json.dumps(report.config),
                    json.dumps(report.skipped_metrics),
                    json.dumps(report.skip_reasons),
                ),
            )

    def get_run(self, run_id: str) -> dict | None:
        """Return a run by ID, or None if not found."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "n_samples": row["n_samples"],
            "metrics": json.loads(row["metrics"]),
            "config": json.loads(row["config"]),
            "skipped_metrics": json.loads(row["skipped"]),
            "skip_reasons": json.loads(row["skip_reasons"]),
        }

    def list_runs(self, limit: int = 50) -> list[dict]:
        """Return most recent runs, newest first."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT run_id, created_at, n_samples, metrics " "FROM benchmark_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": r["run_id"],
                "created_at": r["created_at"],
                "n_samples": r["n_samples"],
                "metrics": json.loads(r["metrics"]),
            }
            for r in rows
        ]

    def compare_runs(self, run_ids: list[str]) -> dict:
        """Compare metrics across multiple runs."""
        runs = [self.get_run(rid) for rid in run_ids]
        all_metrics: set[str] = set()
        for r in runs:
            if r:
                all_metrics.update(r["metrics"].keys())
        comparison: dict[str, list] = {}
        for metric in sorted(all_metrics):
            comparison[metric] = [(r["metrics"].get(metric) if r else None) for r in runs]
        return {"run_ids": run_ids, "metrics": comparison}

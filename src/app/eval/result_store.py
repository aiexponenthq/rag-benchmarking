"""SQLite-backed result store for evaluation run persistence.

Schema
------
runs
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    run_id      TEXT UNIQUE NOT NULL          -- caller-supplied or uuid4
    created_at  TEXT NOT NULL                 -- ISO-8601 UTC
    metrics     TEXT NOT NULL                 -- JSON blob of aggregate metric dict
    n_samples   INTEGER NOT NULL
    skipped     TEXT                          -- JSON list of skipped metric names

samples
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    run_id      TEXT NOT NULL REFERENCES runs(run_id)
    sample_idx  INTEGER NOT NULL
    question    TEXT
    answer      TEXT

metrics
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    run_id      TEXT NOT NULL REFERENCES runs(run_id)
    sample_idx  INTEGER       -- NULL for aggregate rows
    metric_name TEXT NOT NULL
    score       REAL

Usage
-----
    from app.eval.result_store import ResultStore

    store = ResultStore()          # default: ./eval_results.db
    store.save_run(run_id, result, samples)
    runs = store.list_runs()
    run  = store.get_run(run_id)
    comparison = store.compare_runs([run_id_a, run_id_b])
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_DB = Path("eval_results.db")


class ResultStore:
    """Synchronous SQLite store using stdlib ``sqlite3``."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT    UNIQUE NOT NULL,
                    created_at  TEXT    NOT NULL,
                    metrics     TEXT    NOT NULL,
                    n_samples   INTEGER NOT NULL,
                    skipped     TEXT
                );

                CREATE TABLE IF NOT EXISTS samples (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT    NOT NULL REFERENCES runs(run_id),
                    sample_idx  INTEGER NOT NULL,
                    question    TEXT,
                    answer      TEXT
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT    NOT NULL REFERENCES runs(run_id),
                    sample_idx  INTEGER,
                    metric_name TEXT    NOT NULL,
                    score       REAL
                );
                """
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_run(
        self,
        result: dict[str, Any],
        samples: list[dict[str, Any]],
        run_id: str | None = None,
    ) -> str:
        """Persist one evaluation run.

        Parameters
        ----------
        result:
            The dict returned by ``ragas_runner.run_evaluation()``.
            Must contain ``{"metrics": {...}, "per_sample": {...}}``.
        samples:
            The raw input samples (question/answer) for record-keeping.
        run_id:
            Caller-supplied identifier; auto-generated uuid4 if omitted.

        Returns
        -------
        str
            The run_id used.
        """
        run_id = run_id or str(uuid.uuid4())
        now = datetime.now(tz=UTC).isoformat()
        agg_metrics: dict[str, float] = result.get("metrics", {})
        per_sample: dict[str, list[float]] = result.get("per_sample", {})
        skipped: list[str] = result.get("skipped_metrics", [])

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, created_at, metrics, n_samples, skipped) " "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    now,
                    json.dumps(agg_metrics),
                    len(samples),
                    json.dumps(skipped),
                ),
            )

            # Insert sample metadata rows
            for idx, s in enumerate(samples):
                conn.execute(
                    "INSERT INTO samples (run_id, sample_idx, question, answer) VALUES (?, ?, ?, ?)",
                    (
                        run_id,
                        idx,
                        s.get("question") or s.get("user_input", ""),
                        s.get("answer") or s.get("response", ""),
                    ),
                )

            # Aggregate metric rows (sample_idx = NULL)
            for metric_name, score in agg_metrics.items():
                conn.execute(
                    "INSERT INTO metrics (run_id, sample_idx, metric_name, score) VALUES (?, NULL, ?, ?)",
                    (run_id, metric_name, score),
                )

            # Per-sample metric rows
            for metric_name, scores in per_sample.items():
                for idx, score in enumerate(scores):
                    conn.execute(
                        "INSERT INTO metrics (run_id, sample_idx, metric_name, score) VALUES (?, ?, ?, ?)",
                        (run_id, idx, metric_name, score),
                    )

        return run_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return summary rows for recent runs, newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT run_id, created_at, metrics, n_samples, skipped " "FROM runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": r["run_id"],
                "created_at": r["created_at"],
                "metrics": json.loads(r["metrics"]),
                "n_samples": r["n_samples"],
                "skipped_metrics": json.loads(r["skipped"] or "[]"),
            }
            for r in rows
        ]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return full detail for a single run including per-sample scores."""
        with self._conn() as conn:
            run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if run_row is None:
                return None

            sample_rows = conn.execute(
                "SELECT sample_idx, question, answer FROM samples WHERE run_id = ? ORDER BY sample_idx",
                (run_id,),
            ).fetchall()

            metric_rows = conn.execute(
                "SELECT sample_idx, metric_name, score FROM metrics WHERE run_id = ? ORDER BY sample_idx",
                (run_id,),
            ).fetchall()

        # Build per-sample structure
        sample_metrics: dict[int, dict[str, float]] = {}
        for m in metric_rows:
            if m["sample_idx"] is None:
                continue
            sample_metrics.setdefault(m["sample_idx"], {})[m["metric_name"]] = m["score"]

        samples_out = [
            {
                "idx": r["sample_idx"],
                "question": r["question"],
                "answer": r["answer"],
                "metrics": sample_metrics.get(r["sample_idx"], {}),
            }
            for r in sample_rows
        ]

        return {
            "run_id": run_row["run_id"],
            "created_at": run_row["created_at"],
            "n_samples": run_row["n_samples"],
            "metrics": json.loads(run_row["metrics"]),
            "skipped_metrics": json.loads(run_row["skipped"] or "[]"),
            "samples": samples_out,
        }

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        """Return aggregate metrics for multiple runs side-by-side.

        Returns
        -------
        Dict[str, Any]
            ``{"run_ids": [...], "metrics": {"faithfulness": [0.9, 0.85, ...], ...}}``
        """
        comparison: dict[str, list[float | None]] = {}
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run is None:
                for col in comparison:
                    comparison[col].append(None)
                continue
            for metric_name, score in run["metrics"].items():
                comparison.setdefault(metric_name, []).append(score)
            # Pad metrics that weren't present in this run
            for metric_name in comparison:
                if metric_name not in run["metrics"]:
                    comparison[metric_name].append(None)

        return {"run_ids": run_ids, "metrics": comparison}

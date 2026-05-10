from pathlib import Path

import pytest

from harness.result_store import ResultStore
from harness.schemas import BenchmarkReport


@pytest.fixture
def store(tmp_path: Path):
    return ResultStore(db_path=str(tmp_path / "test.db"))


def test_save_and_retrieve_run(store):
    report = BenchmarkReport(
        run_id="run-001",
        n_samples=2,
        metrics={"faithfulness": 0.85, "answer_relevancy": 0.90},
    )
    store.save_run(report)
    result = store.get_run("run-001")
    assert result is not None
    assert result["run_id"] == "run-001"
    assert abs(result["metrics"]["faithfulness"] - 0.85) < 1e-9


def test_list_runs(store):
    for i in range(3):
        report = BenchmarkReport(
            run_id=f"run-{i:03d}",
            n_samples=5,
            metrics={"faithfulness": 0.8},
        )
        store.save_run(report)
    runs = store.list_runs()
    assert len(runs) == 3


def test_compare_runs(store):
    store.save_run(BenchmarkReport(run_id="run-a", n_samples=5, metrics={"faithfulness": 0.7}))
    store.save_run(BenchmarkReport(run_id="run-b", n_samples=5, metrics={"faithfulness": 0.9}))
    comparison = store.compare_runs(["run-a", "run-b"])
    assert comparison["run_ids"] == ["run-a", "run-b"]
    assert abs(comparison["metrics"]["faithfulness"][0] - 0.7) < 1e-9
    assert abs(comparison["metrics"]["faithfulness"][1] - 0.9) < 1e-9


def test_get_nonexistent_run(store):
    assert store.get_run("does-not-exist") is None


def test_save_run_with_skipped_metrics(store):
    report = BenchmarkReport(
        run_id="run-skip",
        n_samples=3,
        metrics={"faithfulness": 0.8},
        skipped_metrics=["context_precision"],
        skip_reasons={"context_precision": "requires ground_truth"},
    )
    store.save_run(report)
    result = store.get_run("run-skip")
    assert "context_precision" in result["skipped_metrics"]

"""Integration tests for the full evaluation pipeline.

Coverage
--------
* ragas_runner.run_evaluation()  — mocked RAGAS internals, real code path
* ResultStore                    — save, list, get, compare (real SQLite in tmp)
* /v1/evaluate endpoint          — async endpoint with monkeypatched runner
* Pytest fixtures for evaluation testing

Run with:
    pytest tests/test_eval_pipeline.py -v
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.eval.result_store import ResultStore
from app.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_sample() -> dict[str, Any]:
    """One sample WITHOUT ground truths (triggers graceful retrieval-metric skip)."""
    return {
        "question": "What is RAG?",
        "contexts": ["RAG combines retrieval and generation."],
        "answer": "RAG is a retrieval-augmented generation approach.",
        "ground_truths": [],
    }


@pytest.fixture()
def full_sample() -> dict[str, Any]:
    """One sample WITH ground truths (enables context_precision + context_recall)."""
    return {
        "question": "What is RAG?",
        "contexts": ["Retrieval-Augmented Generation combines dense retrieval with a language model."],
        "answer": "RAG combines retrieval and language generation.",
        "ground_truths": ["RAG combines retrieval with language model generation."],
    }


@pytest.fixture()
def golden_samples(full_sample: dict[str, Any]) -> list[dict[str, Any]]:
    """A small but representative batch of 5 samples for pipeline tests."""
    base = full_sample
    extras = [
        {
            "question": f"Question {i}",
            "contexts": [f"Context passage {i} with relevant information."],
            "answer": f"Answer {i}.",
            "ground_truths": [f"Reference answer {i}."],
        }
        for i in range(1, 5)
    ]
    return [base] + extras


@pytest.fixture()
def fake_ragas_result() -> dict[str, Any]:
    """A synthetic run_evaluation() return value to avoid real LLM calls in tests."""
    return {
        "metrics": {
            "faithfulness": 0.92,
            "answer_relevancy": 0.88,
            "context_precision": 0.85,
            "context_recall": 0.79,
        },
        "per_sample": {
            "faithfulness": [0.9, 0.95, 0.88, 0.91, 0.96],
            "answer_relevancy": [0.87, 0.89, 0.9, 0.86, 0.88],
            "context_precision": [0.8, 0.9, 0.85, 0.83, 0.87],
            "context_recall": [0.75, 0.82, 0.79, 0.78, 0.81],
        },
        "skipped_metrics": [],
        "skip_reason": None,
    }


@pytest.fixture()
def result_store(tmp_path: pathlib.Path) -> ResultStore:
    """Isolated ResultStore backed by a temp SQLite DB."""
    return ResultStore(db_path=tmp_path / "test_eval.db")


@pytest.fixture()
def api_client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# ResultStore unit tests
# ---------------------------------------------------------------------------


class TestResultStore:
    def test_save_and_list(
        self,
        result_store: ResultStore,
        fake_ragas_result: dict[str, Any],
        golden_samples: list[dict[str, Any]],
    ) -> None:
        run_id = result_store.save_run(fake_ragas_result, golden_samples, run_id="run-001")
        assert run_id == "run-001"

        runs = result_store.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-001"
        assert runs[0]["n_samples"] == 5
        assert runs[0]["metrics"]["faithfulness"] == pytest.approx(0.92)

    def test_get_run_full_detail(
        self,
        result_store: ResultStore,
        fake_ragas_result: dict[str, Any],
        golden_samples: list[dict[str, Any]],
    ) -> None:
        result_store.save_run(fake_ragas_result, golden_samples, run_id="run-002")
        detail = result_store.get_run("run-002")
        assert detail is not None
        assert detail["run_id"] == "run-002"
        assert len(detail["samples"]) == 5
        first = detail["samples"][0]
        assert first["question"] == "What is RAG?"
        assert "faithfulness" in first["metrics"]

    def test_get_run_not_found(self, result_store: ResultStore) -> None:
        assert result_store.get_run("nonexistent") is None

    def test_compare_runs(
        self,
        result_store: ResultStore,
        fake_ragas_result: dict[str, Any],
        golden_samples: list[dict[str, Any]],
    ) -> None:
        result_store.save_run(fake_ragas_result, golden_samples, run_id="run-A")

        # Slightly different second run
        result2 = dict(fake_ragas_result)
        result2["metrics"] = {k: v - 0.05 for k, v in fake_ragas_result["metrics"].items()}
        result2["per_sample"] = {
            k: [v - 0.05 for v in vs] for k, vs in fake_ragas_result["per_sample"].items()
        }
        result_store.save_run(result2, golden_samples, run_id="run-B")

        comparison = result_store.compare_runs(["run-A", "run-B"])
        assert comparison["run_ids"] == ["run-A", "run-B"]
        faith = comparison["metrics"]["faithfulness"]
        assert len(faith) == 2
        assert faith[0] == pytest.approx(0.92)
        assert faith[1] == pytest.approx(0.87)

    def test_auto_generate_run_id(
        self,
        result_store: ResultStore,
        fake_ragas_result: dict[str, Any],
        golden_samples: list[dict[str, Any]],
    ) -> None:
        run_id = result_store.save_run(fake_ragas_result, golden_samples)
        assert len(run_id) == 36  # uuid4 format
        assert result_store.get_run(run_id) is not None

    def test_skipped_metrics_stored(
        self,
        result_store: ResultStore,
        golden_samples: list[dict[str, Any]],
    ) -> None:
        result_with_skip = {
            "metrics": {"faithfulness": 0.9, "answer_relevancy": 0.85},
            "per_sample": {"faithfulness": [0.9] * 5, "answer_relevancy": [0.85] * 5},
            "skipped_metrics": ["context_precision", "context_recall"],
            "skip_reason": "ground_truths not provided",
        }
        run_id = result_store.save_run(result_with_skip, golden_samples)
        detail = result_store.get_run(run_id)
        assert "context_precision" in detail["skipped_metrics"]
        assert "context_recall" in detail["skipped_metrics"]


# ---------------------------------------------------------------------------
# ragas_runner unit tests (real code path, mocked RAGAS internals)
# ---------------------------------------------------------------------------


class TestRagasRunner:
    """Tests patch the source modules that ragas_runner imports lazily.

    Because all imports in run_evaluation() are inside the function body,
    we patch at the source module level (e.g. ``ragas.evaluate``) rather
    than at ``app.eval.ragas_runner.evaluate``.
    """

    def test_skips_retrieval_metrics_without_ground_truth(
        self, minimal_sample: dict[str, Any]
    ) -> None:
        """When ground_truths is empty, context_precision and context_recall are skipped."""
        from app.eval import ragas_runner as rr
        import os

        import pandas as pd

        os.environ.setdefault("GEMINI_API_KEY", "test-key")

        with (
            patch("ragas.evaluate") as mock_eval,
            patch("langchain_google_genai.ChatGoogleGenerativeAI"),
            patch("ragas.llms.LangchainLLMWrapper"),
        ):
            mock_eval.return_value = MagicMock(
                to_pandas=lambda: pd.DataFrame(
                    [{"faithfulness": 0.9, "answer_relevancy": 0.88}]
                )
            )
            result = rr.run_evaluation([minimal_sample])

        assert "context_precision" not in result["metrics"]
        assert "context_recall" not in result["metrics"]
        assert "context_precision" in result["skipped_metrics"]
        assert "context_recall" in result["skipped_metrics"]

    def test_includes_retrieval_metrics_with_ground_truth(
        self, full_sample: dict[str, Any]
    ) -> None:
        """When ground_truths is present, all four metrics are requested."""
        from app.eval import ragas_runner as rr
        import os
        import pandas as pd

        os.environ.setdefault("GEMINI_API_KEY", "test-key")

        with (
            patch("ragas.evaluate") as mock_eval,
            patch("langchain_google_genai.ChatGoogleGenerativeAI"),
            patch("ragas.llms.LangchainLLMWrapper"),
        ):
            mock_eval.return_value = MagicMock(
                to_pandas=lambda: pd.DataFrame(
                    [
                        {
                            "faithfulness": 0.9,
                            "answer_relevancy": 0.88,
                            "llm_context_precision_with_reference": 0.85,
                            "llm_context_recall": 0.79,
                        }
                    ]
                )
            )
            result = rr.run_evaluation([full_sample])

        # evaluate() was called — retrieval metrics were not skipped
        assert mock_eval.called
        assert result["skipped_metrics"] == []

    def test_raises_without_gemini_key(self, full_sample: dict[str, Any]) -> None:
        """RuntimeError raised when GEMINI_API_KEY is absent."""
        from app.eval import ragas_runner as rr
        import os

        saved = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
                rr.run_evaluation([full_sample])
        finally:
            if saved:
                os.environ["GEMINI_API_KEY"] = saved

    def test_field_translation_question_to_user_input(
        self, full_sample: dict[str, Any]
    ) -> None:
        """Samples keyed as 'question'/'contexts'/'answer' are translated to SingleTurnSample."""
        from app.eval import ragas_runner as rr
        import os
        import pandas as pd

        os.environ.setdefault("GEMINI_API_KEY", "test-key")
        captured_dataset: list[Any] = []

        def capture_evaluate(dataset, **kwargs):
            captured_dataset.append(dataset)
            return MagicMock(to_pandas=lambda: pd.DataFrame([{"faithfulness": 1.0}]))

        with (
            patch("ragas.evaluate", side_effect=capture_evaluate),
            patch("langchain_google_genai.ChatGoogleGenerativeAI"),
            patch("ragas.llms.LangchainLLMWrapper"),
        ):
            rr.run_evaluation([full_sample], metrics=["faithfulness"])

        ds = captured_dataset[0]
        sample = ds.samples[0]
        assert sample.user_input == full_sample["question"]
        assert sample.retrieved_contexts == full_sample["contexts"]
        assert sample.response == full_sample["answer"]


# ---------------------------------------------------------------------------
# /v1/evaluate endpoint integration tests
# ---------------------------------------------------------------------------


class TestEvaluateEndpoint:
    def test_evaluate_returns_result_and_written(
        self,
        api_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Patch run_evaluation where EvaluationRunner imports it
        import harness.runner as runner_mod

        def fake_run(samples: Any, metrics: Any = None) -> dict[str, Any]:
            return {
                "metrics": {"faithfulness": 0.91},
                "per_sample": {"faithfulness": [0.91]},
                "skipped_metrics": [],
                "skip_reason": None,
            }

        monkeypatch.setattr(runner_mod, "run_evaluation", fake_run)

        payload = {
            "samples": [
                {
                    "question": "What is RAG?",
                    "contexts": ["RAG retrieves and generates."],
                    "answer": "RAG retrieves then generates.",
                    "ground_truths": ["RAG retrieves then generates."],
                }
            ]
        }
        resp = api_client.post("/v1/evaluate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # New response shape: metrics at top level (not nested under "result")
        assert data["metrics"]["faithfulness"] == pytest.approx(0.91)
        assert "written" in data

    def test_evaluate_multiple_samples(
        self,
        api_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        golden_samples: list[dict[str, Any]],
    ) -> None:
        import harness.runner as runner_mod

        received: list[Any] = []

        def fake_run(samples: Any, metrics: Any = None) -> dict[str, Any]:
            received.extend(samples)
            return {
                "metrics": {"faithfulness": 0.9, "answer_relevancy": 0.85},
                "per_sample": {},
                "skipped_metrics": [],
                "skip_reason": None,
            }

        monkeypatch.setattr(runner_mod, "run_evaluation", fake_run)

        payload = {
            "samples": [
                {
                    "question": s["question"],
                    "contexts": s["contexts"],
                    "answer": s["answer"],
                    "ground_truths": s["ground_truths"],
                }
                for s in golden_samples
            ]
        }
        resp = api_client.post("/v1/evaluate", json=payload)
        assert resp.status_code == 200
        assert len(received) == 5

    def test_evaluate_propagates_500_on_error(
        self,
        api_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import harness.runner as runner_mod

        def boom(samples: Any, metrics: Any = None) -> dict[str, Any]:
            raise RuntimeError("GEMINI_API_KEY is required")

        monkeypatch.setattr(runner_mod, "run_evaluation", boom)

        payload = {
            "samples": [
                {
                    "question": "q",
                    "contexts": ["c"],
                    "answer": "a",
                    "ground_truths": [],
                }
            ]
        }
        resp = api_client.post("/v1/evaluate", json=payload)
        assert resp.status_code == 500
        assert "GEMINI_API_KEY" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Full pipeline integration test (store + runner together)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_run_and_persist(
        self,
        golden_samples: list[dict[str, Any]],
        fake_ragas_result: dict[str, Any],
        result_store: ResultStore,
    ) -> None:
        """Simulate a complete evaluation loop: run -> store -> retrieve."""
        # 1. Save
        run_id = result_store.save_run(fake_ragas_result, golden_samples, run_id="pipe-001")

        # 2. Retrieve and validate
        detail = result_store.get_run(run_id)
        assert detail["n_samples"] == 5
        assert detail["metrics"]["faithfulness"] == pytest.approx(0.92)

        # 3. Verify per-sample stored
        first_sample = detail["samples"][0]
        assert first_sample["metrics"]["faithfulness"] == pytest.approx(0.9)

    def test_multiple_runs_comparison(
        self,
        golden_samples: list[dict[str, Any]],
        fake_ragas_result: dict[str, Any],
        result_store: ResultStore,
    ) -> None:
        """Compare 3 runs to verify regression tracking works."""
        run_ids = []
        for i, delta in enumerate([0.0, -0.03, +0.05]):
            r = {
                "metrics": {k: v + delta for k, v in fake_ragas_result["metrics"].items()},
                "per_sample": fake_ragas_result["per_sample"],
                "skipped_metrics": [],
                "skip_reason": None,
            }
            rid = result_store.save_run(r, golden_samples, run_id=f"compare-{i}")
            run_ids.append(rid)

        cmp = result_store.compare_runs(run_ids)
        faith_scores = cmp["metrics"]["faithfulness"]
        assert len(faith_scores) == 3
        # Third run should have highest faithfulness
        assert faith_scores[2] > faith_scores[0] > faith_scores[1]

import pytest
from unittest.mock import patch, MagicMock
from harness.runner import EvaluationRunner
from harness.schemas import EvalSample, RunConfig, BenchmarkReport


@pytest.fixture
def samples():
    return [
        EvalSample(
            question="What is RAG?",
            contexts=["RAG stands for Retrieval-Augmented Generation."],
            answer="RAG is a technique that combines retrieval with generation.",
            ground_truth="Retrieval-Augmented Generation combines retrieval with LLMs.",
        ),
        EvalSample(
            question="What is a vector database?",
            contexts=["A vector database stores high-dimensional embeddings."],
            answer="A vector database stores embeddings for similarity search.",
            ground_truth="Vector databases store high-dimensional embeddings for fast similarity search.",
        ),
    ]


def test_runner_returns_benchmark_report(samples):
    config = RunConfig(metrics=["faithfulness"])
    runner = EvaluationRunner(config)

    mock_ragas_result = {
        "metrics": {"faithfulness": 0.9},
        "per_sample": {"faithfulness": [0.9, 0.9]},
        "skipped_metrics": [],
        "skip_reason": None,
    }

    with patch("harness.runner.run_evaluation", return_value=mock_ragas_result):
        report = runner.evaluate(samples)

    assert isinstance(report, BenchmarkReport)
    assert report.n_samples == 2
    assert "faithfulness" in report.metrics
    assert 0.0 <= report.metrics["faithfulness"] <= 1.0


def test_runner_skips_retrieval_metrics_without_relevant_ids(samples):
    config = RunConfig(metrics=["precision_at_k"])
    runner = EvaluationRunner(config)
    report = runner.evaluate(samples)
    assert "precision_at_k" in report.skipped_metrics


def test_runner_skips_context_precision_without_ground_truths():
    samples_no_gt = [
        EvalSample(
            question="Q?",
            contexts=["ctx"],
            answer="A.",
        )
    ]
    config = RunConfig(metrics=["context_precision"])
    runner = EvaluationRunner(config)

    mock_ragas_result = {
        "metrics": {},
        "per_sample": {},
        "skipped_metrics": ["context_precision"],
        "skip_reason": "ground_truths not provided",
    }

    with patch("harness.runner.run_evaluation", return_value=mock_ragas_result):
        report = runner.evaluate(samples_no_gt)

    assert "context_precision" in report.skipped_metrics


def test_runner_generates_run_id(samples):
    config = RunConfig(metrics=["faithfulness"])
    runner = EvaluationRunner(config)

    mock_ragas_result = {
        "metrics": {"faithfulness": 0.8},
        "per_sample": {"faithfulness": [0.8, 0.8]},
        "skipped_metrics": [],
        "skip_reason": None,
    }

    with patch("harness.runner.run_evaluation", return_value=mock_ragas_result):
        report = runner.evaluate(samples)

    assert report.run_id is not None
    assert len(report.run_id) > 0

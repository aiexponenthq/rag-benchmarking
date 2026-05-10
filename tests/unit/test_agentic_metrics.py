import pytest

from app.eval.agentic_metrics import source_attribution_accuracy


def test_perfect_attribution():
    result = source_attribution_accuracy(
        cited_ids=["doc-1", "doc-2"],
        retrieved_ids=["doc-1", "doc-2", "doc-3"],
    )
    assert result["score"] == pytest.approx(1.0)
    assert result["hallucinated_sources"] == []


def test_hallucinated_source():
    result = source_attribution_accuracy(
        cited_ids=["doc-1", "fabricated-99"],
        retrieved_ids=["doc-1", "doc-2"],
    )
    assert result["score"] == pytest.approx(0.5)
    assert "fabricated-99" in result["hallucinated_sources"]


def test_no_citations_returns_perfect():
    result = source_attribution_accuracy(cited_ids=[], retrieved_ids=["doc-1"])
    assert result["score"] == pytest.approx(1.0)


def test_empty_retrieved():
    result = source_attribution_accuracy(cited_ids=["doc-1"], retrieved_ids=[])
    assert result["score"] == pytest.approx(0.0)
    assert "doc-1" in result["hallucinated_sources"]


def test_all_hallucinated():
    result = source_attribution_accuracy(
        cited_ids=["fake-1", "fake-2"],
        retrieved_ids=["doc-1", "doc-2"],
    )
    assert result["score"] == pytest.approx(0.0)
    assert set(result["hallucinated_sources"]) == {"fake-1", "fake-2"}


def test_coverage_field():
    result = source_attribution_accuracy(
        cited_ids=["doc-1"],
        retrieved_ids=["doc-1", "doc-2"],
    )
    assert "coverage" in result
    assert result["coverage"] == pytest.approx(0.5)  # 1 of 2 retrieved cited

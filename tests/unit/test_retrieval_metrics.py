import math
import pytest
from app.eval.retrieval_metrics import precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k


def test_precision_at_k_perfect():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

def test_precision_at_k_partial():
    result = precision_at_k(["a", "x", "b"], {"a", "b", "c"}, k=3)
    assert abs(result - 2/3) < 1e-9

def test_precision_at_k_zero():
    assert precision_at_k(["x", "y", "z"], {"a", "b", "c"}, k=3) == 0.0

def test_precision_at_k_zero_k():
    assert precision_at_k(["a"], {"a"}, k=0) == 0.0

def test_recall_at_k_perfect():
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

def test_recall_at_k_partial():
    result = recall_at_k(["a", "x", "b"], {"a", "b", "c", "d"}, k=3)
    assert abs(result - 0.5) < 1e-9

def test_recall_at_k_empty_relevant():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0

def test_mrr_first_relevant_at_1():
    assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

def test_mrr_first_relevant_at_3():
    result = mean_reciprocal_rank(["x", "y", "a"], {"a"})
    assert abs(result - 1/3) < 1e-9

def test_mrr_no_relevant():
    assert mean_reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

def test_ndcg_perfect():
    result = ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3)
    assert abs(result - 1.0) < 1e-9

def test_ndcg_empty_relevant():
    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0

def test_ndcg_partial():
    # Only first doc is relevant
    result = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
    # DCG = 1/log2(2) = 1.0, IDCG = 1/log2(2) = 1.0 → NDCG = 1.0
    assert abs(result - 1.0) < 1e-9

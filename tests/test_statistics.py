"""Tests for bootstrap statistics and paired comparisons."""

import pytest

from src.evaluation.statistics import (
    bootstrap_ci,
    compare_metric,
    compare_runs,
    paired_bootstrap,
    paired_effect_size,
    sign_test,
)


def test_bootstrap_ci_is_reproducible():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    first = bootstrap_ci(values, iterations=500, seed=13)
    second = bootstrap_ci(values, iterations=500, seed=13)

    assert first == second
    assert first["lower"] <= first["mean"] <= first["upper"]
    assert first["n"] == 7


def test_bootstrap_ci_empty_sample():
    result = bootstrap_ci([], iterations=100)
    assert result["n"] == 0
    assert result["mean"] is None


def test_paired_bootstrap_zero_difference():
    values = [0.5, 0.6, 0.7, 0.8]
    result = paired_bootstrap(values, values, iterations=500, seed=13)

    assert result["observed_difference"] == pytest.approx(0.0)
    assert result["p_value"] == 1.0


def test_paired_bootstrap_detects_consistent_gain():
    a = [0.6, 0.7, 0.8, 0.65, 0.9, 0.55, 0.75, 0.85]
    b = [value - 0.2 for value in a]
    result = paired_bootstrap(a, b, iterations=2000, seed=13)

    assert result["observed_difference"] == pytest.approx(0.2)
    assert result["lower"] > 0
    assert result["p_value"] < 0.05


def test_paired_bootstrap_requires_equal_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_effect_size_and_wins():
    a = [0.9, 0.8, 0.7, 0.6]
    b = [0.5, 0.5, 0.5, 0.5]
    result = paired_effect_size(a, b)

    assert result["cohens_dz"] > 0
    assert result["wins"] == 4
    assert result["losses"] == 0
    assert result["magnitude"] in {"small", "medium", "large"}


def test_sign_test_counts():
    result = sign_test([0.9, 0.8, 0.1, 0.5], [0.5, 0.5, 0.5, 0.5])
    assert result["wins"] == 2
    assert result["losses"] == 1
    assert result["n"] == 3
    assert 0 <= result["p_value"] <= 1


def test_compare_metric_pairs_by_query_id():
    a = [
        {"query_id": "q1", "mrr": 1.0},
        {"query_id": "q2", "mrr": 0.5},
        {"query_id": "q3", "mrr": 0.0},
    ]
    b = [
        {"query_id": "q1", "mrr": 0.5},
        {"query_id": "q2", "mrr": 0.5},
        {"query_id": "q9", "mrr": 0.9},
    ]

    result = compare_metric(a, b, "mrr", iterations=500, seed=13)

    assert result["num_paired_queries"] == 2
    assert result["mean_a"] == pytest.approx(0.75)
    assert result["mean_b"] == pytest.approx(0.5)
    assert result["status"] == "ok"
    assert result["power_warning"]


def test_compare_metric_reports_insufficient_overlap():
    result = compare_metric(
        [{"query_id": "q1", "mrr": 1.0}],
        [{"query_id": "q2", "mrr": 1.0}],
        "mrr",
    )
    assert result["status"] == "insufficient_paired_queries"


def test_compare_runs_labels_come_from_runs():
    run_a = {
        "experiment_id": "exp-a",
        "chunking_strategy": "fixed_size",
        "retrieval_config": "bm25",
        "per_query": [{"query_id": "q1", "recall@5": 1.0}],
    }
    run_b = {
        "experiment_id": "exp-b",
        "chunking_strategy": "fixed_size",
        "retrieval_config": "dense",
        "per_query": [{"query_id": "q1", "recall@5": 0.0}],
    }

    result = compare_runs(run_a, run_b, metrics=("recall@5",), iterations=200, seed=13)

    assert result["run_a"]["retrieval_config"] == "bm25"
    assert result["run_b"]["retrieval_config"] == "dense"
    assert result["comparisons"]["recall@5"]["bootstrap"]["observed_difference"] == pytest.approx(1.0)

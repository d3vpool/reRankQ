"""Hand-calculated tests for retrieval metrics and judgment gating."""

import math

import pytest

from src.evaluation.retrieval_metrics import (
    average_precision,
    coverage_by_query,
    evaluate_query,
    evaluate_run,
    gain,
    metric_table,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["a", "b", "c", "d"]
GRADES = {"a": 3, "b": 0, "c": 2, "d": 2}
RELEVANT = {"a", "c", "d"}


def test_gain_is_exponential():
    assert [gain(grade) for grade in range(4)] == [0.0, 1.0, 3.0, 7.0]


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, RELEVANT, 1) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(2 / 3)
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(1.0)


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, RELEVANT, 1) == pytest.approx(1.0)
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(3 / 4)


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, RELEVANT) == pytest.approx(1.0)
    assert reciprocal_rank(["b", "c"], RELEVANT) == pytest.approx(1 / 2)
    assert reciprocal_rank(["b", "x"], RELEVANT) == 0.0


def test_average_precision():
    # hits at ranks 1, 3, 4: (1/1 + 2/3 + 3/4) / 3
    expected = (1.0 + 2 / 3 + 3 / 4) / 3
    assert average_precision(RETRIEVED, RELEVANT) == pytest.approx(expected)


def test_ndcg_hand_calculated():
    retrieved_grades = [3, 0, 2, 2]
    dcg_3 = 7 / 1 + 0 + 3 / math.log2(4)
    ideal = [3, 2, 2, 0]
    idcg_3 = 7 / 1 + 3 / math.log2(3) + 3 / math.log2(4)
    assert ndcg_at_k(retrieved_grades, ideal, 3) == pytest.approx(dcg_3 / idcg_3)
    assert ndcg_at_k(ideal, ideal, 4) == pytest.approx(1.0)


def test_ndcg_without_gain_is_undefined():
    with pytest.raises(ValueError):
        ndcg_at_k([0, 0], [0, 0], 2)


def test_recall_requires_relevant_set():
    with pytest.raises(ValueError):
        recall_at_k(RETRIEVED, set(), 5)


def test_grade_one_is_not_relevant_at_threshold_two():
    grades = {"a": 1, "b": 0}
    result = evaluate_query(["a", "b"], grades, k_values=[1, 2])
    assert result["status"] == "excluded"
    assert result["reason"] == "no_judged_relevant_chunks"


def test_pending_candidates_are_not_scored_as_zero():
    grades = {"a": 3}
    result = evaluate_query(["a", "unjudged"], grades, k_values=[1, 2])
    assert result["status"] == "excluded"
    assert result["reason"] == "unjudged_retrieved_chunks"
    assert result["unjudged_retrieved"] == 1


def test_query_metrics_when_everything_is_judged():
    result = evaluate_query(RETRIEVED, GRADES, k_values=[1, 3, 5])
    assert result["status"] == "evaluated"
    assert result["recall@1"] == pytest.approx(1 / 3)
    assert result["ndcg@3"] == pytest.approx(
        (7 + 3 / math.log2(4)) / (7 + 3 / math.log2(3) + 3 / math.log2(4))
    )
    assert result["num_relevant_judged"] == 3


def test_evaluate_run_blocks_without_judgments():
    report = evaluate_run(
        ["q1", "q2"],
        {"q1": ["a", "b"], "q2": ["c"]},
        {"q1": {"a": 3}, "q2": {}},
        {},
        k_values=[1, 3],
    )
    assert report["status"] == "blocked"
    assert report["num_evaluated"] == 0
    assert report["metrics"] == {}
    assert report["excluded"]["unjudged_retrieved_chunks"] == 1
    assert report["excluded"]["no_judged_relevant_chunks"] == 1


def test_evaluate_run_macro_average():
    qrels = {
        "q1": {"a": 3, "b": 0},
        "q2": {"c": 0, "d": 2},
    }
    run = {"q1": ["a", "b"], "q2": ["d", "c"]}
    report = evaluate_run(["q1", "q2"], run, qrels, {}, k_values=[1, 2])
    assert report["status"] == "ok"
    assert report["num_evaluated"] == 2
    assert report["metrics"]["recall@1"] == pytest.approx((1 + 1) / 2)
    assert report["metrics"]["mrr"] == pytest.approx(1.0)
    assert report["metrics"]["ndcg@2"] == pytest.approx(1.0)


def test_strict_pool_policy_requires_full_pool_judgment():
    qrels = {"q1": {"a": 3, "b": 0}}
    run = {"q1": ["a"]}
    pool = {"q1": {"a", "b", "c"}}

    lenient = evaluate_run(["q1"], run, qrels, pool, k_values=[1], policy="strict_topk")
    strict = evaluate_run(["q1"], run, qrels, pool, k_values=[1], policy="strict_pool")

    assert lenient["status"] == "ok"
    assert strict["status"] == "blocked"
    assert strict["excluded"]["incomplete_pool"] == 1


def test_coverage_by_query():
    candidates = {
        "q1": [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}],
        "q2": [{"chunk_id": "d"}],
    }
    coverage = coverage_by_query(["q1", "q2"], candidates, {"q1": {"a", "b"}})

    assert coverage["pool_candidates"] == 4
    assert coverage["judged_candidates"] == 2
    assert coverage["pending_candidates"] == 2
    assert coverage["fully_judged_queries"] == 0
    assert coverage["queries_without_judgments"] == ["q2"]
    assert coverage["per_query"][0]["coverage"] == pytest.approx(2 / 3)


def test_metric_table_marks_missing_values():
    rows = [
        {
            "experiment_id": "exp-1",
            "chunking_strategy": "fixed_size",
            "retrieval_config": "bm25",
            "num_evaluated": 0,
            "metrics": {"status": "blocked"},
        }
    ]
    table = metric_table(rows)
    assert table[0]["mrr"] is None
    assert table[0]["recall@5"] is None

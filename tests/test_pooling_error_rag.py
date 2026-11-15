"""Tests for pooling coverage, error analysis, benchmarks and RAG evaluation."""

import pytest

from src.evaluation.benchmark import benchmark_configuration
from src.evaluation.error_analysis import (
    analyze_experiments,
    compare_configuration,
    first_relevant_rank,
    query_outcome,
)
from src.evaluation.pooling import merge_pools, pool_coverage_report
from src.evaluation.rag_eval import (
    HumanAnswerEvalStore,
    build_answer_record,
    evaluate_answers,
    evaluate_citations,
    generate_answer,
    parse_citations,
)


def candidate(query_id, strategy, chunk_id, candidate_id=None):
    return {
        "candidate_id": candidate_id or f"cand_{query_id}_{strategy}_{chunk_id}",
        "query_id": query_id,
        "chunking_strategy": strategy,
        "chunk_id": chunk_id,
        "doc_id": "doc-1",
    }


def judgment(query_id, strategy, chunk_id, grade, status="judged"):
    return {
        "query_id": query_id,
        "candidate_id": f"cand_{query_id}_{strategy}_{chunk_id}",
        "chunking_strategy": strategy,
        "chunk_id": chunk_id,
        "doc_id": "doc-1",
        "relevance_grade": grade,
        "judgment_status": status,
    }


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def test_pool_coverage_report_counts_pending_and_judged():
    queries = [{"query_id": "q1"}, {"query_id": "q2"}]
    candidates = [
        candidate("q1", "fixed_size", "c1"),
        candidate("q1", "fixed_size", "c2"),
        candidate("q1", "structure_aware", "s1"),
        candidate("q2", "fixed_size", "c3"),
    ]
    provenance = [
        {
            "query_id": row["query_id"],
            "chunking_strategy": row["chunking_strategy"],
            "chunk_id": row["chunk_id"],
            "systems": [{"system": "bm25", "rank": 1, "score": 1.0}],
        }
        for row in candidates
    ]
    judgments = [
        judgment("q1", "fixed_size", "c1", 3),
        judgment("q1", "fixed_size", "c2", 0),
    ]

    report = pool_coverage_report(
        queries, candidates, provenance, judgments, ("fixed_size", "structure_aware")
    )

    assert report["pool_candidates"] == 4
    assert report["judged_pairs"] == 2
    assert report["pending_pairs"] == 2
    fixed = report["by_strategy"]["fixed_size"]
    assert fixed["judged"] == 2
    assert fixed["pending"] == 1
    assert fixed["queries_fully_judged"] == 1
    assert fixed["system_contributions"] == {"bm25": 3}
    assert report["adequate_for_cutoff"]["fixed_size"] is False


def test_pool_coverage_ignores_pending_records():
    queries = [{"query_id": "q1"}]
    candidates = [candidate("q1", "fixed_size", "c1")]
    judgments = [judgment("q1", "fixed_size", "c1", 0, status="pending")]

    report = pool_coverage_report([{"query_id": "q1"}], candidates, [], judgments, ("fixed_size",))

    assert report["judged_pairs"] == 0
    assert report["pending_pairs"] == 1


def test_merge_pools_deduplicates_and_keeps_provenance():
    existing = [candidate("q1", "fixed_size", "c1", "cand_a")]
    existing_provenance = [
        {
            "query_id": "q1",
            "chunking_strategy": "fixed_size",
            "chunk_id": "c1",
            "systems": [{"system": "bm25", "rank": 1, "score": 1.0}],
            "num_systems": 1,
            "best_rank": 1,
        }
    ]
    new = [candidate("q1", "fixed_size", "c1", "cand_a"), candidate("q1", "fixed_size", "c2", "cand_b")]
    new_provenance = [
        {
            "query_id": "q1",
            "chunking_strategy": "fixed_size",
            "chunk_id": "c1",
            "systems": [{"system": "dense", "rank": 2, "score": 0.5}],
        },
        {
            "query_id": "q1",
            "chunking_strategy": "fixed_size",
            "chunk_id": "c2",
            "systems": [{"system": "dense", "rank": 1, "score": 0.9}],
        },
    ]

    merged, merged_provenance, stats = merge_pools(
        existing, existing_provenance, new, new_provenance
    )

    assert stats == {"existing": 1, "new_input": 2, "added": 1, "total": 2}
    assert len(merged) == 2
    c1 = next(row for row in merged_provenance if row["chunk_id"] == "c1")
    assert {source["system"] for source in c1["systems"]} == {"bm25", "dense"}
    assert c1["num_systems"] == 2
    assert c1["best_rank"] == 1


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def rows(query_id, chunk_order, query="q"):
    return [
        {
            "query_id": query_id,
            "query": query,
            "ranked": [
                {"rank": index + 1, "chunk_id": chunk_id, "doc_id": "doc-1", "score": 1.0}
                for index, chunk_id in enumerate(chunk_order)
            ],
        }
    ]


def test_first_relevant_rank_and_outcome():
    ranked = rows("q1", ["x", "a", "b"])[0]["ranked"]
    grades = {"a": 3, "b": 0}

    assert first_relevant_rank(ranked, grades) == 2
    outcome = query_outcome("q1", ranked, grades, k=3)
    assert outcome["hit_at_k"] is True
    assert outcome["first_relevant_rank"] == 2
    assert outcome["unjudged_at_k"] == 1
    assert outcome["has_judged_relevant"] is True


def test_outcome_without_judged_relevant():
    ranked = rows("q1", ["x", "y"])[0]["ranked"]
    outcome = query_outcome("q1", ranked, {"x": 0}, k=2)
    assert outcome["has_judged_relevant"] is False
    assert outcome["hit_at_k"] is False


def test_compare_configuration_classifies_improvement():
    qrels = {"q1": {"a": 3, "b": 0}, "q2": {"c": 3}, "q3": {}}
    run_a = rows("q1", ["b", "a"]) + rows("q2", ["c"]) + rows("q3", ["x"])
    run_b = rows("q1", ["a", "b"]) + rows("q2", ["z"]) + rows("q3", ["y"])

    comparison = compare_configuration(run_a, run_b, qrels, k=5, label_a="bm25", label_b="dense")

    counts = comparison["counts"]
    assert counts["both_hit"] == 1
    assert counts["only_a"] == 1
    assert counts["improved_by_b"] == 1
    assert counts["excluded_no_judged_relevant"] == 1
    assert comparison["improved_examples"]
    assert comparison["improved_examples"][0]["query_id"] == "q1"


def test_compare_configuration_reports_unjudged_chunks():
    qrels = {"q1": {"a": 3}}
    comparison = compare_configuration(
        rows("q1", ["a", "unknown"]), rows("q1", ["unknown", "a"]), qrels, k=2
    )
    assert comparison["counts"]["unjudged_at_k"] == 2


def test_analyze_experiments_builds_standard_comparisons():
    qrels = {"q1": {"a": 3}}
    run = {
        "chunking_strategy": "fixed_size",
        "retrieval_config": "bm25",
        "rows": rows("q1", ["a"]),
        "qrels": qrels,
    }
    other = {
        "chunking_strategy": "fixed_size",
        "retrieval_config": "dense",
        "rows": rows("q1", ["b", "a"]),
        "qrels": qrels,
    }

    report = analyze_experiments([run, other], k=3)

    assert report["num_comparisons"] == 1
    assert report["comparisons"][0]["comparison"] == "bm25 vs dense"


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class StubRunner:
    top_k = 5

    def __init__(self):
        self.calls = 0

    def load_retrievers(self, strategy):
        return {"strategy": strategy}

    def _retrieve(self, retrievers, config_name, query, top_k, overrides):
        self.calls += 1
        return []


def test_benchmark_excludes_warmup_and_reports_cold_start():
    runner = StubRunner()
    result = benchmark_configuration(
        runner,
        strategy="fixed_size",
        config_name="bm25",
        queries=["q1", "q2", "q3", "q4"],
        warmup=2,
        iterations=3,
    )

    assert result["status"] == "ok"
    assert result["warmup_iterations"] == 2
    assert result["measured_iterations"] == 3
    assert result["num_measurements"] == 12
    assert result["cold_first_call_ms"] >= 0
    # 1 cold call + 2 warm-up passes + 3 measured passes
    assert runner.calls == 1 + 2 * 4 + 3 * 4


def test_benchmark_without_queries_is_not_reported_as_measured():
    result = benchmark_configuration(StubRunner(), "fixed_size", "bm25", queries=[])
    assert result["status"] == "no_queries"


# ---------------------------------------------------------------------------
# RAG evaluation
# ---------------------------------------------------------------------------

def test_parse_citations_is_unique_and_ordered():
    assert parse_citations("First [1], then [2] and again [1].") == [1, 2]
    assert parse_citations("no citations") == []


def test_evaluate_citations_with_grades():
    context = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
    result = evaluate_citations("Uses [1] but cites a missing [3].", context, {"c1": 3, "c2": 0})

    assert result["valid_citation_rate"] == pytest.approx(0.5)
    assert result["invalid_citations"] == [3]
    assert result["citation_correctness"] == pytest.approx(1.0)
    assert result["citation_completeness"] == pytest.approx(1.0)


def test_evaluate_citations_without_grades_is_unknown():
    result = evaluate_citations("Only [1].", [{"chunk_id": "c1"}])
    assert result["citation_correctness"] is None
    assert result["citation_completeness"] is None


def test_build_answer_record_preserves_context():
    query = {"query_id": "q1", "query": "What?", "split": "dev"}
    ranked = [{"rank": 1, "chunk_id": "c1", "doc_id": "doc-1"}]
    chunks = {"c1": {"content": "text", "metadata": {"url": "https://example.test"}}}

    record = build_answer_record(query, ranked, chunks, answer="Because.", experiment_id="gen-1")

    assert record["retrieved_context"][0]["content"] == "text"
    assert record["retrieved_context"][0]["url"] == "https://example.test"
    assert record["answer"] == "Because."


def test_generation_without_credentials_is_not_executed(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = generate_answer("q", [], {}, provider="anthropic")

    assert result["status"] == "not_executed"
    assert result["answer"] is None
    assert "credentials" in result["reason"]


def test_human_answer_store_validates_and_summarizes(tmp_path):
    store = HumanAnswerEvalStore(tmp_path / "ratings.jsonl")
    store.record(
        answer_id="gen-1:q1",
        reviewer_id="reviewer-1",
        scores={"faithfulness": 4, "answer_relevance": 5},
    )

    with pytest.raises(ValueError):
        store.record(answer_id="gen-1:q1", reviewer_id="", scores={"faithfulness": 4})
    with pytest.raises(ValueError):
        store.record(answer_id="gen-1:q1", reviewer_id="r", scores={"faithfulness": 9})

    summary = store.summarize(["gen-1:q1", "gen-1:q2"])
    assert summary["num_human_rated"] == 1
    assert summary["faithfulness"] == 4


def test_evaluate_answers_without_human_ratings(tmp_path):
    store = HumanAnswerEvalStore(tmp_path / "ratings.jsonl")
    records = [
        {
            "answer_id": "gen-1:q1",
            "query_id": "q1",
            "answer": "Cites [1].",
            "retrieved_context": [{"chunk_id": "c1"}],
        }
    ]

    report = evaluate_answers(records, {"q1": {"c1": 2}}, store)

    assert report["num_generated"] == 1
    assert report["citation_correctness_mean"] == pytest.approx(1.0)
    assert report["answer_quality"]["status"] == "not_evaluated"
    assert report["answer_quality"]["reason"]

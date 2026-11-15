"""Tests for candidate query generation, pooling, splits and validation."""

from collections import Counter

import pytest

from src.evaluation.eval_dataset import (
    CATEGORIES,
    STRATEGIES,
    assign_splits,
    attach_system_coverage,
    build_candidate_pools,
    generate_candidate_queries,
    validate_dataset,
)
from src.ingestion.chunker import Chunk
from src.ingestion.crawler import Document

CORPUS_VERSION = "corpus-test123456"

DOCS = [
    Document(
        doc_id="doc-payments",
        url="https://example.test/payments",
        title="Payments",
        section="Payments",
        content=(
            "# Payments\n\n"
            "A PaymentIntent tracks the lifecycle of a customer payment. "
            "Use an idempotency key to retry safely."
        ),
    ),
    Document(
        doc_id="doc-webhooks",
        url="https://example.test/webhooks",
        title="Webhooks",
        section="Webhooks",
        content=(
            "# Webhooks\n\n"
            "Webhook signatures are verified with a signing secret.\n\n"
            "## Retries\n\nStripe retries failed webhook deliveries."
        ),
    ),
    Document(
        doc_id="doc-setup",
        url="https://example.test/setup",
        title="Setup intents",
        section="Payments",
        content=(
            "# Setup intents\n\n"
            "A SetupIntent saves a payment method for future payments."
        ),
    ),
]


def make_chunk(chunk_id: str, doc_id: str, content: str, strategy: str, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        chunk_index=index,
        strategy=strategy,
        start_char=0,
        end_char=len(content),
        token_count=len(content.split()),
        metadata={
            "url": f"https://example.test/{doc_id}",
            "title": f"Title {doc_id}",
            "section": "Testing",
        },
        heading_context="Payments" if "Payment" in content else "",
        heading_path=["# Payments"] if "Payment" in content else [],
    )


CHUNKS = {
    "fixed_size": [
        make_chunk("f1", "doc-payments", "A PaymentIntent tracks a payment lifecycle.", "fixed_size"),
        make_chunk("f2", "doc-payments", "Use an idempotency key to retry safely.", "fixed_size"),
        make_chunk("f3", "doc-webhooks", "Webhook signatures are verified with a secret.", "fixed_size"),
        make_chunk("f4", "doc-setup", "A SetupIntent saves a payment method.", "fixed_size"),
    ],
    "structure_aware": [
        make_chunk("s1", "doc-payments", "A PaymentIntent tracks a payment lifecycle. Use an idempotency key.", "structure_aware"),
        make_chunk("s2", "doc-webhooks", "Webhook signatures are verified with a signing secret.", "structure_aware"),
        make_chunk("s3", "doc-setup", "A SetupIntent saves a payment method for future payments.", "structure_aware"),
    ],
}


class StubRetriever:
    def __init__(self, results_by_query=None, default=None):
        self.results_by_query = results_by_query or {}
        self.default = default or []
        self.calls = []

    def retrieve(self, query, top_k=10):
        self.calls.append((query, top_k))
        results = self.results_by_query.get(query, self.default)
        return results[:top_k]


class StubReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, results, top_k=5):
        self.calls.append((query, len(results), top_k))
        return list(reversed(results))[:top_k]


def result(chunk_id: str, rank: int, score: float = 1.0) -> dict:
    return {"chunk_id": chunk_id, "rank": rank, "score": score}


def retrievers():
    fixed = {
        "bm25": StubRetriever(default=[result("f1", 1), result("f2", 2)]),
        "dense": StubRetriever(default=[result("f2", 1), result("f1", 2)]),
        "hybrid": StubRetriever(default=[result("f1", 1), result("f2", 2)]),
        "reranker": StubReranker(),
    }
    structure = {
        "bm25": StubRetriever(default=[result("s1", 1), result("s2", 2)]),
        "dense": StubRetriever(default=[result("s2", 1), result("s1", 2)]),
        "hybrid": StubRetriever(default=[result("s1", 1), result("s2", 2)]),
        "reranker": StubReranker(),
    }
    return {"fixed_size": fixed, "structure_aware": structure}


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def test_generated_queries_are_grounded_and_marked_candidate():
    queries, skipped = generate_candidate_queries(
        DOCS, CHUNKS["structure_aware"], CORPUS_VERSION
    )

    assert queries, "expected at least some grounded queries"
    for query in queries:
        assert query["validation_status"] == "candidate"
        assert query["corpus_version"] == CORPUS_VERSION
        assert query["category"] in CATEGORIES
        assert query["source_doc_ids"]
        assert query["source_passage"]
        assert query["query_id"].startswith("q_")
        assert query["split"] is None
        assert query["generation_method"] == "rule_blueprint+corpus_grounding"


def test_generation_skips_ungrounded_blueprints():
    _, skipped = generate_candidate_queries(DOCS, CHUNKS["structure_aware"], CORPUS_VERSION)

    reasons = {entry["reason"] for entry in skipped}
    assert "not_grounded_in_corpus" in reasons


def test_generated_queries_are_unique():
    queries, _ = generate_candidate_queries(DOCS, CHUNKS["structure_aware"], CORPUS_VERSION)

    texts = [q["query"].lower() for q in queries]
    assert len(texts) == len(set(texts))


def test_query_ids_are_sequential_and_unique():
    queries, _ = generate_candidate_queries(DOCS, CHUNKS["structure_aware"], CORPUS_VERSION)

    ids = [q["query_id"] for q in queries]
    assert ids == [f"q_{i:03d}" for i in range(1, len(queries) + 1)]


# ---------------------------------------------------------------------------
# Candidate pooling
# ---------------------------------------------------------------------------

def test_pooling_deduplicates_and_hides_systems():
    queries, _ = generate_candidate_queries(DOCS, CHUNKS["structure_aware"], CORPUS_VERSION)
    candidates, provenance = build_candidate_pools(
        queries[:2], retrievers(), CHUNKS, CORPUS_VERSION, depth=3
    )

    assert candidates
    assert len({c["candidate_id"] for c in candidates}) == len(candidates)
    for candidate in candidates:
        assert "bm25" not in candidate
        assert "dense" not in candidate
        assert "rank" not in candidate
        assert "system" not in candidate
        assert candidate["chunking_strategy"] in STRATEGIES
        assert candidate["content"]
        assert candidate["doc_id"] == next(
            c.doc_id for c in CHUNKS[candidate["chunking_strategy"]]
            if c.chunk_id == candidate["chunk_id"]
        )

    # Both systems contributed to f1, so provenance has two entries for it.
    f1 = next(p for p in provenance if p["chunk_id"] == "f1" and p["chunking_strategy"] == "fixed_size")
    systems = {source["system"] for source in f1["systems"]}
    assert {"bm25", "dense", "hybrid"} <= systems
    assert systems <= {"bm25", "dense", "hybrid", "hybrid_rerank"}
    assert f1["best_rank"] == 1


def test_pooling_keeps_strategies_separate():
    queries, _ = generate_candidate_queries(DOCS, CHUNKS["structure_aware"], CORPUS_VERSION)
    candidates, _ = build_candidate_pools(queries[:1], retrievers(), CHUNKS, CORPUS_VERSION)

    for strategy in STRATEGIES:
        ids = {c["chunk_id"] for c in candidates if c["chunking_strategy"] == strategy}
        other = {c.chunk_id for s, chunks in CHUNKS.items() if s != strategy for c in chunks}
        assert not ids & other


def test_pooling_never_pools_unknown_chunks():
    queries, _ = generate_candidate_queries(DOCS, CHUNKS["structure_aware"], CORPUS_VERSION)
    bad = retrievers()
    bad["fixed_size"]["bm25"] = StubRetriever(default=[result("does-not-exist", 1)])

    candidates, _ = build_candidate_pools(queries[:1], bad, CHUNKS, CORPUS_VERSION)

    assert all(c["chunk_id"] != "does-not-exist" for c in candidates)


def test_pooling_reports_system_coverage():
    queries, _ = generate_candidate_queries(DOCS, CHUNKS["structure_aware"], CORPUS_VERSION)
    candidates, provenance = build_candidate_pools(
        queries[:1], retrievers(), CHUNKS, CORPUS_VERSION
    )

    report = attach_system_coverage({"num_candidates": len(candidates)}, provenance)

    assert set(report["system_coverage"]) == {"bm25", "dense", "hybrid", "hybrid_rerank"}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_splits_are_reproducible_and_stratified():
    queries, _ = generate_candidate_queries(DOCS, CHUNKS["structure_aware"], CORPUS_VERSION)
    if len(queries) < 8:
        pytest.skip("not enough grounded queries in the synthetic fixture")

    first_version = assign_splits(queries)
    first_assignment = [q["split"] for q in queries]

    second_version = assign_splits(queries)
    second_assignment = [q["split"] for q in queries]

    assert first_version == second_version
    assert first_assignment == second_assignment
    assert set(first_assignment) == {"dev", "test"}

    test_fraction = first_assignment.count("test") / len(first_assignment)
    assert 0.1 <= test_fraction <= 0.4


def test_split_does_not_move_queries_when_recomputed():
    queries, _ = generate_candidate_queries(DOCS, CHUNKS["structure_aware"], CORPUS_VERSION)
    assign_splits(queries)
    snapshot = {q["query_id"]: q["split"] for q in queries}

    assign_splits(queries)

    assert {q["query_id"]: q["split"] for q in queries} == snapshot


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def clean_dataset():
    queries, _ = generate_candidate_queries(DOCS, CHUNKS["structure_aware"], CORPUS_VERSION)
    assign_splits(queries)
    candidates, provenance = build_candidate_pools(
        queries, retrievers(), CHUNKS, CORPUS_VERSION
    )
    manifest = {"corpus_version": CORPUS_VERSION}
    return queries, candidates, provenance, manifest


def test_clean_dataset_validates():
    queries, candidates, provenance, manifest = clean_dataset()

    report = validate_dataset(queries, candidates, [], CHUNKS, manifest)

    assert report["problems"] == []
    assert report["is_structurally_valid"] is True
    assert report["judged_candidates"] == 0
    assert report["pending_candidates"] == len(candidates)
    assert report["is_fully_judged"] is False


def test_duplicate_query_ids_are_detected():
    queries, candidates, _, manifest = clean_dataset()
    queries.append(dict(queries[0]))

    report = validate_dataset(queries, candidates, [], CHUNKS, manifest)

    assert any("duplicate query ids" in problem for problem in report["problems"])


def test_invalid_grade_and_state_are_detected():
    queries, candidates, _, manifest = clean_dataset()
    candidate = candidates[0]
    bad = [
        {
            "query_id": candidate["query_id"],
            "candidate_id": candidate["candidate_id"],
            "chunk_id": candidate["chunk_id"],
            "doc_id": candidate["doc_id"],
            "chunking_strategy": candidate["chunking_strategy"],
            "relevance_grade": 7,
            "judgment_status": "maybe",
        }
    ]

    report = validate_dataset(queries, candidates, bad, CHUNKS, manifest)

    assert any("invalid relevance grade" in problem for problem in report["problems"])
    assert any("invalid state" in problem for problem in report["problems"])


def test_corpus_version_mismatch_is_detected():
    queries, candidates, _, _ = clean_dataset()
    queries[0]["corpus_version"] = "corpus-other"

    report = validate_dataset(
        queries, candidates, [], CHUNKS, {"corpus_version": CORPUS_VERSION}
    )

    assert any("corpus version" in problem for problem in report["problems"])


def test_missing_chunk_is_detected():
    queries, candidates, _, manifest = clean_dataset()
    broken = dict(candidates[0])
    broken["candidate_id"] = "cand_broken"
    broken["chunk_id"] = "missing-chunk"

    report = validate_dataset(queries, candidates + [broken], [], CHUNKS, manifest)

    assert any("not found" in problem for problem in report["problems"])


def test_pending_candidates_are_not_counted_as_judged():
    queries, candidates, _, manifest = clean_dataset()
    candidate = candidates[0]
    pending = [
        {
            "query_id": candidate["query_id"],
            "candidate_id": candidate["candidate_id"],
            "chunk_id": candidate["chunk_id"],
            "doc_id": candidate["doc_id"],
            "chunking_strategy": candidate["chunking_strategy"],
            "relevance_grade": 0,
            "judgment_status": "pending",
        }
    ]

    report = validate_dataset(queries, candidates, pending, CHUNKS, manifest)

    assert report["judged_candidates"] == 0
    assert report["pending_candidates"] == len(candidates)
    assert report["grade_distribution"]["0"] == 0

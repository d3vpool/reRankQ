"""Tests for query review, approval and pool-staleness handling."""

import json

import pytest

from src.evaluation.query_review import (
    ReviewError,
    ReviewStore,
    approved_dataset_version,
    build_approved_dataset,
    candidate_pool_status,
    effective_queries,
    next_unreviewed_query,
    query_quality_warnings,
    query_review_report,
    replace_query_pools,
    review_progress,
    review_validation_problems,
    validate_approved_dataset,
    validate_review_record,
)
from src.ingestion.chunker import Chunk
from src.ingestion.crawler import Document

DOCUMENTS = [
    Document(
        doc_id="doc-1",
        url="https://example.test/payments",
        title="Payment intents",
        section="Payments",
        content=(
            "A PaymentIntent tracks a customer payment. Stripe recommends idempotency "
            "keys for retrying failed requests. Webhook signatures are verified with a "
            "signing secret."
        ),
    ),
    Document(
        doc_id="doc-2",
        url="https://example.test/webhooks",
        title="Webhooks",
        section="Webhooks",
        content="Webhook deliveries are retried when the endpoint does not respond.",
    ),
]


def make_query(
    qid="q_001",
    text="What is a PaymentIntent?",
    category="terminology",
    doc="doc-1",
    section="Payment intents",
    passage="A PaymentIntent tracks a customer payment.",
    split="dev",
    chunk_id="chunk-1",
):
    return {
        "query_id": qid,
        "query": text,
        "category": category,
        "generation_method": "test",
        "source_doc_ids": [doc],
        "source_section": section,
        "source_passage": passage,
        "source_chunk_id": chunk_id,
        "corpus_version": "corpus-test",
        "validation_status": "candidate",
        "created_at": "2026-01-01T00:00:00+00:00",
        "split": split,
    }


def accept(store, query, note="looks good", query_text=None, category=None, reviewer="reviewer-1"):
    return store.record(
        query=query,
        review_status="accepted",
        reviewer_id=reviewer,
        review_note=note,
        query_text=query_text,
        category=category,
    )


# ---------------------------------------------------------------------------
# Decisions, persistence, revisions
# ---------------------------------------------------------------------------

def test_accept_records_review_metadata(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    query = make_query()

    record = accept(store, query)

    assert record["review_status"] == "accepted"
    assert record["reviewer_id"] == "reviewer-1"
    assert record["review_timestamp"]
    assert record["original_query"] == query["query"]
    assert record["current_query"] == query["query"]
    assert record["source_doc_ids"] == ["doc-1"]
    assert record["corpus_version"] == "corpus-test"
    assert record["revision_history"] == []
    validate_review_record(record)


@pytest.mark.parametrize("status", ["rejected", "needs_revision"])
def test_other_decisions_persist(tmp_path, status):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    query = make_query()

    store.record(query=query, review_status=status, reviewer_id="reviewer-1", review_note="no")

    assert store.status(query["query_id"]) == status


def test_editing_preserves_original_and_records_history(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    query = make_query()

    record = accept(store, query, query_text="How does a PaymentIntent track a payment?")

    assert record["original_query"] == "What is a PaymentIntent?"
    assert record["current_query"] == "How does a PaymentIntent track a payment?"
    assert len(record["revision_history"]) == 1
    revision = record["revision_history"][0]
    assert revision["from_query"] == "What is a PaymentIntent?"
    assert revision["to_query"] == "How does a PaymentIntent track a payment?"
    assert revision["reviewer_id"] == "reviewer-1"


def test_multiple_reviews_accumulate_revision_history(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    query = make_query()

    accept(store, query, query_text="First wording?")
    record = accept(store, query, query_text="Second wording?", category="conceptual")

    assert record["original_query"] == "What is a PaymentIntent?"
    assert record["current_query"] == "Second wording?"
    assert record["current_category"] == "conceptual"
    assert len(record["revision_history"]) == 2


def test_reviews_resume_across_store_instances(tmp_path):
    path = tmp_path / "reviews.jsonl"
    store = ReviewStore(path)
    accept(store, make_query(), query_text="Edited wording?")

    reopened = ReviewStore(path)

    assert reopened.status("q_001") == "accepted"
    assert reopened.get("q_001")["current_query"] == "Edited wording?"
    assert len(reopened.records()) == 1


def test_review_append_does_not_destroy_earlier_decisions(tmp_path):
    path = tmp_path / "reviews.jsonl"
    store = ReviewStore(path)
    accept(store, make_query())
    store.record(query=make_query(), review_status="needs_revision", reviewer_id="reviewer-1")

    lines = [line for line in path.read_text().splitlines() if line.strip()]

    assert len(lines) == 2
    assert store.status("q_001") == "needs_revision"


def test_review_progress(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    queries = [make_query("q_001"), make_query("q_002", text="What is a SetupIntent?"), make_query("q_003")]
    accept(store, queries[0])
    store.record(query=queries[1], review_status="rejected", reviewer_id="reviewer-1", review_note="vague")

    progress = review_progress(queries, store)

    assert progress["by_status"] == {
        "candidate": 1,
        "accepted": 1,
        "rejected": 1,
        "needs_revision": 0,
    }
    assert progress["reviewed"] == 2
    assert progress["accepted"] == 1
    assert progress["is_fully_reviewed"] is False


def test_next_unreviewed_query(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    queries = [make_query("q_001"), make_query("q_002")]
    accept(store, queries[0])

    assert next_unreviewed_query(queries, store) == "q_002"


def test_effective_queries_apply_current_wording(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    query = make_query()
    accept(store, query, query_text="Edited?", category="conceptual")

    merged = effective_queries([query], store)[0]

    assert merged["query"] == "Edited?"
    assert merged["category"] == "conceptual"
    assert merged["validation_status"] == "accepted"
    assert merged["original_query"] == query["query"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"review_status": "maybe", "reviewer_id": "reviewer-1"},
        {"review_status": "accepted", "reviewer_id": "   "},
    ],
)
def test_invalid_review_metadata_is_rejected(tmp_path, kwargs):
    store = ReviewStore(tmp_path / "reviews.jsonl")

    with pytest.raises(ReviewError):
        store.record(query=make_query(), **kwargs)

    assert store.records() == []


def test_empty_wording_and_bad_category_are_rejected(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")

    with pytest.raises(ReviewError):
        accept(store, make_query(), query_text="   ")
    with pytest.raises(ReviewError):
        accept(store, make_query(), category="not_a_category")


def test_edit_without_revision_history_is_invalid():
    record = {
        "query_id": "q_001",
        "original_query": "original?",
        "current_query": "changed?",
        "original_category": "terminology",
        "current_category": "terminology",
        "source_doc_ids": ["doc-1"],
        "source_section": "section",
        "corpus_version": "corpus-test",
        "review_status": "accepted",
        "reviewer_id": "reviewer-1",
        "review_timestamp": "2026-01-01T00:00:00+00:00",
        "revision_history": [],
    }

    with pytest.raises(ReviewError):
        validate_review_record(record)


def test_accepted_without_reviewer_is_reported():
    record = {
        "query_id": "q_001",
        "original_query": "original?",
        "current_query": "original?",
        "original_category": "terminology",
        "current_category": "terminology",
        "source_doc_ids": ["doc-1"],
        "source_section": "section",
        "corpus_version": "corpus-test",
        "review_status": "accepted",
        "reviewer_id": "",
        "review_timestamp": "2026-01-01T00:00:00+00:00",
        "revision_history": [],
    }

    with pytest.raises(ReviewError):
        validate_review_record(record)


def test_review_of_unknown_query_is_reported(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    accept(store, make_query("q_999"))

    problems = review_validation_problems(store.records(), [make_query("q_001")])

    assert any("unknown query" in problem for problem in problems)


def test_missing_note_on_rejection_is_reported(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    store.record(query=make_query(), review_status="rejected", reviewer_id="reviewer-1", review_note="")

    problems = review_validation_problems(store.records(), [make_query()])

    assert any("no review note" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Approved dataset
# ---------------------------------------------------------------------------

def test_approved_dataset_contains_only_accepted(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    queries = [make_query("q_001"), make_query("q_002", text="What is a SetupIntent?"), make_query("q_003")]
    accept(store, queries[0])
    store.record(query=queries[1], review_status="rejected", reviewer_id="reviewer-1", review_note="vague")

    approved, meta = build_approved_dataset(queries, store)

    assert [row["query_id"] for row in approved] == ["q_001"]
    assert meta["num_queries"] == 1
    assert meta["by_split"] == {"dev": 1, "test": 0}
    assert approved[0]["review"]["status"] == "accepted"
    assert meta["stale_pool_count"] == 0


def test_approved_dataset_version_is_stable_and_changes_on_edit(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    queries = [make_query()]
    baseline = pool_baseline(queries)
    accept(store, queries[0])

    first, meta_first = build_approved_dataset(
        queries, store, pool_status=candidate_pool_status(queries, baseline, store)
    )
    second, meta_second = build_approved_dataset(
        queries, store, pool_status=candidate_pool_status(queries, baseline, store)
    )
    assert first[0]["pool"]["is_stale"] is False

    accept(store, queries[0], query_text="Reworded question about PaymentIntents?")
    third, meta_third = build_approved_dataset(
        queries, store, pool_status=candidate_pool_status(queries, baseline, store)
    )

    assert meta_first["approved_version"] == meta_second["approved_version"] == approved_dataset_version(first)
    assert meta_third["approved_version"] != meta_first["approved_version"]
    assert third[0]["query"] == "Reworded question about PaymentIntents?"
    assert third[0]["pool"]["is_stale"] is True


def test_approved_version_changes_when_acceptance_changes(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    queries = [make_query("q_001"), make_query("q_002")]
    accept(store, queries[0])
    _, before = build_approved_dataset(queries, store)

    accept(store, queries[1])
    after, meta = build_approved_dataset(queries, store)

    assert meta["approved_version"] != before["approved_version"]
    assert len(after) == 2


def test_approved_dataset_validation_catches_bad_rows(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    queries = [make_query()]
    accept(store, queries[0])
    approved, _ = build_approved_dataset(queries, store)

    broken = dict(approved[0])
    broken["split"] = None
    broken["review"] = {"status": "rejected", "reviewer_id": "", "review_timestamp": ""}

    problems = validate_approved_dataset([broken], queries)

    assert any("invalid split" in problem for problem in problems)
    assert any("not marked accepted" in problem for problem in problems)
    assert any("missing reviewer_id" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Quality warnings
# ---------------------------------------------------------------------------

def kinds_for(queries, **kwargs):
    warnings = query_quality_warnings(queries, **kwargs)
    return {warning["kind"] for warning in warnings}, warnings


def test_short_query_is_flagged():
    kinds, _ = kinds_for([make_query(text="PaymentIntent?")])
    assert "short_query" in kinds


def test_heading_copy_is_flagged():
    kinds, _ = kinds_for([make_query(text="Payment intents", section="Payment intents")])
    assert "heading_copy" in kinds


def test_answer_verbatim_is_flagged():
    kinds, _ = kinds_for(
        [
            make_query(
                text="Stripe recommends idempotency keys for retrying failed requests",
                passage="Stripe recommends idempotency keys for retrying failed requests.",
            )
        ]
    )
    assert "answer_verbatim" in kinds


def test_duplicate_and_near_duplicate_are_flagged():
    queries = [
        make_query("q_001", text="What is a PaymentMethod object?"),
        make_query("q_002", text="What is a PaymentMethod object?"),
        make_query("q_003", text="How do I retrieve a PaymentMethod object?"),
    ]

    kinds, warnings = kinds_for(queries, near_duplicate_threshold=0.6)

    assert "duplicate_query" in kinds
    assert "near_duplicate" in kinds
    assert {warning["query_id"] for warning in warnings if warning["kind"] == "duplicate_query"} == {
        "q_001",
        "q_002",
    }


def test_missing_provenance_is_flagged():
    query = make_query()
    query["source_doc_ids"] = []
    query["source_section"] = ""

    kinds, _ = kinds_for([query])

    assert "missing_source_docs" in kinds
    assert "missing_source_section" in kinds


def test_invalid_source_chunk_is_flagged():
    chunks = {
        "fixed_size": [
            Chunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                content="text",
                chunk_index=0,
                strategy="fixed_size",
                start_char=0,
                end_char=4,
                token_count=1,
                metadata={},
            )
        ],
        "structure_aware": [],
    }

    kinds, _ = kinds_for([make_query(chunk_id="missing-chunk")], chunks_by_strategy=chunks)

    assert "unknown_source_chunk" in kinds


def test_possibly_unanswerable_is_flagged():
    kinds, _ = kinds_for(
        [make_query(text="What is quantum flux capacitance?")], documents=DOCUMENTS
    )

    assert "possibly_unanswerable" in kinds


def test_warnings_never_reject_queries(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    query = make_query(text="PaymentIntent?")

    warnings = query_quality_warnings([query])

    assert warnings
    assert store.status("q_001") == "candidate"


def test_report_distinguishes_structural_validity_from_approval(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    queries = [make_query("q_001", text="What is a PaymentIntent?"), make_query("q_002")]
    accept(store, queries[0])

    report = query_review_report(queries, store.records(), DOCUMENTS)

    assert report["is_structurally_valid"] is True
    assert report["human_approved"] == 1
    assert report["by_status"]["candidate"] == 1
    assert report["is_fully_reviewed"] is False
    assert "warnings_by_kind" in report


def test_corpus_version_mismatch_is_reported(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    queries = [make_query()]
    accept(store, queries[0])

    report = query_review_report(queries, store.records(), DOCUMENTS, manifest={"corpus_version": "corpus-other"})

    assert any("corpus version" in problem for problem in report["structural_problems"])


# ---------------------------------------------------------------------------
# Candidate-pool staleness
# ---------------------------------------------------------------------------

def pool_baseline(queries):
    return {"wording": {query["query_id"]: query["query"] for query in queries}}


def test_pool_status_is_current_when_wording_matches(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    queries = [make_query()]

    status = candidate_pool_status(queries, pool_baseline(queries), store)

    assert status["counts"] == {"current": 1, "stale": 0, "missing_baseline": 0}
    assert status["is_consistent"] is True


def test_edited_wording_marks_pool_stale(tmp_path):
    store = ReviewStore(tmp_path / "reviews.jsonl")
    queries = [make_query()]
    baseline = pool_baseline(queries)
    accept(store, queries[0], query_text="A completely different question about setup intents?")

    status = candidate_pool_status(queries, baseline, store)

    assert status["stale"] == ["q_001"]
    assert status["is_consistent"] is False


def test_missing_baseline_is_reported():
    queries = [make_query()]

    status = candidate_pool_status(queries, None)

    assert status["missing_baseline"] == ["q_001"]


def test_replace_query_pools_keeps_other_queries():
    candidates = [
        {"query_id": "q_001", "chunking_strategy": "fixed_size", "candidate_id": "c1"},
        {"query_id": "q_002", "chunking_strategy": "fixed_size", "candidate_id": "c2"},
    ]
    provenance = [
        {"query_id": "q_001", "chunking_strategy": "fixed_size", "candidate_id": "c1"},
        {"query_id": "q_002", "chunking_strategy": "fixed_size", "candidate_id": "c2"},
    ]

    new_candidates, new_provenance = replace_query_pools(
        candidates,
        provenance,
        "q_001",
        [{"query_id": "q_001", "chunking_strategy": "fixed_size", "candidate_id": "c9"}],
        [{"query_id": "q_001", "chunking_strategy": "fixed_size", "candidate_id": "c9"}],
    )

    assert [row["candidate_id"] for row in new_candidates] == ["c9", "c2"]
    assert [row["candidate_id"] for row in new_provenance] == ["c9", "c2"]


def test_approval_workflow_does_not_touch_judgments_or_manifest(tmp_path):
    judgments = tmp_path / "judgments.jsonl"
    manifest = tmp_path / "corpus_manifest.json"
    judgments.write_text(json.dumps({"candidate_id": "cand_1", "relevance_grade": 2}) + "\n")
    manifest.write_text(json.dumps({"corpus_version": "corpus-test"}) + "\n")
    judgments_before = judgments.read_bytes()
    manifest_before = manifest.read_bytes()

    store = ReviewStore(tmp_path / "reviews.jsonl")
    queries = [make_query()]
    accept(store, queries[0], query_text="Edited question?")
    build_approved_dataset(queries, store, pool_status=candidate_pool_status(queries, None, store))

    assert judgments.read_bytes() == judgments_before
    assert manifest.read_bytes() == manifest_before

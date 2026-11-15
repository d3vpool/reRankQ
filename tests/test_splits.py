"""Tests for leakage auditing and group-aware splits."""

import pytest

from src.evaluation.splits import (
    apply_split_assignment,
    assign_group_splits,
    audit_split_leakage,
    build_query_groups,
    near_duplicate_pairs,
    normalise_text,
    split_version,
)

CATEGORIES = ("terminology", "conceptual", "api_usage", "procedural")


def make_query(qid, query, doc_ids, section, category="terminology", split=None):
    return {
        "query_id": qid,
        "query": query,
        "category": category,
        "generation_method": "test",
        "source_doc_ids": list(doc_ids),
        "source_section": section,
        "source_passage": "Some grounded passage about the topic.",
        "source_chunk_id": f"chunk-{qid}",
        "corpus_version": "corpus-test",
        "validation_status": "candidate",
        "created_at": "2026-01-01T00:00:00+00:00",
        "split": split,
    }


TOPICS = [
    "payment intent lifecycle",
    "setup intent attributes",
    "idempotency keys",
    "asynchronous capture",
    "payment method objects",
    "checkout session creation",
    "strong customer authentication",
    "manual capture timings",
    "payment record linkage",
    "webhook signature verification",
    "webhook retry backoff",
    "refund creation flow",
    "dispute evidence upload",
    "balance transaction reports",
    "customer object retrieval",
    "invoice item proration",
]


def general_fixture():
    """Four documents, four queries each, distinct topics and mixed categories."""
    queries = []
    docs = ["doc-a", "doc-b", "doc-c", "doc-d"]
    for doc_index, doc in enumerate(docs):
        for offset in range(4):
            position = doc_index * 4 + offset
            queries.append(
                make_query(
                    f"q_{position + 1:03d}",
                    f"How does {TOPICS[position]} work?",
                    [doc],
                    f"Section {doc}",
                    CATEGORIES[(doc_index + offset) % len(CATEGORIES)],
                )
            )
    return queries


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def test_shared_documents_are_grouped():
    queries = [
        make_query("q_001", "What is a PaymentIntent?", ["doc-a"], "Payment intents"),
        make_query("q_002", "How is capture handled?", ["doc-a"], "Capture"),
        make_query("q_003", "What is an idempotency key?", ["doc-b"], "Idempotency"),
    ]

    groups = build_query_groups(queries)

    by_qid = {qid: group["group_id"] for group in groups for qid in group["query_ids"]}
    assert by_qid["q_001"] == by_qid["q_002"]
    assert by_qid["q_003"] != by_qid["q_001"]


def test_shared_sections_are_grouped_across_documents():
    queries = [
        make_query("q_001", "What is a PaymentIntent?", ["doc-a"], "Core concepts"),
        make_query("q_002", "What is a SetupIntent?", ["doc-b"], "Core concepts"),
    ]

    groups = build_query_groups(queries)

    assert len(groups) == 1
    assert groups[0]["doc_ids"] == ["doc-a", "doc-b"]


def test_near_duplicate_wording_is_grouped():
    queries = [
        make_query("q_001", "How do I retrieve a PaymentMethod object?", ["doc-a"], "Methods"),
        make_query("q_002", "How do I retrieve a PaymentMethod object by id?", ["doc-a"], "Methods"),
    ]

    groups = build_query_groups(queries, threshold=0.8)

    assert len(groups) == 1


def test_groups_partition_all_queries():
    queries = general_fixture()

    groups = build_query_groups(queries)

    assigned = [qid for group in groups for qid in group["query_ids"]]
    assert sorted(assigned) == sorted(query["query_id"] for query in queries)
    assert len(groups) == 4
    assert all(group["size"] == 4 for group in groups)


def test_multiple_source_documents_merge_groups():
    queries = [
        make_query("q_001", "How do intents and webhooks relate?", ["doc-a", "doc-b"], "Intro"),
        make_query("q_002", "What is a webhook?", ["doc-b"], "Webhooks"),
        make_query("q_003", "What is a PaymentIntent?", ["doc-a"], "Intents"),
        make_query("q_004", "Unrelated question about refunds", ["doc-c"], "Refunds"),
    ]

    groups = build_query_groups(queries)
    by_qid = {qid: group["group_id"] for group in groups for qid in group["query_ids"]}

    assert by_qid["q_001"] == by_qid["q_002"] == by_qid["q_003"]
    assert by_qid["q_004"] != by_qid["q_001"]


def test_near_duplicate_pairs_report_similarity():
    queries = [
        make_query("q_001", "What is a PaymentMethod object?", ["doc-a"], "Methods"),
        make_query("q_002", "How do I retrieve a PaymentMethod object?", ["doc-b"], "Methods"),
        make_query("q_003", "Why would a webhook delivery be retried?", ["doc-c"], "Webhooks"),
    ]

    pairs = near_duplicate_pairs(queries, threshold=0.6)

    assert [pair["query_ids"] for pair in pairs] == [["q_001", "q_002"]]
    assert 0.6 <= pairs[0]["similarity"] < 1.0


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def test_audit_detects_document_leakage():
    queries = general_fixture()
    for index, query in enumerate(queries):
        query["split"] = "test" if index % 4 == 0 else "dev"

    report = audit_split_leakage(queries)

    assert report["leakage_found"] is True
    assert report["shared_documents"]
    assert report["groups_spanning_splits"]
    shared = {entry["doc_id"] for entry in report["shared_documents"]}
    assert shared == {"doc-a", "doc-b", "doc-c", "doc-d"}


def test_audit_reports_no_leakage_for_isolated_groups():
    queries = general_fixture()
    for index, query in enumerate(queries):
        query["split"] = "test" if index < 4 else "dev"

    report = audit_split_leakage(queries)

    assert report["leakage_found"] is False
    assert report["shared_documents"] == []
    assert report["groups_spanning_splits"] == []


def test_audit_detects_duplicate_and_near_duplicate_across_splits():
    queries = [
        make_query("q_001", "What is a PaymentMethod object?", ["doc-a"], "Core concepts", split="dev"),
        make_query("q_002", "What is a PaymentMethod object?", ["doc-b"], "Methods", split="test"),
        make_query("q_003", "How do I retrieve a PaymentMethod object?", ["doc-c"], "Methods", split="test"),
        make_query("q_004", "Why would a webhook delivery be retried?", ["doc-d"], "Webhooks", split="dev"),
    ]

    report = audit_split_leakage(queries, threshold=0.6)

    assert report["exact_duplicates_across_splits"]
    assert report["near_duplicates_across_splits"]
    assert report["leakage_found"] is True


def test_audit_never_modifies_the_split():
    queries = general_fixture()
    for index, query in enumerate(queries):
        query["split"] = "test" if index % 3 == 0 else "dev"
    before = [query["split"] for query in queries]

    audit_split_leakage(queries)

    assert [query["split"] for query in queries] == before


# ---------------------------------------------------------------------------
# Group-aware assignment
# ---------------------------------------------------------------------------

def test_group_split_is_reproducible():
    first = general_fixture()
    second = general_fixture()

    meta_first = assign_group_splits(first)
    meta_second = assign_group_splits(second)

    assert meta_first["assignment"] == meta_second["assignment"]
    assert meta_first["split_version"] == meta_second["split_version"]


def test_group_split_keeps_groups_intact():
    queries = general_fixture()

    meta = assign_group_splits(queries)

    groups = build_query_groups(queries)
    for group in groups:
        splits = {query["split"] for query in queries if query["query_id"] in group["query_ids"]}
        assert len(splits) == 1


def test_group_split_leaves_no_leakage():
    queries = general_fixture()

    assign_group_splits(queries)

    report = audit_split_leakage(queries)
    assert report["leakage_found"] is False
    assert report["shared_documents"] == []
    assert report["groups_spanning_splits"] == []


def test_group_split_respects_target_window():
    queries = general_fixture()

    meta = assign_group_splits(queries)

    test_count = meta["by_split"]["test"]
    assert 0.2 * len(queries) <= test_count <= 0.3 * len(queries)
    assert set(meta["by_split"]) == {"dev", "test"}
    assert meta["split_method"] == "group-doc-section-neardup"
    assert meta["split_seed"] == 13


def test_group_split_metadata_is_complete():
    queries = general_fixture()

    meta = assign_group_splits(queries)

    assert meta["num_queries"] == len(queries)
    assert meta["num_groups"] == len(meta["groups"])
    assert meta["group_sizes"] == [group["size"] for group in meta["groups"]]
    for group in meta["groups"]:
        assert group["split"] in ("dev", "test")
    assert meta["split_version"].startswith("split-")
    assert all(meta["by_category_split"][split] for split in ("dev", "test"))


def test_group_split_raises_when_everything_is_one_group():
    queries = [
        make_query("q_001", "What is a PaymentIntent?", ["doc-a"], "Intents"),
        make_query("q_002", "How is capture handled?", ["doc-a"], "Intents"),
    ]

    with pytest.raises(ValueError):
        assign_group_splits(queries)


def test_group_split_handles_multi_document_queries():
    queries = [
        make_query("q_001", "Bridge question across docs", ["doc-a", "doc-b"], "Intro"),
        make_query("q_002", "What is a PaymentIntent?", ["doc-a"], "Intents"),
        make_query("q_003", "What is a webhook?", ["doc-b"], "Webhooks"),
        make_query("q_004", "Refund question", ["doc-c"], "Refunds"),
        make_query("q_005", "Dispute question", ["doc-d"], "Disputes"),
        make_query("q_006", "Balance question", ["doc-e"], "Balances"),
    ]

    meta = assign_group_splits(queries)

    bridged = {"q_001", "q_002", "q_003"}
    assert len({meta["assignment"][qid] for qid in bridged}) == 1


def test_split_version_changes_with_assignment():
    queries = general_fixture()
    groups = build_query_groups(queries)
    assignment = {query["query_id"]: "dev" for query in queries}
    other = dict(assignment)
    other["q_001"] = "test"

    assert split_version(assignment, groups, 13) != split_version(other, groups, 13)
    assert split_version(assignment, groups, 13) == split_version(dict(assignment), groups, 13)


def test_apply_split_assignment_requires_every_query():
    queries = general_fixture()

    with pytest.raises(KeyError):
        apply_split_assignment(queries, {"q_001": "dev"})


def test_normalise_text_strips_punctuation():
    assert normalise_text("What is a PaymentIntent?!") == "what is a paymentintent"

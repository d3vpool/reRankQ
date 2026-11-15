"""Tests for the append-only human judgment store."""

import csv
import json

import pytest

from src.evaluation.judgments import JudgmentStore, next_unjudged_query

QUERY = {"query_id": "q_001", "query": "What is a PaymentIntent?", "split": "dev"}


def candidate(cid="cand_a", query_id="q_001", strategy="fixed_size", chunk_id="c1"):
    return {
        "candidate_id": cid,
        "query_id": query_id,
        "chunking_strategy": strategy,
        "chunk_id": chunk_id,
        "doc_id": "doc-1",
        "content": "chunk body",
    }


def record(store, grade=2, cid="cand_a", status="judged", judge="judge-1", rationale=""):
    return store.record(
        query_id="q_001",
        candidate_id=cid,
        chunk_id="c1",
        doc_id="doc-1",
        chunking_strategy="fixed_size",
        relevance_grade=grade,
        judge_id=judge,
        rationale=rationale,
        judgment_status=status,
    )


def test_record_persists_and_resumes(tmp_path):
    path = tmp_path / "judgments.jsonl"
    store = JudgmentStore(path)
    record(store, grade=3)

    reopened = JudgmentStore(path)

    assert len(reopened.records()) == 1
    assert reopened.get("cand_a")["relevance_grade"] == 3
    assert reopened.judged_candidate_ids() == {"cand_a"}


def test_recording_again_supersedes_without_deleting(tmp_path):
    path = tmp_path / "judgments.jsonl"
    store = JudgmentStore(path)
    record(store, grade=1)
    record(store, grade=3, rationale="better on second look")

    lines = [line for line in path.read_text().splitlines() if line.strip()]

    assert len(lines) == 2
    assert store.get("cand_a")["relevance_grade"] == 3
    assert len(store.records()) == 2


@pytest.mark.parametrize("grade", [4, -1, "2", 2.5, True, None])
def test_invalid_grades_are_rejected(tmp_path, grade):
    store = JudgmentStore(tmp_path / "judgments.jsonl")

    with pytest.raises(ValueError):
        record(store, grade=grade)

    assert store.records() == []


def test_invalid_status_and_judge_are_rejected(tmp_path):
    store = JudgmentStore(tmp_path / "judgments.jsonl")

    with pytest.raises(ValueError):
        record(store, status="maybe")
    with pytest.raises(ValueError):
        record(store, judge="   ")

    assert store.records() == []


def test_pending_is_not_judged(tmp_path):
    store = JudgmentStore(tmp_path / "judgments.jsonl")
    record(store, grade=0, status="pending")

    progress = store.progress([candidate()])

    assert progress["judged"] == 0
    assert progress["pending"] == 1
    assert store.judged_candidate_ids() == set()


def test_progress_by_strategy_and_query(tmp_path):
    store = JudgmentStore(tmp_path / "judgments.jsonl")
    record(store, cid="cand_a")

    candidates = [
        candidate("cand_a"),
        candidate("cand_b", strategy="structure_aware", chunk_id="s1"),
        candidate("cand_c", query_id="q_002", chunk_id="c3"),
    ]

    progress = store.progress(candidates)

    assert progress["total"] == 3
    assert progress["judged"] == 1
    assert progress["pending"] == 2
    assert progress["by_strategy"]["fixed_size"]["judged"] == 1
    assert progress["by_strategy"]["structure_aware"]["pending"] == 1
    assert progress["by_query"]["q_001"] == {"total": 2, "judged": 1, "pending": 1}


def test_export_jsonl_and_csv(tmp_path):
    store = JudgmentStore(tmp_path / "judgments.jsonl")
    record(store, grade=2, rationale="useful")

    jsonl_path = store.export_jsonl(tmp_path / "exports" / "judgments.jsonl")
    csv_path = store.export_csv(tmp_path / "exports" / "judgments.csv")

    exported = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert exported[0]["relevance_grade"] == 2

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["candidate_id"] == "cand_a"
    assert rows[0]["relevance_grade"] == "2"
    assert rows[0]["rationale"] == "useful"


def test_next_unjudged_query_skips_completed_queries(tmp_path):
    store = JudgmentStore(tmp_path / "judgments.jsonl")
    record(store, cid="cand_a")

    queries = [QUERY, {"query_id": "q_002", "query": "other", "split": "dev"}]
    candidates = [
        candidate("cand_a"),
        candidate("cand_b", query_id="q_002", chunk_id="c2"),
    ]

    assert next_unjudged_query(queries, candidates, store) == "q_002"

    record(store, cid="cand_b")
    # record() writes chunk_id c1, but identity is the candidate id; q_002 is done.
    assert next_unjudged_query(queries, candidates, store) is None


def test_next_unjudged_query_respects_strategy_filter(tmp_path):
    store = JudgmentStore(tmp_path / "judgments.jsonl")
    candidates = [
        candidate("cand_fixed", strategy="fixed_size"),
        candidate("cand_struct", strategy="structure_aware", chunk_id="s1"),
    ]

    assert (
        next_unjudged_query([QUERY], candidates, store, strategy="structure_aware")
        == "q_001"
    )

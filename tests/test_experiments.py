"""Tests for experiment identity, environment capture and artifact safety."""

import json

import pytest

from src.evaluation.ablation_runner import ablation_specs, run_ablation
from src.evaluation.corpus_manifest import build_manifest
from src.evaluation.experiments import (
    EvaluationBlocked,
    build_specs,
    environment_snapshot,
    experiment_id,
    prepare_experiment_dir,
    verify_dataset,
    write_json,
    write_jsonl,
)
from src.evaluation.retrieval_runner import RetrievalRunner, summarize_latencies
from src.ingestion.crawler import Document


def make_doc(doc_id="doc-1", content="A PaymentIntent tracks a payment."):
    return Document(
        doc_id=doc_id,
        url=f"https://example.test/{doc_id}",
        title="Doc",
        section="Testing",
        content=content,
    )


def test_experiment_id_is_stable_and_sensitive():
    base = dict(
        corpus_version="corpus-1",
        query_version="approved-1",
        split_version="split-1",
        split="dev",
        chunking_strategy="fixed_size",
        retrieval_config="bm25",
        parameters={"top_k": 10},
    )
    assert experiment_id(**base) == experiment_id(**base)
    assert experiment_id(**base) != experiment_id(**{**base, "chunking_strategy": "structure_aware"})
    assert experiment_id(**base) != experiment_id(**{**base, "parameters": {"top_k": 5}})


def test_build_specs_covers_eight_configurations():
    specs = build_specs(
        corpus_version="corpus-1",
        query_version="approved-1",
        split_version="split-1",
        split="dev",
    )
    assert len(specs) == 8
    assert len({spec["experiment_id"] for spec in specs}) == 8
    assert {(s["chunking_strategy"], s["retrieval_config"]) for s in specs} == {
        (strategy, config)
        for strategy in ("fixed_size", "structure_aware")
        for config in ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")
    }


def test_environment_snapshot_records_only_observed_values():
    env = environment_snapshot()
    assert env["python"]
    assert env["packages"]["numpy"]
    assert env["device"] in ("cpu", "cuda")
    if not env["cuda_available"]:
        assert env["gpu"] is None
    # No invented hardware: a CPU-only host records no GPU model.
    assert env["cpu_count"] is None or env["cpu_count"] > 0


def test_prepare_experiment_dir_refuses_to_overwrite(tmp_path):
    directory = prepare_experiment_dir(tmp_path, "exp-1")
    write_json(directory / "experiment.json", {"experiment_id": "exp-1"})

    with pytest.raises(FileExistsError):
        prepare_experiment_dir(tmp_path, "exp-1")


def test_artifact_writers_roundtrip(tmp_path):
    write_json(tmp_path / "a.json", {"x": 1})
    write_jsonl(tmp_path / "rows.jsonl", [{"a": 1}, {"a": 2}])

    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 1}
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert rows == [{"a": 1}, {"a": 2}]


def test_verify_dataset_blocks_without_approved_queries():
    documents = [make_doc()]
    manifest = build_manifest(documents, {"fixed_size": []})

    problems = verify_dataset(
        eval_dir=None,
        documents=documents,
        manifest=manifest,
        approved_queries=[],
        split_meta=None,
    )

    assert any("no approved queries" in problem for problem in problems)


def test_verify_dataset_detects_corpus_mismatch():
    documents = [make_doc()]
    manifest = build_manifest(documents, {"fixed_size": []})
    changed = [make_doc(content="Completely different content.")]

    problems = verify_dataset(
        eval_dir=None,
        documents=changed,
        manifest=manifest,
        approved_queries=[],
        split_meta=None,
        require_approved=False,
    )

    assert any("corpus does not match" in problem for problem in problems)


def test_verify_dataset_detects_split_disagreement():
    documents = [make_doc()]
    manifest = build_manifest(documents, {"fixed_size": []})
    approved = [
        {
            "query_id": "q_001",
            "query": "What is a PaymentIntent?",
            "category": "terminology",
            "corpus_version": manifest["corpus_version"],
            "split": "dev",
        }
    ]
    split_meta = {"assignment": {"q_001": "test"}}

    problems = verify_dataset(
        eval_dir=None,
        documents=documents,
        manifest=manifest,
        approved_queries=approved,
        split_meta=split_meta,
    )

    assert any("disagrees with split_assignment" in problem for problem in problems)


def test_run_matrix_refuses_without_approved_queries(tmp_path):
    from src.evaluation.judgments import JudgmentStore
    from src.utils.config import get_config

    documents = [make_doc()]
    manifest = build_manifest(documents, {"fixed_size": []})

    runner = RetrievalRunner.__new__(RetrievalRunner)
    runner.eval_dir = tmp_path
    runner.indexes_dir = tmp_path / "indexes"
    runner.raw_dir = tmp_path / "raw"
    runner.processed_dir = tmp_path / "processed"
    runner.top_k = 10
    runner.results_root = tmp_path / "results"
    runner.config = get_config()
    runner.documents = documents
    runner.manifest = manifest
    runner.approved = []
    runner.candidates = []
    runner.provenance = []
    runner.judgment_store = JudgmentStore(tmp_path / "judgments.jsonl")
    runner.split_meta = None
    runner._retrievers = {}
    runner._reranker = None

    with pytest.raises(EvaluationBlocked) as error:
        runner.run_matrix(query_set="approved", split="dev")

    assert "no approved queries" in str(error.value)
    assert not (tmp_path / "results").exists()


def test_run_matrix_requires_descriptive_for_candidates(tmp_path):
    from src.evaluation.judgments import JudgmentStore
    from src.utils.config import get_config

    documents = [make_doc()]
    manifest = build_manifest(documents, {"fixed_size": []})
    (tmp_path / "queries.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q_001",
                "query": "What is a PaymentIntent?",
                "category": "terminology",
                "split": "dev",
            }
        )
        + "\n"
    )

    runner = RetrievalRunner.__new__(RetrievalRunner)
    runner.eval_dir = tmp_path
    runner.indexes_dir = tmp_path / "indexes"
    runner.raw_dir = tmp_path / "raw"
    runner.processed_dir = tmp_path / "processed"
    runner.top_k = 10
    runner.results_root = tmp_path / "results"
    runner.config = get_config()
    runner.documents = documents
    runner.manifest = manifest
    runner.approved = []
    runner.candidates = []
    runner.provenance = []
    runner.judgment_store = JudgmentStore(tmp_path / "judgments.jsonl")
    runner.split_meta = None
    runner._retrievers = {}
    runner._reranker = None

    with pytest.raises(EvaluationBlocked) as error:
        runner.run_matrix(query_set="candidates", split="dev")

    assert "descriptive=True" in str(error.value)


def test_summarize_latencies():
    stats = summarize_latencies([1.0, 2.0, 3.0, 4.0])
    assert stats["count"] == 4
    assert stats["mean_ms"] == pytest.approx(2.5)
    assert stats["p50_ms"] == pytest.approx(2.5)
    assert stats["p95_ms"] == pytest.approx(3.85)


def test_summarize_latencies_single_sample():
    stats = summarize_latencies([5.0])
    assert stats["p50_ms"] == 5.0
    assert stats["p95_ms"] == 5.0
    assert stats["stdev_ms"] == 0.0


# ---------------------------------------------------------------------------
# Ablation guards
# ---------------------------------------------------------------------------

def test_ablation_specs_cover_supported_dimensions():
    specs = ablation_specs("fixed_size")
    dimensions = {spec["dimension"] for spec in specs}
    assert dimensions == {"fusion", "candidate_depth", "rerank_depth"}
    assert all(spec["chunking_strategy"] == "fixed_size" for spec in specs)


def test_ablation_refuses_test_split():
    with pytest.raises(ValueError) as error:
        run_ablation(runner=None, strategy="fixed_size", split="test")
    assert "development split" in str(error.value)

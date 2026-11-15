#!/usr/bin/env python
"""
Build the reproducible evaluation dataset: corpus manifest, candidate queries,
pooled candidates from all four retrieval variants and both chunking
strategies, and a deterministic dev/test split.

This script never assigns relevance grades.

Usage:
    python scripts/build_eval_dataset.py --rebuild-indexes
    python scripts/build_eval_dataset.py --new-version   # corpus changed
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.corpus_manifest import (  # noqa: E402
    build_manifest,
    corpus_version,
    load_manifest,
    verify_corpus,
    write_manifest,
)
from src.evaluation.eval_dataset import (  # noqa: E402
    DATASET_FORMAT_VERSION,
    POOL_DEPTH,
    STRATEGIES,
    SYSTEMS,
    attach_system_coverage,
    build_candidate_pools,
    generate_candidate_queries,
    validate_dataset,
    write_jsonl,
)
from src.evaluation.judgments import JudgmentStore  # noqa: E402
from src.evaluation.query_review import write_pool_baseline  # noqa: E402
from src.evaluation.splits import (  # noqa: E402
    assign_group_splits,
    write_split_assignment,
)
from src.ingestion.chunker import load_chunks  # noqa: E402
from src.ingestion.crawler import load_documents  # noqa: E402
from src.reranking.reranker import CrossEncoderReranker  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--indexes", default="data/indexes")
    parser.add_argument("--depth", type=int, default=POOL_DEPTH)
    parser.add_argument("--rebuild-indexes", action="store_true")
    parser.add_argument(
        "--new-version",
        action="store_true",
        help="accept a changed corpus and write a new snapshot version",
    )
    return parser.parse_args()


def load_or_build_retrievers(indexes_dir: Path, chunks_by_strategy, cv, rebuild):
    """One retriever set per chunking strategy, loaded from disk when possible."""
    reranker = CrossEncoderReranker()
    retrievers = {}

    for strategy in STRATEGIES:
        strategy_dir = indexes_dir / strategy
        chunks = chunks_by_strategy[strategy]

        bm25 = BM25Retriever()
        dense = DenseRetriever()
        hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
        hybrid.corpus_version = cv
        hybrid.strategy = strategy

        manifest = strategy_dir / HybridRetriever.MANIFEST_NAME
        if rebuild or not manifest.exists():
            bm25.index_chunks(chunks)
            dense.index_chunks(chunks)
            hybrid.save_indexes(str(strategy_dir), corpus_version=cv)
            print(f"  built {strategy} index ({len(chunks)} chunks) -> {strategy_dir}")
        else:
            hybrid.load_indexes(str(strategy_dir))
            print(f"  loaded {strategy} index ({len(chunks)} chunks) <- {strategy_dir}")

        retrievers[strategy] = {
            "bm25": bm25,
            "dense": dense,
            "hybrid": hybrid,
            "reranker": reranker,
        }

    return retrievers


def main() -> int:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(args.raw)
    chunks_by_strategy = {
        strategy: load_chunks(args.processed, strategy=strategy) for strategy in STRATEGIES
    }
    cv = corpus_version(documents)
    print(f"Corpus: {len(documents)} documents, version {cv}")
    for strategy, chunks in chunks_by_strategy.items():
        print(f"  {strategy}: {len(chunks)} chunks")

    manifest_path = eval_dir / "corpus_manifest.json"
    if manifest_path.exists() and not args.new_version:
        existing = load_manifest(manifest_path)
        report = verify_corpus(existing, documents)
        if not report["ok"]:
            print(
                "Corpus does not match the existing snapshot "
                f"({existing['corpus_version']}): "
                f"missing={len(report['missing'])} extra={len(report['extra'])} "
                f"changed={len(report['changed'])}",
                file=sys.stderr,
            )
            print("Re-run with --new-version to freeze a new snapshot.", file=sys.stderr)
            return 2
        print(f"Matching snapshot found: {existing['corpus_version']}")

    print("Loading or building retrievers...")
    retrievers = load_or_build_retrievers(
        Path(args.indexes), chunks_by_strategy, cv, args.rebuild_indexes
    )

    manifest = build_manifest(
        documents=documents,
        chunks_by_strategy=chunks_by_strategy,
        chunk_paths={
            strategy: str(Path(args.processed) / f"chunks_{strategy}.jsonl")
            for strategy in STRATEGIES
        },
        index_paths={
            strategy: str(Path(args.indexes) / strategy) for strategy in STRATEGIES
        },
    )
    write_manifest(manifest, manifest_path)
    print(f"Wrote {manifest_path}")

    queries, skipped = generate_candidate_queries(
        documents=documents,
        chunks=chunks_by_strategy["structure_aware"],
        corpus_version=cv,
    )
    print(f"Candidate queries: {len(queries)} (skipped blueprints: {len(skipped)})")
    for entry in skipped:
        print(f"  skipped: {entry['reason']}: {entry['query']}")

    candidates, provenance = build_candidate_pools(
        queries=queries,
        retrievers_by_strategy=retrievers,
        chunks_by_strategy=chunks_by_strategy,
        corpus_version=cv,
        depth=args.depth,
    )
    print(f"Pooled candidates: {len(candidates)}")
    for strategy in STRATEGIES:
        count = sum(1 for c in candidates if c["chunking_strategy"] == strategy)
        print(f"  {strategy}: {count}")

    split_meta = assign_group_splits(queries)
    split_version = split_meta["split_version"]
    for warning in split_meta["warnings"]:
        print(f"  split warning: {warning}")
    write_split_assignment(split_meta, eval_dir / "split_assignment.json")

    judgments_path = eval_dir / "judgments.jsonl"
    store = JudgmentStore(judgments_path)
    queries_path = write_jsonl(queries, eval_dir / "queries.jsonl")
    candidates_path = write_jsonl(candidates, eval_dir / "candidates.jsonl")
    provenance_path = write_jsonl(provenance, eval_dir / "candidate_provenance.jsonl")
    baseline_path = write_pool_baseline(
        queries,
        eval_dir / "candidate_pool_baseline.json",
        source="build_eval_dataset",
    )

    report = validate_dataset(
        queries=queries,
        candidates=candidates,
        judgments=store.records(),
        chunks_by_strategy=chunks_by_strategy,
        manifest=manifest,
    )
    attach_system_coverage(report, provenance)

    dataset_version = "ds-" + hashlib.sha256(
        "\n".join(
            [q["query_id"] for q in queries] + [c["candidate_id"] for c in candidates]
        ).encode()
    ).hexdigest()[:12]

    meta = {
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_version": dataset_version,
        "corpus_version": cv,
        "split_version": split_version,
        "split_method": split_meta["split_method"],
        "split_seed": split_meta["split_seed"],
        "num_query_groups": split_meta["num_groups"],
        "pool_depth": args.depth,
        "systems": list(SYSTEMS),
        "strategies": list(STRATEGIES),
        "num_documents": len(documents),
        "num_queries": len(queries),
        "num_candidates": len(candidates),
        "files": {
            "corpus_manifest": str(manifest_path),
            "queries": str(queries_path),
            "candidates": str(candidates_path),
            "candidate_provenance": str(provenance_path),
            "judgments": str(judgments_path),
            "candidate_pool_baseline": str(baseline_path),
            "split_assignment": str(eval_dir / "split_assignment.json"),
        },
        "notes": (
            "Candidate queries and pooled candidates only. No human relevance "
            "judgments exist until they are recorded in judgments.jsonl."
        ),
    }
    (eval_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Wrote {queries_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {provenance_path}")
    print(f"Wrote {eval_dir / 'dataset_meta.json'} ({dataset_version}, {split_version})")
    print(
        f"Validation: {'ok' if report['is_structurally_valid'] else 'problems'} | "
        f"judged={report['judged_candidates']} pending={report['pending_candidates']}"
    )
    if report["problems"]:
        for problem in report["problems"][:20]:
            print(f"  problem: {problem}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

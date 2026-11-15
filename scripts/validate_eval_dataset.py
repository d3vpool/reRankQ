#!/usr/bin/env python
"""
Validate the evaluation dataset and print its state.

Reports query/candidate/judgment counts, category and split distributions, grade
distribution, per-system candidate coverage, query-review status, query-quality
warnings, split leakage, approved-dataset state and any structural problems.
Exits non-zero when structural problems exist. Pending candidates and
unreviewed queries are reported as such: neither is ever treated as a decision.

Usage:
    python scripts/validate_eval_dataset.py
    python scripts/validate_eval_dataset.py --eval-dir data/eval --json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.corpus_manifest import load_manifest, verify_corpus  # noqa: E402
from src.evaluation.eval_dataset import (  # noqa: E402
    STRATEGIES,
    attach_system_coverage,
    load_jsonl,
    validate_dataset,
)
from src.evaluation.judgments import JudgmentStore  # noqa: E402
from src.evaluation.query_review import (  # noqa: E402
    POOL_BASELINE_FILENAME,
    ReviewStore,
    candidate_pool_status,
    load_pool_baseline,
    query_review_report,
    validate_approved_dataset,
)
from src.evaluation.splits import audit_split_leakage  # noqa: E402
from src.ingestion.chunker import load_chunks  # noqa: E402
from src.ingestion.crawler import load_documents  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_dir = Path(args.eval_dir)

    queries = load_jsonl(eval_dir / "queries.jsonl")
    candidates = load_jsonl(eval_dir / "candidates.jsonl")
    provenance = load_jsonl(eval_dir / "candidate_provenance.jsonl")
    approved = load_jsonl(eval_dir / "approved_queries.jsonl")
    documents = load_documents(args.raw)
    store = JudgmentStore(eval_dir / "judgments.jsonl")
    reviews = ReviewStore(eval_dir / "query_reviews.jsonl")
    manifest = load_manifest(eval_dir / "corpus_manifest.json")
    chunks_by_strategy = {
        strategy: load_chunks(args.processed, strategy=strategy) for strategy in STRATEGIES
    }

    report = validate_dataset(
        queries=queries,
        candidates=candidates,
        judgments=store.records(),
        chunks_by_strategy=chunks_by_strategy,
        manifest=manifest,
    )
    attach_system_coverage(report, provenance)

    review_report = query_review_report(
        queries, reviews.records(), documents, chunks_by_strategy, manifest
    )
    leakage = audit_split_leakage(queries)
    baseline_path = eval_dir / POOL_BASELINE_FILENAME
    pool_status = (
        candidate_pool_status(queries, load_pool_baseline(baseline_path), reviews)
        if baseline_path.exists()
        else {"counts": {"current": 0, "stale": 0, "missing_baseline": len(queries)}, "stale": []}
    )
    approved_problems = validate_approved_dataset(approved, queries) if approved else []

    corpus_report = verify_corpus(manifest, documents)
    if not corpus_report["ok"]:
        report["problems"].append(
            "raw corpus no longer matches the frozen snapshot "
            f"(missing={len(corpus_report['missing'])}, extra={len(corpus_report['extra'])}, "
            f"changed={len(corpus_report['changed'])})"
        )
        report["is_structurally_valid"] = False

    report["query_review"] = review_report
    report["leakage"] = {
        "leakage_found": leakage["leakage_found"],
        "groups_spanning_splits": len(leakage["groups_spanning_splits"]),
        "shared_documents": len(leakage["shared_documents"]),
        "shared_sections": len(leakage["shared_sections"]),
        "near_duplicates_across_splits": len(leakage["near_duplicates_across_splits"]),
        "num_groups": leakage["num_groups"],
    }
    report["approved_queries"] = len(approved)
    report["approved_problems"] = approved_problems
    report["pool_consistency"] = pool_status["counts"]
    report["is_structurally_valid"] = (
        report["is_structurally_valid"]
        and review_report["is_structurally_valid"]
        and not approved_problems
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Evaluation dataset")
        print("==================")
        print(f"corpus version     : {report['corpus_version']}")
        print(f"queries            : {report['num_queries']}")
        print(f"  by category      : {report['queries_by_category']}")
        print(f"  by split         : {report['queries_by_split']}")
        print(f"  review status    : {review_report['by_status']}")
        print(f"  by category/status: {review_report['by_category_status']}")
        print(f"candidates         : {report['num_candidates']}")
        print(f"  by strategy      : {report['candidates_by_strategy']}")
        print(f"  by system        : {report['system_coverage']}")
        print(f"judged candidates  : {report['judged_candidates']}")
        print(f"pending candidates : {report['pending_candidates']}")
        print(f"fully judged queries: {report['fully_judged_queries']}")
        print(f"grade distribution : {report['grade_distribution']}")
        print(f"superseded records : {report['superseded_judgment_records']}")
        print()
        print(f"query warnings     : {review_report['warnings_by_kind']}")
        print(f"queries needing attention: {len(review_report['queries_requiring_attention'])}")
        print(f"approved queries   : {len(approved)}")
        print(f"pool consistency   : {pool_status['counts']}")
        print(
            "split leakage      : "
            f"{'yes' if leakage['leakage_found'] else 'no'} "
            f"(groups spanning splits: {len(leakage['groups_spanning_splits'])}, "
            f"shared docs: {len(leakage['shared_documents'])})"
        )
        print(
            f"query groups       : {leakage['num_groups']} sizes={leakage['group_sizes']}"
        )
        print()
        if report["pending_candidates"]:
            print(
                "Relevance judgments are NOT complete: "
                f"{report['pending_candidates']} candidates are still pending."
            )
        if review_report["by_status"]["candidate"]:
            print(
                "Queries are NOT fully reviewed: "
                f"{review_report['by_status']['candidate']} still have status candidate."
            )
        problems = report["problems"] + review_report["structural_problems"] + approved_problems
        if problems:
            print(f"{len(problems)} problem(s):")
            for problem in problems[:30]:
                print(f"  - {problem}")
        else:
            print("Structural validation: ok")

    return 0 if report["is_structurally_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

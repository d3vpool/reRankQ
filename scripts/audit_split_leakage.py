#!/usr/bin/env python
"""
Audit the current dev/test split for leakage.

Reports shared source documents, shared sections, duplicate and near-duplicate
queries across splits, the document-connected query groups and their sizes, and
the category distribution of each split.

Usage:
    python scripts/audit_split_leakage.py
    python scripts/audit_split_leakage.py --json --write
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.eval_dataset import load_jsonl  # noqa: E402
from src.evaluation.splits import audit_split_leakage, build_query_groups  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write", action="store_true", help="write split_audit.json")
    parser.add_argument("--threshold", type=float, default=0.6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    queries = load_jsonl(eval_dir / "queries.jsonl")
    if not queries:
        print("No queries found", file=sys.stderr)
        return 2

    report = audit_split_leakage(queries, threshold=args.threshold)
    groups = build_query_groups(queries)
    report["groups"] = groups

    if args.write:
        path = eval_dir / "split_audit.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {path}")

    if args.json:
        print(json.dumps(report, indent=2))
        return 1 if report["leakage_found"] else 0

    print("Split leakage audit")
    print("===================")
    print(f"queries        : {report['num_queries']} {report['by_split']}")
    print(f"categories     : {report['by_category_split']}")
    print(f"query groups   : {report['num_groups']} sizes={report['group_sizes']}")
    print()
    print(f"Groups spanning both splits: {len(report['groups_spanning_splits'])}")
    for group in report["groups_spanning_splits"]:
        print(
            f"  {group['group_id']} size={group['size']} "
            f"dev={group['dev']} test={group['test']}"
        )
    print()
    print(f"Shared documents: {len(report['shared_documents'])}")
    for entry in report["shared_documents"]:
        print(f"  {entry['doc_id']} dev={entry['dev']} test={entry['test']}")
    print()
    print(f"Shared sections: {len(report['shared_sections'])}")
    for entry in report["shared_sections"]:
        print(f"  {entry['section']!r} dev={entry['dev']} test={entry['test']}")
    print()
    print(f"Duplicate queries across splits: {len(report['exact_duplicates_across_splits'])}")
    for pair in report["exact_duplicates_across_splits"]:
        print(f"  {pair['query_ids']} {pair['queries']}")
    print(
        f"Near-duplicate queries across splits (jaccard >= {report['near_duplicate_threshold']}): "
        f"{len(report['near_duplicates_across_splits'])}"
    )
    for pair in report["near_duplicates_across_splits"]:
        print(f"  {pair['query_ids']} j={pair['similarity']} {pair['queries']}")
    print()
    print(f"Leakage found: {report['leakage_found']}")
    print(report["summary"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

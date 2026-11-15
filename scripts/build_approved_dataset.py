#!/usr/bin/env python
"""
Build the approved evaluation set and freeze the dev/test split.

Only queries whose latest review status is ``accepted`` enter
``approved_queries.jsonl``; rejected and needs-revision queries are preserved in
``query_reviews.jsonl`` for auditability. Candidate pools and relevance
judgments are never regenerated or modified here.

Usage:
    python scripts/build_approved_dataset.py
    python scripts/build_approved_dataset.py --method existing
    python scripts/build_approved_dataset.py --seed 13 --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.eval_dataset import load_jsonl, write_jsonl  # noqa: E402
from src.evaluation.query_review import (  # noqa: E402
    POOL_BASELINE_FILENAME,
    ReviewStore,
    build_approved_dataset,
    candidate_pool_status,
    load_pool_baseline,
    validate_approved_dataset,
    write_pool_baseline,
)
from src.evaluation.splits import (  # noqa: E402
    GROUP_SPLIT_SEED,
    GROUP_TEST_FRACTION,
    assign_group_splits,
    audit_split_leakage,
    build_query_groups,
    split_version,
    write_split_assignment,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--method", choices=("group", "existing"), default="group")
    parser.add_argument("--seed", type=int, default=GROUP_SPLIT_SEED)
    parser.add_argument("--test-fraction", type=float, default=GROUP_TEST_FRACTION)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_dir = Path(args.eval_dir)

    queries = load_jsonl(eval_dir / "queries.jsonl")
    candidates = load_jsonl(eval_dir / "candidates.jsonl")
    if not queries:
        print("No candidate queries found", file=sys.stderr)
        return 2

    store = ReviewStore(eval_dir / "query_reviews.jsonl")
    baseline_path = eval_dir / POOL_BASELINE_FILENAME
    if baseline_path.exists():
        baseline = load_pool_baseline(baseline_path)
    else:
        write_pool_baseline(
            queries,
            baseline_path,
            source="reconstructed from the wording used to build the existing pools",
        )
        baseline = load_pool_baseline(baseline_path)
        print(f"Wrote {baseline_path} (reconstructed pool baseline)")

    if args.method == "group":
        split_meta = assign_group_splits(
            queries, seed=args.seed, test_fraction=args.test_fraction
        )
        split_meta["leakage_audit_after_assignment"] = audit_split_leakage(queries)
        split_groups = {
            query_id: next(
                group["group_id"]
                for group in split_meta["groups"]
                if query_id in group["query_ids"]
            )
            for query_id in split_meta["assignment"]
        }
    else:
        groups = build_query_groups(queries)
        assignment = {query["query_id"]: query.get("split") for query in queries}
        split_meta = {
            "split_method": "existing-assignment",
            "split_version": split_version(assignment, groups, args.seed),
            "split_seed": args.seed,
            "num_queries": len(queries),
            "num_groups": len(groups),
            "group_sizes": [group["size"] for group in groups],
            "groups": groups,
            "assignment": assignment,
            "by_split": {
                split: sum(1 for value in assignment.values() if value == split)
                for split in ("dev", "test")
            },
            "warnings": ["existing split retained; audit output reports its leakage"],
            "leakage_audit_after_assignment": audit_split_leakage(queries),
        }
        split_groups = {
            query_id: next(
                group["group_id"] for group in groups if query_id in group["query_ids"]
            )
            for query_id in assignment
        }

    pool_status = candidate_pool_status(queries, baseline, store)
    approved, approved_meta = build_approved_dataset(
        queries, store, split_groups=split_groups, pool_status=pool_status
    )
    problems = validate_approved_dataset(approved, queries)

    print("Approved evaluation set")
    print("=======================")
    print(f"split method      : {split_meta['split_method']}")
    print(f"split version     : {split_meta['split_version']}")
    print(f"split             : {split_meta['by_split']}")
    print(f"categories        : {split_meta.get('by_category_split')}")
    print(f"query groups      : {split_meta['num_groups']} sizes={split_meta['group_sizes']}")
    for warning in split_meta.get("warnings", []):
        print(f"  warning: {warning}")
    print(
        "leak after split  : "
        f"{split_meta['leakage_audit_after_assignment']['leakage_found']}"
    )
    print(f"candidate rows    : {len(candidates)} (unchanged)")
    print(f"review statuses   : {store.status_counts()}")
    print(f"approved queries  : {approved_meta['num_queries']}")
    print(f"approved version  : {approved_meta['approved_version']}")
    print(f"stale pools       : {pool_status['counts']}")
    if pool_status["stale"]:
        print(f"  stale query ids : {pool_status['stale']}")
    if problems:
        print(f"approved problems : {len(problems)}")
        for problem in problems[:20]:
            print(f"  - {problem}")
    if approved_meta["awaiting_human_review"]:
        print(
            "no query has been accepted yet; approved_queries.jsonl is empty and "
            "the evaluation set is not human-validated"
        )

    if args.dry_run:
        print("dry run: nothing written")
        return 1 if problems else 0

    write_split_assignment(split_meta, eval_dir / "split_assignment.json")
    write_jsonl(queries, eval_dir / "queries.jsonl")
    write_jsonl(approved, eval_dir / "approved_queries.jsonl")

    meta_path = eval_dir / "dataset_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        {
            "split_version": split_meta["split_version"],
            "split_method": split_meta["split_method"],
            "split_seed": split_meta["split_seed"],
            "split_file": str(eval_dir / "split_assignment.json"),
            "num_query_groups": split_meta["num_groups"],
            "query_groups": split_meta["group_sizes"],
            "approved_version": approved_meta["approved_version"],
            "approved_queries": approved_meta["num_queries"],
            "approved_file": str(eval_dir / "approved_queries.jsonl"),
            "review_file": str(eval_dir / "query_reviews.jsonl"),
            "stale_pools": pool_status["stale"],
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    approved_meta.update(
        {
            "review_counts": store.status_counts(),
            "split_version": split_meta["split_version"],
            "split_method": split_meta["split_method"],
            "problems": problems,
            "note": (
                "Only queries with review status 'accepted' are included. "
                "Relevance judgments are a separate task and are untouched."
            ),
        }
    )
    (eval_dir / "approved_dataset_meta.json").write_text(
        json.dumps(approved_meta, indent=2) + "\n"
    )

    print(f"Wrote {eval_dir / 'split_assignment.json'}")
    print(f"Wrote {eval_dir / 'queries.jsonl'} (split fields updated)")
    print(f"Wrote {eval_dir / 'approved_queries.jsonl'}")
    print(f"Wrote {eval_dir / 'approved_dataset_meta.json'}")
    print(f"Wrote {eval_dir / 'dataset_meta.json'}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

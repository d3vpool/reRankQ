#!/usr/bin/env python
"""
Build a deeper candidate pool (top-10/top-20) without touching the frozen pool.

Existing candidates, their provenance and all human judgments are preserved.
The deeper pool is written to a separate directory, and any candidate it adds
has no judgment record, so it is pending by construction.

Usage:
    python scripts/build_deeper_pool.py --depth 10 --dry-run
    python scripts/build_deeper_pool.py --depth 10
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.eval_dataset import load_jsonl, write_jsonl  # noqa: E402
from src.evaluation.pooling import (  # noqa: E402
    build_deeper_pool,
    merge_pools,
    pool_coverage_report,
)
from src.evaluation.retrieval_runner import RetrievalRunner  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--query-set", default="candidates", choices=("candidates", "approved"))
    parser.add_argument("--strategies", default="fixed_size,structure_aware")
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--output-root", default="data/eval/pools")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    runner = RetrievalRunner(eval_dir=eval_dir)

    queries = runner.queries(query_set=args.query_set, split="all")
    if not queries:
        print("No queries available", file=sys.stderr)
        return 2

    existing_candidates = load_jsonl(eval_dir / "candidates.jsonl")
    existing_provenance = load_jsonl(eval_dir / "candidate_provenance.jsonl")
    judgments = load_jsonl(eval_dir / "judgments.jsonl")
    strategies = args.strategies.split(",")

    new_candidates, new_provenance = build_deeper_pool(
        runner, queries, depth=args.depth, strategies=strategies
    )
    merged_candidates, merged_provenance, stats = merge_pools(
        existing_candidates, existing_provenance, new_candidates, new_provenance
    )
    coverage = pool_coverage_report(
        queries, merged_candidates, merged_provenance, judgments, strategies
    )

    digest = hashlib.sha256(
        "\n".join(
            sorted(
                f"{c['query_id']}|{c['chunking_strategy']}|{c['chunk_id']}"
                for c in merged_candidates
            )
        ).encode()
    ).hexdigest()
    pool_version = f"pool-depth{args.depth}-{digest[:8]}"

    print(f"Deeper pool · depth {args.depth} · {len(queries)} queries")
    print(f"existing candidates : {stats['existing']}")
    print(f"added by depth-{args.depth}: {stats['added']}")
    print(f"total              : {stats['total']}")
    print(f"pool version        : {pool_version}")
    print(f"judged pairs        : {coverage['judged_pairs']}")
    print(f"pending pairs       : {coverage['pending_pairs']}")
    print("Existing candidates, provenance and judgments are untouched.")
    if stats["added"]:
        print(
            f"{stats['added']} candidate rows are new and pending: they need human "
            "judgment before any metric can include them."
        )

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    output_dir = Path(args.output_root) / f"depth{args.depth}"
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"{output_dir} already exists; refusing to overwrite", file=sys.stderr)
        return 3
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(merged_candidates, output_dir / "candidates.jsonl")
    write_jsonl(merged_provenance, output_dir / "candidate_provenance.jsonl")
    (output_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2) + "\n")
    (output_dir / "pool_meta.json").write_text(
        json.dumps(
            {
                "pool_version": pool_version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "depth": args.depth,
                "corpus_version": runner.corpus_version,
                "query_set": args.query_set,
                "num_queries": len(queries),
                "num_candidates": len(merged_candidates),
                "num_added": stats["added"],
                "strategies": strategies,
                "note": (
                    "Additive pool for judging coverage. Metrics must not use it until "
                    "the added candidates are judged."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

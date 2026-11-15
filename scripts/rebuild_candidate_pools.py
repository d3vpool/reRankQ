#!/usr/bin/env python
"""
Rebuild candidate pools for queries whose wording changed.

A pool is only valid for the exact query wording it was retrieved with. After a
review edits a query, the old pool is stale; this script archives those rows,
re-pools the affected queries with their current wording, and leaves every other
query's pool and all relevance judgments untouched.

Usage:
    python scripts/rebuild_candidate_pools.py --stale
    python scripts/rebuild_candidate_pools.py --query-ids q_003,q_017
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_eval_dataset import load_or_build_retrievers  # noqa: E402
from src.evaluation.corpus_manifest import corpus_version  # noqa: E402
from src.evaluation.eval_dataset import (  # noqa: E402
    POOL_DEPTH,
    STRATEGIES,
    build_candidate_pools,
    load_jsonl,
    write_jsonl,
)
from src.evaluation.query_review import (  # noqa: E402
    POOL_BASELINE_FILENAME,
    ReviewStore,
    candidate_pool_status,
    effective_queries,
    load_pool_baseline,
    replace_query_pools,
)
from src.ingestion.chunker import load_chunks  # noqa: E402
from src.ingestion.crawler import load_documents  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--indexes", default="data/indexes")
    parser.add_argument("--depth", type=int, default=POOL_DEPTH)
    parser.add_argument("--query-ids", default="")
    parser.add_argument("--stale", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_dir = Path(args.eval_dir)

    queries = load_jsonl(eval_dir / "queries.jsonl")
    candidates = load_jsonl(eval_dir / "candidates.jsonl")
    provenance = load_jsonl(eval_dir / "candidate_provenance.jsonl")
    judgments = load_jsonl(eval_dir / "judgments.jsonl")
    store = ReviewStore(eval_dir / "query_reviews.jsonl")
    baseline_path = eval_dir / POOL_BASELINE_FILENAME
    baseline = load_pool_baseline(baseline_path) if baseline_path.exists() else {"wording": {}}

    requested = [qid.strip() for qid in args.query_ids.split(",") if qid.strip()]
    if args.stale:
        status = candidate_pool_status(queries, baseline, store)
        requested = sorted(set(requested) | set(status["stale"]))
    if not requested:
        print("No stale pools and no --query-ids given; nothing to rebuild.")
        return 0

    documents = load_documents(args.raw)
    chunks_by_strategy = {
        strategy: load_chunks(args.processed, strategy=strategy) for strategy in STRATEGIES
    }
    cv = corpus_version(documents)
    retrievers = load_or_build_retrievers(Path(args.indexes), chunks_by_strategy, cv, False)

    current = {query["query_id"]: query for query in effective_queries(queries, store)}
    unknown = [qid for qid in requested if qid not in current]
    if unknown:
        print(f"Unknown query ids: {unknown}", file=sys.stderr)
        return 2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = eval_dir / "pool_archive" / timestamp
    archived_before = len(candidates)
    before_ids = set()
    after_ids = set()

    for query_id in requested:
        query = current[query_id]
        old_rows = [row for row in candidates if row["query_id"] == query_id]
        before_ids |= {row["candidate_id"] for row in old_rows}
        new_candidates, new_provenance = build_candidate_pools(
            queries=[query],
            retrievers_by_strategy=retrievers,
            chunks_by_strategy=chunks_by_strategy,
            corpus_version=cv,
            depth=args.depth,
        )
        after_ids |= {row["candidate_id"] for row in new_candidates}

        print(
            f"{query_id}: {len(old_rows)} -> {len(new_candidates)} candidates "
            f"for “{query['query']}”"
        )
        if not args.dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(old_rows, archive_dir / f"{query_id}_candidates.jsonl")
            write_jsonl(
                [row for row in provenance if row["query_id"] == query_id],
                archive_dir / f"{query_id}_provenance.jsonl",
            )
        candidates, provenance = replace_query_pools(
            candidates, provenance, query_id, new_candidates, new_provenance
        )
        baseline.setdefault("wording", {})[query_id] = query["query"]

    removed = before_ids - after_ids
    orphaned_judgments = [
        row for row in judgments if row.get("candidate_id") in removed
    ]
    baseline["updated_at"] = datetime.now(timezone.utc).isoformat()

    print()
    print(f"candidate rows   : {archived_before} -> {len(candidates)}")
    print(f"re-pooled queries: {len(requested)}")
    if orphaned_judgments:
        print(
            f"warning: {len(orphaned_judgments)} existing judgment(s) reference candidates "
            "that no longer exist for this wording; they are preserved but cannot be "
            "reused and must be re-judged"
        )
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    write_jsonl(candidates, eval_dir / "candidates.jsonl")
    write_jsonl(provenance, eval_dir / "candidate_provenance.jsonl")
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"archive          : {archive_dir}")
    print(f"Wrote {eval_dir / 'candidates.jsonl'}")
    print(f"Wrote {eval_dir / 'candidate_provenance.jsonl'}")
    print(f"Updated {baseline_path}")
    print("Judgments were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

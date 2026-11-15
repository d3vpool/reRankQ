"""
Candidate-pool coverage and deeper pooling.

The Task 3 pool was built at depth 3 per system. That is shallow for metrics at
k=10: a relevant chunk that no system returned cannot be judged, so recall is
pool-limited. This module reports that coverage honestly and can build a deeper
pool without touching existing candidates, provenance or human judgments.
"""

from typing import Iterable, Optional

from src.evaluation.eval_dataset import build_candidate_pools


def _judged_pairs(judgments: Iterable[dict]) -> dict[tuple[str, str, str], int]:
    """Latest judged grades keyed by (query_id, strategy, chunk_id)."""
    latest: dict[str, dict] = {}
    for record in judgments:
        latest[record["candidate_id"]] = record
    pairs = {}
    for record in latest.values():
        if record.get("judgment_status") != "judged":
            continue
        pairs[(record["query_id"], record["chunking_strategy"], record["chunk_id"])] = int(
            record["relevance_grade"]
        )
    return pairs


def pool_coverage_report(
    queries: Iterable[dict],
    candidates: Iterable[dict],
    provenance: Iterable[dict],
    judgments: Iterable[dict],
    strategies: Iterable[str],
) -> dict:
    """
    Coverage of the pooled candidate set by human judgments.

    Reports totals, per-strategy and per-query coverage, contributing-system
    counts and the queries whose pools are not fully judged.
    """
    queries = list(queries)
    query_ids = [query["query_id"] for query in queries]
    candidates = list(candidates)
    provenance = list(provenance)
    judged = _judged_pairs(judgments)

    by_strategy = {}
    for strategy in strategies:
        strategy_candidates = [c for c in candidates if c["chunking_strategy"] == strategy]
        strategy_pairs = {
            (c["query_id"], c["chunk_id"]) for c in strategy_candidates
        }
        judged_pairs = {pair for pair in judged if pair[1] == strategy}
        judged_chunks = {(query_id, chunk_id) for query_id, _, chunk_id in judged_pairs}
        covered = strategy_pairs & judged_chunks

        system_counts = {}
        for record in provenance:
            if record["chunking_strategy"] != strategy:
                continue
            for system in {source["system"] for source in record.get("systems", [])}:
                system_counts[system] = system_counts.get(system, 0) + 1

        per_query = []
        for query_id in query_ids:
            pool = {chunk_id for qid, chunk_id in strategy_pairs if qid == query_id}
            done = {chunk_id for qid, chunk_id in covered if qid == query_id}
            per_query.append(
                {
                    "query_id": query_id,
                    "pool_size": len(pool),
                    "judged": len(done),
                    "pending": len(pool) - len(done),
                    "coverage": (len(done) / len(pool)) if pool else 0.0,
                    "fully_judged": bool(pool) and len(done) == len(pool),
                }
            )

        by_strategy[strategy] = {
            "pool_candidates": len(strategy_candidates),
            "unique_chunks": len(strategy_pairs),
            "judged": len(covered),
            "pending": len(strategy_pairs) - len(covered),
            "coverage": (len(covered) / len(strategy_pairs)) if strategy_pairs else 0.0,
            "queries_fully_judged": sum(1 for row in per_query if row["fully_judged"]),
            "queries_without_judgments": [
                row["query_id"] for row in per_query if row["judged"] == 0
            ],
            "system_contributions": dict(sorted(system_counts.items())),
            "per_query": per_query,
        }

    total_unique = {
        (c["query_id"], c["chunking_strategy"], c["chunk_id"]) for c in candidates
    }
    total_judged = set(judged)
    return {
        "num_queries": len(queries),
        "pool_candidates": len(candidates),
        "unique_query_chunk_pairs": len(total_unique),
        "judged_pairs": len(total_judged & total_unique),
        "pending_pairs": len(total_unique - total_judged),
        "coverage": (
            len(total_judged & total_unique) / len(total_unique) if total_unique else 0.0
        ),
        "by_strategy": by_strategy,
        "adequate_for_cutoff": {
            strategy: by_strategy[strategy]["coverage"] >= 1.0 for strategy in by_strategy
        },
        "note": (
            "Coverage is measured on the pooled candidate set. A relevant chunk that "
            "no retrieval system returned is not in the pool and cannot be judged, so "
            "recall computed from these judgments is pool recall."
        ),
    }


def merge_pools(
    existing_candidates: list[dict],
    existing_provenance: list[dict],
    new_candidates: list[dict],
    new_provenance: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    """
    Union of two pools, deduplicated by (query, strategy, chunk).

    Existing rows win; provenance from both pools is combined. Candidates added
    by the deeper pool have no judgment record and are therefore pending.
    """
    key = lambda row: (row["query_id"], row["chunking_strategy"], row["chunk_id"])

    candidates = {key(row): dict(row) for row in existing_candidates}
    added = 0
    for row in new_candidates:
        if key(row) not in candidates:
            candidates[key(row)] = dict(row)
            added += 1

    provenance = {key(row): dict(row) for row in existing_provenance}
    for row in new_provenance:
        bucket = provenance.get(key(row))
        if bucket is None:
            provenance[key(row)] = dict(row)
            continue
        seen = {
            (source["system"], source.get("rank"), source.get("score"))
            for source in bucket.get("systems", [])
        }
        for source in row.get("systems", []):
            if (source["system"], source.get("rank"), source.get("score")) not in seen:
                bucket.setdefault("systems", []).append(source)
        systems = bucket.get("systems", [])
        bucket["num_systems"] = len({source["system"] for source in systems})
        ranks = [source["rank"] for source in systems if source.get("rank") is not None]
        bucket["best_rank"] = min(ranks) if ranks else None

    ordered_candidates = sorted(candidates.values(), key=key)
    ordered_provenance = sorted(provenance.values(), key=key)
    return ordered_candidates, ordered_provenance, {
        "existing": len(existing_candidates),
        "new_input": len(new_candidates),
        "added": added,
        "total": len(ordered_candidates),
    }


def build_deeper_pool(
    runner,
    queries: list[dict],
    depth: int,
    strategies: Iterable[str],
    chunker=None,
) -> tuple[list[dict], list[dict]]:
    """
    Build a pool at ``depth`` per system without touching stored artifacts.

    ``runner`` supplies corpus version and index loading; ``chunker`` defaults to
    ``src.ingestion.chunker.load_chunks`` over the processed directory.
    """
    from src.ingestion.chunker import load_chunks

    load = chunker or load_chunks
    chunks_by_strategy = {
        strategy: load(str(runner.processed_dir), strategy=strategy) for strategy in strategies
    }
    retrievers_by_strategy = {
        strategy: runner.load_retrievers(strategy) for strategy in strategies
    }
    return build_candidate_pools(
        queries=[
            {"query_id": query["query_id"], "query": query["query"]} for query in queries
        ],
        retrievers_by_strategy=retrievers_by_strategy,
        chunks_by_strategy=chunks_by_strategy,
        corpus_version=runner.corpus_version,
        depth=depth,
    )

"""
Retrieval metrics with explicit handling of incomplete judgments.

Design rules:

* Only human judgments are used. Pending candidates are unknown, never grade 0.
* A query is evaluated only when every chunk it contributed to the cutoff has a
  judgment; otherwise it is excluded and the reason is recorded.
* Recall is *pool recall*: the denominator is the judged-relevant set inside the
  pooled candidate set, not every relevant document in the corpus.
"""

import math
from collections import Counter, defaultdict
from typing import Iterable, Optional

RELEVANCE_THRESHOLD = 2
GAIN_BASE = 2
K_VALUES = (1, 3, 5, 10)


def gain(grade: int, base: int = GAIN_BASE) -> float:
    """Exponential gain: grade 0 -> 0, 1 -> 1, 2 -> 3, 3 -> 7."""
    return float(base**grade - 1)


def is_relevant(grade: Optional[int], threshold: int = RELEVANCE_THRESHOLD) -> bool:
    """Binary relevance decision. ``None`` (unjudged) is never relevant."""
    return grade is not None and grade >= threshold


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined without judged relevant chunks")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        raise ValueError("average precision is undefined without judged relevant chunks")
    hits = 0
    total = 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def dcg_at_k(grades: list[int], k: int, base: int = GAIN_BASE) -> float:
    return sum(
        gain(grade, base) / math.log2(index + 2)
        for index, grade in enumerate(grades[:k])
    )


def ndcg_at_k(retrieved_grades: list[int], ideal_grades: list[int], k: int) -> float:
    """
    nDCG@k over human grades.

    ``ideal_grades`` must be the judged grades for the same query, sorted
    descending; the ideal DCG is therefore attainable with the available
    judgments rather than an imagined perfect corpus.
    """
    ideal = dcg_at_k(sorted(ideal_grades, reverse=True), k)
    if ideal == 0:
        raise ValueError("nDCG is undefined when no judged chunk has non-zero gain")
    return dcg_at_k(retrieved_grades, k) / ideal


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query(
    retrieved_ids: list[str],
    qrels: dict[str, int],
    k_values: Iterable[int] = K_VALUES,
    policy: str = "strict_topk",
) -> dict:
    """
    Metrics for one query, or an explicit exclusion.

    ``qrels`` contains only judged chunks (grade 0 included); anything absent is
    unjudged. ``policy="strict_topk"`` requires every retrieved chunk inside the
    largest cutoff to be judged, so no unjudged chunk is ever scored as 0.
    """
    k_values = sorted(set(k_values))
    max_k = max(k_values)
    top = retrieved_ids[:max_k]
    judged_ids = set(qrels)
    unjudged = [chunk_id for chunk_id in top if chunk_id not in judged_ids]

    relevant = {chunk_id for chunk_id, grade in qrels.items() if is_relevant(grade)}
    if not relevant:
        return {
            "status": "excluded",
            "reason": "no_judged_relevant_chunks",
            "retrieved": len(top),
            "unjudged_retrieved": len(unjudged),
        }
    if policy == "strict_topk" and unjudged:
        return {
            "status": "excluded",
            "reason": "unjudged_retrieved_chunks",
            "retrieved": len(top),
            "unjudged_retrieved": len(unjudged),
        }
    if policy not in ("strict_topk", "strict_pool"):
        raise ValueError(f"unknown judgment policy {policy!r}")

    retrieved_grades = [qrels[chunk_id] for chunk_id in top if chunk_id in judged_ids]
    ideal_grades = list(qrels.values())

    metrics = {
        "query_id_unused": None,
        "mrr": reciprocal_rank(top, relevant),
        "map": average_precision(top, relevant),
        "precision@10": precision_at_k(top, relevant, max_k),
        "num_relevant_judged": len(relevant),
        "num_retrieved": len(top),
        "unjudged_retrieved": 0,
    }
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(top, relevant, k)
        metrics[f"precision@{k}"] = precision_at_k(top, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_grades, ideal_grades, k)
    return {"status": "evaluated", "reason": None, **metrics}


# ---------------------------------------------------------------------------
# Run-level evaluation
# ---------------------------------------------------------------------------

def evaluate_run(
    query_ids: list[str],
    run_results: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    pool_ids: dict[str, set[str]],
    k_values: Iterable[int] = K_VALUES,
    policy: str = "strict_topk",
) -> dict:
    """
    Evaluate a full run.

    Returns macro averages over evaluated queries, a complete per-query record
    and an explicit accounting of excluded queries. When no query qualifies the
    result is ``blocked`` and no metric value is produced.
    """
    k_values = sorted(set(k_values))
    per_query = []
    excluded = Counter()

    for query_id in query_ids:
        evaluated = evaluate_query(
            run_results.get(query_id, []),
            qrels.get(query_id, {}),
            k_values,
            policy,
        )
        if evaluated["status"] == "evaluated":
            evaluated["query_id"] = query_id
            per_query.append(evaluated)
        else:
            excluded[evaluated["reason"]] += 1

    if policy == "strict_pool":
        # Additionally require every pooled candidate to be judged.
        kept = []
        for record in per_query:
            query_id = record["query_id"]
            pool = pool_ids.get(query_id, set())
            if not pool <= set(qrels.get(query_id, {})):
                excluded["incomplete_pool"] += 1
                continue
            kept.append(record)
        per_query = kept

    if not per_query:
        return {
            "status": "blocked",
            "reason": (
                "no query has complete human judgments for this configuration"
                if not excluded
                else f"no evaluable queries: {dict(excluded)}"
            ),
            "policy": policy,
            "num_queries": len(query_ids),
            "num_evaluated": 0,
            "excluded": dict(excluded),
            "metrics": {},
            "per_query": [],
        }

    metric_names = ["mrr", "map", "num_relevant_judged", "num_retrieved"]
    metric_names += [f"{name}@{k}" for k in k_values for name in ("recall", "precision", "ndcg")]

    macro = {}
    for name in metric_names:
        values = [record[name] for record in per_query if name in record]
        macro[name] = sum(values) / len(values) if values else None

    return {
        "status": "ok",
        "reason": None,
        "policy": policy,
        "num_queries": len(query_ids),
        "num_evaluated": len(per_query),
        "excluded": dict(excluded),
        "metrics": macro,
        "per_query": per_query,
    }


def coverage_by_query(
    query_ids: list[str],
    candidates_by_query: dict[str, list[dict]],
    judged_by_query: dict[str, set[str]],
) -> dict:
    """Pool coverage per query: how much of the pool has a human judgment."""
    per_query = []
    for query_id in query_ids:
        pool = candidates_by_query.get(query_id, [])
        pool_ids = {candidate["chunk_id"] for candidate in pool}
        judged = pool_ids & judged_by_query.get(query_id, set())
        per_query.append(
            {
                "query_id": query_id,
                "pool_size": len(pool_ids),
                "judged": len(judged),
                "pending": len(pool_ids) - len(judged),
                "coverage": (len(judged) / len(pool_ids)) if pool_ids else 0.0,
                "fully_judged": bool(pool_ids) and len(judged) == len(pool_ids),
            }
        )

    total_pool = sum(entry["pool_size"] for entry in per_query)
    total_judged = sum(entry["judged"] for entry in per_query)
    return {
        "num_queries": len(query_ids),
        "pool_candidates": total_pool,
        "judged_candidates": total_judged,
        "pending_candidates": total_pool - total_judged,
        "coverage": (total_judged / total_pool) if total_pool else 0.0,
        "fully_judged_queries": sum(1 for entry in per_query if entry["fully_judged"]),
        "queries_without_judgments": [
            entry["query_id"] for entry in per_query if entry["judged"] == 0
        ],
        "per_query": per_query,
    }


def metric_table(rows: Iterable[dict], k_values: Iterable[int] = K_VALUES) -> list[dict]:
    """Compact comparison rows for printing or reporting."""
    k_values = sorted(set(k_values))
    table = []
    for row in rows:
        metrics = row.get("metrics") or {}
        entry = {
            "experiment_id": row.get("experiment_id"),
            "strategy": row.get("chunking_strategy"),
            "config": row.get("retrieval_config"),
            "num_evaluated": row.get("num_evaluated"),
        }
        for name in ("mrr", "map"):
            entry[name] = metrics.get(name)
        for k in k_values:
            entry[f"recall@{k}"] = metrics.get(f"recall@{k}")
            entry[f"ndcg@{k}"] = metrics.get(f"ndcg@{k}")
        table.append(entry)
    return table

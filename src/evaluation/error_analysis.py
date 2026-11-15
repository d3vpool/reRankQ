"""
Retrieval error analysis.

Explains per-query behaviour: where the first relevant chunk appears, which
systems retrieve it and which do not, and whether reranking or the other
chunking strategy helps. Only judged chunks can be called relevant; unjudged
chunks are reported as unjudged, never as failures.
"""

from typing import Iterable, Optional

from src.evaluation.retrieval_metrics import RELEVANCE_THRESHOLD, is_relevant

COMPARISONS = (
    ("bm25", "dense"),
    ("bm25", "hybrid_rrf"),
    ("dense", "hybrid_rrf"),
    ("hybrid_rrf", "hybrid_rerank"),
)


def first_relevant_rank(
    ranked: list[dict],
    grades: dict[str, int],
    threshold: int = RELEVANCE_THRESHOLD,
) -> Optional[int]:
    for item in ranked:
        if is_relevant(grades.get(item["chunk_id"]), threshold):
            return item["rank"]
    return None


def query_outcome(
    query_id: str,
    ranked: list[dict],
    grades: dict[str, int],
    k: int = 10,
    threshold: int = RELEVANCE_THRESHOLD,
) -> dict:
    top = ranked[:k]
    ranked_ids = [item["chunk_id"] for item in top]
    judged = [chunk_id for chunk_id in ranked_ids if chunk_id in grades]
    unjudged = [chunk_id for chunk_id in ranked_ids if chunk_id not in grades]
    relevant_ranks = [
        item["rank"]
        for item in top
        if is_relevant(grades.get(item["chunk_id"]), threshold)
    ]
    return {
        "query_id": query_id,
        "hit_at_k": bool(relevant_ranks),
        "first_relevant_rank": min(relevant_ranks) if relevant_ranks else None,
        "judged_at_k": len(judged),
        "unjudged_at_k": len(unjudged),
        "coverage_at_k": (len(judged) / len(top)) if top else 0.0,
        "top1_relevant": bool(
            top and is_relevant(grades.get(top[0]["chunk_id"]), threshold)
        ),
        "has_judged_relevant": any(
            is_relevant(grade, threshold) for grade in grades.values()
        ),
    }


def _rows_by_query(rows: Iterable[dict]) -> dict[str, dict]:
    return {row["query_id"]: row for row in rows}


def compare_configuration(
    rows_a: list[dict],
    rows_b: list[dict],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
    label_a: str = "a",
    label_b: str = "b",
    examples: int = 5,
) -> dict:
    """
    Compare two runs query by query.

    Classification uses the first relevant rank: a query is "improved" when the
    first relevant chunk moves to a better rank, not merely when a metric rises.
    Queries without any judged relevant chunk are excluded from the comparison
    and counted separately.
    """
    by_a = _rows_by_query(rows_a)
    by_b = _rows_by_query(rows_b)
    shared = sorted(set(by_a) & set(by_b))

    counts = {
        "both_hit": 0,
        "only_a": 0,
        "only_b": 0,
        "neither": 0,
        "improved_by_b": 0,
        "worsened_by_b": 0,
        "unchanged": 0,
        "excluded_no_judged_relevant": 0,
        "unjudged_at_k": 0,
    }
    improved_examples, regressed_examples, failure_examples = [], [], []

    for query_id in shared:
        grades = qrels.get(query_id, {})
        outcome_a = query_outcome(query_id, by_a[query_id]["ranked"], grades, k)
        outcome_b = query_outcome(query_id, by_b[query_id]["ranked"], grades, k)
        counts["unjudged_at_k"] += outcome_a["unjudged_at_k"] + outcome_b["unjudged_at_k"]

        if not outcome_a["has_judged_relevant"]:
            counts["excluded_no_judged_relevant"] += 1
            continue

        if outcome_a["hit_at_k"] and outcome_b["hit_at_k"]:
            counts["both_hit"] += 1
        elif outcome_a["hit_at_k"]:
            counts["only_a"] += 1
        elif outcome_b["hit_at_k"]:
            counts["only_b"] += 1
        else:
            counts["neither"] += 1

        rank_a, rank_b = outcome_a["first_relevant_rank"], outcome_b["first_relevant_rank"]
        if rank_b is not None and (rank_a is None or rank_b < rank_a):
            counts["improved_by_b"] += 1
            if len(improved_examples) < examples:
                improved_examples.append(
                    _example(query_id, outcome_a, outcome_b, by_b[query_id], grades, label_a, label_b)
                )
        elif rank_a is not None and (rank_b is None or rank_b > rank_a):
            counts["worsened_by_b"] += 1
            if len(regressed_examples) < examples:
                regressed_examples.append(
                    _example(query_id, outcome_a, outcome_b, by_a[query_id], grades, label_a, label_b)
                )
        else:
            counts["unchanged"] += 1

        if not outcome_a["hit_at_k"] and not outcome_b["hit_at_k"] and len(failure_examples) < examples:
            failure_examples.append(
                _example(query_id, outcome_a, outcome_b, by_b[query_id], grades, label_a, label_b)
            )

    return {
        "comparison": f"{label_a} vs {label_b}",
        "label_a": label_a,
        "label_b": label_b,
        "k": k,
        "num_queries": len(shared),
        "counts": counts,
        "hypothesis": (
            f"{label_b} changes the first relevant rank on "
            f"{counts['improved_by_b'] + counts['worsened_by_b']} of "
            f"{len(shared)} queries"
        ),
        "improved_examples": improved_examples,
        "regressed_examples": regressed_examples,
        "failure_examples": failure_examples,
    }


def _example(query_id, outcome_a, outcome_b, row, grades, label_a, label_b) -> dict:
    best = None
    for item in row["ranked"]:
        if item["chunk_id"] in grades:
            best = item
            break
    return {
        "query_id": query_id,
        "query": row.get("query"),
        f"first_relevant_rank_{label_a}": outcome_a["first_relevant_rank"],
        f"first_relevant_rank_{label_b}": outcome_b["first_relevant_rank"],
        "example_chunk": (
            {
                "chunk_id": best["chunk_id"],
                "doc_id": best.get("doc_id"),
                "rank": best["rank"],
                "grade": grades.get(best["chunk_id"]),
            }
            if best
            else None
        ),
    }


def analyze_experiments(runs: list[dict], k: int = 10) -> dict:
    """
    Compare the standard pairs inside each chunking strategy, and the two
    strategies for each configuration.

    ``runs`` items need ``chunking_strategy``, ``retrieval_config``, ``rows``
    and ``qrels``.
    """
    lookup = {
        (run["chunking_strategy"], run["retrieval_config"]): run for run in runs
    }
    strategies = sorted({run["chunking_strategy"] for run in runs})
    comparisons = []

    for strategy in strategies:
        for config_a, config_b in COMPARISONS:
            run_a = lookup.get((strategy, config_a))
            run_b = lookup.get((strategy, config_b))
            if not run_a or not run_b:
                continue
            comparisons.append(
                compare_configuration(
                    run_a["rows"],
                    run_b["rows"],
                    run_a.get("qrels", {}),
                    k=k,
                    label_a=config_a,
                    label_b=config_b,
                )
            )

    configs = sorted({run["retrieval_config"] for run in runs})
    if len(strategies) == 2:
        for config_name in configs:
            run_a = lookup.get((strategies[0], config_name))
            run_b = lookup.get((strategies[1], config_name))
            if not run_a or not run_b:
                continue
            comparisons.append(
                compare_configuration(
                    run_a["rows"],
                    run_b["rows"],
                    run_a.get("qrels", {}),
                    k=k,
                    label_a=strategies[0],
                    label_b=strategies[1],
                )
            )

    return {
        "k": k,
        "num_comparisons": len(comparisons),
        "comparisons": comparisons,
        "note": (
            "Counts come from first-relevant-rank changes over judged chunks. "
            "Queries with no judged relevant chunk are excluded, not scored."
        ),
    }

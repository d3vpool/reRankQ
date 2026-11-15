"""
Ablation variants around the production retrieval pipeline.

Only parameters the code can genuinely change at query time are varied here.
Chunk size and overlap are *not* independently configurable at evaluation time:
they are baked into the frozen chunk set and corpus manifest, so changing them
requires re-chunking and a new snapshot. That limitation is reported instead of
being simulated.

Parameter selection must use the development split; ``run_ablation`` refuses the
test split so tuning cannot be done against held-out data.
"""

from src.evaluation.experiments import build_specs
from src.evaluation.retrieval_metrics import K_VALUES, evaluate_run

ABLATION_FORMAT_VERSION = 1

FUSION_VARIANTS = (
    {"variant_id": "rrf_k60", "rrf_k": 60},
    {"variant_id": "rrf_k10", "rrf_k": 10},
    {"variant_id": "rrf_k30_weights_70_30", "rrf_k": 30, "bm25_weight": 0.7, "dense_weight": 0.3},
)

DEPTH_VARIANTS = (
    {"variant_id": "candidates_10", "bm25_candidates": 10, "dense_candidates": 10},
    {"variant_id": "candidates_20", "bm25_candidates": 20, "dense_candidates": 20},
    {"variant_id": "candidates_40", "bm25_candidates": 40, "dense_candidates": 40},
)

RERANK_DEPTH_VARIANTS = (
    {"variant_id": "rerank_from_6", "fusion_candidates": 6},
    {"variant_id": "rerank_from_20", "fusion_candidates": 20},
)

UNSUPPORTED_DIMENSIONS = {
    "chunk_size": (
        "chunk_size/chunk_overlap are fixed by the frozen chunk set and corpus "
        "manifest; changing them needs re-chunking and a new corpus snapshot, so "
        "no chunk-size ablation is reported here"
    ),
    "bm25_k1_b": (
        "BM25 k1/b are read from the index build configuration; varying them at "
        "query time would not reflect a real index, so they are not ablated"
    ),
}


def ablation_specs(strategy: str) -> list[dict]:
    """Every supported ablation variant for one chunking strategy."""
    specs = [
        {
            "dimension": "fusion",
            "variant_id": variant["variant_id"],
            "retrieval_config": "hybrid_rrf",
            "overrides": {k: v for k, v in variant.items() if k != "variant_id"},
        }
        for variant in FUSION_VARIANTS
    ]
    specs += [
        {
            "dimension": "candidate_depth",
            "variant_id": variant["variant_id"],
            "retrieval_config": "hybrid_rrf",
            "overrides": {k: v for k, v in variant.items() if k != "variant_id"},
        }
        for variant in DEPTH_VARIANTS
    ]
    specs += [
        {
            "dimension": "rerank_depth",
            "variant_id": variant["variant_id"],
            "retrieval_config": "hybrid_rerank",
            "overrides": {k: v for k, v in variant.items() if k != "variant_id"},
        }
        for variant in RERANK_DEPTH_VARIANTS
    ]
    for spec in specs:
        spec["chunking_strategy"] = strategy
    return specs


def run_ablation(
    runner,
    *,
    strategy: str,
    split: str = "dev",
    query_set: str = "approved",
    descriptive: bool = False,
    k_values=K_VALUES,
) -> dict:
    """
    Run every supported variant on one strategy and split.

    The test split is refused: ablations exist to select parameters on
    development data, and the chosen configuration is evaluated on test once.
    """
    if split == "test":
        raise ValueError(
            "ablation and parameter tuning must use the development split; "
            "evaluate the chosen configuration on test exactly once"
        )
    if query_set == "approved":
        problems = runner.verify(require_approved=True)
        if problems:
            from src.evaluation.experiments import EvaluationBlocked

            raise EvaluationBlocked("; ".join(problems))

    queries = runner.queries(query_set=query_set, split=split)
    if not queries:
        from src.evaluation.experiments import EvaluationBlocked

        raise EvaluationBlocked(f"no queries for query_set={query_set!r} split={split!r}")

    results = []
    for spec in ablation_specs(strategy):
        rows = runner.run_config(
            strategy,
            spec["retrieval_config"],
            queries,
            retrieval_overrides=spec["overrides"],
        )
        if descriptive or query_set != "approved":
            metrics = {
                "status": "blocked",
                "reason": "descriptive ablation: requires approved queries and complete judgments",
            }
        else:
            grades = runner.judged_grades(strategy)
            ranked_ids = {
                row["query_id"]: [item["chunk_id"] for item in row["ranked"]]
                for row in rows
            }
            pool_ids = {
                query_id: {candidate["chunk_id"] for candidate in candidates}
                for query_id, candidates in runner.candidates_by_query(strategy).items()
            }
            metrics = evaluate_run(
                [query["query_id"] for query in queries],
                ranked_ids,
                grades,
                pool_ids,
                k_values=k_values,
            )
        results.append(
            {
                "dimension": spec["dimension"],
                "variant_id": spec["variant_id"],
                "chunking_strategy": strategy,
                "retrieval_config": spec["retrieval_config"],
                "overrides": spec["overrides"],
                "num_queries": len(queries),
                "metrics": metrics,
            }
        )

    return {
        "ablation_format_version": ABLATION_FORMAT_VERSION,
        "strategy": strategy,
        "split": split,
        "query_set": query_set,
        "descriptive": descriptive or query_set != "approved",
        "corpus_version": runner.corpus_version,
        "query_version": runner.query_version,
        "split_version": runner.split_version,
        "variants": results,
        "unsupported": UNSUPPORTED_DIMENSIONS,
        "note": (
            "Only development-split results may guide parameter choices. "
            "Unsupported dimensions are reported, not simulated."
        ),
    }


def final_configurations(strategies, configs) -> list[dict]:
    """The selected configurations for a single held-out test evaluation."""
    return build_specs(
        corpus_version="unused",
        query_version="unused",
        split_version="unused",
        split="test",
        strategies=strategies,
        configs=configs,
    )

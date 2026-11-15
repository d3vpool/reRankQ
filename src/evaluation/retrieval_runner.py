"""
Reproducible retrieval experiment runner.

Runs one or more of the eight primary configurations (2 chunking strategies x 4
retrieval systems), records rankings, and computes metrics only when human
judgments cover the retrieved cutoffs.

The runner never fabricates a metric: when judgments are missing it writes a
``blocked`` aggregate with the reason, and it refuses to overwrite an existing
experiment directory.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.evaluation.corpus_manifest import load_manifest
from src.evaluation.eval_dataset import load_jsonl
from src.evaluation.experiments import (
    CHUNKING_STRATEGIES,
    RETRIEVAL_CONFIGS,
    EvaluationBlocked,
    build_specs,
    default_parameters,
    environment_snapshot,
    experiment_dir,
    prepare_experiment_dir,
    verify_dataset,
    write_json,
    write_jsonl,
)
from src.evaluation.judgments import JudgmentStore
from src.evaluation.retrieval_metrics import (
    K_VALUES,
    coverage_by_query,
    evaluate_run,
)
from src.ingestion.chunker import load_chunks
from src.ingestion.crawler import load_documents
from src.utils.config import get_config

DEFAULT_RESULTS_ROOT = "results/experiments"


class RetrievalRunner:
    """Loads evaluation data and runs retrieval configurations."""

    def __init__(
        self,
        eval_dir: str | Path = "data/eval",
        indexes_dir: str | Path = "data/indexes",
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        top_k: int = 10,
        results_root: str | Path = DEFAULT_RESULTS_ROOT,
    ):
        self.eval_dir = Path(eval_dir)
        self.indexes_dir = Path(indexes_dir)
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.top_k = top_k
        self.results_root = Path(results_root)
        self.config = get_config()

        self.documents = load_documents(str(self.raw_dir))
        self.manifest = load_manifest(self.eval_dir / "corpus_manifest.json")
        self.approved = load_jsonl(self.eval_dir / "approved_queries.jsonl")
        self.candidates = load_jsonl(self.eval_dir / "candidates.jsonl")
        self.provenance = load_jsonl(self.eval_dir / "candidate_provenance.jsonl")
        self.judgment_store = JudgmentStore(self.eval_dir / "judgments.jsonl")
        self.split_meta = self._load_split_meta()
        self._retrievers: dict[str, dict] = {}
        self._reranker = None

    # -- data ---------------------------------------------------------------

    def _load_split_meta(self) -> Optional[dict]:
        path = self.eval_dir / "split_assignment.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    @property
    def corpus_version(self) -> str:
        return self.manifest["corpus_version"]

    @property
    def query_version(self) -> str:
        meta_path = self.eval_dir / "approved_dataset_meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text()).get("approved_version", "none")
        return "none"

    @property
    def split_version(self) -> str:
        if self.split_meta:
            return self.split_meta.get("split_version", "none")
        return "none"

    def verify(self, require_approved: bool = True) -> list[str]:
        return verify_dataset(
            eval_dir=self.eval_dir,
            documents=self.documents,
            manifest=self.manifest,
            approved_queries=self.approved,
            split_meta=self.split_meta,
            require_approved=require_approved,
        )

    def queries(self, query_set: str = "approved", split: str = "all") -> list[dict]:
        """
        Queries for a run.

        ``approved`` uses the human-approved set only. ``candidates`` is for
        pipeline validation and latency work; such runs are descriptive and
        never produce metrics.
        """
        if query_set == "approved":
            rows = [
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "category": row["category"],
                    "split": row["split"],
                }
                for row in self.approved
            ]
        elif query_set == "candidates":
            rows = [
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "category": row["category"],
                    "split": row.get("split"),
                }
                for row in load_jsonl(self.eval_dir / "queries.jsonl")
            ]
        else:
            raise ValueError(f"unknown query set {query_set!r}")

        if split != "all":
            rows = [row for row in rows if row["split"] == split]
        return rows

    # -- judgments ----------------------------------------------------------

    def judged_grades(self, strategy: str) -> dict[str, dict[str, int]]:
        """
        Latest human grades per (query, chunk) for one chunking strategy.

        Pending candidates never appear here, so they are indistinguishable from
        unjudged rather than being treated as grade 0.
        """
        grades: dict[str, dict[str, int]] = {}
        for candidate_id, record in self.judgment_store.latest_by_candidate().items():
            if record.get("judgment_status") != "judged":
                continue
            if record.get("chunking_strategy") != strategy:
                continue
            query_id = record["query_id"]
            grades.setdefault(query_id, {})[record["chunk_id"]] = int(record["relevance_grade"])
        return grades

    def candidates_by_query(self, strategy: str) -> dict[str, list[dict]]:
        buckets: dict[str, list[dict]] = {}
        for candidate in self.candidates:
            if candidate["chunking_strategy"] != strategy:
                continue
            buckets.setdefault(candidate["query_id"], []).append(candidate)
        return buckets

    # -- retrievers ---------------------------------------------------------

    def load_retrievers(self, strategy: str) -> dict:
        """Load the strategy's indexes once, verifying corpus and strategy."""
        if strategy in self._retrievers:
            return self._retrievers[strategy]

        from src.evaluation.build_retrievers import load_strategy_retrievers

        if self._reranker is None:
            from src.reranking.reranker import CrossEncoderReranker

            self._reranker = CrossEncoderReranker()

        retrievers = load_strategy_retrievers(
            strategy=strategy,
            indexes_dir=self.indexes_dir,
            corpus_version=self.corpus_version,
            reranker=self._reranker,
        )
        self._retrievers[strategy] = retrievers
        return retrievers

    # -- execution ----------------------------------------------------------

    def run_config(
        self,
        strategy: str,
        config_name: str,
        queries: list[dict],
        retrieval_overrides: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Run one configuration over the queries, recording ranks and latency."""
        retrievers = self.load_retrievers(strategy)
        top_k = top_k or self.top_k
        overrides = retrieval_overrides or {}
        rows = []

        for query in queries:
            started = time.perf_counter()
            ranked = self._retrieve(retrievers, config_name, query["query"], top_k, overrides)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            rows.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "split": query.get("split"),
                    "chunking_strategy": strategy,
                    "retrieval_config": config_name,
                    "latency_ms": elapsed_ms,
                    "ranked": [
                        {
                            "rank": item["rank"],
                            "chunk_id": item["chunk_id"],
                            "doc_id": item.get("doc_id"),
                            "score": item.get("score"),
                            "rerank_score": item.get("rerank_score"),
                        }
                        for item in ranked
                    ],
                }
            )
        return rows

    def _retrieve(self, retrievers: dict, config_name: str, query: str, top_k: int, overrides: dict) -> list[dict]:
        if config_name == "bm25":
            return retrievers["bm25"].retrieve(query, top_k)
        if config_name == "dense":
            return retrievers["dense"].retrieve(query, top_k)
        if config_name == "hybrid_rrf":
            return retrievers["hybrid"].retrieve(
                query,
                top_k,
                bm25_top_k=overrides.get("bm25_candidates"),
                dense_top_k=overrides.get("dense_candidates"),
                k=overrides.get("rrf_k", self.config.retrieval.hybrid.k),
                bm25_weight=overrides.get("bm25_weight", self.config.retrieval.hybrid.bm25_weight),
                dense_weight=overrides.get("dense_weight", self.config.retrieval.hybrid.dense_weight),
            )
        if config_name == "hybrid_rerank":
            depth = overrides.get("fusion_candidates", max(top_k, self.config.retrieval.dense.top_k))
            fused = retrievers["hybrid"].retrieve(query, depth)
            return retrievers["reranker"].rerank(query, fused, top_k=top_k)
        raise ValueError(f"unknown retrieval configuration {config_name!r}")

    # -- artifacts ----------------------------------------------------------

    def run_matrix(
        self,
        *,
        query_set: str = "approved",
        split: str = "all",
        strategies=CHUNKING_STRATEGIES,
        configs=RETRIEVAL_CONFIGS,
        descriptive: bool = False,
        parameters: Optional[dict] = None,
    ) -> dict:
        """
        Execute the matrix and write one artifact directory per configuration.

        ``descriptive=True`` (candidate queries) records rankings, latency and
        coverage but never metrics.
        """
        if query_set == "approved":
            problems = self.verify(require_approved=True)
            if problems:
                raise EvaluationBlocked("; ".join(problems))
        else:
            if not descriptive:
                raise EvaluationBlocked(
                    "candidate queries are not human-approved; pass descriptive=True "
                    "for a run that records rankings only"
                )
            problems = self.verify(require_approved=False)
            if problems:
                raise EvaluationBlocked("; ".join(problems))

        queries = self.queries(query_set=query_set, split=split)
        if not queries:
            raise EvaluationBlocked(f"no queries for query_set={query_set!r} split={split!r}")

        parameters = parameters or {}
        base_params = parameters.get("default", default_parameters(self.config, self.top_k))
        specs = build_specs(
            corpus_version=self.corpus_version,
            query_version=self.query_version,
            split_version=self.split_version,
            split=split,
            strategies=strategies,
            configs=configs,
            parameters={name: parameters.get(name, base_params) for name in configs},
        )

        environment = environment_snapshot()
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        runs = []

        for spec in specs:
            strategy = spec["chunking_strategy"]
            config_name = spec["retrieval_config"]
            experiment_dir_path = prepare_experiment_dir(self.results_root, spec["experiment_id"])

            init_started = time.perf_counter()
            self.load_retrievers(strategy)
            init_seconds = time.perf_counter() - init_started

            rows = self.run_config(
                strategy,
                config_name,
                queries,
                retrieval_overrides=parameters.get(config_name, {}),
            )

            coverage = coverage_by_query(
                [query["query_id"] for query in queries],
                self.candidates_by_query(strategy),
                {
                    query_id: set(grades)
                    for query_id, grades in self.judged_grades(strategy).items()
                },
            )

            if descriptive:
                metrics = {
                    "status": "blocked",
                    "reason": (
                        "descriptive run on candidate queries: metrics require "
                        "human-approved queries and complete human relevance judgments"
                    ),
                    "num_queries": len(queries),
                }
            else:
                grades = self.judged_grades(strategy)
                metrics = evaluate_run(
                    [query["query_id"] for query in queries],
                    {row["query_id"]: [item["chunk_id"] for item in row["ranked"]] for row in rows},
                    grades,
                    {
                        query_id: {candidate["chunk_id"] for candidate in candidates}
                        for query_id, candidates in self.candidates_by_query(strategy).items()
                    },
                    k_values=K_VALUES,
                )
                metrics.update(
                    {
                        "experiment_id": spec["experiment_id"],
                        "chunking_strategy": strategy,
                        "retrieval_config": config_name,
                    }
                )

            latencies = sorted(row["latency_ms"] for row in rows)
            latency_summary = summarize_latencies(latencies)
            latency_summary["index_initialisation_seconds"] = init_seconds

            experiment_record = {
                "experiment_format_version": 1,
                "experiment_id": spec["experiment_id"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "completed" if metrics.get("status") != "blocked" else "completed_no_metrics",
                "evaluation_status": (
                    "descriptive_only" if descriptive else metrics.get("status")
                ),
                "corpus_version": self.corpus_version,
                "query_version": self.query_version,
                "split_version": self.split_version,
                "split": split,
                "query_set": query_set,
                "num_queries": len(queries),
                "chunking_strategy": strategy,
                "retrieval_config": config_name,
                "parameters": spec["parameters"],
                "retrieval_overrides": parameters.get(config_name, {}),
                "embedding_model": self.config.embedding.model_name,
                "reranker_model": self.config.retrieval.reranking.model_name,
                "random_seed": 13,
                "top_k": self.top_k,
                "judgment_policy": "strict_topk",
                "batch_id": batch_id,
                "dataset": {
                    "eval_dir": str(self.eval_dir),
                    "num_documents": len(self.documents),
                    "num_candidates": len(self.candidates),
                    "num_judgments": len(self.judgment_store.records()),
                },
            }

            write_json(experiment_dir_path / "experiment.json", experiment_record)
            write_jsonl(experiment_dir_path / "query_results.jsonl", rows)
            write_json(experiment_dir_path / "aggregate_metrics.json", metrics)
            write_json(experiment_dir_path / "latency.json", latency_summary)
            write_json(experiment_dir_path / "coverage_report.json", coverage)
            write_json(experiment_dir_path / "environment.json", environment)
            write_json(
                experiment_dir_path / "judgments_snapshot.json",
                {
                    "num_judged_pairs": sum(len(g) for g in self.judged_grades(strategy).values()),
                    "strategy": strategy,
                    "policy": "strict_topk",
                },
            )

            runs.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "experiment_dir": str(experiment_dir_path),
                    "chunking_strategy": strategy,
                    "retrieval_config": config_name,
                    "metrics": metrics,
                    "latency": latency_summary,
                    "coverage": coverage,
                    "rows": rows,
                }
            )

        summary = {
            "batch_id": batch_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query_set": query_set,
            "split": split,
            "descriptive": descriptive,
            "corpus_version": self.corpus_version,
            "query_version": self.query_version,
            "split_version": self.split_version,
            "num_configurations": len(runs),
            "num_queries": len(queries),
            "experiments": [
                {
                    "experiment_id": run["experiment_id"],
                    "chunking_strategy": run["chunking_strategy"],
                    "retrieval_config": run["retrieval_config"],
                    "num_evaluated": run["metrics"].get("num_evaluated"),
                    "metrics_status": run["metrics"].get("status"),
                }
                for run in runs
            ],
        }
        write_json(self.results_root / "batches" / f"{batch_id}.json", summary)
        return {"summary": summary, "runs": runs}


def summarize_latencies(latencies_ms: list[float]) -> dict:
    """p50/p95/mean/stdev over measured per-query latencies."""
    import statistics

    if not latencies_ms:
        return {"count": 0}
    return {
        "count": len(latencies_ms),
        "mean_ms": round(statistics.fmean(latencies_ms), 3),
        "median_ms": round(statistics.median(latencies_ms), 3),
        "stdev_ms": round(statistics.pstdev(latencies_ms), 3) if len(latencies_ms) > 1 else 0.0,
        "p50_ms": round(_percentile(latencies_ms, 50), 3),
        "p95_ms": round(_percentile(latencies_ms, 95), 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

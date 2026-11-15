"""
Latency benchmarking for the retrieval configurations.

Model loading, index loading and first-touch warm-up are measured separately
from steady-state retrieval latency, so a cold start is never reported as
retrieval cost. Benchmarks need no LLM provider.

Latency is inherently machine-specific: results record the observed hardware
and are not comparable across machines without re-running them.
"""

import statistics
import time
from typing import Iterable, Optional

from src.evaluation.retrieval_runner import summarize_latencies

DEFAULT_WARMUP = 3
DEFAULT_ITERATIONS = 10


def benchmark_configuration(
    runner,
    strategy: str,
    config_name: str,
    queries: list[str],
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    overrides: Optional[dict] = None,
) -> dict:
    """
    Time one configuration over a fixed query workload.

    The first call after loading is recorded as ``cold_first_call_ms`` and is
    excluded from the steady-state statistics, as are ``warmup`` full passes.
    """
    started = time.perf_counter()
    retrievers = runner.load_retrievers(strategy)
    index_init_seconds = time.perf_counter() - started

    if not queries:
        return {
            "strategy": strategy,
            "retrieval_config": config_name,
            "status": "no_queries",
        }

    cold_started = time.perf_counter()
    runner._retrieve(retrievers, config_name, queries[0], runner.top_k, overrides or {})
    cold_first_call_ms = (time.perf_counter() - cold_started) * 1000.0

    for _ in range(warmup):
        for query in queries:
            runner._retrieve(retrievers, config_name, query, runner.top_k, overrides or {})

    samples_ms: list[float] = []
    for _ in range(iterations):
        for query in queries:
            call_started = time.perf_counter()
            runner._retrieve(retrievers, config_name, query, runner.top_k, overrides or {})
            samples_ms.append((time.perf_counter() - call_started) * 1000.0)

    return {
        "strategy": strategy,
        "retrieval_config": config_name,
        "status": "ok",
        "index_init_seconds": round(index_init_seconds, 4),
        "cold_first_call_ms": round(cold_first_call_ms, 3),
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "num_queries": len(queries),
        "num_measurements": len(samples_ms),
        **summarize_latencies(sorted(samples_ms)),
    }


def benchmark_matrix(
    runner,
    queries: Iterable[dict],
    strategies: Iterable[str],
    configs: Iterable[str],
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
) -> list[dict]:
    """Benchmark every (strategy, configuration) pair on the same workload."""
    texts = [query["query"] for query in queries]
    results = []
    for strategy in strategies:
        for config_name in configs:
            results.append(
                benchmark_configuration(
                    runner,
                    strategy=strategy,
                    config_name=config_name,
                    queries=texts,
                    warmup=warmup,
                    iterations=iterations,
                )
            )
    return results


def end_to_end_summary(records: list[dict]) -> dict:
    """Aggregate the measured configurations into a compact comparison table."""
    rows = []
    for record in records:
        if record.get("status") != "ok":
            continue
        rows.append(
            {
                "strategy": record["strategy"],
                "retrieval_config": record["retrieval_config"],
                "p50_ms": record.get("p50_ms"),
                "p95_ms": record.get("p95_ms"),
                "mean_ms": record.get("mean_ms"),
                "stdev_ms": record.get("stdev_ms"),
                "num_measurements": record.get("num_measurements"),
            }
        )
    return {
        "measurements": rows,
        "methodology": {
            "warmup_iterations_excluded": DEFAULT_WARMUP,
            "cold_start_reported_separately": True,
            "statistics": "per-query retrieval latency in milliseconds",
        },
    }

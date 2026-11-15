#!/usr/bin/env python
"""
Latency benchmark for every retrieval configuration.

Model loading, index loading, the first call and warm-up passes are measured
separately from steady-state retrieval latency. No LLM provider is required.

Writes a timestamped artifact under results/benchmarks/; existing benchmark
files are never overwritten.

Usage:
    python scripts/run_latency_benchmark.py
    python scripts/run_latency_benchmark.py --iterations 20 --warmup 5
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmark import benchmark_matrix, end_to_end_summary  # noqa: E402
from src.evaluation.experiments import (  # noqa: E402
    CHUNKING_STRATEGIES,
    RETRIEVAL_CONFIGS,
    environment_snapshot,
)
from src.evaluation.retrieval_runner import RetrievalRunner  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--output-root", default="results/benchmarks")
    parser.add_argument("--strategies", default=",".join(CHUNKING_STRATEGIES))
    parser.add_argument("--configs", default=",".join(RETRIEVAL_CONFIGS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="cap the query workload")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = RetrievalRunner(eval_dir=args.eval_dir)

    queries = runner.queries(query_set="candidates", split="all")
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("No queries available for the benchmark workload", file=sys.stderr)
        return 2

    print("Latency benchmark")
    print("=================")
    print(f"workload queries : {len(queries)} (candidate query texts; latency only)")
    print(f"warm-up passes   : {args.warmup}")
    print(f"measured passes  : {args.iterations}")
    print()

    records = benchmark_matrix(
        runner,
        queries,
        strategies=args.strategies.split(","),
        configs=args.configs.split(","),
        warmup=args.warmup,
        iterations=args.iterations,
    )

    header = (
        f"{'strategy':<16}{'config':<14}{'p50 ms':>9}{'p95 ms':>9}"
        f"{'mean ms':>9}{'stdev':>8}{'cold ms':>9}"
    )
    print(header)
    for record in records:
        if record.get("status") != "ok":
            print(f"{record['strategy']:<16}{record['retrieval_config']:<14}  no queries")
            continue
        print(
            f"{record['strategy']:<16}{record['retrieval_config']:<14}"
            f"{record['p50_ms']:>9.2f}{record['p95_ms']:>9.2f}"
            f"{record['mean_ms']:>9.2f}{record['stdev_ms']:>8.2f}"
            f"{record['cold_first_call_ms']:>9.1f}"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root) / f"latency-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    environment = environment_snapshot()
    payload = {
        "benchmark_format_version": 1,
        "created_at": environment["timestamp"],
        "corpus_version": runner.corpus_version,
        "query_source": "candidate queries (latency workload only)",
        "num_queries": len(queries),
        "top_k": runner.top_k,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "environment": environment,
        "records": records,
        "summary": end_to_end_summary(records),
        "notes": [
            "Per-query retrieval latency after warm-up; model and index loading reported separately.",
            "Absolute latencies are machine-specific and must be re-measured elsewhere.",
            "No relevance judgments are involved, so this benchmark is not an accuracy result.",
        ],
    }
    (output_dir / "latency.json").write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"Wrote {output_dir / 'latency.json'}")
    print(f"environment: {environment.get('cpu_model')} · {environment.get('cpu_count')} cores · "
          f"{environment.get('memory_gb')} GB RAM · device {environment.get('device')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

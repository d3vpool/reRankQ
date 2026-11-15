#!/usr/bin/env python
"""
Statistical comparison and error analysis for a completed experiment batch.

Loads stored artifacts only: nothing is re-run and no metric is recomputed from
scratch. Comparisons are paired by query and resampled by query.

Usage:
    python scripts/analyze_experiments.py --batch 20260101T120000Z
    python scripts/analyze_experiments.py --latest --metric ndcg@5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.error_analysis import analyze_experiments  # noqa: E402
from src.evaluation.eval_dataset import load_jsonl  # noqa: E402
from src.evaluation.experiments import write_json  # noqa: E402
from src.evaluation.statistics import DEFAULT_ITERATIONS, DEFAULT_SEED, compare_runs  # noqa: E402

COMPARISONS = (
    ("bm25", "dense"),
    ("bm25", "hybrid_rrf"),
    ("dense", "hybrid_rrf"),
    ("hybrid_rrf", "hybrid_rerank"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/experiments")
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--batch", default="")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--metric", default="ndcg@5")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def load_batch(root: Path, batch_id: str) -> dict:
    path = root / "batches" / f"{batch_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No batch summary at {path}")
    return json.loads(path.read_text())


def load_runs(root: Path, batch: dict) -> list[dict]:
    runs = []
    for entry in batch["experiments"]:
        directory = root / entry["experiment_id"]
        experiment = json.loads((directory / "experiment.json").read_text())
        metrics = json.loads((directory / "aggregate_metrics.json").read_text())
        rows = load_jsonl(directory / "query_results.jsonl")
        runs.append(
            {
                "experiment_id": entry["experiment_id"],
                "chunking_strategy": experiment["chunking_strategy"],
                "retrieval_config": experiment["retrieval_config"],
                "metrics": metrics,
                "rows": rows,
                "query_set": experiment.get("query_set"),
                "split": experiment.get("split"),
            }
        )
    return runs


def load_qrels(eval_dir: Path) -> dict[str, dict[str, dict[str, int]]]:
    """{strategy: {query_id: {chunk_id: grade}}} from judged records only."""
    latest: dict[str, dict] = {}
    for record in load_jsonl(eval_dir / "judgments.jsonl"):
        latest[record["candidate_id"]] = record

    qrels: dict[str, dict[str, dict[str, int]]] = {}
    for record in latest.values():
        if record.get("judgment_status") != "judged":
            continue
        strategy = record["chunking_strategy"]
        qrels.setdefault(strategy, {}).setdefault(record["query_id"], {})[
            record["chunk_id"]
        ] = int(record["relevance_grade"])
    return qrels


def main() -> int:
    args = parse_args()
    root = Path(args.results_root)
    batches = sorted((root / "batches").glob("*.json"))
    if not batches:
        print("No experiment batches found. Run scripts/run_retrieval_experiments.py first.")
        return 2

    batch_path = (
        root / "batches" / f"{args.batch}.json" if args.batch else batches[-1]
    )
    batch = json.loads(batch_path.read_text())
    batch_id = batch["batch_id"]

    runs = load_runs(root, batch)
    evaluated = [run for run in runs if run["metrics"].get("status") == "ok"]

    print(f"Batch {batch_id}: {len(runs)} experiments, {len(evaluated)} with metrics")
    if not evaluated:
        print()
        print("No statistical comparison was computed.")
        for run in runs[:2]:
            print(f"  {run['retrieval_config']}: {run['metrics'].get('reason')}")
        return 3

    qrels = load_qrels(Path(args.eval_dir))
    lookup = {(run["chunking_strategy"], run["retrieval_config"]): run for run in runs}

    written = 0
    for strategy in sorted({run["chunking_strategy"] for run in evaluated}):
        strategy_comparisons = []
        for config_a, config_b in COMPARISONS:
            run_a = lookup.get((strategy, config_a))
            run_b = lookup.get((strategy, config_b))
            if not run_a or not run_b:
                continue
            if run_a["metrics"].get("status") != "ok" or run_b["metrics"].get("status") != "ok":
                continue
            comparison = compare_runs(
                {"experiment_id": run_a["experiment_id"], "chunking_strategy": strategy,
                 "retrieval_config": config_a, "per_query": run_a["metrics"]["per_query"]},
                {"experiment_id": run_b["experiment_id"], "chunking_strategy": strategy,
                 "retrieval_config": config_b, "per_query": run_b["metrics"]["per_query"]},
                iterations=args.iterations,
                seed=args.seed,
            )
            strategy_comparisons.append(comparison)
            entry = strategy_comparisons[-1]["comparisons"][args.metric]
            print(
                f"{strategy} {config_a} vs {config_b} · {args.metric}: "
                f"{entry['mean_a']} -> {entry['mean_b']} "
                f"(paired n={entry['num_paired_queries']})"
            )
            if entry.get("power_warning"):
                print(f"  warning: {entry['power_warning']}")

        output = {
            "batch_id": batch_id,
            "strategy": strategy,
            "metric_reported": args.metric,
            "alpha": 0.05,
            "iterations": args.iterations,
            "seed": args.seed,
            "comparisons": strategy_comparisons,
        }
        for run in runs:
            if run["chunking_strategy"] != strategy:
                continue
            write_json(
                root / run["experiment_id"] / "statistical_comparisons.json",
                {**output, "experiment_id": run["experiment_id"]},
            )
            written += 1

    for run in runs:
        run["qrels"] = qrels.get(run["chunking_strategy"], {})
    error_report = analyze_experiments(runs, k=args.k)
    write_json(root / "analysis" / f"{batch_id}_error_analysis.json", error_report)
    print(f"\nerror analysis: {error_report['num_comparisons']} comparisons")
    print(f"Wrote {root / 'analysis' / f'{batch_id}_error_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

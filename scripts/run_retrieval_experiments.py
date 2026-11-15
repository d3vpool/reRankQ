#!/usr/bin/env python
"""
Run the retrieval experiment matrix.

Eight primary configurations (2 chunking strategies x 4 retrieval systems) are
identified by a hash of corpus/query/split versions and retrieval parameters.
Metrics are computed only when human judgments cover the retrieved cutoffs;
otherwise the aggregate is recorded as blocked with the reason.

Usage:
    # approved queries (requires human approval + judgments)
    python scripts/run_retrieval_experiments.py --split dev

    # pipeline validation on candidate queries: rankings only, no metrics
    python scripts/run_retrieval_experiments.py \
        --query-set candidates --descriptive --split all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.experiments import (  # noqa: E402
    CHUNKING_STRATEGIES,
    RETRIEVAL_CONFIGS,
    EvaluationBlocked,
)
from src.evaluation.retrieval_metrics import metric_table  # noqa: E402
from src.evaluation.retrieval_runner import RetrievalRunner  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--results-root", default="results/experiments")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--query-set", choices=("approved", "candidates"), default="approved")
    parser.add_argument("--descriptive", action="store_true")
    parser.add_argument("--strategies", default=",".join(CHUNKING_STRATEGIES))
    parser.add_argument("--configs", default=",".join(RETRIEVAL_CONFIGS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = RetrievalRunner(
        eval_dir=args.eval_dir,
        top_k=args.top_k,
        results_root=args.results_root,
    )

    print("Dataset")
    print("=======")
    print(f"corpus version   : {runner.corpus_version}")
    print(f"query version    : {runner.query_version}")
    print(f"split version    : {runner.split_version}")
    print(f"approved queries : {len(runner.approved)}")
    print(f"candidates       : {len(runner.candidates)}")
    print(f"judgments        : {len(runner.judgment_store.records())}")

    problems = runner.verify(require_approved=args.query_set == "approved")
    if problems:
        print()
        print("Cannot run a metric-producing experiment:")
        for problem in problems:
            print(f"  - {problem}")
        if not args.descriptive:
            print()
            print(
                "Re-run with --query-set candidates --descriptive to validate the "
                "pipeline without computing metrics."
            )
            return 2

    if args.dry_run:
        print("\ndry run: nothing executed or written")
        return 0

    limits = args.split == "test"
    if limits:
        print(
            "\nnote: test-split results are for final reporting only; do not use them "
            "to select retrieval parameters"
        )

    try:
        result = runner.run_matrix(
            query_set=args.query_set,
            split=args.split,
            strategies=args.strategies.split(","),
            configs=args.configs.split(","),
            descriptive=args.descriptive,
        )
    except EvaluationBlocked as error:
        print(f"\nBlocked: {error}")
        return 2

    summary = result["summary"]
    print()
    print(f"Batch {summary['batch_id']} · {summary['num_configurations']} configurations")
    print(f"queries: {summary['num_queries']} · set: {summary['query_set']} · split: {summary['split']}")

    rows = [
        {
            "experiment_id": run["experiment_id"],
            "chunking_strategy": run["chunking_strategy"],
            "retrieval_config": run["retrieval_config"],
            "num_evaluated": run["metrics"].get("num_evaluated"),
            "metrics": run["metrics"],
        }
        for run in result["runs"]
    ]
    table = metric_table(rows)
    if table:
        print()
        header = f"{'strategy':<16}{'config':<14}{'n':>4}{'MRR':>8}{'R@5':>8}{'nDCG@5':>9}"
        print(header)
        for row in table:
            def fmt(value):
                return f"{value:.3f}" if isinstance(value, float) else "—"
            print(
                f"{row['strategy']:<16}{row['config']:<14}{str(row['num_evaluated']):>4}"
                f"{fmt(row['mrr']):>8}{fmt(row['recall@5']):>8}{fmt(row['ndcg@5']):>9}"
            )

    statuses = {run["metrics"].get("status") for run in result["runs"]}
    if statuses == {"blocked"}:
        print()
        print("Metrics were NOT computed:")
        print(f"  {result['runs'][0]['metrics'].get('reason')}")

    print()
    for run in result["runs"]:
        print(f"  {run['experiment_id']}  {run['chunking_strategy']:<16} {run['retrieval_config']}")
    print(f"\nArtifacts: {args.results_root}/<experiment_id>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

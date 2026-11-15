#!/usr/bin/env python
"""
Run retrieval ablations on the development split.

Supported: fusion parameters (RRF k, weights), candidate depth, rerank depth.
Not supported and explicitly reported: chunk size/overlap and BM25 k1/b, which
are fixed by the frozen chunk set/index rather than by query-time parameters.

The test split is refused: parameter choices must come from development data.

Usage:
    python scripts/run_ablation.py --strategy structure_aware
    python scripts/run_ablation.py --descriptive --query-set candidates
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.ablation_runner import run_ablation  # noqa: E402
from src.evaluation.experiments import EvaluationBlocked  # noqa: E402
from src.evaluation.retrieval_runner import RetrievalRunner  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=("fixed_size", "structure_aware"))
    parser.add_argument("--split", default="dev", choices=("dev",))
    parser.add_argument("--query-set", default="approved", choices=("approved", "candidates"))
    parser.add_argument("--descriptive", action="store_true")
    parser.add_argument("--output-root", default="results/ablations")
    parser.add_argument("--eval-dir", default="data/eval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = RetrievalRunner(eval_dir=args.eval_dir)

    try:
        report = run_ablation(
            runner,
            strategy=args.strategy,
            split=args.split,
            query_set=args.query_set,
            descriptive=args.descriptive,
        )
    except EvaluationBlocked as error:
        print(f"Blocked: {error}")
        return 2
    except ValueError as error:
        print(f"Refused: {error}")
        return 2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root) / f"{args.strategy}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "ablation.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(f"Ablation · {args.strategy} · split {args.split}")
    print(f"queries: {report['variants'][0]['num_queries']} · descriptive: {report['descriptive']}")
    print()
    print(f"{'dimension':<18}{'variant':<26}{'status':<10}{'nDCG@5':>9}{'MRR':>8}")
    for variant in report["variants"]:
        metrics = variant["metrics"]
        if metrics.get("status") == "ok":
            print(
                f"{variant['dimension']:<18}{variant['variant_id']:<26}"
                f"{'ok':<10}{metrics['metrics'].get('ndcg@5', 0):>9.3f}"
                f"{metrics['metrics'].get('mrr', 0):>8.3f}"
            )
        else:
            print(
                f"{variant['dimension']:<18}{variant['variant_id']:<26}"
                f"{'blocked':<10}{'—':>9}{'—':>8}"
            )
    print()
    print("Unsupported dimensions:")
    for name, reason in report["unsupported"].items():
        print(f"  {name}: {reason}")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

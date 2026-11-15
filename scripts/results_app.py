#!/usr/bin/env python
"""
Read-only results browser for stored experiment artifacts.

Displays only what exists under results/: metric comparisons, latency
measurements and error-analysis examples. When no valid experiment has run it
says so instead of showing placeholder numbers.

Separate from scripts/review_queries.py and scripts/judge_relevance.py.

Run:
    streamlit run scripts/results_app.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src.evaluation.eval_dataset import load_jsonl  # noqa: E402

RESULTS = Path("results")

st.set_page_config(page_title="reRankQ results", layout="wide")


def list_batches() -> list[Path]:
    directory = RESULTS / "experiments" / "batches"
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def load_batch(path: Path) -> tuple[dict, list[dict]]:
    batch = json.loads(path.read_text())
    experiments = []
    for entry in batch.get("experiments", []):
        directory = RESULTS / "experiments" / entry["experiment_id"]
        metrics_path = directory / "aggregate_metrics.json"
        if not metrics_path.exists():
            continue
        experiments.append(
            {
                "experiment_id": entry["experiment_id"],
                "strategy": entry["chunking_strategy"],
                "config": entry["retrieval_config"],
                "metrics": json.loads(metrics_path.read_text()),
                "experiment": json.loads((directory / "experiment.json").read_text())
                if (directory / "experiment.json").exists()
                else {},
            }
        )
    return batch, experiments


def metric_frame(experiments: list[dict]):
    rows = []
    for item in experiments:
        metrics = item["metrics"]
        if metrics.get("status") != "ok":
            continue
        values = metrics.get("metrics", {})
        rows.append(
            {
                "experiment": item["experiment_id"],
                "chunking": item["strategy"],
                "retrieval": item["config"],
                "queries": metrics.get("num_evaluated"),
                "MRR": values.get("mrr"),
                "MAP": values.get("map"),
                "Recall@5": values.get("recall@5"),
                "nDCG@5": values.get("ndcg@5"),
                "nDCG@10": values.get("ndcg@10"),
            }
        )
    return rows


def main() -> None:
    st.title("reRankQ — experiment results")
    st.caption(
        "Read-only view of stored artifacts. Nothing is computed here, and "
        "retrieval metrics are separate from RAG answer quality."
    )

    batches = list_batches()
    if not batches:
        st.info(
            "No experiment batch has been run yet. Run "
            "`python scripts/run_retrieval_experiments.py --split dev` once queries "
            "are approved and judgments exist."
        )
    else:
        batch_path = st.sidebar.selectbox(
            "Batch", options=batches, format_func=lambda path: path.stem
        )
        batch, experiments = load_batch(batch_path)

        st.subheader("Batch")
        st.write(
            {
                "batch": batch.get("batch_id"),
                "queries": batch.get("num_queries"),
                "query_set": batch.get("query_set"),
                "split": batch.get("split"),
                "corpus_version": batch.get("corpus_version"),
                "query_version": batch.get("query_version"),
                "split_version": batch.get("split_version"),
                "configurations": batch.get("num_configurations"),
            }
        )

        rows = metric_frame(experiments)
        if rows:
            import pandas as pd

            st.subheader("Retrieval metrics (macro over evaluated queries)")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            blocked = [
                item for item in experiments if item["metrics"].get("status") == "blocked"
            ]
            st.warning(
                "This batch contains no metric-bearing results. "
                "Metrics need human-approved queries and complete human judgments."
            )
            for item in blocked[:1]:
                st.write(item["metrics"].get("reason"))
            st.caption(
                "Rankings, latency and coverage are still stored for every configuration."
            )

        coverage_path = (
            RESULTS / "experiments" / experiments[0]["experiment_id"] / "coverage_report.json"
            if experiments
            else None
        )
        if coverage_path and coverage_path.exists():
            coverage = json.loads(coverage_path.read_text())
            st.subheader("Judgment coverage")
            st.write(
                {
                    "pool candidates": coverage.get("pool_candidates"),
                    "judged": coverage.get("judged_candidates"),
                    "pending": coverage.get("pending_candidates"),
                    "coverage": f"{coverage.get('coverage', 0):.1%}",
                }
            )

    benchmarks = sorted((RESULTS / "benchmarks").glob("latency-*"))
    if benchmarks:
        st.subheader("Latency")
        chosen = st.selectbox(
            "Benchmark", options=benchmarks, format_func=lambda path: path.name
        )
        payload = json.loads((chosen / "latency.json").read_text())
        records = [
            {
                "chunking": record.get("strategy"),
                "retrieval": record.get("retrieval_config"),
                "p50 ms": record.get("p50_ms"),
                "p95 ms": record.get("p95_ms"),
                "mean ms": record.get("mean_ms"),
                "stdev": record.get("stdev_ms"),
                "measurements": record.get("num_measurements"),
                "cold first call ms": record.get("cold_first_call_ms"),
            }
            for record in payload.get("records", [])
            if record.get("status") == "ok"
        ]
        if records:
            import pandas as pd

            st.dataframe(pd.DataFrame(records), hide_index=True)
            st.caption(
                f"Warm-up passes excluded: {payload.get('warmup_iterations')}; "
                f"measured passes: {payload.get('measured_iterations')}. "
                f"Workload: {payload.get('query_source')}."
            )
    else:
        st.info("No latency benchmark has been run yet.")

    analyses = sorted((RESULTS / "experiments" / "analysis").glob("*_error_analysis.json"))
    if analyses:
        st.subheader("Error analysis")
        payload = json.loads(analyses[-1].read_text())
        for comparison in payload.get("comparisons", []):
            counts = comparison["counts"]
            with st.expander(
                f"{comparison['comparison']} · improved {counts['improved_by_b']} / "
                f"worsened {counts['worsened_by_b']}"
            ):
                st.write(counts)
                for example in comparison.get("improved_examples", [])[:3]:
                    st.json(example)
    else:
        st.info("No error analysis artifacts yet.")


main()

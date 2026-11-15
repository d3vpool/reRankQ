"""
Statistical comparison of retrieval runs.

Resampling is over queries, never over individual retrieved chunks, and paired
comparisons keep each query's two values together. Nothing here decides that a
difference is "real": it reports the interval, the paired effect size and an
explicit low-power warning for small query sets.
"""

import math
from typing import Iterable, Optional

import numpy as np

DEFAULT_SEED = 13
DEFAULT_ITERATIONS = 5000
DEFAULT_ALPHA = 0.05
LOW_POWER_THRESHOLD = 30


def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def bootstrap_ci(
    values: Iterable[float],
    iterations: int = DEFAULT_ITERATIONS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Percentile bootstrap confidence interval for a mean."""
    array = _as_array(values)
    if array.size == 0:
        return {"n": 0, "mean": None, "lower": None, "upper": None, "level": 1 - alpha}
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(iterations, array.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "level": 1 - alpha,
        "iterations": iterations,
        "seed": seed,
    }


def paired_bootstrap(
    values_a: Iterable[float],
    values_b: Iterable[float],
    iterations: int = DEFAULT_ITERATIONS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Paired bootstrap of ``a - b``.

    ``p_value`` is the two-sided bootstrap tail probability: the share of
    resampled paired differences whose sign disagrees with the observed one,
    doubled and capped at 1. It is a resampling approximation, reported with
    the interval rather than as a decisive test.
    """
    a = _as_array(values_a)
    b = _as_array(values_b)
    if a.size != b.size:
        raise ValueError("paired bootstrap requires equal-length paired samples")
    if a.size == 0:
        return {"n": 0, "observed_difference": None, "lower": None, "upper": None}

    differences = a - b
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(iterations, differences.size))
    resampled = differences[indices].mean(axis=1)
    lower, upper = np.quantile(resampled, [alpha / 2, 1 - alpha / 2])

    if observed == 0:
        p_value = 1.0
    else:
        opposite = resampled <= 0 if observed > 0 else resampled >= 0
        p_value = min(1.0, 2 * (float(opposite.mean()) + 1) / (iterations + 1))

    return {
        "n": int(differences.size),
        "observed_difference": observed,
        "lower": float(lower),
        "upper": float(upper),
        "p_value": p_value,
        "level": 1 - alpha,
        "iterations": iterations,
        "seed": seed,
        "significant_at_alpha": p_value < alpha,
    }


def paired_effect_size(values_a: Iterable[float], values_b: Iterable[float]) -> dict:
    """Cohen's d_z for paired samples, plus the win/loss/tie split."""
    a = _as_array(values_a)
    b = _as_array(values_b)
    if a.size != b.size:
        raise ValueError("paired effect size requires equal-length samples")
    if a.size < 2:
        return {"cohens_dz": None, "wins": None, "losses": None, "ties": None, "n": int(a.size)}

    differences = a - b
    sd = float(differences.std(ddof=1))
    dz = float(differences.mean() / sd) if sd > 0 else 0.0
    return {
        "cohens_dz": dz,
        "magnitude": (
            "negligible" if abs(dz) < 0.2
            else "small" if abs(dz) < 0.5
            else "medium" if abs(dz) < 0.8
            else "large"
        ),
        "wins": int((differences > 0).sum()),
        "losses": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
        "n": int(a.size),
    }


def sign_test(values_a: Iterable[float], values_b: Iterable[float]) -> dict:
    """Exact two-sided binomial sign test over non-tied pairs."""
    a = _as_array(values_a)
    b = _as_array(values_b)
    wins = int((a > b).sum())
    losses = int((a < b).sum())
    n = wins + losses
    if n == 0:
        return {"n": 0, "wins": 0, "losses": 0, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(0, min(wins, losses) + 1)) / (2**n)
    return {"n": n, "wins": wins, "losses": losses, "p_value": min(1.0, 2 * tail)}


def compare_metric(
    per_query_a: Iterable[dict],
    per_query_b: Iterable[dict],
    metric: str,
    alpha: float = DEFAULT_ALPHA,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Compare two runs on one metric, paired by query id.

    Only queries present and evaluable in both runs are used.
    """
    a_by_id = {row["query_id"]: row for row in per_query_a}
    b_by_id = {row["query_id"]: row for row in per_query_b}
    shared = sorted(set(a_by_id) & set(b_by_id))

    values_a, values_b = [], []
    for query_id in shared:
        value_a = a_by_id[query_id].get(metric)
        value_b = b_by_id[query_id].get(metric)
        if value_a is None or value_b is None:
            continue
        values_a.append(value_a)
        values_b.append(value_b)

    result = {
        "metric": metric,
        "num_paired_queries": len(values_a),
        "mean_a": (sum(values_a) / len(values_a)) if values_a else None,
        "mean_b": (sum(values_b) / len(values_b)) if values_b else None,
        "bootstrap": None,
        "effect_size": None,
        "sign_test": None,
        "power_warning": None,
    }
    if not values_a:
        result["status"] = "insufficient_paired_queries"
        return result

    result["bootstrap"] = paired_bootstrap(values_a, values_b, iterations, alpha, seed)
    result["effect_size"] = paired_effect_size(values_a, values_b)
    result["sign_test"] = sign_test(values_a, values_b)
    result["status"] = "ok"
    if len(values_a) < LOW_POWER_THRESHOLD:
        result["power_warning"] = (
            f"only {len(values_a)} paired queries; confidence intervals are wide and "
            "the test has low power, so a non-significant result is not evidence of equality"
        )
    return result


def compare_runs(
    run_a: dict,
    run_b: dict,
    metrics: Iterable[str] = ("recall@5", "ndcg@5", "mrr", "map"),
    alpha: float = DEFAULT_ALPHA,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Compare two evaluated runs; labels come from run metadata, not the caller."""
    comparisons = {
        metric: compare_metric(
            run_a.get("per_query", []),
            run_b.get("per_query", []),
            metric,
            alpha=alpha,
            iterations=iterations,
            seed=seed,
        )
        for metric in metrics
    }
    return {
        "run_a": {
            "experiment_id": run_a.get("experiment_id"),
            "chunking_strategy": run_a.get("chunking_strategy"),
            "retrieval_config": run_a.get("retrieval_config"),
        },
        "run_b": {
            "experiment_id": run_b.get("experiment_id"),
            "chunking_strategy": run_b.get("chunking_strategy"),
            "retrieval_config": run_b.get("retrieval_config"),
        },
        "comparisons": comparisons,
        "alpha": alpha,
        "iterations": iterations,
        "seed": seed,
    }

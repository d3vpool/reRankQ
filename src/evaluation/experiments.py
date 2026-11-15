"""
Experiment configuration, identity and environment capture.

Every experiment is identified by a hash of the corpus version, approved query
version, split, chunking strategy, retrieval configuration and the retrieval
parameters that affect its output. Re-running the same specification therefore
produces the same experiment id, and changing anything produces a new one.
"""

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

EXPERIMENT_FORMAT_VERSION = 1

CHUNKING_STRATEGIES = ("fixed_size", "structure_aware")
RETRIEVAL_CONFIGS = ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")

SPLITS = ("dev", "test", "all")


class EvaluationBlocked(RuntimeError):
    """Raised when the data needed for a valid experiment is missing."""


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def experiment_id(
    *,
    corpus_version: str,
    query_version: str,
    split_version: str,
    split: str,
    chunking_strategy: str,
    retrieval_config: str,
    parameters: dict,
) -> str:
    payload = {
        "corpus_version": corpus_version,
        "query_version": query_version,
        "split_version": split_version,
        "split": split,
        "chunking_strategy": chunking_strategy,
        "retrieval_config": retrieval_config,
        "parameters": parameters,
    }
    return f"exp-{_digest(payload)[:12]}"


def default_parameters(config, top_k: int = 10) -> dict:
    """Retrieval parameters that can change rankings, from the live config."""
    return {
        "top_k": top_k,
        "bm25": {"k1": config.retrieval.bm25.k1, "b": config.retrieval.bm25.b},
        "hybrid": {
            "method": config.retrieval.hybrid.method,
            "k": config.retrieval.hybrid.k,
            "bm25_weight": config.retrieval.hybrid.bm25_weight,
            "dense_weight": config.retrieval.hybrid.dense_weight,
        },
        "reranking": {
            "model_name": config.retrieval.reranking.model_name,
            "top_k": config.retrieval.reranking.top_k,
        },
    }


def build_specs(
    *,
    corpus_version: str,
    query_version: str,
    split_version: str,
    split: str,
    strategies=CHUNKING_STRATEGIES,
    configs=RETRIEVAL_CONFIGS,
    parameters: Optional[dict] = None,
) -> list[dict]:
    """The eight primary configurations (or a filtered subset)."""
    parameters = parameters or {}
    specs = []
    for strategy in strategies:
        for config_name in configs:
            params = parameters.get(config_name, parameters.get("default", {}))
            specs.append(
                {
                    "experiment_id": experiment_id(
                        corpus_version=corpus_version,
                        query_version=query_version,
                        split_version=split_version,
                        split=split,
                        chunking_strategy=strategy,
                        retrieval_config=config_name,
                        parameters=params,
                    ),
                    "corpus_version": corpus_version,
                    "query_version": query_version,
                    "split_version": split_version,
                    "split": split,
                    "chunking_strategy": strategy,
                    "retrieval_config": config_name,
                    "parameters": params,
                }
            )
    return specs


def cpu_model() -> Optional[str]:
    """CPU model string from /proc/cpuinfo (Linux); None elsewhere."""
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def total_memory_gb() -> Optional[float]:
    """Total RAM in GiB from /proc/meminfo; None when unavailable."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kib = float(line.split()[1])
                    return round(kib / (1024 * 1024), 2)
    except OSError:
        return None
    return None


def _torch_info() -> dict:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        return {
            "torch": getattr(torch, "__version__", None),
            "cuda_available": cuda,
            "device": "cuda" if cuda else "cpu",
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
        }
    except Exception:
        return {"torch": None, "cuda_available": False, "device": "cpu", "gpu": None}


def _package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    names = (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "sentence-transformers",
        "faiss-cpu",
        "rank-bm25",
        "tiktoken",
        "pydantic",
        "torch",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def environment_snapshot() -> dict:
    """Runtime environment. Only values actually readable are recorded."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "memory_gb": total_memory_gb(),
        "packages": _package_versions(),
        **_torch_info(),
    }


def verify_dataset(
    *,
    eval_dir: Path,
    documents: list,
    manifest: dict,
    approved_queries: list[dict],
    split_meta: Optional[dict],
    require_approved: bool = True,
) -> list[str]:
    """
    Pre-flight checks. Returns problems; never repairs anything.

    A mismatch is fatal for metric-producing runs: silently evaluating against
    another corpus or dataset version would make the numbers meaningless.
    """
    from src.evaluation.corpus_manifest import corpus_version, verify_corpus

    problems: list[str] = []
    report = verify_corpus(manifest, documents)
    if not report["ok"]:
        problems.append(
            "corpus does not match the frozen manifest "
            f"(missing={report['missing']}, extra={report['extra']}, changed={report['changed']})"
        )
    if corpus_version(documents) != manifest.get("corpus_version"):
        problems.append("corpus version does not match the manifest version")

    if require_approved and not approved_queries:
        problems.append(
            "no approved queries: approved_queries.jsonl is empty. A human must "
            "review candidate queries in scripts/review_queries.py before metrics "
            "can be computed."
        )

    for row in approved_queries:
        if row.get("corpus_version") != manifest.get("corpus_version"):
            problems.append(
                f"{row.get('query_id')}: approved query targets corpus "
                f"{row.get('corpus_version')!r}, manifest is {manifest.get('corpus_version')!r}"
            )
        if row.get("split") not in ("dev", "test"):
            problems.append(f"{row.get('query_id')}: approved query has no valid split")

    if split_meta and approved_queries:
        assignment = split_meta.get("assignment", {})
        for row in approved_queries:
            expected = assignment.get(row.get("query_id"))
            if expected and expected != row.get("split"):
                problems.append(
                    f"{row.get('query_id')}: split {row.get('split')!r} disagrees with "
                    f"split_assignment.json ({expected!r})"
                )
    return problems


def experiment_dir(root: str | Path, exp_id: str) -> Path:
    return Path(root) / exp_id


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def write_jsonl(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    return path


def prepare_experiment_dir(root: str | Path, exp_id: str) -> Path:
    """
    Create the artifact directory for an experiment.

    Refuses to reuse a directory that already contains results: previous
    experiments are never overwritten.
    """
    directory = experiment_dir(root, exp_id)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"experiment directory {directory} already contains results; "
            "refusing to overwrite a previous experiment"
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory

"""
Reproducible corpus snapshots for evaluation.

A manifest pins the exact documents, chunking configuration and software
versions an evaluation dataset was built against. Its ``corpus_version`` is a
content hash, so re-ingesting changed content produces a different version and
stale evaluation artifacts can be detected instead of silently reused.
"""

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

TOKENIZER_NAME = "cl100k_base"
MANIFEST_FORMAT_VERSION = 1


def sha256_text(text: str) -> str:
    """SHA-256 of a document's exact content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # package missing or has no metadata
        return None


def package_versions() -> dict:
    """Versions of the libraries whose behaviour affects the dataset."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pydantic": _package_version("pydantic"),
        "tiktoken": _package_version("tiktoken"),
        "faiss-cpu": _package_version("faiss-cpu"),
        "rank-bm25": _package_version("rank-bm25"),
        "sentence-transformers": _package_version("sentence-transformers"),
        "numpy": _package_version("numpy"),
    }


def corpus_version(documents: Iterable) -> str:
    """
    Content hash over document ids, URLs and bodies.

    Sorted so the value depends only on corpus content, never on crawl order.
    """
    rows = sorted(
        f"{doc.doc_id}\t{doc.url}\t{sha256_text(doc.content)}" for doc in documents
    )
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return f"corpus-{digest[:12]}"


def build_manifest(
    documents: list,
    chunks_by_strategy: dict[str, list],
    chunk_paths: Optional[dict[str, str]] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    config=None,
    source: Optional[str] = None,
    index_paths: Optional[dict[str, str]] = None,
) -> dict:
    """Build a manifest describing one corpus snapshot."""
    documents = list(documents)
    if not documents:
        raise ValueError("Cannot build a corpus manifest for an empty corpus")

    if config is None:
        from src.utils.config import get_config

        config = get_config()

    chunk_paths = chunk_paths or {}
    index_paths = index_paths or {}

    chunk_summary = {}
    for strategy, chunks in chunks_by_strategy.items():
        ids = [chunk.chunk_id for chunk in chunks]
        chunk_summary[strategy] = {
            "num_chunks": len(chunks),
            "chunk_ids_sha256": hashlib.sha256(
                "\n".join(sorted(ids)).encode("utf-8")
            ).hexdigest(),
            "chunk_file": chunk_paths.get(strategy),
            "index_dir": index_paths.get(strategy),
            "config": config.chunking.model_dump()[strategy],
        }

    return {
        "manifest_format_version": MANIFEST_FORMAT_VERSION,
        "corpus_version": corpus_version(documents),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source or config.corpus.source,
        "num_documents": len(documents),
        "documents": [
            {
                "doc_id": doc.doc_id,
                "url": doc.url,
                "title": doc.title,
                "section": doc.section,
                "chars": len(doc.content),
                "sha256": sha256_text(doc.content),
            }
            for doc in sorted(documents, key=lambda d: d.doc_id)
        ],
        "chunking_config": config.chunking.model_dump(),
        "chunks": chunk_summary,
        "embedding_model": embedding_model,
        "tokenizer": {"name": TOKENIZER_NAME, "library": "tiktoken"},
        "software": package_versions(),
    }


def write_manifest(manifest: dict, path: str | Path) -> Path:
    """Write a manifest to disk (pretty-printed JSON)."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def load_manifest(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No corpus manifest at {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("manifest_format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported manifest format {manifest.get('manifest_format_version')} "
            f"in {path}"
        )
    return manifest


def verify_corpus(manifest: dict, documents: Iterable) -> dict:
    """
    Compare a manifest against the corpus currently on disk.

    Returns a report with ``ok`` plus ``missing``, ``extra`` and ``changed``
    document ids. Nothing is modified.
    """
    documents = list(documents)
    stored = {entry["doc_id"]: entry for entry in manifest.get("documents", [])}
    current = {
        doc.doc_id: {"sha256": sha256_text(doc.content), "url": doc.url}
        for doc in documents
    }

    missing = sorted(set(stored) - set(current))
    extra = sorted(set(current) - set(stored))
    changed = sorted(
        doc_id
        for doc_id in set(stored) & set(current)
        if stored[doc_id]["sha256"] != current[doc_id]["sha256"]
    )
    current_version = corpus_version(documents)

    return {
        "ok": not missing and not extra and not changed,
        "manifest_corpus_version": manifest.get("corpus_version"),
        "current_corpus_version": current_version,
        "version_matches": manifest.get("corpus_version") == current_version,
        "missing": missing,
        "extra": extra,
        "changed": changed,
        "num_documents": {"manifest": len(stored), "current": len(current)},
    }

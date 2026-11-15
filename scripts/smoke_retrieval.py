#!/usr/bin/env python
"""
Real-data retrieval smoke test for reRankQ.

Runs all four retrieval variants (BM25, dense, hybrid RRF, hybrid + rerank)
against an already-ingested corpus. It does not compute evaluation metrics and
does not invoke an LLM.

Usage:
    python scripts/smoke_retrieval.py
    python scripts/smoke_retrieval.py --strategy structure_aware --top-k 3
    python scripts/smoke_retrieval.py --query "How do webhooks work?"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.chunker import load_chunks  # noqa: E402
from src.reranking.reranker import CrossEncoderReranker  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402

DEFAULT_QUERIES = [
    "How do I create a payment intent?",
    "How do webhooks work and how are signatures verified?",
    "What is an idempotency key used for?",
    "How do subscription lifecycles behave when a payment fails?",
]


def print_results(name: str, results: list[dict], top_k: int) -> None:
    print(f"\n{name}")
    if not results:
        print("  (no results)")
        return
    for result in results[:top_k]:
        title = result.get("metadata", {}).get("title", "?")
        url = result.get("metadata", {}).get("url", "?")
        snippet = " ".join(result["content"].split())[:120]
        print(f"  {result['rank']}. [{result['chunk_id']}] {title} <{url}>")
        print(f"     {snippet}...")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--indexes", default="data/indexes")
    parser.add_argument(
        "--strategy",
        default="structure_aware",
        choices=["fixed_size", "structure_aware"],
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--query", action="append", default=None)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild indexes instead of loading them from disk",
    )
    args = parser.parse_args()

    queries = args.query or DEFAULT_QUERIES

    index_dir = Path(args.indexes)
    if args.rebuild or not (index_dir / "index_manifest.json").exists():
        chunks = load_chunks(args.processed, strategy=args.strategy)
        print(f"Indexing {len(chunks)} chunks from {args.processed} ({args.strategy})")

        bm25 = BM25Retriever()
        bm25.index_chunks(chunks)
        dense = DenseRetriever()
        dense.index_chunks(chunks)

        hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
        hybrid.save_indexes(str(index_dir))
        print(f"Saved indexes to {index_dir}")
    else:
        print(f"Loading indexes from {index_dir}")
        bm25 = BM25Retriever()
        dense = DenseRetriever()
        hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
        hybrid.load_indexes(str(index_dir))

    manifest_path = index_dir / "index_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(
            f"Manifest: {manifest['num_chunks']} chunks, "
            f"{manifest['embedding_model']}, strategies={manifest['chunk_strategies']}"
        )

    reranker = CrossEncoderReranker()

    for query in queries:
        print("\n" + "=" * 78)
        print(f"QUERY: {query}")
        print("=" * 78)

        print_results("A. BM25 only", hybrid.bm25_retriever.retrieve(query, args.top_k), args.top_k)
        print_results("B. Dense only", hybrid.dense_retriever.retrieve(query, args.top_k), args.top_k)

        candidates = hybrid.retrieve(query, top_k=args.top_k * 2)
        print_results("C. Hybrid RRF", candidates, args.top_k)

        print_results(
            "D. Hybrid RRF + cross-encoder rerank",
            reranker.rerank(query, candidates, top_k=args.top_k),
            args.top_k,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

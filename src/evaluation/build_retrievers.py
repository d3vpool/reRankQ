"""
Load the four retrieval systems for one chunking strategy.

The index manifest is verified against the expected corpus version and strategy,
so an experiment cannot silently run on another corpus snapshot.
"""

from pathlib import Path
from typing import Optional

from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


def load_strategy_retrievers(
    strategy: str,
    indexes_dir: str | Path = "data/indexes",
    corpus_version: Optional[str] = None,
    reranker: Optional[CrossEncoderReranker] = None,
) -> dict:
    """Return ``{"bm25", "dense", "hybrid", "reranker"}`` for one strategy."""
    index_dir = Path(indexes_dir) / strategy
    if not index_dir.exists():
        raise FileNotFoundError(
            f"No index directory for chunking strategy {strategy!r} at {index_dir}"
        )

    bm25 = BM25Retriever()
    dense = DenseRetriever()
    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
    hybrid.corpus_version = corpus_version
    hybrid.strategy = strategy
    hybrid.load_indexes(str(index_dir))

    return {
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
        "reranker": reranker or CrossEncoderReranker(),
    }

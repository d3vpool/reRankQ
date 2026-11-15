"""
Hybrid retriever combining BM25 and dense retrieval.
Implements reciprocal rank fusion (RRF) for combining results.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.ingestion.chunker import Chunk, load_chunks
from src.utils.config import get_config
from src.utils.logging import setup_logging, log_metrics


class HybridRetriever:
    """Hybrid retriever combining BM25 and dense retrieval."""
    
    def __init__(
        self,
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[DenseRetriever] = None,
    ):
        self.config = get_config()
        self.logger = setup_logging()
        
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.dense_retriever = dense_retriever or DenseRetriever()
        #: Optional dataset binding, checked against the index manifest.
        self.corpus_version: Optional[str] = None
        #: Optional expected chunking strategy, checked against the manifest.
        self.strategy: Optional[str] = None
    
    def reciprocal_rank_fusion(
        self,
        bm25_results: list[dict],
        dense_results: list[dict],
        k: int = 60,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> list[dict]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(weight / (k + rank)) for each result list
        
        Args:
            bm25_results: Results from BM25 retriever
            dense_results: Results from dense retriever
            k: RRF constant (higher = less weight on top ranks)
            bm25_weight: Weight for BM25 results
            dense_weight: Weight for dense results
        
        Returns:
            Fused and re-ranked results
        """
        # Create rank mappings. Duplicate chunk ids keep their first (best)
        # rank so a repeated candidate cannot shift the fusion.
        bm25_ranks: dict[str, int] = {}
        for rank, row in enumerate(bm25_results, start=1):
            bm25_ranks.setdefault(row["chunk_id"], rank)

        dense_ranks: dict[str, int] = {}
        for rank, row in enumerate(dense_results, start=1):
            dense_ranks.setdefault(row["chunk_id"], rank)

        # Keep the first record seen for each chunk id (BM25 preferred), with
        # dictionary lookups instead of a scan over both result lists.
        chunk_data: dict[str, dict] = {}
        for result in bm25_results:
            chunk_data.setdefault(result["chunk_id"], result)
        for result in dense_results:
            chunk_data.setdefault(result["chunk_id"], result)

        # Get all unique chunk IDs
        all_chunk_ids = set(bm25_ranks) | set(dense_ranks)

        # Calculate RRF scores
        rrf_scores: dict[str, float] = {}
        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in bm25_ranks:
                score += bm25_weight / (k + bm25_ranks[chunk_id])
            if chunk_id in dense_ranks:
                score += dense_weight / (k + dense_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        # Deterministic ordering: score desc, then best BM25 rank, then best
        # dense rank, then chunk id as a final tie-breaker.
        sorted_chunk_ids = sorted(
            rrf_scores,
            key=lambda cid: (
                -rrf_scores[cid],
                bm25_ranks.get(cid, math.inf),
                dense_ranks.get(cid, math.inf),
                cid,
            ),
        )

        # Build final results
        results = []
        for i, chunk_id in enumerate(sorted_chunk_ids):
            result = chunk_data[chunk_id].copy()
            result["rrf_score"] = rrf_scores[chunk_id]
            result["bm25_rank"] = bm25_ranks.get(chunk_id)
            result["dense_rank"] = dense_ranks.get(chunk_id)
            result["rank"] = i + 1
            results.append(result)

        return results
    
    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks in both retrievers."""
        self.logger.info("Indexing chunks in hybrid retriever...")
        self.bm25_retriever.index_chunks(chunks)
        self.dense_retriever.index_chunks(chunks)
        self.logger.info(f"Indexed {len(chunks)} chunks in both retrievers")
    
    #: Written next to both indexes so a stale/incompatible directory is caught.
    MANIFEST_NAME = "index_manifest.json"

    def _manifest(self, corpus_version: Optional[str] = None) -> dict:
        embeddings = self.dense_retriever.embeddings
        chunks = self.bm25_retriever.chunks
        return {
            "format_version": 1,
            "corpus_version": corpus_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": self.dense_retriever.model_name,
            "num_chunks": len(chunks),
            "chunk_strategies": sorted({chunk.strategy for chunk in chunks}),
            "chunking": self.config.chunking.model_dump(),
            "bm25": {
                "k1": self.config.retrieval.bm25.k1,
                "b": self.config.retrieval.bm25.b,
            },
            "dense": {
                "dimension": int(embeddings.shape[1]) if embeddings is not None and embeddings.ndim == 2 else None,
            },
        }

    def save_indexes(
        self,
        output_dir: str = "data/indexes",
        corpus_version: Optional[str] = None,
    ) -> None:
        """Save both indexes to disk, plus a manifest describing them."""
        self.bm25_retriever.save_index(output_dir)
        self.dense_retriever.save_index(output_dir)

        manifest_path = Path(output_dir) / self.MANIFEST_NAME
        with open(manifest_path, "w") as f:
            json.dump(self._manifest(corpus_version), f, indent=2)

        self.logger.info(f"Saved hybrid indexes to {output_dir}")

    def _verify_manifest(self, index_dir: str) -> None:
        """Refuse to use an index directory built for a different configuration."""
        manifest_path = Path(index_dir) / self.MANIFEST_NAME
        if not manifest_path.exists():
            self.logger.warning(
                f"No index manifest in {index_dir}; skipping compatibility check"
            )
            return

        with open(manifest_path) as f:
            manifest = json.load(f)

        expected_model = self.config.embedding.model_name
        stored_model = manifest.get("embedding_model")
        if stored_model and stored_model != expected_model:
            raise ValueError(
                f"Index in {index_dir} was built with embedding model {stored_model!r} "
                f"but the configuration requests {expected_model!r}"
            )

        stored_corpus = manifest.get("corpus_version")
        stored_strategies = manifest.get("chunk_strategies") or []
        if stored_corpus and self.corpus_version:
            if stored_corpus != self.corpus_version:
                raise ValueError(
                    f"Index in {index_dir} was built for corpus {stored_corpus!r} "
                    f"but the dataset targets {self.corpus_version!r}"
                )
        if stored_strategies and self.strategy:
            if stored_strategies != [self.strategy]:
                raise ValueError(
                    f"Index in {index_dir} covers strategies {stored_strategies}, "
                    f"expected [{self.strategy!r}]"
                )

        stored_chunks = manifest.get("num_chunks")
        actual_chunks = {len(self.bm25_retriever.chunks), len(self.dense_retriever.chunks)}
        if stored_chunks is not None and actual_chunks != {stored_chunks}:
            raise ValueError(
                f"Index in {index_dir} is inconsistent: manifest lists {stored_chunks} "
                f"chunks, loaded {actual_chunks}"
            )

    def load_indexes(self, index_dir: str = "data/indexes") -> None:
        """Load both indexes from disk and verify they belong together."""
        self.bm25_retriever.load_index(index_dir)
        self.dense_retriever.load_index(index_dir)
        self._verify_manifest(index_dir)
        self.logger.info(f"Loaded hybrid indexes from {index_dir}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        bm25_top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        fusion_method: str = "rrf",
        **kwargs,
    ) -> list[dict]:
        """
        Retrieve using hybrid search.
        
        Args:
            query: Search query
            top_k: Final number of results
            bm25_top_k: Number of BM25 candidates (default: 2x top_k)
            dense_top_k: Number of dense candidates (default: 2x top_k)
            fusion_method: Fusion method ("rrf" or "weighted")
            **kwargs: Additional arguments for fusion method
        
        Returns:
            Fused and re-ranked results
        """
        if bm25_top_k is None:
            bm25_top_k = top_k * 2
        if dense_top_k is None:
            dense_top_k = top_k * 2
        
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, bm25_top_k)
        dense_results = self.dense_retriever.retrieve(query, dense_top_k)
        
        # Combine results
        if fusion_method == "rrf":
            k = kwargs.get("k", self.config.retrieval.hybrid.k)
            bm25_weight = kwargs.get("bm25_weight", self.config.retrieval.hybrid.bm25_weight)
            dense_weight = kwargs.get("dense_weight", self.config.retrieval.hybrid.dense_weight)
            
            fused_results = self.reciprocal_rank_fusion(
                bm25_results, dense_results, k=k,
                bm25_weight=bm25_weight, dense_weight=dense_weight,
            )
        else:
            # Simple weighted score fusion
            fused_results = self._weighted_fusion(
                bm25_results, dense_results, **kwargs
            )
        
        # Return top-k
        return fused_results[:top_k]
    
    def _weighted_fusion(
        self,
        bm25_results: list[dict],
        dense_results: list[dict],
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        **kwargs,
    ) -> list[dict]:
        """Simple weighted score fusion."""
        # Normalize scores to [0, 1]
        bm25_scores = [r["score"] for r in bm25_results]
        dense_scores = [r["score"] for r in dense_results]

        # Guard against empty or all-zero score lists.
        bm25_max = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0
        dense_max = max(dense_scores) if dense_scores and max(dense_scores) > 0 else 1.0

        # Create score mappings
        bm25_score_map = {
            r["chunk_id"]: r["score"] / bm25_max
            for r in bm25_results
        }
        dense_score_map = {
            r["chunk_id"]: r["score"] / dense_max
            for r in dense_results
        }

        # Store chunk data once per chunk id (BM25 preferred).
        chunk_data: dict[str, dict] = {}
        for result in bm25_results:
            chunk_data.setdefault(result["chunk_id"], result.copy())
        for result in dense_results:
            chunk_data.setdefault(result["chunk_id"], result.copy())

        # Combine scores
        all_chunk_ids = set(bm25_score_map) | set(dense_score_map)

        for chunk_id in all_chunk_ids:
            bm25_score = bm25_score_map.get(chunk_id, 0.0)
            dense_score = dense_score_map.get(chunk_id, 0.0)

            chunk_data[chunk_id]["combined_score"] = (
                bm25_weight * bm25_score + dense_weight * dense_score
            )
            chunk_data[chunk_id]["bm25_score"] = bm25_score
            chunk_data[chunk_id]["dense_score"] = dense_score

        # Sort by combined score, deterministic on ties.
        sorted_chunk_ids = sorted(
            chunk_data,
            key=lambda cid: (-chunk_data[cid]["combined_score"], cid),
        )

        results = []
        for i, chunk_id in enumerate(sorted_chunk_ids):
            result = chunk_data[chunk_id]
            result["rank"] = i + 1
            results.append(result)

        return results
    
    def batch_retrieve(
        self,
        queries: list[str],
        top_k: int = 10,
        **kwargs,
    ) -> list[list[dict]]:
        """Retrieve for multiple queries."""
        return [self.retrieve(query, top_k, **kwargs) for query in queries]


def get_hybrid_retriever(
    chunks: Optional[list[Chunk]] = None,
    index_dir: str = "data/indexes",
) -> HybridRetriever:
    """Get a hybrid retriever, loading from disk if available."""
    retriever = HybridRetriever()
    
    # Try to load existing indexes
    try:
        retriever.load_indexes(index_dir)
        return retriever
    except FileNotFoundError:
        pass
    
    # Build new indexes
    if chunks is None:
        chunks = load_chunks()
    
    retriever.index_chunks(chunks)
    retriever.save_indexes(index_dir)
    return retriever

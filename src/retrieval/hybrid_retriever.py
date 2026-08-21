"""
Hybrid retriever combining BM25 and dense retrieval.
Implements reciprocal rank fusion (RRF) for combining results.
"""

import json
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
        # Create rank mappings
        bm25_ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(bm25_results)}
        dense_ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(dense_results)}
        
        # Get all unique chunk IDs
        all_chunk_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        chunk_data = {}
        
        for chunk_id in all_chunk_ids:
            score = 0.0
            
            if chunk_id in bm25_ranks:
                rank = bm25_ranks[chunk_id]
                score += bm25_weight / (k + rank)
            
            if chunk_id in dense_ranks:
                rank = dense_ranks[chunk_id]
                score += dense_weight / (k + rank)
            
            rrf_scores[chunk_id] = score
            
            # Store chunk data (from whichever result list has it)
            for result in bm25_results + dense_results:
                if result["chunk_id"] == chunk_id:
                    chunk_data[chunk_id] = result
                    break
        
        # Sort by RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
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
    
    def save_indexes(self, output_dir: str = "data/indexes") -> None:
        """Save both indexes to disk."""
        self.bm25_retriever.save_index(output_dir)
        self.dense_retriever.save_index(output_dir)
        self.logger.info(f"Saved hybrid indexes to {output_dir}")
    
    def load_indexes(self, index_dir: str = "data/indexes") -> None:
        """Load both indexes from disk."""
        self.bm25_retriever.load_index(index_dir)
        self.dense_retriever.load_index(index_dir)
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
        
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        dense_max = max(dense_scores) if dense_scores else 1.0
        
        # Create score mappings
        bm25_score_map = {
            r["chunk_id"]: r["score"] / bm25_max
            for r in bm25_results
        }
        dense_score_map = {
            r["chunk_id"]: r["score"] / dense_max
            for r in dense_results
        }
        
        # Combine scores
        all_chunk_ids = set(bm25_score_map.keys()) | set(dense_score_map.keys())
        
        results = []
        chunk_data = {}
        
        for chunk_id in all_chunk_ids:
            bm25_score = bm25_score_map.get(chunk_id, 0.0)
            dense_score = dense_score_map.get(chunk_id, 0.0)
            
            combined_score = bm25_weight * bm25_score + dense_weight * dense_score
            
            # Store chunk data
            for result in bm25_results + dense_results:
                if result["chunk_id"] == chunk_id:
                    chunk_data[chunk_id] = result.copy()
                    chunk_data[chunk_id]["combined_score"] = combined_score
                    chunk_data[chunk_id]["bm25_score"] = bm25_score
                    chunk_data[chunk_id]["dense_score"] = dense_score
                    break
        
        # Sort by combined score
        sorted_chunk_ids = sorted(
            chunk_data.keys(),
            key=lambda x: chunk_data[x]["combined_score"],
            reverse=True
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

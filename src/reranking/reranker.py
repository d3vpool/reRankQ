"""
Cross-encoder reranker for re-ranking retrieval results.
Provides the final stage in the hybrid retrieval pipeline.
"""

import json
from pathlib import Path
from typing import Optional

from src.utils.config import get_config
from src.utils.logging import setup_logging


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.config = get_config()
        self.logger = setup_logging()
        
        if model_name is None:
            model_name = self.config.retrieval.reranking.model_name
        
        self.model_name = model_name
        self.model = None
    
    def _load_model(self) -> None:
        """Load cross-encoder model."""
        from sentence_transformers import CrossEncoder
        
        self.logger.info(f"Loading cross-encoder model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        self.logger.info("Cross-encoder model loaded")
    
    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Rerank retrieval results using cross-encoder.
        
        Args:
            query: Original query
            results: Initial retrieval results (from BM25/dense/hybrid)
            top_k: Number of results to return after reranking
        
        Returns:
            Re-ranked results with new scores
        """
        if top_k is None:
            top_k = self.config.retrieval.reranking.top_k
        
        if not results:
            return []

        # Deduplicate candidates by chunk id (first occurrence wins) so a
        # rerank cannot return the same chunk twice.
        seen: set[str] = set()
        unique_results = []
        for result in results:
            chunk_id = result["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            unique_results.append(result)
        results = unique_results
        
        if self.model is None:
            self._load_model()
        
        # Prepare query-document pairs
        pairs = [(query, result["content"]) for result in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Update results with new scores
        reranked_results = []
        for result, score in zip(results, scores):
            new_result = result.copy()
            new_result["rerank_score"] = float(score)
            new_result["original_score"] = result.get("score", result.get("rrf_score", 0.0))
            reranked_results.append(new_result)
        
        # Sort by rerank score
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked_results):
            result["rank"] = i + 1
        
        return reranked_results[:top_k]
    
    def batch_rerank(
        self,
        query: str,
        batch_results: list[list[dict]],
        top_k: Optional[int] = None,
    ) -> list[list[dict]]:
        """Rerank multiple result sets."""
        return [self.rerank(query, results, top_k) for results in batch_results]


class RerankingPipeline:
    """Complete reranking pipeline."""
    
    def __init__(self, reranker: Optional[CrossEncoderReranker] = None):
        self.reranker = reranker or CrossEncoderReranker()
        self.config = get_config()
        self.logger = setup_logging()
    
    def rerank_hybrid_results(
        self,
        query: str,
        hybrid_results: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Rerank hybrid retrieval results.
        
        Args:
            query: Original query
            hybrid_results: Results from hybrid retriever
            top_k: Final number of results
        
        Returns:
            Reranked results with citations
        """
        if top_k is None:
            top_k = self.config.retrieval.reranking.top_k
        
        # Rerank
        reranked = self.reranker.rerank(query, hybrid_results, top_k)
        
        # Add citation information
        for result in reranked:
            result["citation"] = {
                "chunk_id": result["chunk_id"],
                "doc_id": result["doc_id"],
                "url": result.get("metadata", {}).get("url", ""),
                "title": result.get("metadata", {}).get("title", ""),
                "section": result.get("metadata", {}).get("section", ""),
            }
        
        return reranked
    
    def save_reranker(self, output_dir: str = "data/indexes") -> None:
        """Save reranker configuration."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        config_file = output_path / "reranker_config.json"
        with open(config_file, "w") as f:
            json.dump({
                "model_name": self.reranker.model_name,
                "top_k": self.config.retrieval.reranking.top_k,
            }, f, indent=2)
        
        self.logger.info(f"Saved reranker config to {output_path}")
    
    def load_reranker(self, index_dir: str = "data/indexes") -> None:
        """Load reranker configuration."""
        index_path = Path(index_dir)
        config_file = index_path / "reranker_config.json"
        
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            self.reranker.model_name = config.get("model_name", self.reranker.model_name)
            self.logger.info(f"Loaded reranker config from {index_path}")


def get_reranker(
    model_name: Optional[str] = None,
) -> CrossEncoderReranker:
    """Get a reranker instance."""
    return CrossEncoderReranker(model_name)


def get_reranking_pipeline(
    model_name: Optional[str] = None,
) -> RerankingPipeline:
    """Get a complete reranking pipeline."""
    reranker = CrossEncoderReranker(model_name)
    return RerankingPipeline(reranker)

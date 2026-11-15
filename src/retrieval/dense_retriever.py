"""
Dense retriever using sentence-transformers and FAISS.
Provides dense retrieval baseline for ablation study.
"""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from src.ingestion.chunker import Chunk, load_chunks
from src.utils.config import get_config
from src.utils.logging import setup_logging, log_metrics, log_retrieval_result


class DenseRetriever:
    """Dense retriever using sentence-transformers embeddings."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.config = get_config()
        self.logger = setup_logging()
        
        if model_name is None:
            model_name = self.config.embedding.model_name
        
        self.model_name = model_name
        self.model = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def _load_model(self) -> None:
        """Load sentence-transformer model."""
        from sentence_transformers import SentenceTransformer
        
        self.logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.logger.info("Embedding model loaded")
    
    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts."""
        if self.model is None:
            self._load_model()
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.embedding.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.config.embedding.normalize,
        )
        
        return embeddings.astype(np.float32)
    
    def _build_index(self) -> None:
        """Build FAISS index from chunk embeddings."""
        self.logger.info(f"Building FAISS index from {len(self.chunks)} chunks...")

        if not self.chunks:
            # An empty corpus is valid: retrieval simply returns no results.
            self.index = None
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
            self.logger.warning("FAISS index not built: empty corpus")
            return
        
        # Extract text from chunks
        texts = [chunk.content for chunk in self.chunks]
        
        # Generate embeddings
        self.embeddings = self._embed_texts(texts)
        
        # Build FAISS index (Inner Product for cosine similarity with normalized vectors)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        
        self.logger.info(f"FAISS index built with dimension {dimension}")
    
    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks for retrieval."""
        self.chunks = chunks
        self._build_index()
    
    def save_index(self, output_dir: str = "data/indexes") -> Path:
        """Save dense index to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = output_path / "dense_index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save embeddings
        embeddings_file = output_path / "dense_embeddings.npy"
        np.save(str(embeddings_file), self.embeddings)
        
        # Save chunks metadata
        chunks_file = output_path / "dense_chunks.jsonl"
        with open(chunks_file, "w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")
        
        # Save config
        config_file = output_path / "dense_config.json"
        with open(config_file, "w") as f:
            json.dump({
                "model_name": self.model_name,
                "dimension": self.embeddings.shape[1],
                "num_chunks": len(self.chunks),
            }, f, indent=2)
        
        self.logger.info(f"Saved dense index to {output_path}")
        return output_path
    
    def load_index(self, index_dir: str = "data/indexes") -> None:
        """Load dense index from disk."""
        index_path = Path(index_dir)
        
        # Load config
        config_file = index_path / "dense_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"No dense index found at {index_path}")
        
        with open(config_file) as f:
            config = json.load(f)
        
        self.model_name = config["model_name"]
        
        # Load FAISS index
        index_file = index_path / "dense_index.faiss"
        self.index = faiss.read_index(str(index_file))
        
        # Load embeddings
        embeddings_file = index_path / "dense_embeddings.npy"
        self.embeddings = np.load(str(embeddings_file))
        
        # Load chunks
        chunks_file = index_path / "dense_chunks.jsonl"
        self.chunks = []
        with open(chunks_file, "r") as f:
            for line in f:
                data = json.loads(line)
                self.chunks.append(Chunk(**data))
        
        self._validate_loaded_index()

        self.logger.info(f"Loaded dense index with {len(self.chunks)} chunks")

    def _validate_loaded_index(self) -> None:
        """Reject indexes whose contents are internally inconsistent."""
        if self.index is None or self.embeddings is None:
            raise ValueError(f"Dense index at {self.model_name!r} is incomplete")

        if self.index.d != self.embeddings.shape[1]:
            raise ValueError(
                f"Dense index dimension ({self.index.d}) does not match the stored "
                f"embeddings ({self.embeddings.shape[1]})"
            )

        if self.index.ntotal != len(self.chunks) or self.embeddings.shape[0] != len(self.chunks):
            raise ValueError(
                "Dense index is inconsistent with its chunk metadata: "
                f"{self.index.ntotal} vectors, {self.embeddings.shape[0]} embeddings, "
                f"{len(self.chunks)} chunks"
            )

        if len(self.chunks) and not np.isfinite(self.embeddings).all():
            raise ValueError("Stored embeddings contain non-finite values")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Retrieve top-k chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of dicts with chunk_id, score, and chunk metadata
        """
        if self.index is None:
            if not self.chunks:
                return []
            raise RuntimeError("Index not built. Call index_chunks() or load_index() first.")

        if not query.strip():
            return []

        # Embed query
        query_embedding = self._embed_texts([query])
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "score": float(score),
                "content": chunk.content,
                "metadata": chunk.metadata,
                "rank": len(results) + 1,
            })
        
        return results
    
    def batch_retrieve(
        self,
        queries: list[str],
        top_k: int = 10,
    ) -> list[list[dict]]:
        """Retrieve for multiple queries."""
        if self.index is None and not self.chunks:
            return [[] for _ in queries]

        queries = [query for query in queries if query.strip()]
        if not queries:
            return []

        if self.model is None:
            self._load_model()
        
        # Embed all queries at once
        query_embeddings = self._embed_texts(queries)
        
        # Batch search
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Build results for each query
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx == -1:
                    continue
                
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "score": float(score),
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "rank": len(results) + 1,
                })
            all_results.append(results)
        
        return all_results


def get_dense_retriever(
    chunks: Optional[list[Chunk]] = None,
    index_dir: str = "data/indexes",
) -> DenseRetriever:
    """Get a dense retriever, loading from disk if available."""
    retriever = DenseRetriever()
    
    # Try to load existing index
    try:
        retriever.load_index(index_dir)
        return retriever
    except FileNotFoundError:
        pass
    
    # Build new index
    if chunks is None:
        chunks = load_chunks()
    
    retriever.index_chunks(chunks)
    retriever.save_index(index_dir)
    return retriever

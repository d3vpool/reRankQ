"""
BM25 retriever using rank_bm25 library.
Provides sparse retrieval baseline for ablation study.
"""

import json
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk, load_chunks
from src.utils.config import get_config
from src.utils.logging import setup_logging, log_metrics, log_retrieval_result


class BM25Retriever:
    """BM25-based sparse retriever."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging()
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: list[Chunk] = []
        self.corpus_tokens: list[list[str]] = []
    
    #: Word tokens keep underscores and internal hyphens, so identifiers like
    #: ``payment_intent`` or ``payment-intent`` survive normalization.
    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*")
    #: Splits CamelCase / snake_case / kebab-case compounds into sub-tokens.
    COMPOUND_PATTERN = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9_]+")

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize technical text without destroying its terminology.

        Each word token is lowercased and kept whole (underscores and inner
        hyphens included). Compounds additionally emit their parts, so
        ``PaymentIntent`` also matches ``payment``/``intent`` and
        ``2020-08-27`` also matches its numeric parts. The same routine is used
        for documents and queries, so matches stay symmetric.
        """
        tokens: list[str] = []

        for raw in self.TOKEN_PATTERN.findall(text):
            lowered = raw.lower()
            emitted = [lowered]

            if "-" in raw:
                emitted.extend(part.lower() for part in raw.split("-") if part)

            parts = self.COMPOUND_PATTERN.findall(raw)
            if len(parts) > 1:
                emitted.extend(part.lower() for part in parts if part)

            # Deduplicate per token so term frequencies are not inflated.
            tokens.extend(dict.fromkeys(emitted))

        return tokens
    
    def _build_index(self) -> None:
        """Build BM25 index from chunks."""
        self.logger.info(f"Building BM25 index from {len(self.chunks)} chunks...")

        if not self.chunks:
            # An empty corpus is valid: retrieval simply returns no results.
            self.corpus_tokens = []
            self.bm25 = None
            self.logger.warning("BM25 index not built: empty corpus")
            return
        
        # Tokenize all chunks
        self.corpus_tokens = [self._tokenize(chunk.content) for chunk in self.chunks]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.corpus_tokens,
            k1=self.config.retrieval.bm25.k1,
            b=self.config.retrieval.bm25.b,
        )
        
        self.logger.info("BM25 index built successfully")
    
    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks for retrieval."""
        self.chunks = chunks
        self._build_index()
    
    def save_index(self, output_dir: str = "data/indexes") -> Path:
        """Save BM25 index to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save chunks metadata
        chunks_file = output_path / "bm25_chunks.jsonl"
        with open(chunks_file, "w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")
        
        # Save tokenized corpus
        tokens_file = output_path / "bm25_tokens.pkl"
        with open(tokens_file, "wb") as f:
            pickle.dump(self.corpus_tokens, f)
        
        self.logger.info(f"Saved BM25 index to {output_path}")
        return output_path
    
    def load_index(self, index_dir: str = "data/indexes") -> None:
        """Load BM25 index from disk."""
        index_path = Path(index_dir)
        
        # Load chunks
        chunks_file = index_path / "bm25_chunks.jsonl"
        if not chunks_file.exists():
            raise FileNotFoundError(f"No BM25 index found at {index_path}")
        
        self.chunks = []
        with open(chunks_file, "r") as f:
            for line in f:
                data = json.loads(line)
                self.chunks.append(Chunk(**data))
        
        # Load tokenized corpus
        tokens_file = index_path / "bm25_tokens.pkl"
        if tokens_file.exists():
            with open(tokens_file, "rb") as f:
                self.corpus_tokens = pickle.load(f)
        
        # Rebuild index
        self._build_index()
        self.logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks")
    
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
        if self.bm25 is None:
            if not self.chunks:
                return []
            raise RuntimeError("Index not built. Call index_chunks() or load_index() first.")

        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices (stable sort keeps ties in corpus order)
        top_k_indices = np.argsort(-scores, kind="stable")[:top_k]
        
        # Build results
        results = []
        for idx in top_k_indices:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "score": float(scores[idx]),
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
        return [self.retrieve(query, top_k) for query in queries]


def get_bm25_retriever(
    chunks: Optional[list[Chunk]] = None,
    index_dir: str = "data/indexes",
) -> BM25Retriever:
    """Get a BM25 retriever, loading from disk if available."""
    retriever = BM25Retriever()
    
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

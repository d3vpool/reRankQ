"""
Real-model tests: exercise the actual sentence-transformers models.

These require network access on first run (model download) and are therefore
opt-in:

    RERANKQ_REAL_MODEL=1 .venv/bin/python -m pytest -m real_model

Nothing here is stubbed: if these pass, the models loaded and ran.
"""

import os

import numpy as np
import pytest

from src.ingestion.chunker import Chunk
from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.dense_retriever import DenseRetriever
from src.utils.config import get_config

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        os.environ.get("RERANKQ_REAL_MODEL") != "1",
        reason="set RERANKQ_REAL_MODEL=1 to download and run real models",
    ),
]


def make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        content=content,
        chunk_index=0,
        strategy="fixed_size",
        start_char=0,
        end_char=len(content),
        token_count=len(content.split()),
        metadata={"url": "https://example.test/doc", "title": "Doc"},
    )


CORPUS = [
    make_chunk("payments", "A payment intent tracks a customer payment lifecycle."),
    make_chunk("webhooks", "Webhook signatures are verified with a signing secret."),
    make_chunk("refunds", "Refund a captured charge through the refunds API."),
]


def test_embedding_model_dimension_and_norm():
    retriever = DenseRetriever()
    retriever.index_chunks(CORPUS)

    assert retriever.index.d == 384
    assert retriever.index.ntotal == len(CORPUS)
    assert np.isfinite(retriever.embeddings).all()
    norms = np.linalg.norm(retriever.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_dense_retrieval_with_real_embeddings():
    retriever = DenseRetriever()
    retriever.index_chunks(CORPUS)

    results = retriever.retrieve("verify webhook signature secret", top_k=3)

    assert results[0]["chunk_id"] == "webhooks"


def test_cross_encoder_scores_and_ordering():
    reranker = CrossEncoderReranker()
    candidates = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "score": 0.0,
            "content": chunk.content,
            "metadata": chunk.metadata,
        }
        for chunk in CORPUS
    ]

    results = reranker.rerank("verify webhook signature", candidates, top_k=3)

    assert results[0]["chunk_id"] == "webhooks"
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert {r["chunk_id"] for r in results} == {c.chunk_id for c in CORPUS}
    assert reranker.model_name == get_config().retrieval.reranking.model_name

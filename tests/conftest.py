"""Shared pytest fixtures for the reRankQ test suite."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import reset_config  # noqa: E402

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

FAKE_EMBEDDING_DIM = 16


def _hash_bucket(token: str) -> int:
    digest = hashlib.md5(token.encode()).digest()
    return int.from_bytes(digest[:4], "little") % FAKE_EMBEDDING_DIM


def fake_embed_texts(self, texts):
    """
    Deterministic bag-of-words embedding standing in for a real model.

    Vectors are L2-normalized so they are compatible with the inner-product
    FAISS index. This is a stub, not an embedding model.
    """
    vectors = np.zeros((len(texts), FAKE_EMBEDDING_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.lower().split():
            vectors[row, _hash_bucket(token)] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class FakeCrossEncoder:
    """Score stub: more query tokens shared with the document => higher score."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        scores = []
        for query, document in pairs:
            query_tokens = set(query.lower().split())
            document_tokens = set(document.lower().split())
            scores.append(float(len(query_tokens & document_tokens)))
        return np.array(scores, dtype=np.float32)


@pytest.fixture(autouse=True)
def repository_config():
    """Start and end every test from the repository default configuration."""
    reset_config(str(DEFAULT_CONFIG_PATH))
    yield
    reset_config(str(DEFAULT_CONFIG_PATH))


@pytest.fixture
def fake_embedder(monkeypatch):
    """Patch the dense retriever so no embedding model is downloaded."""
    from src.retrieval.dense_retriever import DenseRetriever

    monkeypatch.setattr(DenseRetriever, "_embed_texts", fake_embed_texts)
    return fake_embed_texts


@pytest.fixture
def fake_cross_encoder():
    return FakeCrossEncoder()

"""Dense retriever tests.

The embedding model is replaced by the deterministic ``fake_embedder`` fixture,
so these tests exercise FAISS indexing, similarity ordering and persistence
without downloading a model. They are not evidence of real embedding quality.
"""

import json

import numpy as np
import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.dense_retriever import DenseRetriever

FAKE_DIM = 16


def make_chunk(chunk_id: str, content: str, doc_id: str = "doc-1", index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        chunk_index=index,
        strategy="fixed_size",
        start_char=0,
        end_char=len(content),
        token_count=len(content.split()),
        metadata={"url": f"https://example.test/{doc_id}", "title": "Synthetic Doc"},
    )


CORPUS = [
    make_chunk("c1", "Create a payment intent to charge a card.", index=0),
    make_chunk("c2", "Idempotency keys prevent duplicate charges.", index=1),
    make_chunk("c3", "Webhooks notify your endpoint about events.", index=2),
]


@pytest.mark.usefixtures("fake_embedder")
def indexed(chunks=CORPUS) -> DenseRetriever:
    retriever = DenseRetriever()
    retriever.index_chunks(list(chunks))
    return retriever


@pytest.mark.usefixtures("fake_embedder")
def test_index_dimension_and_count():
    retriever = indexed()

    assert retriever.index.ntotal == len(CORPUS)
    assert retriever.index.d == FAKE_DIM
    assert retriever.embeddings.shape == (len(CORPUS), FAKE_DIM)


@pytest.mark.usefixtures("fake_embedder")
def test_embeddings_are_finite_and_normalized():
    retriever = indexed()

    assert np.isfinite(retriever.embeddings).all()
    norms = np.linalg.norm(retriever.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.usefixtures("fake_embedder")
def test_retrieval_ranks_matching_chunk_first():
    retriever = indexed()

    results = retriever.retrieve("webhooks endpoint", top_k=3)

    assert results[0]["chunk_id"] == "c3"
    assert results[0]["score"] >= results[-1]["score"]


@pytest.mark.usefixtures("fake_embedder")
def test_scores_descend_with_rank():
    retriever = indexed()

    scores = [r["score"] for r in retriever.retrieve("payment charge", top_k=3)]

    assert scores == sorted(scores, reverse=True)


@pytest.mark.usefixtures("fake_embedder")
def test_top_k_larger_than_corpus():
    retriever = indexed()

    results = retriever.retrieve("payment", top_k=50)

    assert len(results) == len(CORPUS)
    assert [r["rank"] for r in results] == [1, 2, 3]


@pytest.mark.usefixtures("fake_embedder")
def test_empty_query_returns_no_results():
    assert indexed().retrieve("", top_k=3) == []


@pytest.mark.usefixtures("fake_embedder")
def test_empty_corpus_returns_no_results():
    retriever = DenseRetriever()
    retriever.index_chunks([])

    assert retriever.index is None
    assert retriever.retrieve("payment", top_k=3) == []


@pytest.mark.usefixtures("fake_embedder")
def test_batch_retrieval_matches_single_queries():
    retriever = indexed()

    batch = retriever.batch_retrieve(["payment", "webhooks"], top_k=2)
    single = [retriever.retrieve("payment", top_k=2), retriever.retrieve("webhooks", top_k=2)]

    assert [[r["chunk_id"] for r in rows] for rows in batch] == [
        [r["chunk_id"] for r in rows] for rows in single
    ]


@pytest.mark.usefixtures("fake_embedder")
def test_index_persistence_and_reload(tmp_path):
    retriever = indexed()
    retriever.save_index(str(tmp_path))

    reloaded = DenseRetriever()
    reloaded.load_index(str(tmp_path))

    assert reloaded.model is None  # no model loaded just to read the index
    assert reloaded.index.ntotal == retriever.index.ntotal
    assert np.array_equal(reloaded.embeddings, retriever.embeddings)
    assert reloaded.retrieve("idempotency", top_k=2) == retriever.retrieve(
        "idempotency", top_k=2
    )


@pytest.mark.usefixtures("fake_embedder")
def test_inconsistent_index_is_rejected(tmp_path):
    retriever = indexed()
    retriever.save_index(str(tmp_path))

    chunks_file = tmp_path / "dense_chunks.jsonl"
    extra = json.dumps(make_chunk("c99", "unexpected extra chunk").to_dict())
    with open(chunks_file, "a") as f:
        f.write(extra + "\n")

    with pytest.raises(ValueError):
        DenseRetriever().load_index(str(tmp_path))


@pytest.mark.usefixtures("fake_embedder")
def test_metadata_survives_retrieval(tmp_path):
    retriever = indexed()

    result = retriever.retrieve("charge", top_k=1)[0]

    assert result["metadata"]["url"] == "https://example.test/doc-1"
    assert result["doc_id"] == "doc-1"

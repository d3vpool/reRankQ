"""Synthetic end-to-end retrieval integration test.

Raw documents -> chunking -> BM25 + FAISS indexes -> hybrid RRF -> reranking.

The embedding model and cross-encoder are stubbed (see conftest), so this test
is integration coverage of the pipeline wiring, not of model quality.
"""

import json

import pytest

from src.ingestion.chunker import chunk_documents, load_chunks
from src.ingestion.crawler import Document
from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

pytestmark = pytest.mark.integration


SYNTHETIC_DOCS = [
    Document(
        doc_id="doc-payments",
        url="https://docs.stripe.com/payments",
        title="Payments",
        section="Payments",
        content=(
            "# Payments\n\n"
            "A payment intent tracks the lifecycle of a customer payment.\n\n"
            "## Creating intents\n\n"
            "Create a payment intent on the server and confirm it on the client. "
            "The intent handles authentication and capture.\n\n"
            "## Idempotency\n\n"
            "Send an idempotency key to make create requests safe to retry."
        ),
    ),
    Document(
        doc_id="doc-webhooks",
        url="https://docs.stripe.com/webhooks",
        title="Webhooks",
        section="Webhooks",
        content=(
            "# Webhooks\n\n"
            "Webhooks deliver event notifications to your endpoint.\n\n"
            "## Signatures\n\n"
            "Verify the webhook signature with the signing secret before trusting "
            "the payload.\n\n"
            "## Retries\n\n"
            "Stripe retries failed webhook deliveries with backoff."
        ),
    ),
    Document(
        doc_id="doc-billing",
        url="https://docs.stripe.com/billing",
        title="Billing",
        section="Billing",
        content=(
            "# Billing\n\n"
            "Subscriptions bill customers on a recurring schedule.\n\n"
            "## Invoices\n\n"
            "Invoices are generated for each billing period.\n\n"
            "## Refunds\n\n"
            "Refund a captured charge through the refunds API."
        ),
    ),
]


@pytest.fixture
def pipeline(tmp_path, fake_embedder):
    """Build both chunks sets, both indexes and a hybrid retriever."""
    processed = tmp_path / "processed"
    chunk_documents(SYNTHETIC_DOCS, strategy="both", output_dir=str(processed))
    chunks = load_chunks(str(processed), strategy="structure_aware")

    bm25 = BM25Retriever()
    bm25.index_chunks(chunks)

    dense = DenseRetriever()
    dense.index_chunks(chunks)

    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)

    return {
        "processed": processed,
        "chunks": chunks,
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
    }


def test_both_chunking_strategies_are_written(tmp_path):
    processed = tmp_path / "processed"
    result = chunk_documents(SYNTHETIC_DOCS, strategy="both", output_dir=str(processed))

    assert result["fixed_size"] and result["structure_aware"]
    assert (processed / "chunks_fixed_size.jsonl").exists()
    assert (processed / "chunks_structure_aware.jsonl").exists()

    for strategy, chunks in result.items():
        for chunk in chunks:
            assert chunk.doc_id
            assert chunk.metadata["url"].startswith("https://docs.stripe.com/")
            assert chunk.content.strip()


def test_all_four_variants_return_mapped_results(pipeline, fake_cross_encoder):
    chunks = pipeline["chunks"]
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    query = "webhook signature signing secret"

    bm25_results = pipeline["bm25"].retrieve(query, top_k=3)
    dense_results = pipeline["dense"].retrieve(query, top_k=3)
    hybrid_results = pipeline["hybrid"].retrieve(query, top_k=3)

    reranker = CrossEncoderReranker()
    reranker.model = fake_cross_encoder
    rerank_results = reranker.rerank(query, hybrid_results, top_k=3)

    for variant in (bm25_results, dense_results, hybrid_results, rerank_results):
        assert variant, "each variant must return results"
        for result in variant:
            assert result["chunk_id"] in chunk_ids
            assert result["metadata"]["url"].startswith("https://docs.stripe.com/")
            assert result["doc_id"] in {doc.doc_id for doc in SYNTHETIC_DOCS}


def test_retrieved_content_matches_stored_chunk(pipeline):
    stored = {chunk.chunk_id: chunk.content for chunk in pipeline["chunks"]}

    for result in pipeline["hybrid"].retrieve("payment intent", top_k=3):
        assert result["content"] == stored[result["chunk_id"]]


def test_indexes_persist_and_reload(pipeline, tmp_path, fake_embedder):
    indexes = tmp_path / "indexes"
    pipeline["hybrid"].save_indexes(str(indexes))

    manifest = json.loads((indexes / "index_manifest.json").read_text())
    assert manifest["num_chunks"] == len(pipeline["chunks"])
    assert manifest["embedding_model"] == "all-MiniLM-L6-v2"
    assert manifest["chunk_strategies"] == ["structure_aware"]

    reloaded = HybridRetriever(
        bm25_retriever=BM25Retriever(), dense_retriever=DenseRetriever()
    )
    reloaded.load_indexes(str(indexes))

    assert reloaded.retrieve("webhook signature", top_k=3) == pipeline["hybrid"].retrieve(
        "webhook signature", top_k=3
    )


def test_incompatible_manifest_is_rejected(pipeline, tmp_path, fake_embedder):
    indexes = tmp_path / "indexes"
    pipeline["hybrid"].save_indexes(str(indexes))

    manifest_path = indexes / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["embedding_model"] = "some/other-model"
    manifest_path.write_text(json.dumps(manifest))

    reloaded = HybridRetriever(
        bm25_retriever=BM25Retriever(), dense_retriever=DenseRetriever()
    )
    with pytest.raises(ValueError, match="embedding model"):
        reloaded.load_indexes(str(indexes))


def test_repeated_queries_are_consistent(pipeline):
    query = "subscription invoices refunds"

    runs = [pipeline["hybrid"].retrieve(query, top_k=3) for _ in range(3)]

    assert runs[0] == runs[1] == runs[2]


def test_bm25_and_dense_agree_on_obvious_query(pipeline):
    results_bm25 = pipeline["bm25"].retrieve("webhooks deliver event notifications", top_k=1)
    results_dense = pipeline["dense"].retrieve("webhooks deliver event notifications", top_k=1)

    assert results_bm25[0]["doc_id"] == "doc-webhooks"
    assert results_dense[0]["doc_id"] == "doc-webhooks"

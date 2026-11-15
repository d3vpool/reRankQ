"""Unit tests for the BM25 retriever (real library, synthetic corpus)."""

from src.ingestion.chunker import Chunk
from src.retrieval.bm25_retriever import BM25Retriever


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
        metadata={
            "url": f"https://example.test/{doc_id}",
            "title": "Synthetic Doc",
            "section": "Testing",
        },
    )


CORPUS = [
    make_chunk("c1", "Create a PaymentIntent to charge a customer card.", index=0),
    make_chunk("c2", "Idempotency keys prevent duplicate charges on retries.", index=1),
    make_chunk("c3", "Webhooks notify your endpoint about subscription events.", index=2),
    make_chunk("c4", "Refund a captured charge with the refunds endpoint.", index=3),
]


def indexed(chunks=CORPUS) -> BM25Retriever:
    retriever = BM25Retriever()
    retriever.index_chunks(list(chunks))
    return retriever


def test_empty_corpus_returns_no_results():
    retriever = BM25Retriever()
    retriever.index_chunks([])

    assert retriever.bm25 is None
    assert retriever.retrieve("payment", top_k=5) == []


def test_empty_query_returns_no_results():
    assert indexed().retrieve("", top_k=5) == []
    assert indexed().retrieve("   ", top_k=5) == []


def test_single_document_corpus():
    retriever = indexed([make_chunk("only", "Webhook signing secrets verify payloads.")])

    results = retriever.retrieve("webhook signing", top_k=5)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "only"
    # rank_bm25 applies an IDF floor, so scores on a 1-document corpus can be
    # negative; the contract is that the single chunk is returned.
    assert results[0]["score"] != 0.0


def test_exact_keyword_match_ranks_first():
    results = indexed().retrieve("idempotency", top_k=4)

    assert results[0]["chunk_id"] == "c2"


def test_multiple_documents_are_ranked():
    results = indexed().retrieve("subscription webhooks", top_k=4)

    assert results[0]["chunk_id"] == "c3"
    assert results[0]["score"] > results[-1]["score"]


def test_missing_query_terms_yield_zero_scores_but_no_error():
    results = indexed().retrieve("kubernetes helm chart", top_k=2)

    assert len(results) == 2
    assert all(result["score"] == 0.0 for result in results)


def test_top_k_larger_than_corpus_returns_all_chunks():
    results = indexed().retrieve("payment", top_k=50)

    assert len(results) == len(CORPUS)
    assert {r["chunk_id"] for r in results} == {"c1", "c2", "c3", "c4"}


def test_result_metadata_is_preserved():
    results = indexed().retrieve("refund", top_k=1)

    assert results[0]["metadata"]["url"] == "https://example.test/doc-1"
    assert results[0]["metadata"]["title"] == "Synthetic Doc"
    assert results[0]["doc_id"] == "doc-1"
    assert results[0]["content"].startswith("Refund")


def test_index_persistence_roundtrip(tmp_path):
    retriever = indexed()
    retriever.save_index(str(tmp_path))

    reloaded = BM25Retriever()
    reloaded.load_index(str(tmp_path))

    assert [c.chunk_id for c in reloaded.chunks] == [c.chunk_id for c in retriever.chunks]
    assert reloaded.retrieve("webhook", top_k=3) == retriever.retrieve("webhook", top_k=3)


def test_repeated_queries_are_deterministic():
    retriever = indexed()
    runs = [retriever.retrieve("charge", top_k=3) for _ in range(5)]

    assert all(run == runs[0] for run in runs)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def test_tokenizer_keeps_underscored_identifiers():
    tokens = BM25Retriever()._tokenize("payment_intent is a resource")

    assert "payment_intent" in tokens
    assert "resource" in tokens


def test_tokenizer_splits_camel_case():
    tokens = BM25Retriever()._tokenize("PaymentIntent")

    assert "paymentintent" in tokens
    assert "payment" in tokens
    assert "intent" in tokens


def test_tokenizer_keeps_and_splits_hyphenated_terms():
    tokens = BM25Retriever()._tokenize("idempotency-key")

    assert "idempotency-key" in tokens
    assert "idempotency" in tokens
    assert "key" in tokens


def test_tokenizer_handles_version_strings():
    tokens = BM25Retriever()._tokenize("API version 2020-08-27")

    assert "2020-08-27" in tokens
    assert "2020" in tokens
    assert "08" in tokens
    assert "27" in tokens


def test_tokenizer_does_not_inflate_term_frequency():
    tokens = BM25Retriever()._tokenize("payment payment_intent")

    assert tokens.count("payment") == 1
    assert tokens.count("payment_intent") == 1


def test_query_and_document_tokenization_are_symmetric():
    retriever = BM25Retriever()

    camel = retriever._tokenize("PaymentIntent")
    flat = retriever._tokenize("paymentintent")

    # The compound form is shared by both, so a CamelCase query still matches
    # an already-lowercased document token.
    assert set(flat) <= set(camel)
    assert retriever._tokenize("webhooks deliver") == ["webhooks", "deliver"]

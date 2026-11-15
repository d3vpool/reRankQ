"""Cross-encoder reranker tests.

The cross-encoder is replaced by ``FakeCrossEncoder`` (see conftest). These
tests verify ordering, truncation and result mapping — not model quality.
"""

import pytest

from src.reranking.reranker import CrossEncoderReranker, RerankingPipeline


def candidate(chunk_id: str, content: str, doc_id: str = "doc-1") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "score": 0.5,
        "content": content,
        "metadata": {
            "url": f"https://example.test/{chunk_id}",
            "title": f"Title {chunk_id}",
            "section": "Testing",
        },
        "rank": 1,
    }


CANDIDATES = [
    candidate("weak", "unrelated billing information"),
    candidate("strong", "webhook signing secret verification"),
    candidate("medium", "webhook configuration"),
]


def reranker(fake_cross_encoder) -> CrossEncoderReranker:
    instance = CrossEncoderReranker()
    instance.model = fake_cross_encoder
    return instance


def test_reranking_reorders_by_score(fake_cross_encoder):
    results = reranker(fake_cross_encoder).rerank("webhook signing secret", CANDIDATES, top_k=3)

    assert [r["chunk_id"] for r in results][:2] == ["strong", "medium"]
    assert [r["rank"] for r in results] == [1, 2, 3]


def test_reranking_truncates_to_top_k(fake_cross_encoder):
    results = reranker(fake_cross_encoder).rerank("webhook", CANDIDATES, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "strong"


def test_reranking_preserves_identity_and_metadata(fake_cross_encoder):
    results = reranker(fake_cross_encoder).rerank("webhook signing", CANDIDATES, top_k=3)

    assert {r["chunk_id"] for r in results} == {"weak", "strong", "medium"}
    for result in results:
        assert result["doc_id"] == "doc-1"
        assert result["metadata"]["url"].startswith("https://example.test/")
        assert result["content"]
        assert result["original_score"] == 0.5


def test_reranking_adds_no_new_documents(fake_cross_encoder):
    results = reranker(fake_cross_encoder).rerank("webhook", CANDIDATES, top_k=10)

    assert len({r["chunk_id"] for r in results}) == len(CANDIDATES) == len(results)


def test_duplicate_candidates_are_collapsed(fake_cross_encoder):
    duplicated = CANDIDATES + [CANDIDATES[0]]

    results = reranker(fake_cross_encoder).rerank("webhook", duplicated, top_k=10)

    assert [r["chunk_id"] for r in results].count("weak") == 1
    assert len(results) == len(CANDIDATES)


def test_empty_candidates_return_empty_list(fake_cross_encoder):
    instance = CrossEncoderReranker()

    assert instance.rerank("webhook", [], top_k=5) == []
    assert instance.model is None  # no model was loaded


def test_scores_are_sorted_descending(fake_cross_encoder):
    results = reranker(fake_cross_encoder).rerank("webhook signing", CANDIDATES, top_k=3)

    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_reranking_pipeline_adds_citations(fake_cross_encoder):
    pipeline = RerankingPipeline(reranker(fake_cross_encoder))

    results = pipeline.rerank_hybrid_results("webhook", CANDIDATES, top_k=2)

    assert len(results) == 2
    for result in results:
        citation = result["citation"]
        assert citation["chunk_id"] == result["chunk_id"]
        assert citation["url"] == result["metadata"]["url"]
        assert citation["section"] == "Testing"

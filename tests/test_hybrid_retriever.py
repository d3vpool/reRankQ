"""Hybrid retrieval tests, including hand-computed RRF expectations."""

import pytest

from src.retrieval.hybrid_retriever import HybridRetriever


def result(chunk_id: str, score: float = 1.0, content: str = "") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": f"doc-{chunk_id}",
        "score": score,
        "content": content or f"content for {chunk_id}",
        "metadata": {"url": f"https://example.test/{chunk_id}", "title": "Doc"},
        "rank": 1,
    }


def fusion() -> HybridRetriever:
    """Retriever without touching indexes or models."""
    return HybridRetriever(bm25_retriever=object(), dense_retriever=object())


# ---------------------------------------------------------------------------
# RRF mathematics
# ---------------------------------------------------------------------------

def test_rrf_matches_hand_computed_scores():
    """k=60, equal weights: c1=1/61, c2=1/62+1/61, c3=1/62."""
    bm25 = [result("c1"), result("c2")]
    dense = [result("c2"), result("c3")]

    fused = fusion().reciprocal_rank_fusion(bm25, dense, k=60, bm25_weight=1.0, dense_weight=1.0)

    scores = {row["chunk_id"]: row["rrf_score"] for row in fused}
    assert scores["c1"] == pytest.approx(1 / 61)
    assert scores["c2"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c3"] == pytest.approx(1 / 62)
    assert [row["chunk_id"] for row in fused] == ["c2", "c1", "c3"]


def test_rrf_respects_weights():
    bm25 = [result("a"), result("b")]
    dense = [result("b"), result("a")]

    bm25_heavy = fusion().reciprocal_rank_fusion(
        bm25, dense, k=60, bm25_weight=1.0, dense_weight=0.0
    )
    dense_heavy = fusion().reciprocal_rank_fusion(
        bm25, dense, k=60, bm25_weight=0.0, dense_weight=1.0
    )

    assert [r["chunk_id"] for r in bm25_heavy] == ["a", "b"]
    assert [r["chunk_id"] for r in dense_heavy] == ["b", "a"]


def test_rrf_k_constant_changes_score_spread():
    bm25 = [result("a"), result("b")]
    dense: list[dict] = []

    low_k = fusion().reciprocal_rank_fusion(bm25, dense, k=1)
    high_k = fusion().reciprocal_rank_fusion(bm25, dense, k=1000)

    assert low_k[0]["rrf_score"] > high_k[0]["rrf_score"]
    assert [r["chunk_id"] for r in low_k] == [r["chunk_id"] for r in high_k] == ["a", "b"]


def test_chunk_in_only_one_ranking_is_kept():
    bm25 = [result("shared"), result("bm25_only")]
    dense = [result("shared"), result("dense_only")]

    fused = fusion().reciprocal_rank_fusion(bm25, dense, k=60)

    by_id = {row["chunk_id"]: row for row in fused}
    assert set(by_id) == {"shared", "bm25_only", "dense_only"}
    assert by_id["bm25_only"]["dense_rank"] is None
    assert by_id["dense_only"]["bm25_rank"] is None
    assert by_id["shared"]["bm25_rank"] == 1
    assert by_id["shared"]["dense_rank"] == 1
    assert fused[0]["chunk_id"] == "shared"


def test_duplicate_chunk_ids_appear_once():
    bm25 = [result("dup", content="first"), result("dup", content="second")]
    dense = [result("dup")]

    fused = fusion().reciprocal_rank_fusion(bm25, dense, k=60)

    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "dup"
    assert fused[0]["content"] == "first"
    assert fused[0]["bm25_rank"] == 1


def test_empty_inputs_return_empty_list():
    assert fusion().reciprocal_rank_fusion([], [], k=60) == []


def test_rrf_output_is_deterministic():
    bm25 = [result("a"), result("b"), result("c")]
    dense = [result("c"), result("b"), result("a")]

    orders = {
        tuple(r["chunk_id"] for r in fusion().reciprocal_rank_fusion(bm25, dense, k=60))
        for _ in range(25)
    }

    assert len(orders) == 1


def test_rrf_ranks_are_sequential():
    fused = fusion().reciprocal_rank_fusion([result("a")], [result("b")], k=60)

    assert [row["rank"] for row in fused] == [1, 2]


# ---------------------------------------------------------------------------
# Weighted fusion (secondary path)
# ---------------------------------------------------------------------------

def test_weighted_fusion_tolerates_zero_scores():
    bm25 = [result("a", 0.0), result("b", 0.0)]
    dense = [result("b", 0.0)]

    fused = fusion()._weighted_fusion(bm25, dense, bm25_weight=0.5, dense_weight=0.5)

    assert len(fused) == 2
    assert all(row["combined_score"] == 0.0 for row in fused)


def test_weighted_fusion_normalizes_and_orders():
    bm25 = [result("a", 10.0), result("b", 5.0)]
    dense = [result("b", 1.0)]

    fused = fusion()._weighted_fusion(bm25, dense, bm25_weight=1.0, dense_weight=0.0)

    scores = {row["chunk_id"]: row["combined_score"] for row in fused}
    assert scores["a"] == pytest.approx(1.0)
    assert scores["b"] == pytest.approx(0.5)
    assert [row["chunk_id"] for row in fused] == ["a", "b"]


# ---------------------------------------------------------------------------
# Retriever integration with stubbed components
# ---------------------------------------------------------------------------

class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, top_k=10):
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_retrieve_combines_both_retrievers_and_truncates():
    bm25 = StubRetriever([result("a"), result("b"), result("c")])
    dense = StubRetriever([result("c"), result("a")])
    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)

    fused = hybrid.retrieve("payment", top_k=2)

    assert len(fused) == 2
    assert fused[0]["chunk_id"] == "a"
    # 2x candidates are requested from each retriever before fusion
    assert bm25.calls == [("payment", 4)]
    assert dense.calls == [("payment", 4)]


def test_retrieve_with_both_retrievers_empty():
    hybrid = HybridRetriever(
        bm25_retriever=StubRetriever([]), dense_retriever=StubRetriever([])
    )

    assert hybrid.retrieve("payment", top_k=5) == []


def test_retrieve_preserves_chunk_metadata():
    bm25 = StubRetriever([result("a", 2.0)])
    dense = StubRetriever([result("a", 0.9)])
    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)

    fused = hybrid.retrieve("payment", top_k=1)

    assert fused[0]["metadata"]["url"] == "https://example.test/a"
    assert fused[0]["doc_id"] == "doc-a"

"""Unit tests for BM25 + RRF hybrid retrieval helpers."""

from ai_core.vector_store.bm25_index import (
    BM25Index,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_keeps_grant_codes():
    tokens = tokenize("FMRPO digitalizacija ZDK EU4Agri")
    assert "fmrpo" in tokens
    assert "eu4agri" in tokens
    assert "zdk" in tokens


def test_bm25_ranks_exact_program_name_higher():
    index = BM25Index()
    index.build(
        ids=["a", "b", "c"],
        documents=[
            "Opci poticaj za MSP u Federaciji",
            "FMRPO poziv za digitalizaciju malih preduzeca ZDK",
            "Poljoprivredni program EU4Agri ruralni razvoj",
        ],
    )
    top = index.top_n("FMRPO digitalizacija", n=2)
    assert top[0][0] == "b"
    assert top[0][1] > 0


def test_rrf_prefers_consensus():
    # y je rank-1 u obje liste → jasno pobjedjuje
    fused = reciprocal_rank_fusion(
        [
            ["y", "x", "z"],
            ["y", "w", "x"],
        ],
        k=60,
    )
    assert fused[0][0] == "y"


def test_rrf_weights():
    fused = reciprocal_rank_fusion(
        [
            ["a", "b"],
            ["b", "a"],
        ],
        weights=[2.0, 0.1],
        k=60,
    )
    assert fused[0][0] == "a"


def test_empty_query_scores_zero():
    index = BM25Index()
    index.build(ids=["a"], documents=["FMRPO digitalizacija"])
    scores = index.score("")
    assert scores == [("a", 0.0)]

"""Unit tests for grounded AI context builder."""

from backend.app.api.search import (
    _build_grounded_context,
    _meta_field,
)


def test_meta_field_missing_is_not_available():
    assert _meta_field({}, "url") == "NOT_AVAILABLE"
    assert _meta_field({"url": ""}, "url") == "NOT_AVAILABLE"
    assert _meta_field({"url": "  "}, "url") == "NOT_AVAILABLE"


def test_meta_field_keeps_value():
    assert _meta_field({"url": "https://zeda.ba/"}, "url") == "https://zeda.ba/"


def test_grounded_context_contains_url_and_scores():
    docs = ["Opis FMRPO poziva za MSP."]
    metas = [
        {
            "grant_id": "local_grant_001",
            "title": "FMRPO Grant 2026",
            "category": "Federalni",
            "status": "zatvoren",
            "deadline": "11.03.2026",
            "budget": "27.400.000 KM",
            "url": "https://javnipozivi.fmrpo.gov.ba/",
            "verified_score": 20,
            "source_priority": 50,
            "next_expected": "oko februar 2027",
        }
    ]
    context, sources = _build_grounded_context(docs, metas)

    assert "URL: https://javnipozivi.fmrpo.gov.ba/" in context
    assert "VERIFIED_SCORE: 20" in context
    assert "NEXT_EXPECTED: oko februar 2027" in context
    assert "DEADLINE: 11.03.2026" in context
    assert sources[0]["url"] == "https://javnipozivi.fmrpo.gov.ba/"


def test_grounded_context_marks_missing_url():
    docs = ["Bez linka."]
    metas = [{"title": "Test", "grant_id": "x"}]
    context, sources = _build_grounded_context(docs, metas)

    assert "URL: NOT_AVAILABLE" in context
    assert sources == []


def test_empty_lists():
    context, sources = _build_grounded_context([], [])
    assert context == "Nema pronađenih grantova."
    assert sources == []

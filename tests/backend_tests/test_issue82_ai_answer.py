"""Issue #82 tests for structured AI-answer responses."""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend.app.api.schemas import (
    AIAnswerRequest,
    AIAnswerResponse,
)
from backend.app.api.search import (
    _build_grant_results,
    ai_answer_endpoint,
)
from backend.app.services import ai as ai_services


class DummyCollection:
    def count(self):
        return 2


class DummyEmbeddingClient:
    def __init__(self):
        self.calls = 0

    def generate_embeddings(self, texts):
        self.calls += 1
        assert texts == ["digitalizacija MSP ZDK"]
        return [[0.1, 0.2, 0.3]]


class DummyChromaClient:
    def __init__(self):
        self.collection = DummyCollection()
        self.calls = 0

    def query_hybrid(
        self,
        *,
        query_text,
        query_embeddings,
        n_results,
    ):
        self.calls += 1
        assert query_text == "digitalizacija MSP ZDK"
        assert query_embeddings == [[0.1, 0.2, 0.3]]
        assert n_results == 2

        return {
            "documents": [[
                "Opis prvog programa",
                "Opis drugog programa",
            ]],
            "metadatas": [[
                {
                    "grant_id": "grant-high",
                    "title": "Program visokog prioriteta",
                    "category": "Digitalizacija",
                    "status": "verified",
                    "deadline": "",
                    "budget": None,
                    "url": "https://example.org/high",
                    "relevance": "high",
                    "verified_score": "95",
                    "source_priority": "90",
                },
                {
                    "grant_id": "grant-low",
                    "title": "Program nižeg prioriteta",
                    "category": "Opći",
                    "status": "verified",
                    "deadline": "2026-12-31",
                    "budget": "100000 EUR",
                    "url": "https://example.org/low",
                    "relevance": "medium",
                    "verified_score": "invalid",
                    "source_priority": None,
                },
            ]],
        }


class DummyGenAIClient:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        assert "digitalizacija MSP ZDK" in prompt
        return "Testni verificirani AI odgovor."


def _run_ai_answer():
    request = AIAnswerRequest(
        query="digitalizacija MSP ZDK",
        language="bs",
    )

    return asyncio.run(
        ai_answer_endpoint(
            request=request,
            current_user="qa@example.test",
        )
    )


def test_language_is_limited_to_bs_or_en():
    with pytest.raises(ValidationError):
        AIAnswerRequest(
            query="digitalizacija",
            language="de",
        )


def test_response_results_default_to_empty_list():
    response = AIAnswerResponse(
        answer="Odgovor",
        sources=[],
        request_id="request-id",
        processing_time=0.1,
    )

    assert response.results == []


def test_build_grant_results_normalizes_metadata():
    documents = ["Opis"]
    metadatas = [{
        "grant_id": "grant-1",
        "title": "Grant 1",
        "deadline": "",
        "budget": None,
        "verified_score": "invalid",
        "source_priority": "80",
        "url": "https://example.org/grant-1",
    }]

    results = _build_grant_results(
        documents,
        metadatas,
    )

    assert results == [{
        "grant_id": "grant-1",
        "title": "Grant 1",
        "category": "",
        "status": "",
        "deadline": None,
        "budget": None,
        "url": "https://example.org/grant-1",
        "relevance": "",
        "verified_score": None,
        "source_priority": 80,
    }]


def test_result_without_stable_grant_id_is_omitted():
    results = _build_grant_results(
        ["Opis"],
        [{
            "title": "Grant bez stabilnog ID-a",
            "url": "https://example.org/no-id",
        }],
    )

    assert results == []


def test_ai_answer_uses_one_embedding_and_one_query(
    monkeypatch,
):
    embedding = DummyEmbeddingClient()
    chroma = DummyChromaClient()
    genai = DummyGenAIClient()

    monkeypatch.setattr(
        ai_services,
        "embedding_client",
        embedding,
    )
    monkeypatch.setattr(
        ai_services,
        "chroma_client",
        chroma,
    )
    monkeypatch.setattr(
        ai_services,
        "genai_client",
        genai,
    )

    response = _run_ai_answer()

    assert embedding.calls == 1
    assert chroma.calls == 1
    assert genai.calls == 1
    assert len(response.results) == 2

    first = response.results[0]
    second = response.results[1]

    assert first.grant_id == "grant-high"
    assert first.deadline is None
    assert first.budget is None
    assert first.verified_score == 95
    assert first.source_priority == 90

    assert second.grant_id == "grant-low"
    assert second.deadline == "2026-12-31"
    assert second.budget == "100000 EUR"
    assert second.verified_score is None
    assert second.source_priority is None

    assert response.sources[0]["url"] == (
        "https://example.org/high"
    )


def test_ai_failure_does_not_expose_exception(
    monkeypatch,
):
    embedding = DummyEmbeddingClient()
    chroma = DummyChromaClient()
    genai = MagicMock()
    genai.generate.side_effect = RuntimeError(
        "sensitive internal failure"
    )

    monkeypatch.setattr(
        ai_services,
        "embedding_client",
        embedding,
    )
    monkeypatch.setattr(
        ai_services,
        "chroma_client",
        chroma,
    )
    monkeypatch.setattr(
        ai_services,
        "genai_client",
        genai,
    )

    with pytest.raises(HTTPException) as exc_info:
        _run_ai_answer()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == (
        "Došlo je do interne greške. "
        "Pokušajte ponovo."
    )
    assert "sensitive" not in exc_info.value.detail

"""Issue #82 tests for stable grant details lookup."""

import pytest
from fastapi import HTTPException

from backend.app.api import grants as grants_api


def test_get_existing_grant_by_stable_id(
    monkeypatch,
):
    cached = [
        {
            "id": "grant-1",
            "title": "Prvi grant",
            "url": "https://example.org/grant-1",
        },
        {
            "id": "grant-2",
            "title": "Drugi grant",
            "url": "https://example.org/grant-2",
        },
    ]

    monkeypatch.setattr(
        grants_api.ai_services,
        "_grants_cache",
        cached,
    )

    result = grants_api.get_grant_by_id(
        "grant-2"
    )

    assert result == cached[1]


def test_get_missing_grant_returns_404(
    monkeypatch,
):
    monkeypatch.setattr(
        grants_api.ai_services,
        "_grants_cache",
        [{
            "id": "grant-1",
            "title": "Prvi grant",
        }],
    )

    with pytest.raises(HTTPException) as exc_info:
        grants_api.get_grant_by_id(
            "missing-grant"
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == (
        "Grant nije pronađen."
    )


def test_local_route_still_works(
    monkeypatch,
):
    monkeypatch.setattr(
        grants_api.ai_services,
        "_grants_cache",
        [{
            "id": "local-1",
            "title": "ZDK program",
            "category": "ZDK",
            "description": "Program za ZDK",
        }],
    )

    response = grants_api.list_local_grants()

    assert response["total"] == 1
    assert response["grants"][0]["id"] == (
        "local-1"
    )


def test_urgent_route_still_works(
    monkeypatch,
):
    monkeypatch.setattr(
        grants_api.ai_services,
        "_grants_cache",
        [],
    )

    response = grants_api.list_urgent_grants()

    assert response["total"] == 0
    assert response["grants"] == []

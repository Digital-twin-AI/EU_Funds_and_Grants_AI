"""Guest Demo regression tests without external service calls."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from backend.app.api.schemas import (
    AIAnswerRequest,
    AIAnswerResponse,
    DemoAIAnswerResponse,
)
from backend.app.api.search import (
    demo_ai_answer_endpoint,
)
from backend.app.core import config
from backend.app.core.rate_limit import (
    _guest_quota_store,
    _rate_limit_store,
    consume_guest_quota,
    derive_guest_client_key,
    inspect_guest_quota,
    reset_guest_quota_for_tests,
)


@pytest.fixture(autouse=True)
def clean_guest_quota():
    """Isolate Guest quota without clearing general rate-limit state."""
    reset_guest_quota_for_tests()
    yield
    reset_guest_quota_for_tests()


def make_request(host: str = "demo-client") -> SimpleNamespace:
    return SimpleNamespace(
        client=SimpleNamespace(host=host),
    )


def successful_ai_response() -> AIAnswerResponse:
    return AIAnswerResponse(
        answer="Test odgovor",
        results=[],
        sources=[],
        request_id="test-request-id",
        processing_time=0.01,
    )


def call_demo(host: str = "demo-client") -> DemoAIAnswerResponse:
    request = AIAnswerRequest(
        query="test grant upit",
        language="bs",
    )

    with patch(
        "backend.app.api.search._execute_ai_answer",
        new=AsyncMock(return_value=successful_ai_response()),
    ):
        return asyncio.run(
            demo_ai_answer_endpoint(
                request=request,
                http_request=make_request(host),
            )
        )


def test_guest_config_contract():
    assert config.GUEST_DEMO_LIMIT == 3
    assert config.GUEST_DEMO_WINDOW_SECONDS == 3600


def test_inspection_does_not_consume():
    first = inspect_guest_quota("inspect-client")
    second = inspect_guest_quota("inspect-client")

    assert first["remaining"] == 3
    assert second["remaining"] == 3
    assert first["allowed"] is True
    assert second["allowed"] is True


def test_three_successful_demo_responses():
    assert call_demo().demo.remaining == 2
    assert call_demo().demo.remaining == 1

    third = call_demo()
    assert third.demo.limit == 3
    assert third.demo.remaining == 0
    assert third.demo.reset_after_seconds >= 0


def test_fourth_demo_request_returns_429_without_ai_call():
    call_demo()
    call_demo()
    call_demo()

    mocked = AsyncMock(return_value=successful_ai_response())

    with patch(
        "backend.app.api.search._execute_ai_answer",
        new=mocked,
    ):
        with pytest.raises(HTTPException) as captured:
            asyncio.run(
                demo_ai_answer_endpoint(
                    request=AIAnswerRequest(
                        query="četvrti demo upit",
                        language="bs",
                    ),
                    http_request=make_request(),
                )
            )

    assert captured.value.status_code == 429
    assert "registr" in str(captured.value.detail).lower()
    assert int(captured.value.headers["Retry-After"]) >= 0
    mocked.assert_not_awaited()


def test_separate_client_keys_are_isolated():
    assert call_demo("client-a").demo.remaining == 2
    assert call_demo("client-a").demo.remaining == 1
    assert call_demo("client-b").demo.remaining == 2


def test_window_expiration_resets_quota():
    key = "expired-client"
    _guest_quota_store[key] = [
        time.time() - config.GUEST_DEMO_WINDOW_SECONDS - 5
    ]

    quota = inspect_guest_quota(key)

    assert quota["allowed"] is True
    assert quota["remaining"] == 3
    assert quota["reset_after_seconds"] == 3600


@pytest.mark.parametrize("status_code", [500, 503])
def test_failed_ai_response_does_not_consume_quota(status_code):
    host = f"failure-{status_code}"
    key = derive_guest_client_key(make_request(host))
    before = inspect_guest_quota(key)

    with patch(
        "backend.app.api.search._execute_ai_answer",
        new=AsyncMock(
            side_effect=HTTPException(
                status_code=status_code,
                detail="controlled failure",
            )
        ),
    ):
        with pytest.raises(HTTPException) as captured:
            asyncio.run(
                demo_ai_answer_endpoint(
                    request=AIAnswerRequest(
                        query="valid failure test",
                        language="bs",
                    ),
                    http_request=make_request(host),
                )
            )

    after = inspect_guest_quota(key)

    assert captured.value.status_code == status_code
    assert before["remaining"] == 3
    assert after["remaining"] == 3


def test_consume_never_returns_negative_remaining():
    key = "bounded-client"

    for _ in range(10):
        quota = consume_guest_quota(key)
        assert quota["remaining"] >= 0
        assert quota["reset_after_seconds"] >= 0

    assert len(_guest_quota_store[key]) <= 3


def test_guest_reset_does_not_clear_general_limiter():
    marker = time.time()
    _rate_limit_store["general-client"] = [marker]
    consume_guest_quota("guest-client")

    reset_guest_quota_for_tests()

    assert "guest-client" not in _guest_quota_store
    assert _rate_limit_store["general-client"] == [marker]


def test_deterministic_fallback_client_key():
    assert derive_guest_client_key(None) == "unknown-client"
    assert derive_guest_client_key(SimpleNamespace(client=None)) == (
        "unknown-client"
    )


def test_demo_response_has_authoritative_quota():
    response = call_demo("response-client")

    assert isinstance(response, DemoAIAnswerResponse)
    assert response.demo.limit == 3
    assert response.demo.remaining == 2
    assert 0 <= response.demo.reset_after_seconds <= 3600


def test_route_and_security_source_contract():
    source = Path(
        "backend/app/api/search.py"
    ).read_text(encoding="utf-8")

    assert source.count('@router.post("/demo/ai-answer"') == 1
    assert source.count('@router.post("/ai-answer"') == 1
    assert source.count("Depends(get_current_user)") == 3
    assert "get_optional_user" not in source


def test_authenticated_response_schema_has_no_demo_field():
    response = successful_ai_response()
    body = response.model_dump()

    assert set(body) == {
        "answer",
        "results",
        "sources",
        "request_id",
        "processing_time",
    }
    assert "demo" not in body

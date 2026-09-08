"""Issue #85 HTTP-level tests for the local frontend CORS origin."""

from fastapi.testclient import TestClient

from backend.app.main import app


LOCAL_ANDROID_ORIGIN = "http://127.0.0.1:3000"


def test_local_android_frontend_register_preflight_is_allowed():
    """The Android browser origin may preflight auth registration."""
    client = TestClient(app)

    response = client.options(
        "/auth/register",
        headers={
            "Origin": LOCAL_ANDROID_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        LOCAL_ANDROID_ORIGIN
    )
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "content-type" in (
        response.headers["access-control-allow-headers"].lower()
    )


def test_unknown_local_origin_is_not_allowed():
    """The fix must not introduce a wildcard CORS policy."""
    client = TestClient(app)

    response = client.options(
        "/auth/register",
        headers={
            "Origin": "http://127.0.0.1:3999",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers

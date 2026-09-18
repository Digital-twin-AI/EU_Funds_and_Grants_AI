"""
backend/app/core/rate_limit.py — In-memory rate limiter (per-IP, sliding window).
"""

import time
from collections import defaultdict
from typing import Dict, List

from backend.app.core import config

_rate_limit_store: Dict[str, List[float]] = defaultdict(list)

# Kopirano iz config-a na nivo modula da testovi mogu privremeno smanjiti limit
RATE_LIMIT_REQUESTS = config.RATE_LIMIT_REQUESTS
RATE_LIMIT_WINDOW = config.RATE_LIMIT_WINDOW


def _check_rate_limit(ip: str) -> bool:
    """Vraća True ako IP nije prešao limit. Čisti stare upise automatski."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    calls = _rate_limit_store[ip]
    # Ukloni zahtjeve starije od window-a
    _rate_limit_store[ip] = [t for t in calls if t > window_start]
    if len(_rate_limit_store[ip]) >= RATE_LIMIT_REQUESTS:
        return False
    _rate_limit_store[ip].append(now)
    return True

# ---------------------------------------------------------------------------
# Guest Demo quota
#
# This is intentionally a best-effort in-memory MVP:
# - process restart clears Guest quota
# - multiple workers do not share state
# - visitors behind shared NAT may share a client key
# - proxy IP detection is intentionally limited
# - inspect then consume is not atomic
# - concurrent requests may exceed the nominal limit
# ---------------------------------------------------------------------------

_guest_quota_store: Dict[str, List[float]] = defaultdict(list)


def _active_guest_calls(
    client_key: str,
    *,
    now: float | None = None,
) -> tuple[List[float], float]:
    """Return active Guest calls and the effective current time."""
    effective_now = time.time() if now is None else now
    window_start = effective_now - config.GUEST_DEMO_WINDOW_SECONDS
    active = [
        timestamp
        for timestamp in _guest_quota_store[client_key]
        if timestamp > window_start
    ]
    _guest_quota_store[client_key] = active
    return active, effective_now


def _guest_quota_result(
    calls: List[float],
    now: float,
) -> dict:
    """Build a non-negative authoritative Guest quota snapshot."""
    limit = config.GUEST_DEMO_LIMIT
    remaining = max(0, limit - len(calls))

    if calls:
        elapsed = max(0.0, now - min(calls))
        reset_after = max(
            0,
            int(config.GUEST_DEMO_WINDOW_SECONDS - elapsed + 0.999999),
        )
    else:
        reset_after = config.GUEST_DEMO_WINDOW_SECONDS

    return {
        "allowed": remaining > 0,
        "limit": limit,
        "remaining": remaining,
        "reset_after_seconds": reset_after,
    }


def inspect_guest_quota(client_key: str) -> dict:
    """Inspect Guest quota without consuming an AI response."""
    calls, now = _active_guest_calls(client_key)
    return _guest_quota_result(calls, now)


def consume_guest_quota(client_key: str) -> dict:
    """
    Record one successful Guest AI response.

    The store is bounded to the configured limit. The surrounding endpoint
    performs inspection first, but this defensive check also prevents an
    unbounded list when concurrent requests complete together.
    """
    calls, now = _active_guest_calls(client_key)

    if len(calls) < config.GUEST_DEMO_LIMIT:
        calls.append(now)
        _guest_quota_store[client_key] = calls

    return _guest_quota_result(calls, now)


def reset_guest_quota_for_tests() -> None:
    """Clear Guest quota only, preserving the general rate-limit store."""
    _guest_quota_store.clear()


def derive_guest_client_key(request: object) -> str:
    """
    Derive a deterministic best-effort client key.

    Arbitrary X-Forwarded-For values are not trusted. request.client.host is
    used when available. The fixed fallback avoids creating a new unlimited
    identity for every request.
    """
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)

    if host:
        return str(host)

    return "unknown-client"

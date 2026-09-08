"""
backend/app/api/grants.py — REST endpointi za pregled grantova.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException

from backend.app.services import ai as ai_services

router = APIRouter()

LOCAL_CATEGORIES = {"Kantonalni (ZDK)", "Lokalni", "ZDK", "Tešanj", "ZEDA"}
LOCAL_KEYWORDS = ["zdk", "tešanj", "tesanj", "zeda", "zenica", "kanton", "federalni zavod za zapošljavanje"]


@router.get("/grants")
def list_grants(
    category: Optional[str] = None,
    relevance: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """
    Vraća listu svih dostupnih grantova iz baze.
    Parametri: category, relevance (High/Medium/Low), page, page_size.
    """
    filtered = ai_services._grants_cache

    if category:
        filtered = [g for g in filtered if category.lower() in g.get("category", "").lower()]
    if relevance:
        filtered = [g for g in filtered if g.get("relevance", "").lower() == relevance.lower()]

    total = len(filtered)
    start = (page - 1) * page_size
    paginated = filtered[start: start + page_size]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
        "grants": paginated,
    }


@router.get("/grants/local")
def list_local_grants():
    """
    Vraća grantove relevantne za ZDK kanton i Tešanj (Tier 1 prioritet).
    Uključuje kantonalne i federalne pozive koji pokrivaju ZDK područje.
    """
    local = []
    for g in ai_services._grants_cache:
        cat = g.get("category", "").lower()
        title = g.get("title", "").lower()
        desc = g.get("description", "").lower()
        combined = f"{cat} {title} {desc}"
        if any(kw in combined for kw in LOCAL_KEYWORDS) or g.get("category") in LOCAL_CATEGORIES:
            local.append(g)

    return {
        "region": "ZDK — Zeničko-dobojski kanton / Tešanj",
        "total": len(local),
        "note": "Ovi grantovi su direktno relevantni za poduzetnike iz ZDK kantona.",
        "grants": local,
    }


@router.get("/grants/urgent")
def list_urgent_grants(days: int = 30):
    """
    Vraća grantove čiji rok ističe unutar zadanog broja dana (default: 30).
    """
    today = datetime.now(timezone.utc).date()
    urgent = []
    for g in ai_services._grants_cache:
        deadline_str = g.get("deadline", "")
        if not deadline_str or deadline_str in ("N/A", "Varijabilno", ""):
            continue
        try:
            deadline = datetime.strptime(deadline_str, "%Y-%m-%d").date()
            days_left = (deadline - today).days
            if 0 <= days_left <= days:
                urgent.append({**g, "days_left": days_left})
        except ValueError:
            continue

    urgent.sort(key=lambda x: x["days_left"])

    return {
        "window_days": days,
        "total": len(urgent),
        "as_of": today.isoformat(),
        "grants": urgent,
    }

@router.get("/grants/{grant_id}")
def get_grant_by_id(grant_id: str):
    """Return one full grant record by its stable dataset ID."""
    for grant in ai_services._grants_cache:
        if grant.get("id") == grant_id:
            return grant

    raise HTTPException(
        status_code=404,
        detail="Grant nije pronađen.",
    )

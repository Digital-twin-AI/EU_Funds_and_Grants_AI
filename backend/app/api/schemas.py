"""
backend/app/api/schemas.py — Pydantic modeli (request/response ugovori API-ja).
"""

import re
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


class RegisterRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("Neispravan format email adrese.")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Lozinka mora imati minimalno 6 karaktera.")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    email: str
    plan: str


class SearchRequest(BaseModel):
    """Model zahtjeva koji stiže sa Frontenda."""
    query: str = Field(..., min_length=3, description="Korisnički upit, npr. 'startups in BiH'")
    n_results: int = Field(default=5, ge=1, le=20, description="Broj rezultata za vratiti")


class SearchResponse(BaseModel):
    """Format odgovora koji Frontend očekuje."""
    results: List[str]                       # Jednostavna lista za prikaz na karticama
    documents: List[List[str]]               # Raw dokumenti iz ChromaDB (za debug)
    metadatas: List[List[Dict[str, Any]]]    # Metapodaci (Izvor, Godina, Kategorija)
    request_id: str
    processing_time: float


class GrantSummary(BaseModel):
    """Structured grant result returned by the AI endpoint."""

    grant_id: str
    title: str
    category: str = ""
    status: str = ""
    deadline: str | None = None
    budget: str | None = None
    url: str = ""
    relevance: str = ""
    verified_score: int | None = None
    source_priority: int | None = None


class AIAnswerRequest(BaseModel):
    """Zahtjev za AI odgovor (RAG + Gemini generacija)."""
    query: str = Field(..., min_length=3, max_length=500)
    language: Literal["bs", "en"] = Field(
        default="bs",
        description="Jezik odgovora: 'bs' ili 'en'",
    )


class AIAnswerResponse(BaseModel):
    """AI-generirani odgovor sa strukturiranim rezultatima."""

    answer: str
    results: List[GrantSummary] = Field(default_factory=list)
    sources: List[Dict[str, str]]
    request_id: str
    processing_time: float

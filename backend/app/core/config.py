"""
backend/app/core/config.py — Centralna konfiguracija sistema.

Sve environment varijable i konstante žive ovdje. Ostali moduli čitaju
config.DATABASE_URL dinamički (preko modula, ne from-importa) jer se pri
startupu može aktivirati SQLite fallback koji mutira ovu vrijednost.
"""

import os
import secrets
from pathlib import Path

# Repo root: backend/app/core/config.py → parents[3]
ROOT_DIR = Path(__file__).resolve().parents[3]
GRANTS_FILE = ROOT_DIR / "data" / "grants.json"

# --- Release identity ---
APP_VERSION = os.getenv("APP_VERSION", "2.2.1")
GIT_COMMIT = os.getenv(
    "RENDER_GIT_COMMIT",
    os.getenv("GIT_COMMIT", "local"),
)

# --- JWT ---
# JWT tajni ključ — MORA biti postavljen u Render environment za produkciju
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    if os.getenv("RENDER"):  # Render automatski postavlja RENDER=true
        raise RuntimeError(
            "JWT_SECRET environment variable nije postavljen! "
            "Render Dashboard → Environment → Add Environment Variable: JWT_SECRET"
        )
    JWT_SECRET = secrets.token_hex(32)  # Lokalni razvoj — ephemeral key je OK
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30

# --- Baza korisnika ---
DB_PATH = os.getenv("DB_PATH", "users.db")
DATABASE_URL = os.getenv("DATABASE_URL")  # PostgreSQL / Supabase connection string


def use_sqlite_fallback() -> None:
    """Aktivira SQLite fallback kad PostgreSQL konekcija ne uspije na startupu."""
    global DATABASE_URL
    DATABASE_URL = None


# --- Rate limiting ---
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))  # max zahtjeva
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))      # u sekundi

# --- Guest Demo, best-effort in-memory quota ---
GUEST_DEMO_LIMIT = int(os.getenv("GUEST_DEMO_LIMIT", "3"))
GUEST_DEMO_WINDOW_SECONDS = int(
    os.getenv("GUEST_DEMO_WINDOW_SECONDS", "3600")
)

# --- CORS — samo dozvoljeni domeni ---
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",   # VS Code Live Server
    "https://dinohatibovic.github.io",
]

# --- Stripe ---
STRIPE_SECRET = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

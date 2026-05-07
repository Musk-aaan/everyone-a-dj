"""Central configuration: paths, env loading, API keys.

Every other module imports from here so paths and keys live in one place.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR    = ROOT / "audio"
CACHE_DIR    = ROOT / "cache"
LYRICS_DIR   = CACHE_DIR / "lyrics"
ANALYSES_DIR = CACHE_DIR / "analyses"
STEMS_DIR    = CACHE_DIR / "stems"
RENDERS_DIR  = CACHE_DIR / "renders"

for d in (AUDIO_DIR, LYRICS_DIR, ANALYSES_DIR, STEMS_DIR, RENDERS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Env ──────────────────────────────────────────────────────────────────────
load_dotenv(ROOT / ".env", override=True)


def _get(key: str, *fallbacks: str) -> Optional[str]:
    """Read env var with fallback names. Returns None if none set."""
    for k in (key, *fallbacks):
        v = os.environ.get(k)
        if v:
            return v
    return None


# Discovery
YOUTUBE_API_KEY        = _get("YOUTUBE_API_KEY", "GOOGLE_API_KEY")
SPOTIFY_CLIENT_ID      = _get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET  = _get("SPOTIFY_CLIENT_SECRET")

# Lyrics
GENIUS_TOKEN           = _get("GENIUS_TOKEN", "GENIUS_ACCESS_TOKEN")
OPENAI_API_KEY         = _get("OPENAI_API_KEY")

# LLM brain — single provider (OpenRouter + Gemini 2.5 Flash for everything)
OPENROUTER_API_KEY     = _get("OPENROUTER_API_KEY")
LLM_MODEL              = "google/gemini-2.5-flash"

# Audio constants
SR = 44100
DEFAULT_TARGET_BPM = 115.0


def have(*keys: str) -> bool:
    """True iff all given env-key names resolve to a non-empty value here."""
    return all(globals().get(k) for k in keys)


def missing_keys_message() -> str:
    """Diagnostic for the CLI: which API keys are missing and what each unlocks."""
    rows = [
        ("YOUTUBE_API_KEY",       YOUTUBE_API_KEY,       "Phase 1: trending charts"),
        ("SPOTIFY_CLIENT_ID",     SPOTIFY_CLIENT_ID,     "Phase 1: cross-reference Western charts"),
        ("SPOTIFY_CLIENT_SECRET", SPOTIFY_CLIENT_SECRET, "Phase 1: (paired with client id)"),
        ("GENIUS_TOKEN",          GENIUS_TOKEN,          "Phase 2: official lyrics"),
        ("OPENAI_API_KEY",        OPENAI_API_KEY,        "Phase 2: Whisper word timestamps"),
        ("OPENROUTER_API_KEY",    OPENROUTER_API_KEY,    f"Phase 5: brain ({LLM_MODEL})"),
    ]
    lines = []
    for name, val, desc in rows:
        mark = "✓" if val else "·"
        lines.append(f"  {mark} {name:<22} {desc}")
    return "\n".join(lines)

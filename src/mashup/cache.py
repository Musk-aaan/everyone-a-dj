"""Filesystem cache. Every artifact is keyed by a stable song_id (sha1 of YT id).

Layout:
  audio/{song_id}.mp3
  cache/lyrics/{song_id}.json
  cache/analyses/{song_id}.json
  cache/stems/{song_id}/{vocals,drums,bass,other}.wav
  cache/renders/{job_id}.mp3

We never expire. Music doesn't change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from . import config


def song_id(youtube_id: str) -> str:
    """Stable 12-char hex id from a YouTube video id."""
    return hashlib.sha1(youtube_id.encode()).hexdigest()[:12]


# ── Path resolvers ───────────────────────────────────────────────────────────

def audio_path(sid: str) -> Path:
    return config.AUDIO_DIR / f"{sid}.mp3"

def lyrics_path(sid: str) -> Path:
    return config.LYRICS_DIR / f"{sid}.json"

def analysis_path(sid: str) -> Path:
    return config.ANALYSES_DIR / f"{sid}.json"

def stems_dir(sid: str) -> Path:
    return config.STEMS_DIR / sid

def render_path(job_id: str) -> Path:
    return config.RENDERS_DIR / f"{job_id}.mp3"


# ── JSON helpers ─────────────────────────────────────────────────────────────

def read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ── Existence checks (the entire reason this module exists) ──────────────────

def has_audio(sid: str) -> bool:    return audio_path(sid).exists()
def has_lyrics(sid: str) -> bool:   return lyrics_path(sid).exists()
def has_analysis(sid: str) -> bool: return analysis_path(sid).exists()
def has_stems(sid: str) -> bool:
    d = stems_dir(sid)
    return d.exists() and all((d / f"{s}.wav").exists() for s in ("vocals", "drums", "bass", "other"))

"""Phase 2 — Lyrics.

Three sources, picked in order:

  1. Genius API (free, ~30M+ tracks)
     - Official lyrics, high-quality text
     - NO timestamps — just the words
  2. Gemini 2.5 Flash via OpenRouter (uses existing OPENROUTER_API_KEY)
     - Multilingual transcription with line-level timestamps
     - ~$0.001/song — ridiculously cheap
     - Default fallback (no extra key needed)
  3. OpenAI Whisper-1 (optional, $0.006/min audio)
     - Word-level timestamps (more precise than Gemini)
     - Only used if OPENAI_API_KEY is funded; rarely needed

Output is a Lyrics dataclass cached forever to cache/lyrics/{song_id}.json.

The downstream LLM brain (Phase 5) uses lyrics for:
  - Lyric-aware section picking ("the chorus that says X is more iconic than Y")
  - Transition continuity ("BDT's 'badtameez' lands well into UF's 'too hot'")
  - A cappella moment detection (long Whisper silences mean instrumental breaks)
  - Audience filtering (explicit-content checks)
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from . import cache, config


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Lyrics:
    song_id: str
    source: str                                      # 'genius' | 'whisper'
    title: str
    artist: str
    language: str = ""                               # ISO code (Whisper only)
    full_text: str = ""
    lines: list[dict] = field(default_factory=list)  # [{text, start, end}]
    words: list[dict] = field(default_factory=list)  # [{text, start, end}] — Whisper only
    has_timestamps: bool = False
    fetched_at: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Lyrics":
        return cls(**d)


# ── Title cleanup ────────────────────────────────────────────────────────────

_BRACKET_RE = re.compile(r'[\[\(].*?[\]\)]')
_NOISE_RE = re.compile(
    r'\b(official|full|video|song|lyrics?|audio|hd|4k|8k|hdr|imax|remix|'
    r'm/?v|mv|live|cover|version|edit)\b',
    re.IGNORECASE,
)

def clean_title(s: str) -> str:
    """Strip YouTube-title cruft so Genius search matches the actual song.

    'Full Video: Naatu Naatu Song (Telugu) | RRR Songs | NTR,Ram'
      → 'Naatu Naatu'
    """
    s = _BRACKET_RE.sub(' ', s)
    s = s.split('|')[0]                              # everything before first pipe
    s = _NOISE_RE.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip(' :|-,.')
    return s


# ── Genius ───────────────────────────────────────────────────────────────────

def fetch_genius(title: str, artist: str = "") -> Optional[dict]:
    """Search Genius and return {'title','artist','lyrics','url'} or None."""
    if not config.GENIUS_TOKEN:
        return None

    try:
        import lyricsgenius
    except ImportError as e:
        raise RuntimeError("pip install lyricsgenius") from e

    g = lyricsgenius.Genius(
        config.GENIUS_TOKEN,
        timeout=15,
        retries=2,
        remove_section_headers=True,
        skip_non_songs=True,
    )
    g.verbose = False
    g.excluded_terms = ["(Remix)", "(Live)"]

    cleaned = clean_title(title)
    try:
        song = g.search_song(cleaned, artist) if artist else g.search_song(cleaned)
    except Exception:
        return None

    if not song or not song.lyrics:
        return None

    return {
        "title":  song.title,
        "artist": song.artist,
        "lyrics": _strip_genius_chrome(song.lyrics),
        "url":    song.url,
    }


def _strip_genius_chrome(text: str) -> str:
    """lyricsgenius sometimes prepends '<title> Lyrics' and trailing 'EmbedShare...'."""
    text = re.sub(r'^.*?Lyrics\n', '', text, count=1, flags=re.DOTALL)
    text = re.sub(r'\d*Embed.*$', '', text, flags=re.DOTALL)
    return text.strip()


# ── Gemini 2.5 Flash via OpenRouter (preferred — uses existing key) ──────────

_GEMINI_TRANSCRIBE_PROMPT = """Transcribe this song's vocal lyrics with line-level timestamps.

Return ONLY valid JSON in EXACTLY this shape — no prose, no code fences:

{
  "language": "<ISO 639-1 code, e.g. en, hi, te, ko>",
  "lines": [
    {"text": "<line of lyrics>", "start": <seconds, float>, "end": <seconds, float>}
  ]
}

Rules:
- Times are seconds from the start of the audio.
- One JSON object per vocal line. Skip purely instrumental sections.
- Transcribe in the ORIGINAL language; do not translate.
- If a section repeats, include each repetition as a separate line.
- If you cannot hear vocals, return {"language": "", "lines": []}."""


def fetch_gemini_transcription(audio_path: Path) -> Optional[dict]:
    """Use Gemini 2.5 Flash (via OpenRouter) to transcribe audio with line timestamps.

    Same provider+key already used for the diagnostics listener — no extra setup.
    """
    if not config.OPENROUTER_API_KEY:
        return None
    if not audio_path.exists():
        raise RuntimeError(f"audio file not found: {audio_path}")

    with audio_path.open("rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")
    fmt = audio_path.suffix.lstrip(".").lower() or "mp3"

    payload = {
        "model": config.LLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _GEMINI_TRANSCRIBE_PROMPT},
                {"type": "input_audio",
                 "input_audio": {"data": audio_b64, "format": fmt}},
            ],
        }],
    }
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mashup-debug",
            "X-Title": "mashup-lyrics",
        },
        json=payload,
        timeout=180,
    )
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"].strip()

    # Strip code fences if Gemini wraps the JSON
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Sometimes the model emits a JSON object inside other text — extract it
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            raise RuntimeError(f"Gemini did not return JSON. Got: {raw[:200]!r}")
        parsed = json.loads(m.group(0))

    lines = parsed.get("lines", []) or []
    return {
        "language": parsed.get("language", "") or "",
        "text":     "\n".join(ln.get("text", "") for ln in lines),
        "segments": [{"text": ln.get("text", ""),
                      "start": float(ln.get("start", 0.0)),
                      "end":   float(ln.get("end", 0.0))} for ln in lines],
        "words":    [],   # line-level only
    }


# ── OpenAI Whisper (only if quota available) ─────────────────────────────────

def fetch_whisper(audio_path: Path) -> Optional[dict]:
    """Returns {'language','text','segments','words'} from OpenAI Whisper-1."""
    if not config.OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e

    if not audio_path.exists():
        raise RuntimeError(f"audio file not found: {audio_path}")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    try:
        with audio_path.open("rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "exceeded your current quota" in msg:
            raise RuntimeError(
                "OpenAI account has $0 quota. Top up at platform.openai.com/billing "
                "(Whisper-1 = $0.006/min audio). Or switch to Groq's free Whisper-v3."
            ) from e
        raise

    segments = [{"text": s.text, "start": s.start, "end": s.end}
                for s in (resp.segments or [])]
    words = [{"text": w.word, "start": w.start, "end": w.end}
             for w in (resp.words or [])]

    return {
        "language": resp.language,
        "text":     resp.text,
        "segments": segments,
        "words":    words,
    }


# ── Public entry point ───────────────────────────────────────────────────────

def fetch(
    candidate,
    audio_path: Optional[Path] = None,
    *,
    prefer: str = "genius",
    force_whisper: bool = False,
) -> Lyrics:
    """Fetch lyrics for a Candidate. Cached.

    candidate must have .youtube_id, .song_id, .title, .channel.

    Strategy:
      prefer='genius' (default): try Genius → fall back to Whisper if needed.
      prefer='whisper' OR force_whisper=True: skip Genius, run Whisper.

    audio_path defaults to cache.audio_path(candidate.song_id) — must exist
    if Whisper will run.
    """
    sid = candidate.song_id

    if not force_whisper:
        cached = cache.read_json(cache.lyrics_path(sid))
        if cached:
            return Lyrics.from_dict(cached)

    title = candidate.title
    artist = candidate.channel

    result = Lyrics(
        song_id=sid, source="", title=title, artist=artist,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )

    # Try Genius first unless caller asks otherwise
    if prefer == "genius" and not force_whisper:
        g = fetch_genius(title, artist)
        if g:
            result.source         = "genius"
            result.title          = g["title"]
            result.artist         = g["artist"]
            result.full_text      = g["lyrics"]
            result.lines          = [{"text": ln, "start": 0.0, "end": 0.0}
                                     for ln in g["lyrics"].split("\n") if ln.strip()]
            result.has_timestamps = False
            cache.write_json(cache.lyrics_path(sid), asdict(result))
            return result

    # Whisper (fallback or primary)
    if audio_path is None:
        audio_path = cache.audio_path(sid)

    if not audio_path.exists():
        raise RuntimeError(
            f"Whisper needs the audio file at {audio_path}. "
            f"Download first: `mashup download --youtube-id {candidate.youtube_id}`"
        )

    # Try Gemini-via-OpenRouter first (free quota, already paid for)
    w = None
    transcribe_source = ""
    if config.OPENROUTER_API_KEY:
        try:
            w = fetch_gemini_transcription(audio_path)
            if w:
                transcribe_source = "gemini"
        except Exception as e:
            print(f"Gemini transcription failed ({e}); trying OpenAI Whisper...")

    # Fall back to OpenAI Whisper if Gemini failed
    if not w:
        try:
            w = fetch_whisper(audio_path)
            if w:
                transcribe_source = "whisper"
        except Exception as e:
            raise RuntimeError(
                f"All transcription paths failed. Genius miss + "
                f"Gemini/OpenAI both unavailable: {e}"
            ) from e

    if not w:
        raise RuntimeError("Genius miss and no transcription provider available.")

    result.source         = transcribe_source
    result.language       = w["language"]
    result.full_text      = w["text"]
    result.lines          = w["segments"]
    result.words          = w["words"]
    result.has_timestamps = True
    cache.write_json(cache.lyrics_path(sid), asdict(result))
    return result

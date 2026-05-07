"""Phase 1 — Discovery.

Finds candidate songs for a vibe by searching YouTube + Spotify + trending charts,
then downloads them via yt-dlp.

Three entry points:
  search(query, n=10)          → ranked list of Candidate
  trending(region, n=20)       → YouTube music chart for a region
  download(candidate)          → fetches mp3 to audio/{song_id}.mp3, returns path

A Candidate is the universal song record passed between phases.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import cache, config

# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """A song we might use. youtube_id is the canonical key."""
    youtube_id: str
    title: str
    channel: str
    duration_s: int
    view_count: int = 0
    description: str = ""
    region: str = ""
    source: str = "yt-dlp"   # 'yt-dlp' | 'youtube-api' | 'spotify'

    @property
    def song_id(self) -> str:
        return cache.song_id(self.youtube_id)

    @property
    def youtube_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.youtube_id}"


# ── Search via yt-dlp (no API key needed) ────────────────────────────────────

_TITLE_REJECT_RE = re.compile(
    r'\b(album|jukebox|mix|nonstop|compilation|playlist|mash[- ]?up|'
    r'dj set|live set|full movie|episode|podcast|hour[s]? of|'
    r'top \d+ songs?|best of \d{4})\b',
    re.IGNORECASE,
)


def search(query: str, n: int = 10,
           min_dur_s: int = 60, max_dur_s: int = 480,
           auto_append_song: bool = True) -> list[Candidate]:
    """Search YouTube via yt-dlp. Works without any API key.

    Heavily filters the raw results to surface single-track candidates:
      - duration in [min_dur_s, max_dur_s]   (rejects playlists + shorts)
      - title doesn't match playlist/mix patterns (Album, Jukebox, Nonstop, ...)
      - if auto_append_song and the query lacks "song"/"video", retry with " song"
        appended once if results are sparse

    For ranked/curated results use `trending` instead (needs YT API key).
    """
    raw = _yt_search_raw(query, n=max(n * 4, n + 10))
    out = _filter_candidates(raw, n=n, min_dur_s=min_dur_s, max_dur_s=max_dur_s)

    # If we got too few results, retry with "song" appended once
    if auto_append_song and len(out) < n and \
       not re.search(r'\b(song|video|official)\b', query, re.IGNORECASE):
        raw2 = _yt_search_raw(f"{query} song", n=max(n * 4, n + 10))
        out += _filter_candidates(raw2, n=n - len(out),
                                  min_dur_s=min_dur_s, max_dur_s=max_dur_s,
                                  exclude_ids={c.youtube_id for c in out})
    return out[:n]


def _yt_search_raw(query: str, n: int) -> list[dict]:
    cmd = [
        "yt-dlp", f"ytsearch{n}:{query}",
        "--flat-playlist", "--dump-json", "--no-warnings",
        "--default-search", "ytsearch",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp search failed: {proc.stderr.strip()[:200]}")
    rows = []
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _filter_candidates(rows: list[dict], *, n: int,
                       min_dur_s: int, max_dur_s: int,
                       exclude_ids: set[str] = frozenset()) -> list[Candidate]:
    out: list[Candidate] = []
    for d in rows:
        title = d.get("title", "") or ""
        if _TITLE_REJECT_RE.search(title):
            continue
        dur = int(d.get("duration") or 0)
        if dur and (dur < min_dur_s or dur > max_dur_s):
            continue
        yid = d.get("id") or d.get("url", "").split("v=")[-1]
        if yid in exclude_ids:
            continue
        out.append(Candidate(
            youtube_id=yid,
            title=title,
            channel=d.get("channel") or d.get("uploader", ""),
            duration_s=dur,
            view_count=int(d.get("view_count") or 0),
            description=d.get("description", "") or "",
        ))
        if len(out) >= n:
            break
    return out


# ── Trending via YouTube Data API (needs YOUTUBE_API_KEY) ────────────────────

def trending(region: str = "US", n: int = 20) -> list[Candidate]:
    """Top-N music chart for a region. Requires YOUTUBE_API_KEY.

    Region examples: 'US', 'IN', 'GB', 'BR', 'JP', 'KR', 'NG'.
    """
    if not config.YOUTUBE_API_KEY:
        raise RuntimeError(
            "YOUTUBE_API_KEY not set. Use search() instead, or add to .env"
        )

    # Local import so the dep is optional
    from googleapiclient.discovery import build

    yt = build("youtube", "v3", developerKey=config.YOUTUBE_API_KEY,
               cache_discovery=False)
    try:
        resp = yt.videos().list(
            part="snippet,statistics,contentDetails",
            chart="mostPopular",
            regionCode=region,
            videoCategoryId="10",   # Music
            maxResults=min(n, 50),
        ).execute()
    except Exception as e:
        msg = str(e)
        if "accessNotConfigured" in msg or "has not been used" in msg:
            raise RuntimeError(
                "YouTube Data API v3 is not enabled on this Google Cloud project. "
                "Enable at: https://console.cloud.google.com/apis/library/youtube.googleapis.com "
                "(or use a different YOUTUBE_API_KEY in .env)."
            ) from e
        raise

    out: list[Candidate] = []
    for it in resp.get("items", []):
        sn = it["snippet"]; st = it.get("statistics", {})
        out.append(Candidate(
            youtube_id=it["id"],
            title=sn.get("title", ""),
            channel=sn.get("channelTitle", ""),
            duration_s=_iso8601_to_seconds(it["contentDetails"]["duration"]),
            view_count=int(st.get("viewCount") or 0),
            description=sn.get("description", "") or "",
            region=region,
            source="youtube-api",
        ))
    return out


def _iso8601_to_seconds(d: str) -> int:
    """'PT3M45S' → 225. Tiny parser, no deps."""
    import re
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d or "")
    if not m: return 0
    h, mm, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mm * 60 + s


# ── Spotify cross-reference (needs both client id + secret) ──────────────────

def spotify_top_tracks(playlist_id: str = "37i9dQZEVXbMDoHDwVN2tF",
                       n: int = 20) -> list[dict]:
    """Returns Spotify track metadata for a playlist (default = Global Top 50).

    These don't have YouTube IDs — use them as queries to feed into search().
    """
    if not (config.SPOTIFY_CLIENT_ID and config.SPOTIFY_CLIENT_SECRET):
        raise RuntimeError("SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET not set.")

    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=config.SPOTIFY_CLIENT_ID,
        client_secret=config.SPOTIFY_CLIENT_SECRET,
    ))
    items = sp.playlist_items(playlist_id, limit=n)["items"]
    out = []
    for it in items:
        t = it.get("track") or {}
        if not t.get("name"): continue
        out.append({
            "title":  t["name"],
            "artist": ", ".join(a["name"] for a in t.get("artists", [])),
            "popularity": t.get("popularity", 0),
            "duration_ms": t.get("duration_ms", 0),
            "spotify_id":  t.get("id"),
        })
    return out


# ── Download ─────────────────────────────────────────────────────────────────

def download(c: Candidate, force: bool = False) -> Path:
    """Download a candidate's audio to audio/{song_id}.mp3. Cached.

    Returns the local path. Skips if already present (unless force=True).
    """
    out = cache.audio_path(c.song_id)
    if out.exists() and not force:
        return out

    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3", "--audio-quality", "0",
        "-o", str(out.with_suffix(".%(ext)s")),
        "--no-playlist", "--no-warnings", "--quiet",
        c.youtube_url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {proc.stderr.strip()[:300]}")

    if not out.exists():
        # yt-dlp sometimes writes a different extension; find it
        for p in config.AUDIO_DIR.glob(f"{c.song_id}.*"):
            if p.suffix in (".mp3", ".m4a", ".webm", ".opus"):
                p.rename(out)
                break

    if not out.exists():
        raise RuntimeError(f"download produced no file at {out}")
    return out


# ── Manifest ─────────────────────────────────────────────────────────────────

def save_manifest(candidates: list[Candidate], path: Optional[Path] = None) -> Path:
    """Persist a list of candidates to JSON for inspection / re-use."""
    path = path or (config.CACHE_DIR / "candidates_latest.json")
    with path.open("w") as f:
        json.dump([asdict(c) for c in candidates], f, indent=2, ensure_ascii=False)
    return path

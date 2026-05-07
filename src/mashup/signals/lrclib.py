"""Time-synced lyrics from LRCLIB (free, no auth)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import requests

LRCLIB_BASE = "https://lrclib.net/api"
_LRC_TS = re.compile(r"\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]")


@dataclass(frozen=True)
class LyricLine:
    time: float
    text: str


@dataclass(frozen=True)
class SyncedLyrics:
    artist: str
    title: str
    duration: float
    lines: list[LyricLine]


def fetch(
    artist: str,
    title: str,
    *,
    album: Optional[str] = None,
    duration: Optional[float] = None,
    timeout: float = 15.0,
) -> Optional[SyncedLyrics]:
    """Fetch synced lyrics. Returns None if no synced version is available."""
    params = {"artist_name": artist, "track_name": title}
    if album:
        params["album_name"] = album
    if duration:
        params["duration"] = int(duration)
    r = requests.get(
        f"{LRCLIB_BASE}/get",
        params=params,
        timeout=timeout,
        headers={"User-Agent": "mashup/0.0.1 (iconic-moment-detector)"},
    )
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    synced = data.get("syncedLyrics")
    if not synced:
        return None
    return SyncedLyrics(
        artist=data.get("artistName", artist),
        title=data.get("trackName", title),
        duration=float(data.get("duration") or 0.0),
        lines=parse_lrc(synced),
    )


def parse_lrc(synced: str) -> list[LyricLine]:
    out: list[LyricLine] = []
    for raw in synced.splitlines():
        m = _LRC_TS.match(raw)
        if not m:
            continue
        mm, ss, frac = m.groups()
        seconds = int(mm) * 60 + int(ss)
        if frac:
            # LRC fractional digits are right-padded: "47" = 470ms, "5" = 500ms.
            seconds += int(frac.ljust(3, "0")) / 1000.0
        text = raw[m.end():].strip()
        out.append(LyricLine(time=seconds, text=text))
    return out

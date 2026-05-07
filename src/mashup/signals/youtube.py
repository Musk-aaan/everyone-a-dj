"""YouTube most-replayed (heatmap) signal via yt-dlp.

Returns None when yt-dlp isn't installed or the video has no heatmap data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HeatSpan:
    start: float
    end: float
    value: float  # 0..1, normalized replay intensity


def fetch_heatmap(youtube_url: str) -> Optional[list[HeatSpan]]:
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return None

    opts = {"quiet": True, "skip_download": True, "no_warnings": True}
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
    except Exception:
        return None

    raw = info.get("heatmap")
    if not raw:
        return None
    return [
        HeatSpan(
            start=float(h["start_time"]),
            end=float(h["end_time"]),
            value=float(h.get("value", 0.0)),
        )
        for h in raw
    ]


def heat_at(heatmap: list[HeatSpan], time: float) -> float:
    for span in heatmap:
        if span.start <= time <= span.end:
            return span.value
    return 0.0

"""High-level entry point: find iconic moments for a song."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .ranker import Moment, rank_moments
from .signals.audio import AudioFeatures, analyze as analyze_audio
from .signals.lrclib import SyncedLyrics, fetch as fetch_lyrics
from .signals.youtube import HeatSpan, fetch_heatmap


@dataclass
class DetectionResult:
    artist: str
    title: str
    duration: float
    moments: list[Moment]
    signals_used: list[str]
    lyrics: Optional[SyncedLyrics] = None
    audio: Optional[AudioFeatures] = None


def find_moments(
    artist: str,
    title: str,
    *,
    audio_path: Optional[str] = None,
    youtube_url: Optional[str] = None,
    top_k: int = 3,
) -> Optional[DetectionResult]:
    lyrics = fetch_lyrics(artist=artist, title=title)

    if not lyrics:
        # Lyrics unavailable (common for non-English songs like Bollywood).
        # Fall back to audio-energy analysis if an audio file was supplied.
        if not audio_path:
            return None
        audio = analyze_audio(audio_path)
        if not audio:
            return None
        from .ranker import rank_moments_audio_only
        moments = rank_moments_audio_only(audio, top_k=top_k)
        if not moments:
            return None
        return DetectionResult(
            artist=artist,
            title=title,
            duration=audio.duration,
            moments=moments,
            signals_used=["audio"],
            lyrics=None,
            audio=audio,
        )

    signals = ["lyrics"]

    audio: Optional[AudioFeatures] = None
    if audio_path:
        audio = analyze_audio(audio_path)
        if audio:
            signals.append("audio")

    heatmap: Optional[list[HeatSpan]] = None
    if youtube_url:
        heatmap = fetch_heatmap(youtube_url)
        if heatmap:
            signals.append("youtube_heatmap")

    moments = rank_moments(lyrics, audio=audio, heatmap=heatmap, top_k=top_k)
    return DetectionResult(
        artist=lyrics.artist,
        title=lyrics.title,
        duration=lyrics.duration,
        moments=moments,
        signals_used=signals,
        lyrics=lyrics,
        audio=audio,
    )

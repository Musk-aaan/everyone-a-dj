"""Combine signals into ranked iconic-moment candidates.

The core insight from the vision doc: the iconic moment is the lyric line
everyone sings along to. Lines that repeat across the song are almost always
chorus material; the *best* instance of that line is usually the one that lands
in the song's energetic peak (~50-85% through). Audio energy peaks and YouTube
replay heat reinforce that signal when available.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .signals.audio import AudioFeatures
from .signals.lrclib import LyricLine, SyncedLyrics
from .signals.transliteration import normalize_lyric
from .signals.youtube import HeatSpan, heat_at


@dataclass
class Moment:
    start: float
    end: float
    lyric: str
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)


def _normalize(line: str) -> str:
    return normalize_lyric(line)


def _line_repetition_counts(lines: list[LyricLine]) -> Counter:
    return Counter(_normalize(l.text) for l in lines if l.text.strip())


def _section_position_score(time: float, duration: float) -> float:
    """Songs typically peak between 50-85%. Choruses live there."""
    if duration <= 0:
        return 0.5
    rel = time / duration
    if 0.5 <= rel <= 0.85:
        return 1.0
    if 0.3 <= rel < 0.5:
        return 0.7
    if 0.85 < rel <= 1.0:
        return 0.6
    return 0.3


def _window_for(
    line: LyricLine, lines: list[LyricLine], duration: float, target: float = 16.0
) -> tuple[float, float]:
    """Carve an ~8-16s window starting at this line."""
    fallback_end = line.time + target
    if duration > 0:
        fallback_end = min(fallback_end, duration)
    later = [l for l in lines if l.time > line.time]
    end = fallback_end
    for nxt in later:
        if nxt.time - line.time >= target:
            end = nxt.time
            break
    return (line.time, end)


def _energy_score(time: float, audio: AudioFeatures) -> float:
    if not audio.energy_peaks:
        return 0.0
    nearest = min(audio.energy_peaks, key=lambda t: abs(t - time))
    distance = abs(nearest - time)
    return max(0.0, 1.0 - distance / 4.0)


def _replay_score(time: float, heatmap: list[HeatSpan]) -> float:
    return heat_at(heatmap, time)


def rank_moments_audio_only(audio: AudioFeatures, top_k: int = 3) -> list[Moment]:
    """Moment detection when no synced lyrics are available (e.g. non-English songs).

    Selects windows around the song's highest-energy peaks that also land in
    the 50-85% position range where choruses typically live.
    """
    if not audio.energy_peaks:
        return []

    candidates: list[Moment] = []
    for peak_time in audio.energy_peaks:
        pos_score = _section_position_score(peak_time, audio.duration)
        en_score = _energy_score(peak_time, audio)
        score = (pos_score + en_score) / 2.0
        start = max(0.0, peak_time - 2.0)
        end = min(audio.duration, peak_time + 14.0)
        candidates.append(
            Moment(
                start=start,
                end=end,
                lyric="(audio energy peak)",
                score=score,
                breakdown={"position": pos_score, "energy": en_score},
            )
        )

    candidates.sort(key=lambda m: m.score, reverse=True)
    selected: list[Moment] = []
    for m in candidates:
        if not any(abs(m.start - s.start) < 16.0 for s in selected):
            selected.append(m)
        if len(selected) >= top_k:
            break
    return selected


def rank_moments(
    lyrics: SyncedLyrics,
    *,
    audio: Optional[AudioFeatures] = None,
    heatmap: Optional[list[HeatSpan]] = None,
    top_k: int = 3,
) -> list[Moment]:
    counts = _line_repetition_counts(lyrics.lines)
    if not counts:
        return []
    max_count = max(counts.values())

    best_per_lyric: dict[str, Moment] = {}
    for line in lyrics.lines:
        text = _normalize(line.text)
        if not text:
            continue
        rep = counts[text]
        if rep < 2:
            continue

        components: dict[str, float] = {
            "repetition": rep / max_count,
            "position": _section_position_score(line.time, lyrics.duration),
        }
        if audio is not None:
            components["energy"] = _energy_score(line.time, audio)
        if heatmap is not None:
            components["replay"] = _replay_score(line.time, heatmap)

        score = sum(components.values()) / len(components)
        start, end = _window_for(line, lyrics.lines, lyrics.duration)

        existing = best_per_lyric.get(text)
        if not existing or score > existing.score:
            best_per_lyric[text] = Moment(
                start=start,
                end=end,
                lyric=line.text,
                score=score,
                breakdown=components,
            )

    return sorted(best_per_lyric.values(), key=lambda m: m.score, reverse=True)[:top_k]

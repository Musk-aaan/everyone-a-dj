"""Phase 3 — Audio Intelligence.

Produces a per-song musical fingerprint cached to cache/analyses/{song_id}.json.

Pulls out the data the LLM brain (Phase 5) needs to design transitions:
  - tempo (BPM) + beat grid + downbeat positions → snap cuts to bar boundaries
  - musical key + mode → harmonic compatibility between songs
  - section boundaries (A, B, C, ...) → identify chorus / verse / bridge candidates
  - energy curve over time → find the loudest/quietest moments
  - peak moments → highest-impact spots in the song

Pure librosa for v1. madmom (better downbeats) and essentia (better key) can
be layered in later — they're platform-finicky on M-series Macs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import cache, config


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Analysis:
    song_id: str
    duration_s: float
    sample_rate: int

    bpm: float
    bpm_confidence: str            # 'high' | 'medium' | 'low'
    beats: list[float]             # beat times in seconds
    downbeats: list[float]         # every 4th beat (assumes 4/4)
    bar_duration_s: float

    key: str                       # 'C', 'C#', 'D', ...
    mode: str                      # 'major' | 'minor'
    key_confidence: float          # correlation with Krumhansl-Schmuckler profile

    sections: list[dict]           # [{start, end, label, energy}]
    energy_curve: list[dict]       # [{t, rms}] sampled per ~1s
    peak_moments: list[dict]       # top-5 highest-energy 2s windows

    analyzed_at: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Analysis":
        return cls(**d)


# ── Krumhansl-Schmuckler key profiles ────────────────────────────────────────

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def _detect_key(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    """Best-fit (root, mode, correlation) via Krumhansl-Schmuckler matching."""
    best_score = -np.inf
    best: tuple[str, str, float] = ("C", "major", 0.0)
    for shift in range(12):
        rolled = np.roll(chroma_mean, -shift)
        maj = np.corrcoef(rolled, _MAJOR_PROFILE)[0, 1]
        mn  = np.corrcoef(rolled, _MINOR_PROFILE)[0, 1]
        if maj > best_score:
            best_score, best = maj, (_NOTE_NAMES[shift], "major", float(maj))
        if mn > best_score:
            best_score, best = mn,  (_NOTE_NAMES[shift], "minor", float(mn))
    return best


# ── Main analysis ────────────────────────────────────────────────────────────

def analyze(audio_path: Path, *, force: bool = False) -> Analysis:
    """Run full musical analysis on `audio_path`. Cached forever by song_id."""
    sid = audio_path.stem

    if not force:
        cached = cache.read_json(cache.analysis_path(sid))
        if cached:
            return Analysis.from_dict(cached)

    import librosa

    # Load mono at our standard sample rate
    y, sr = librosa.load(str(audio_path), sr=config.SR, mono=True)
    duration = len(y) / sr

    # ── Tempo + beats ──────────────────────────────────────────────────────
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    bpm = float(np.atleast_1d(tempo)[0])

    # librosa often detects half/double tempo; nudge into 60–180 range
    if bpm < 60 and bpm > 0:
        bpm *= 2
        bpm_conf = "medium"
    elif bpm > 180:
        bpm /= 2
        bpm_conf = "medium"
    elif 60 <= bpm <= 180:
        bpm_conf = "high"
    else:
        bpm_conf = "low"

    beats_s = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    bar_duration = 60.0 / bpm * 4 if bpm > 0 else 0.0
    downbeats = beats_s[::4]   # assume 4/4 — true for ~99% of pop music

    # ── Key detection via chroma ───────────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, mode, key_conf = _detect_key(chroma.mean(axis=1))

    # ── Energy curve (RMS per ~1-second window) ────────────────────────────
    hop = sr
    rms = librosa.feature.rms(y=y, frame_length=2 * hop, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    energy_curve = [{"t": float(t), "rms": float(r)} for t, r in zip(times, rms)]

    # ── Peak moments: top-5 non-overlapping high-energy 2s windows ─────────
    sorted_idx = np.argsort(-rms)
    seen_t: list[float] = []
    peak_moments = []
    for idx in sorted_idx:
        t = float(times[idx])
        if any(abs(t - s) < 4.0 for s in seen_t):
            continue
        seen_t.append(t)
        peak_moments.append({"t": t, "rms": float(rms[idx])})
        if len(peak_moments) >= 5:
            break

    # ── Sections via agglomerative clustering on chroma ────────────────────
    n_segments = max(4, min(10, int(duration / 25)))   # ~25s per section target
    bounds = librosa.segment.agglomerative(chroma, k=n_segments)
    bound_times = sorted({0.0, duration, *librosa.frames_to_time(bounds, sr=sr).tolist()})

    sections = []
    for i in range(len(bound_times) - 1):
        s, e = bound_times[i], bound_times[i + 1]
        if e - s < 4.0:        # skip ultra-short fragments
            continue
        sec_rms = float(np.mean([r for t, r in zip(times, rms) if s <= t < e]) or 0.0)
        sections.append({
            "start":  float(round(s, 2)),
            "end":    float(round(e, 2)),
            "label":  chr(ord("A") + (len(sections) % 26)),  # A, B, C, ...
            "energy": float(round(sec_rms, 4)),
        })

    result = Analysis(
        song_id=sid,
        duration_s=float(round(duration, 2)),
        sample_rate=sr,
        bpm=float(round(bpm, 2)),
        bpm_confidence=bpm_conf,
        beats=[float(round(t, 4)) for t in beats_s],
        downbeats=[float(round(t, 4)) for t in downbeats],
        bar_duration_s=float(round(bar_duration, 4)),
        key=key,
        mode=mode,
        key_confidence=float(round(key_conf, 3)),
        sections=sections,
        energy_curve=energy_curve,
        peak_moments=peak_moments,
        analyzed_at=datetime.now(timezone.utc).isoformat(),
    )

    cache.write_json(cache.analysis_path(sid), asdict(result))
    return result


def analyze_song_id(sid: str, *, force: bool = False) -> Analysis:
    """Convenience: analyse the cached mp3 for a given song_id."""
    return analyze(cache.audio_path(sid), force=force)

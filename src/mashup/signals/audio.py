"""Audio-derived signals: tempo, beat grid, energy peaks, section boundaries.

This module degrades gracefully: if librosa isn't installed, ``analyze`` returns
None and the rest of the pipeline keeps working with lyric signals only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioFeatures:
    duration: float
    tempo: float
    beats: list[float]
    energy_peaks: list[float]
    section_boundaries: list[float]


def analyze(path: str) -> Optional[AudioFeatures]:
    try:
        import librosa
        import numpy as np
    except ImportError:
        return None

    y, sr = librosa.load(path, sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    threshold = float(np.percentile(rms, 85))
    hot_idx = np.where(rms > threshold)[0]
    energy_peaks = _cluster_indices(rms_times, hot_idx, gap=2.0)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    boundary_frames = librosa.segment.agglomerative(chroma, k=8)
    section_boundaries = librosa.frames_to_time(boundary_frames, sr=sr).tolist()

    return AudioFeatures(
        duration=duration,
        tempo=float(tempo),
        beats=beats,
        energy_peaks=energy_peaks,
        section_boundaries=section_boundaries,
    )


def _cluster_indices(times, indices, gap: float = 2.0) -> list[float]:
    if len(indices) == 0:
        return []
    clusters: list[list[int]] = []
    cur = [int(indices[0])]
    for i in indices[1:]:
        i = int(i)
        if times[i] - times[cur[-1]] <= gap:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    return [float(times[c[len(c) // 2]]) for c in clusters]

"""Plan a mashup from two analyzed songs.

The recipe is the "producer's brain": which sections of which song play
when, what role each plays (vocal hook vs. instrumental bed), and what
transitions move you between them. A renderer (pydub crossfade today,
Demucs stem-mix later) consumes the recipe to produce audio.

Algorithm v0:
1. Anchor = song A's top-ranked iconic moment (the vocal hook).
2. Groove pocket = the longest no-vocal stretch in song B (verses /
   instrumental breaks have low lyric density and make natural beds).
3. Lay out a ~60s mashup: B intro -> B groove bed with A's vocal floated
   on top -> resolve into A's full mix for the outro.
4. Insert rule-based transitions (filter sweep into the bed, reverb tail
   on the outgoing vocal, crossfade into the resolution).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from .moment_detector import DetectionResult
from .signals.audio import AudioFeatures
from .signals.lrclib import SyncedLyrics


@dataclass
class Section:
    source: str       # "song_a" | "song_b"
    role: str         # "intro" | "groove_bed" | "anchor_vocal" | "outro"
    start: float      # seconds into the source song
    end: float
    timeline_at: float  # seconds into the mashup output
    description: str


@dataclass
class Transition:
    timeline_at: float
    kind: str          # "filter_sweep" | "reverb_tail" | "crossfade" | "beat_repeat"
    duration: float
    description: str


@dataclass
class SongRef:
    artist: str
    title: str
    duration: float


@dataclass
class MashupRecipe:
    song_a: SongRef
    song_b: SongRef
    duration: float
    anchor_lyric: str
    sections: list[Section] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _find_groove_pocket(
    lyrics: SyncedLyrics, *, min_duration: float = 8.0, target: float = 16.0
) -> Optional[tuple[float, float, str]]:
    """Longest stretch of no vocal lines. Bias toward the second half of
    the song where instrumental breaks (post-chorus, bridge) usually live."""
    if not lyrics.lines or lyrics.duration <= 0:
        return None
    sorted_lines = sorted(lyrics.lines, key=lambda l: l.time)
    half = lyrics.duration / 2.0

    candidates: list[tuple[float, float, str, float]] = []
    prev_end = 0.0
    prev_text = "(intro)"
    for line in sorted_lines:
        gap = line.time - prev_end
        if gap >= min_duration:
            score = gap + (5.0 if prev_end >= half else 0.0)
            candidates.append((prev_end, line.time, f'after "{prev_text}"', score))
        prev_end = line.time
        prev_text = line.text
    tail_gap = lyrics.duration - prev_end
    if tail_gap >= min_duration:
        score = tail_gap + (5.0 if prev_end >= half else 0.0)
        candidates.append((prev_end, lyrics.duration, f'outro after "{prev_text}"', score))

    if not candidates:
        return None
    best = max(candidates, key=lambda c: c[3])
    start, end, label, _ = best
    if end - start > target:
        end = start + target
    return (start, end, label)


def _find_groove_pocket_audio(
    audio: AudioFeatures, *, min_duration: float = 8.0, target: float = 16.0
) -> Optional[tuple[float, float, str]]:
    """Groove pocket from audio section boundaries when lyrics are unavailable."""
    if not audio.section_boundaries:
        return None
    half = audio.duration / 2.0
    boundaries = sorted([0.0] + list(audio.section_boundaries) + [audio.duration])
    candidates: list[tuple[float, float, str, float]] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        dur = end - start
        if dur < min_duration:
            continue
        peak_overlap = any(start <= p <= end for p in audio.energy_peaks)
        score = dur + (4.0 if start >= half else 0.0) - (8.0 if peak_overlap else 0.0)
        candidates.append((start, end, f"audio section {i + 1}", score))
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c[3])
    start, end, label, _ = best
    if end - start > target:
        end = start + target
    return (start, end, label)


def plan_mashup(
    song_a: DetectionResult,
    song_b: DetectionResult,
    *,
    target_duration: float = 60.0,
) -> Optional[MashupRecipe]:
    if not song_a.moments:
        return None

    anchor = song_a.moments[0]
    # Hard-cap anchor length: a hook is ~8 bars (~10-16s at 100-140 BPM).
    # The ranker can produce wider windows when a chorus is followed by a
    # long instrumental break; trim back to a usable hook length.
    anchor_len = min(max(anchor.end - anchor.start, 4.0), 16.0)
    anchor_end = anchor.start + anchor_len

    pocket = None
    if song_b.lyrics is not None:
        pocket = _find_groove_pocket(song_b.lyrics)
    # Fall back to audio when lyrics are absent or too dense to find an 8 s gap.
    if pocket is None and song_b.audio is not None:
        pocket = _find_groove_pocket_audio(song_b.audio)
    if pocket is None:
        return None
    pocket_start, pocket_end, pocket_label = pocket
    pocket_len = pocket_end - pocket_start

    intro_len = 6.0
    bed_len = max(anchor_len + 2.0, min(pocket_len, 20.0))
    transition_len = 4.0
    outro_len = max(target_duration - intro_len - bed_len - transition_len, anchor_len)

    intro_b_start = max(pocket_start - intro_len, 0.0)
    bed_timeline = intro_len
    anchor_timeline = bed_timeline + max((bed_len - anchor_len) / 2.0, 0.0)
    transition_timeline = bed_timeline + bed_len
    outro_timeline = transition_timeline + transition_len

    sections = [
        Section(
            source="song_b",
            role="intro",
            start=intro_b_start,
            end=pocket_start,
            timeline_at=0.0,
            description="Song B fade-in (low energy lead-in)",
        ),
        Section(
            source="song_b",
            role="groove_bed",
            start=pocket_start,
            end=pocket_start + bed_len,
            timeline_at=bed_timeline,
            description=f"Song B instrumental pocket {pocket_label}",
        ),
        Section(
            source="song_a",
            role="anchor_vocal",
            start=anchor.start,
            end=anchor_end,
            timeline_at=anchor_timeline,
            description=f'Song A iconic hook: "{anchor.lyric}" floated over Song B bed',
        ),
        Section(
            source="song_a",
            role="outro",
            start=anchor.start,
            end=anchor.start + outro_len,
            timeline_at=outro_timeline,
            description="Song A full mix takes over and resolves",
        ),
    ]

    transitions = [
        Transition(
            timeline_at=max(bed_timeline - 1.0, 0.0),
            kind="filter_sweep",
            duration=1.0,
            description="High-pass sweep into Song B groove bed",
        ),
        Transition(
            timeline_at=anchor_timeline + anchor_len - 0.5,
            kind="reverb_tail",
            duration=1.5,
            description="Reverb tail on Song A vocal as it lifts off the bed",
        ),
        Transition(
            timeline_at=transition_timeline,
            kind="crossfade",
            duration=transition_len,
            description="Crossfade Song B bed -> Song A full mix",
        ),
    ]

    notes = [
        "Anchor moment chosen as Song A's top-ranked iconic hook (repetition + peak-zone position).",
        "Groove pocket = longest no-vocal stretch in Song B; biased toward the second half of the song.",
        "Effects layer is rule-based; tune per genre (sangeet vs. drop-heavy electronic vs. R&B).",
    ]
    used = set(song_a.signals_used) | set(song_b.signals_used)
    if "audio" in used:
        notes.append("Audio signals available; energy peaks could fine-tune anchor alignment.")
    else:
        notes.append("Audio signals not used; alignment is lyric-time approximate (sub-bar).")
    if "youtube_heatmap" in used:
        notes.append("YouTube replay heat factored into anchor selection.")

    return MashupRecipe(
        song_a=SongRef(song_a.artist, song_a.title, song_a.duration),
        song_b=SongRef(song_b.artist, song_b.title, song_b.duration),
        duration=intro_len + bed_len + transition_len + outro_len,
        anchor_lyric=anchor.lyric,
        sections=sections,
        transitions=transitions,
        notes=notes,
    )

"""Session state + queue manager for the live(ish) DJ mode.

A Session represents one user's mashup-in-progress. The queue is a list of
SegmentSpecs — each is a song and the metadata needed to mix into it. As the
audio plays through, we keep ~1-2 segments rendered ahead and let the user
edit the queue (add / replace / reorder) up until each segment locks in.

Lifecycle of a segment:

  draft     → user just added it; LLM hasn't planned the transition yet
  planning  → LLM is picking the section + designing the transition
  rendering → renderer is producing the audio (cancellable)
  ready     → audio file exists, can be played/served
  playing   → currently being heard by the user
  done      → playback finished

Editing rules:
  - draft / planning  → fully editable (replace, remove, move)
  - rendering         → cancellable; replacing it cancels the in-flight render
  - ready             → still editable, but cancelling re-renders cost ~5-10s
  - playing / done    → locked

This is in-memory only for v1. A production version would persist to Redis +
a background worker; for now, threads + asyncio is enough.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from . import analyze as analyze_mod
from . import cache, config, discover, lyrics, plan, render, stems


class SegmentStatus(str, Enum):
    DRAFT     = "draft"
    PLANNING  = "planning"
    RENDERING = "rendering"
    READY     = "ready"
    PLAYING   = "playing"
    DONE      = "done"
    FAILED    = "failed"


@dataclass
class SegmentSpec:
    """One song in the queue, plus everything needed to render the transition into it."""
    position: int                                   # 0 = first song, etc.
    candidate: discover.Candidate
    status: SegmentStatus = SegmentStatus.DRAFT
    section: Optional[dict] = None                  # set by section picker
    audio_path: Optional[str] = None                # set when rendered
    error: Optional[str] = None
    added_at: float = field(default_factory=time.time)
    rendered_at: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidate"] = asdict(self.candidate)
        d["status"] = self.status.value
        return d


@dataclass
class Session:
    """One user's mashup-in-progress."""
    id: str
    vibe: str
    target_bpm: float = 115.0
    queue: list[SegmentSpec] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # ── Queue editing API ────────────────────────────────────────────────

    def add_song(self, candidate: discover.Candidate,
                 position: Optional[int] = None) -> SegmentSpec:
        pos = position if position is not None else len(self.queue)
        if pos < 0 or pos > len(self.queue):
            raise ValueError(f"position {pos} out of range [0, {len(self.queue)}]")
        spec = SegmentSpec(position=pos, candidate=candidate)
        self.queue.insert(pos, spec)
        self._renumber()
        return spec

    def remove(self, position: int) -> SegmentSpec:
        s = self._get(position)
        if s.status in (SegmentStatus.PLAYING, SegmentStatus.DONE):
            raise ValueError(f"can't remove a {s.status.value} segment")
        self.queue.pop(position)
        self._renumber()
        return s

    def replace(self, position: int, candidate: discover.Candidate) -> SegmentSpec:
        s = self._get(position)
        if s.status in (SegmentStatus.PLAYING, SegmentStatus.DONE):
            raise ValueError(f"can't replace a {s.status.value} segment")
        new = SegmentSpec(position=position, candidate=candidate)
        self.queue[position] = new
        return new

    def move(self, from_pos: int, to_pos: int) -> SegmentSpec:
        s = self._get(from_pos)
        if s.status in (SegmentStatus.PLAYING, SegmentStatus.DONE):
            raise ValueError(f"can't move a {s.status.value} segment")
        self.queue.pop(from_pos)
        self.queue.insert(to_pos, s)
        self._renumber()
        return s

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "vibe":       self.vibe,
            "target_bpm": self.target_bpm,
            "created_at": self.created_at,
            "queue":      [s.to_dict() for s in self.queue],
            "now_playing": next(
                (s.position for s in self.queue if s.status == SegmentStatus.PLAYING),
                None,
            ),
        }

    # ── Internals ────────────────────────────────────────────────────────

    def _renumber(self) -> None:
        for i, s in enumerate(self.queue):
            s.position = i

    def _get(self, position: int) -> SegmentSpec:
        if position < 0 or position >= len(self.queue):
            raise ValueError(f"position {position} out of range")
        return self.queue[position]


# ── Session manager (in-memory store) ────────────────────────────────────────

class SessionManager:
    """In-memory store for active sessions. Production would use Redis."""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    def create(self, vibe: str, seed_youtube_ids: Optional[list[str]] = None,
               target_bpm: float = 115.0) -> Session:
        sess = Session(id=uuid.uuid4().hex[:10], vibe=vibe, target_bpm=target_bpm)

        # Seed the queue with user-supplied songs (if any)
        for yid in (seed_youtube_ids or []):
            try:
                results = discover.search(yid, n=1)
                if results:
                    sess.add_song(results[0])
            except Exception:
                pass

        self.sessions[sess.id] = sess
        return sess

    def get(self, sid: str) -> Session:
        if sid not in self.sessions:
            raise KeyError(f"session {sid!r} not found")
        return self.sessions[sid]


# ── Render worker ────────────────────────────────────────────────────────────

LOOKAHEAD = 2   # always keep N segments rendered ahead of the current one


async def render_one(seg: SegmentSpec, sess: Session,
                     prev_seg: Optional[SegmentSpec],
                     do_stems: bool = False) -> None:
    """Enrich + render a single segment. Mutates `seg.status` and `seg.audio_path`."""
    try:
        seg.status = SegmentStatus.PLANNING

        # Download (sync — quick)
        await asyncio.to_thread(discover.download, seg.candidate)

        # Analyze
        an = await asyncio.to_thread(
            analyze_mod.analyze, cache.audio_path(seg.candidate.song_id),
        )

        # Lyrics (best-effort)
        try:
            ly = await asyncio.to_thread(lyrics.fetch, seg.candidate)
            ly_d = asdict(ly)
        except Exception:
            ly_d = {"lines": [], "has_timestamps": False, "full_text": ""}

        # Section picker
        sec = await asyncio.to_thread(
            plan.pick_section,
            title=seg.candidate.title, analysis=asdict(an), lyrics=ly_d,
        )
        seg.section = sec

        # Optional stems
        if do_stems:
            try:
                await asyncio.to_thread(
                    stems.separate, cache.audio_path(seg.candidate.song_id),
                )
            except Exception:
                pass

        # Render: load this section's audio, BPM-stretch to target
        seg.status = SegmentStatus.RENDERING
        audio = await asyncio.to_thread(
            render.load_section,
            seg.candidate.song_id, sec["start"], sec["end"],
        )
        audio = render.bpm_stretch(audio, an.bpm, sess.target_bpm)

        # If there's a previous segment, blend into it
        if prev_seg and prev_seg.audio_path:
            import soundfile as _sf
            prev_audio, _ = _sf.read(prev_seg.audio_path, dtype="float32", always_2d=True)
            prev_audio = prev_audio.T  # [channels, samples]

            if stems.has_stems(seg.candidate.song_id):
                voc = render.load_stem(
                    seg.candidate.song_id, "vocals", sec["start"], sec["end"],
                )
                if voc is not None:
                    voc = render.bpm_stretch(voc, an.bpm, sess.target_bpm)
                    mix = render.acapella_drop_blend(
                        prev_audio, audio, voc, fade_bars=8, bpm=sess.target_bpm,
                    )
                else:
                    mix = render.bass_swap_crossfade(
                        prev_audio, audio, fade_bars=8, bpm=sess.target_bpm,
                    )
            else:
                mix = render.bass_swap_crossfade(
                    prev_audio, audio, fade_bars=8, bpm=sess.target_bpm,
                )
        else:
            # First segment — small fade-in
            import numpy as _np
            fi = int(0.1 * config.SR)
            audio[:, :fi] *= _np.linspace(0, 1, fi)
            mix = audio

        out_path = cache.RENDERS_DIR / f"{sess.id}_{seg.position:02d}.mp3"
        await asyncio.to_thread(render.write_mp3, render.master(mix), out_path)
        seg.audio_path = str(out_path)
        seg.rendered_at = time.time()
        seg.status = SegmentStatus.READY
    except Exception as e:
        seg.status = SegmentStatus.FAILED
        seg.error = str(e)


async def keep_lookahead_rendered(sess: Session, do_stems: bool = False) -> None:
    """Background loop: ensure the next LOOKAHEAD segments are READY.

    Stops when every segment is DONE / PLAYING / READY / FAILED. Run as an
    asyncio.Task alongside the audio playback.
    """
    while True:
        # Find the first segment that needs rendering (within the lookahead window)
        playing_idx = next(
            (i for i, s in enumerate(sess.queue) if s.status == SegmentStatus.PLAYING),
            -1,
        )
        target_end = playing_idx + 1 + LOOKAHEAD
        target_segs = [
            s for s in sess.queue[max(0, playing_idx + 1) : target_end]
            if s.status in (SegmentStatus.DRAFT,)
        ]

        if not target_segs:
            await asyncio.sleep(0.5)
            continue

        seg = target_segs[0]
        prev_seg = sess.queue[seg.position - 1] if seg.position > 0 else None
        await render_one(seg, sess, prev_seg, do_stems=do_stems)

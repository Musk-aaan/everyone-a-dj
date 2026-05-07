"""Phase 7 — Orchestrator.

Glues the six earlier phases into a single end-to-end pipeline:

  vibe (str) → discover candidates → LLM picks 4 songs → for each:
      download → analyze → fetch lyrics → (optionally) separate stems →
      LLM picks the best section
  → LLM plans every transition between them
  → render to MP3 with pedalboard mastering
  → optional: LLM critique loop (max 1 revision)

Result is a MashupResult with the audio path and the full plan JSON
(transparency — the user can see every decision).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import analyze as analyze_mod
from . import cache, config, discover, lyrics, plan, render, stems


class FallbackToBassSwap(Exception):
    """Sentinel: requested technique couldn't run (stems missing). Use bass_swap."""
    pass

# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class MashupResult:
    job_id: str
    vibe: str
    audio_path: str
    duration_s: float
    plan: dict                        # the full render plan (sections + transitions)
    songs: list[dict]                 # what got picked, with reasons
    critique: Optional[dict] = None   # populated if critic was run
    timings: dict = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _section_first_lyric(lyrics_obj: dict, start_s: float, end_s: float) -> str:
    """Return the first lyric line that falls inside [start_s, end_s], or ''."""
    for ln in lyrics_obj.get("lines", []):
        s = float(ln.get("start", 0))
        if lyrics_obj.get("has_timestamps") and s and start_s <= s <= end_s:
            return ln.get("text", "")
    # fall back to the first line of the song
    lines = lyrics_obj.get("lines", [])
    return lines[0].get("text", "") if lines else ""


def _section_last_lyric(lyrics_obj: dict, start_s: float, end_s: float) -> str:
    last = ""
    for ln in lyrics_obj.get("lines", []):
        s = float(ln.get("start", 0))
        if lyrics_obj.get("has_timestamps") and s and start_s <= s <= end_s:
            last = ln.get("text", "")
    return last


def _build_seconds_for_bars(bars: int, bpm: float) -> float:
    return max(2.0, bars * 4.0 * 60.0 / bpm)


# ── Candidate gathering ──────────────────────────────────────────────────────

_QUERY_GEN_PROMPT = """Suggest {n} specific song titles (with artists) that fit this vibe.

VIBE: {vibe}

CRITICAL: All {n} songs must share a similar tempo band (within ±15 BPM of
each other). A mashup with songs at wildly different BPMs is unmixable. Pick
a target BPM range first (e.g. "110-130 BPM"), then list songs that ALL sit
inside that range. Use your musical knowledge.

Other rules:
- Real, well-known songs that exist on YouTube (not made-up titles).
- Mix of energy levels within the BPM band (some warm-up, some peak, some cool-down candidates).
- Diverse artists — don't pick five songs by the same person.

Return ONLY this JSON:
{{
  "target_bpm": "<e.g. '115-125 BPM'>",
  "queries": [
    "<song title> <artist>",
    ...
  ]
}}"""


def _bpm_hint(c: discover.Candidate) -> Optional[float]:
    """Look up cached BPM for a candidate. Returns None if not analyzed yet."""
    a = cache.read_json(cache.analysis_path(c.song_id))
    return float(a["bpm"]) if a and "bpm" in a else None


def _filter_bpm_outliers(candidates: list, max_spread: float = 30.0) -> list:
    """Drop candidates whose cached BPM is more than `max_spread` from the
    median of the candidate pool. Songs without cached BPMs are kept (we
    don't know their tempo yet, so we can't reject them).

    This prevents the LLM from picking songs that physically cannot be mixed
    (e.g. 92 BPM ballad alongside 144 BPM pop) — even if the LLM thinks the
    vibes match.
    """
    bpms = [b for c in candidates if (b := _bpm_hint(c)) is not None]
    if len(bpms) < 3:
        return candidates   # not enough cached info to judge
    bpms.sort()
    median = bpms[len(bpms) // 2]

    kept = []
    for c in candidates:
        b = _bpm_hint(c)
        if b is None or abs(b - median) <= max_spread:
            kept.append(c)
    return kept


def _gather_candidates(*, vibe: str, query: Optional[str], n: int,
                       on_progress) -> list[discover.Candidate]:
    """Build a candidate list. Strategy:

    1. If `query` was passed explicitly, search YouTube with it (one shot).
    2. Otherwise, ask the LLM to generate `n` specific song titles for the vibe,
       then search YouTube for each. This avoids the "playlist soup" problem
       where vague queries return DJ sets / mixes / jukeboxes.
    """
    if query:
        return discover.search(query, n=n)

    on_progress(f"  asking LLM for {n} specific song titles for the vibe...")
    raw = plan._call(_QUERY_GEN_PROMPT.format(n=n, vibe=vibe), max_tokens=500)
    queries = plan._parse_json(raw).get("queries", [])
    on_progress(f"  searching YouTube for {len(queries)} suggested songs...")

    seen: set[str] = set()
    out: list[discover.Candidate] = []
    for q in queries:
        try:
            results = discover.search(q, n=2)
        except Exception:
            continue
        for c in results:
            if c.youtube_id in seen:
                continue
            seen.add(c.youtube_id)
            out.append(c)
    return out


# ── Per-song enrichment ──────────────────────────────────────────────────────

def _enrich_song(
    cand: discover.Candidate,
    *,
    do_stems: bool,
    on_progress=lambda msg: print(msg),
) -> dict:
    """Download + analyze + lyrics + (optional) stems + section pick.

    Returns a dict that the renderer + transition planner consume.
    """
    on_progress(f"  [{cand.title[:48]}] downloading...")
    audio_path = discover.download(cand)

    on_progress(f"  [{cand.title[:48]}] analyzing musical structure...")
    an = analyze_mod.analyze(audio_path)
    an_dict = asdict(an)

    on_progress(f"  [{cand.title[:48]}] fetching lyrics...")
    try:
        ly = lyrics.fetch(cand)
        ly_dict = asdict(ly)
    except Exception as e:
        on_progress(f"  [warn] lyrics failed for {cand.title}: {e}")
        ly_dict = {"lines": [], "has_timestamps": False, "full_text": ""}

    if do_stems and not stems.has_stems(cand.song_id):
        on_progress(f"  [{cand.title[:48]}] separating stems (slow first time)...")
        try:
            stems.separate(audio_path)
        except Exception as e:
            on_progress(f"  [warn] stem separation failed: {e}")

    on_progress(f"  [{cand.title[:48]}] picking best section...")
    sec = plan.pick_section(title=cand.title, analysis=an_dict, lyrics=ly_dict)

    return {
        "candidate":  cand,
        "analysis":   an_dict,
        "lyrics":     ly_dict,
        "section":    sec,
        "first_lyric": _section_first_lyric(ly_dict, sec["start"], sec["end"]),
        "last_lyric":  _section_last_lyric(ly_dict,  sec["start"], sec["end"]),
        "has_stems":  stems.has_stems(cand.song_id),
    }


# ── Render ───────────────────────────────────────────────────────────────────

def _render_pipeline(enriched: list[dict],
                     transitions: list[dict],
                     target_bpm: float,
                     on_progress=lambda msg: print(msg)) -> np.ndarray:
    """Build the final mashup audio from the enriched songs and transition plans."""
    # Per-section audio at target BPM
    section_audio: list[np.ndarray] = []
    for i, e in enumerate(enriched):
        sec = e["section"]
        on_progress(f"  loading section {i+1}/{len(enriched)}: {sec['start']:.1f}-{sec['end']:.1f}s")
        a = render.load_section(e["candidate"].song_id, sec["start"], sec["end"])
        a = render.bpm_stretch(a, e["analysis"]["bpm"], target_bpm)
        section_audio.append(a)

    # Apply build to each section except the last; bass-swap or crossfade between
    mix = section_audio[0]
    # tiny fade-in on the very start
    fi = int(0.1 * config.SR)
    mix[:, :fi] *= np.linspace(0, 1, fi)

    # The LLM picks the technique per transition (see plan.plan_transition).
    # Orchestrator just dispatches. If the LLM picks a stem-based technique
    # but the stems aren't available, we silently fall back to bass_swap.
    n_trans = len(transitions)

    for i, t in enumerate(transitions):
        prev_e   = enriched[i]
        next_e   = enriched[i + 1]
        next_aud = section_audio[i + 1]
        prev_bpm = prev_e["analysis"]["bpm"]
        next_bpm = next_e["analysis"]["bpm"]
        sid_prev = prev_e["candidate"].song_id
        sid_next = next_e["candidate"].song_id
        sec_prev = prev_e["section"]
        sec_next = next_e["section"]

        technique = (t.get("technique") or "bass_swap").strip().lower()
        reasoning = t.get("reasoning", "")
        on_progress(f"  transition {i+1}/{n_trans}: {technique.upper()}  "
                    f"— {reasoning[:70]}")

        # ── Dispatch table ────────────────────────────────────────────────
        try:
            if technique == "tempo_ramp" or abs(prev_bpm - next_bpm) > 25:
                # Override: BPM mismatch always wins, regardless of LLM pick
                mix = render.tempo_ramp_blend(
                    mix, next_aud, a_bpm=prev_bpm, b_bpm=next_bpm,
                    ramp_bars=4, fade_bars=4,
                )

            elif technique == "acapella_drop" and next_e.get("has_stems"):
                voc = render.load_stem(sid_next, "vocals", sec_next["start"], sec_next["end"])
                if voc is None: raise FallbackToBassSwap()
                voc = render.bpm_stretch(voc, next_bpm, target_bpm)
                mix = render.acapella_drop_blend(
                    mix, next_aud, voc, fade_bars=8, bpm=target_bpm,
                )

            elif technique == "drum_swap" and prev_e.get("has_stems") and next_e.get("has_stems"):
                voc_p   = render.load_stem(sid_prev, "vocals", sec_prev["start"], sec_prev["end"])
                bass_p  = render.load_stem(sid_prev, "bass",   sec_prev["start"], sec_prev["end"])
                other_p = render.load_stem(sid_prev, "other",  sec_prev["start"], sec_prev["end"])
                drums_n = render.load_stem(sid_next, "drums",  sec_next["start"], sec_next["end"])
                if any(s is None for s in (voc_p, bass_p, other_p, drums_n)):
                    raise FallbackToBassSwap()
                voc_p   = render.bpm_stretch(voc_p,   prev_bpm, target_bpm)
                bass_p  = render.bpm_stretch(bass_p,  prev_bpm, target_bpm)
                other_p = render.bpm_stretch(other_p, prev_bpm, target_bpm)
                drums_n = render.bpm_stretch(drums_n, next_bpm, target_bpm)
                m = min(voc_p.shape[1], bass_p.shape[1], other_p.shape[1], mix.shape[1])
                a_no_drums = voc_p[:, :m] + bass_p[:, :m] + other_p[:, :m]
                mix = render.drum_swap_blend(
                    mix, next_aud, a_no_drums, drums_n,
                    fade_bars=8, bpm=target_bpm,
                )

            elif technique == "reverb_throw" and prev_e.get("has_stems"):
                voc_p = render.load_stem(sid_prev, "vocals", sec_prev["start"], sec_prev["end"])
                if voc_p is None: raise FallbackToBassSwap()
                voc_p = render.bpm_stretch(voc_p, prev_bpm, target_bpm)
                mix = render.reverb_throw_blend(
                    mix, next_aud, voc_p, fade_bars=8, bpm=target_bpm,
                )

            elif technique == "hard_drop":
                # SYSTEM-LEVEL: a hard_drop only lands if B slams in at a HIGH-ENERGY
                # moment (kick / vocal hook), not at the section's quiet intro.
                # Find B's loudest peak inside the chosen section, snap it to the
                # nearest downbeat at or before, and reload B starting there.
                an_n = next_e["analysis"]
                section_peaks = [p for p in an_n.get("peak_moments", [])
                                 if sec_next["start"] <= p["t"] < sec_next["end"]]
                if section_peaks:
                    peak_t = max(section_peaks, key=lambda p: p["rms"])["t"]
                    downbeats = an_n.get("downbeats", []) or []
                    snap_t = max((d for d in downbeats if d <= peak_t),
                                 default=sec_next["start"])
                    on_progress(f"      hard_drop landing on B's peak at {snap_t:.1f}s "
                                f"(was section start {sec_next['start']:.1f}s)")
                    next_aud_drop = render.load_section(sid_next, snap_t, sec_next["end"])
                    next_aud_drop = render.bpm_stretch(next_aud_drop, next_bpm, target_bpm)
                else:
                    next_aud_drop = next_aud
                # crash_gain 0.95 + B's full mix at full volume can clip the limiter
                # → distortion. 0.7 leaves headroom, the contrast with silence still pops.
                mix = render.hard_drop(
                    mix, next_aud_drop,
                    silence_beats=1.0, crash_gain=0.7, bpm=target_bpm,
                )

            elif technique == "bass_swap":
                mix = render.bass_swap_crossfade(
                    mix, next_aud, fade_bars=8, bpm=target_bpm,
                )

            elif technique == "dramatic_drop" and next_e.get("has_stems"):
                b_bass  = render.load_stem(sid_next, "bass",  sec_next["start"], sec_next["end"])
                b_drums = render.load_stem(sid_next, "drums", sec_next["start"], sec_next["end"])
                if b_bass is not None:  b_bass  = render.bpm_stretch(b_bass,  next_bpm, target_bpm)
                if b_drums is not None: b_drums = render.bpm_stretch(b_drums, next_bpm, target_bpm)
                mix = render.dramatic_drop(
                    mix, next_aud, b_bass=b_bass, b_drums=b_drums,
                    silence_beats=1.5, intro_bars=2, fade_beats=2,
                    crash_gain=0.95, bpm=target_bpm,
                )

            else:
                # Unknown / unsatisfied technique → bass_swap fallback
                raise FallbackToBassSwap()

        except FallbackToBassSwap:
            on_progress(f"      (falling back to bass_swap — stems missing for {technique})")
            mix = render.bass_swap_crossfade(
                mix, next_aud, fade_bars=8, bpm=target_bpm,
            )

    # Long fade out at the end
    fo_n = min(int(12.0 * config.SR), mix.shape[1])
    mix[:, -fo_n:] *= np.linspace(1, 0, fo_n)

    on_progress("  mastering (compressor + limiter)...")
    return render.master(mix)


# ── End-to-end ───────────────────────────────────────────────────────────────

def make_mashup(
    vibe: str,
    *,
    n_songs: int = 4,
    n_candidates: int = 18,
    candidate_query: Optional[str] = None,
    do_stems: bool = False,
    do_critique: bool = False,
    target_bpm: Optional[float] = None,
    on_progress=lambda msg: print(msg),
) -> MashupResult:
    """End-to-end: vibe → mashup MP3.

    `do_stems`: if True, run demucs on each song. Slow first time, cached.
    `do_critique`: if True, send the rendered mix to Gemini for review.
    `target_bpm`: if set, all songs are stretched to this. Default = first
                  picked song's detected BPM.
    """
    job_id = uuid.uuid4().hex[:10]
    timings: dict[str, float] = {}
    t_start = time.time()

    # ── 1. DISCOVER ──────────────────────────────────────────────────────
    t0 = time.time()
    on_progress(f"\n[1/6] Discovering candidates for vibe: {vibe!r}")
    candidates = _gather_candidates(
        vibe=vibe, query=candidate_query, n=n_candidates, on_progress=on_progress,
    )
    # Drop BPM outliers (uses cached analyses where available)
    pre_n = len(candidates)
    candidates = _filter_bpm_outliers(candidates, max_spread=30.0)
    dropped = pre_n - len(candidates)
    if len(candidates) < n_songs + 2:
        raise RuntimeError(
            f"Only {len(candidates)} valid single-track candidates found "
            f"(after BPM-outlier filter dropped {dropped}). "
            "Try a more specific vibe (mention artists or song styles)."
        )
    on_progress(f"      got {len(candidates)} candidates"
                + (f" ({dropped} BPM outliers dropped)" if dropped else ""))
    timings["discover_s"] = round(time.time() - t0, 2)

    # ── 2. SELECT (LLM) ──────────────────────────────────────────────────
    t0 = time.time()
    bpm_hints = {c.youtube_id: bpm for c in candidates
                 if (bpm := _bpm_hint(c)) is not None}
    on_progress(
        f"\n[2/6] LLM picking {n_songs} songs from {len(candidates)} candidates "
        f"({len(bpm_hints)} with cached BPM hints)..."
    )
    selection = plan.pick_songs(vibe, candidates, n=n_songs, bpm_hints=bpm_hints)
    on_progress(f"      Narrative: {selection.get('narrative', '')}")
    timings["pick_songs_s"] = round(time.time() - t0, 2)

    # Map picked youtube_ids back to Candidate objects (validate)
    by_id = {c.youtube_id: c for c in candidates}
    picked: list[discover.Candidate] = []
    for s in selection.get("songs", []):
        c = by_id.get(s.get("youtube_id"))
        if c:
            picked.append(c)
    if len(picked) < 2:
        raise RuntimeError(f"LLM picked invalid yt_ids; only {len(picked)} match candidates.")

    # ── 3. ENRICH (per song) ─────────────────────────────────────────────
    t0 = time.time()
    on_progress(f"\n[3/6] Enriching {len(picked)} songs (download + analyze + lyrics + section)...")
    enriched: list[dict] = []
    for c in picked:
        try:
            enriched.append(_enrich_song(c, do_stems=do_stems, on_progress=on_progress))
        except Exception as e:
            on_progress(f"  [warn] skipping {c.title}: {e}")
    if len(enriched) < 2:
        raise RuntimeError("less than 2 songs successfully enriched")
    timings["enrich_s"] = round(time.time() - t0, 2)

    # ── 4. PLAN TRANSITIONS (LLM) ────────────────────────────────────────
    t0 = time.time()
    on_progress(f"\n[4/6] Planning {len(enriched) - 1} transitions...")
    transitions: list[dict] = []
    n_total = len(enriched) - 1
    for i in range(n_total):
        out_e, in_e = enriched[i], enriched[i + 1]
        on_progress(f"  {out_e['candidate'].title[:32]} → {in_e['candidate'].title[:32]}")
        t = plan.plan_transition(
            out_title=out_e["candidate"].title,
            out_bpm=out_e["analysis"]["bpm"],
            out_key=out_e["analysis"]["key"],
            out_mode=out_e["analysis"]["mode"],
            out_lyric=out_e["last_lyric"],
            out_stems=out_e.get("has_stems", False),
            in_title=in_e["candidate"].title,
            in_bpm=in_e["analysis"]["bpm"],
            in_key=in_e["analysis"]["key"],
            in_mode=in_e["analysis"]["mode"],
            in_lyric=in_e["first_lyric"],
            in_stems=in_e.get("has_stems", False),
            position=i, total=n_total,
        )
        transitions.append(t)
    timings["plan_transitions_s"] = round(time.time() - t0, 2)

    # ── 5. RENDER ────────────────────────────────────────────────────────
    t0 = time.time()
    on_progress(f"\n[5/6] Rendering audio...")
    if target_bpm is not None:
        bpm = float(target_bpm)
    else:
        # Pick the median BPM. Clamp to a sane pop range (90-130).
        bpms = sorted(e["analysis"]["bpm"] for e in enriched)
        median_bpm = bpms[len(bpms) // 2]
        bpm = float(min(130.0, max(90.0, median_bpm)))
    on_progress(f"      target BPM: {bpm:.1f}")
    mix = _render_pipeline(enriched, transitions, bpm, on_progress=on_progress)
    out_path = cache.render_path(job_id)
    render.write_mp3(mix, out_path)
    duration_s = mix.shape[1] / config.SR
    timings["render_s"] = round(time.time() - t0, 2)
    on_progress(f"      wrote {out_path} ({duration_s:.1f}s)")

    # ── 6. CRITIQUE (optional) ───────────────────────────────────────────
    critique = None
    if do_critique:
        t0 = time.time()
        on_progress(f"\n[6/6] Critic listening to the mix...")
        plan_summary = (
            f"Songs (in order): " +
            " → ".join(e["candidate"].title for e in enriched) +
            f"\nTarget BPM: {bpm}\nTransitions: " +
            ", ".join(t.get("technique", "?") for t in transitions)
        )
        try:
            critique = plan.critique(out_path, plan_summary)
            on_progress(f"      score: {critique.get('score', 0):.2f}")
        except Exception as e:
            on_progress(f"      [warn] critique failed: {e}")
        timings["critique_s"] = round(time.time() - t0, 2)
    else:
        on_progress(f"\n[6/6] (critique skipped)")

    timings["total_s"] = round(time.time() - t_start, 2)

    return MashupResult(
        job_id=job_id,
        vibe=vibe,
        audio_path=str(out_path),
        duration_s=round(duration_s, 1),
        plan={
            "narrative":  selection.get("narrative", ""),
            "target_bpm": bpm,
            "sections": [
                {"song_id": e["candidate"].song_id,
                 "title":   e["candidate"].title,
                 "start":   e["section"]["start"],
                 "end":     e["section"]["end"],
                 "why":     e["section"].get("why", ""),
                 "iconic":  e["section"].get("iconic_lyric", "")}
                for e in enriched
            ],
            "transitions": transitions,
        },
        songs=[
            {"youtube_id": e["candidate"].youtube_id,
             "song_id":    e["candidate"].song_id,
             "title":      e["candidate"].title,
             "channel":    e["candidate"].channel,
             "bpm":        e["analysis"]["bpm"],
             "key":        f"{e['analysis']['key']} {e['analysis']['mode']}",
             "duration_s": e["analysis"]["duration_s"]}
            for e in enriched
        ],
        critique=critique,
        timings=timings,
    )

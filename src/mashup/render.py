"""Phase 6 — Pro DSP Rendering.

Takes a render plan (from the Phase 5 brain) and produces the final mashup
audio using pedalboard's pro-grade DSP — replacing our hand-rolled effects
in mix_pro.py with VST-quality processors.

Effects available (all pedalboard, JUCE-backed):
  HighpassFilter, LowpassFilter      — clean Butterworth filters
  Reverb                             — actual room/hall reverb
  Compressor                         — proper attack/release/ratio
  Limiter                            — mastering-grade brick wall
  Delay                              — feedback delay
  Phaser, Chorus                     — modulation effects

Plus our custom synthesised sounds (riser, crash, downlifter) that pedalboard
doesn't ship with — kept from mix_pro.py because they're effective.

A render plan looks like:

  {
    "target_bpm": 115.0,
    "sections": [
      {"song_id": "...", "section_start": 88.0, "section_end": 116.0,
       "stem_mix": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0}},
      ...
    ],
    "transitions": [
      {"from_idx": 0, "to_idx": 1,
       "build_bars": 8, "fx": [...], "silence_ms": 150,
       "use_bass_swap": true, "fade_kind": "drop"},
      ...
    ]
  }
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from . import cache, config, stems


# ── Loading sources ──────────────────────────────────────────────────────────

def load_stem(song_id: str, stem: str, start_s: float, end_s: float) -> Optional[np.ndarray]:
    """Load a single stem ('vocals'|'drums'|'bass'|'other') if available, else None."""
    if not stems.has_stems(song_id):
        return None
    import librosa
    p = stems.stem_paths(song_id)[stem]
    y, _ = librosa.load(str(p), sr=config.SR, mono=False, offset=start_s, duration=end_s - start_s)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.astype(np.float32)


def load_section(
    song_id: str,
    start_s: float,
    end_s: float,
    *,
    stem_mix: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """Load `start_s..end_s` of a song. If `stem_mix` is given, mix stems at
    those levels instead of using the full track.

    `stem_mix` example: {'vocals': 1.0, 'drums': 1.0, 'bass': 0.0, 'other': 0.7}
    (drops the bass for the bass-swap transition; quiet 'other' for cleaner mix)

    Returns a [2, samples] float32 array at config.SR.
    """
    import librosa

    sr = config.SR
    s_off = start_s
    dur = end_s - start_s

    if stem_mix is None or not stems.has_stems(song_id):
        path = cache.audio_path(song_id)
        y, _ = librosa.load(str(path), sr=sr, mono=False, offset=s_off, duration=dur)
        if y.ndim == 1:
            y = np.stack([y, y])
        return y.astype(np.float32)

    # Mix stems at requested levels
    paths = stems.stem_paths(song_id)
    out = None
    for name, gain in stem_mix.items():
        if gain <= 0 or name not in paths:
            continue
        y, _ = librosa.load(str(paths[name]), sr=sr, mono=False, offset=s_off, duration=dur)
        if y.ndim == 1:
            y = np.stack([y, y])
        y = y.astype(np.float32) * float(gain)
        out = y if out is None else out + y[:, : out.shape[1]]
    if out is None:
        raise RuntimeError(f"stem_mix produced no audio for song {song_id}")
    return out


# ── BPM matching (reuse librosa) ─────────────────────────────────────────────

def bpm_stretch(audio: np.ndarray, from_bpm: float, to_bpm: float,
                max_rate: float = 1.20) -> np.ndarray:
    """Time-stretch audio to match a target BPM.

    Skips the stretch if the rate would be too extreme (>1.2x or <0.83x) —
    chipmunk-style speed changes sound worse than letting the BPM mismatch
    show. Caller should design the transition (longer crossfade, build that
    masks the tempo gap, etc.) when this returns the original.
    """
    import librosa
    if from_bpm <= 0 or to_bpm <= 0:
        return audio
    rate = to_bpm / from_bpm
    if abs(rate - 1.0) < 0.01:
        return audio
    if rate > max_rate or rate < (1.0 / max_rate):
        # Too extreme — preserve the original instead of squashing it
        return audio
    return np.stack([
        librosa.effects.time_stretch(audio[0], rate=rate),
        librosa.effects.time_stretch(audio[1], rate=rate),
    ]).astype(np.float32)


# ── Pedalboard effect wrappers ───────────────────────────────────────────────

def apply_pedalboard(audio: np.ndarray, board) -> np.ndarray:
    """Run a pedalboard.Pedalboard on stereo audio. Audio is [2, n]; pedalboard
    expects [n, 2] for stereo. We swap and back."""
    return board(audio.T, config.SR).T.astype(np.float32)


def make_master_chain():
    """Return a Pedalboard that's a sane mastering chain (compressor + limiter)."""
    from pedalboard import Compressor, Limiter, Pedalboard
    return Pedalboard([
        Compressor(threshold_db=-15, ratio=3.0, attack_ms=5, release_ms=80),
        Limiter(threshold_db=-1.0, release_ms=100),
    ])


def hpf(audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
    from pedalboard import HighpassFilter, Pedalboard
    return apply_pedalboard(audio, Pedalboard([HighpassFilter(cutoff_frequency_hz=cutoff_hz)]))


def lpf(audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
    from pedalboard import LowpassFilter, Pedalboard
    return apply_pedalboard(audio, Pedalboard([LowpassFilter(cutoff_frequency_hz=cutoff_hz)]))


def reverb(audio: np.ndarray, *, room_size: float = 0.8, wet: float = 0.5) -> np.ndarray:
    from pedalboard import Pedalboard, Reverb
    return apply_pedalboard(audio, Pedalboard([
        Reverb(room_size=room_size, damping=0.5, wet_level=wet, dry_level=1.0 - wet)
    ]))


# ── Synthesised sounds (kept from mix_pro.py — pedalboard doesn't ship these) ─

def make_riser(dur_s: float, gain: float = 0.45) -> np.ndarray:
    """White-noise riser sweeping 200Hz → 16kHz with a convex envelope."""
    import scipy.signal
    n = int(dur_s * config.SR)
    noise = np.random.randn(2, n).astype(np.float32)
    cutoffs = np.geomspace(200, min(16000, config.SR / 2 - 100), 40)
    chunk = max(1, n // 40)
    out = np.zeros_like(noise)
    for i, hz in enumerate(cutoffs):
        s, e = i * chunk, min((i + 1) * chunk, n)
        sos = scipy.signal.butter(3, hz / (config.SR / 2), "high", output="sos")
        out[:, s:e] = scipy.signal.sosfilt(sos, noise[:, s:e])
    t = np.linspace(0, 1, n)
    return out * (t ** 1.5) * gain


def make_crash(dur_s: float = 1.5, gain: float = 0.85) -> np.ndarray:
    import scipy.signal
    n = int(dur_s * config.SR)
    noise = np.random.randn(2, n).astype(np.float32)
    sos = scipy.signal.butter(4, 3000 / (config.SR / 2), "high", output="sos")
    noise = scipy.signal.sosfilt(sos, noise)
    return noise * np.exp(-np.linspace(0, 6, n)) * gain


def stutter_beat(audio: np.ndarray, bpm: float, n_reps: int = 4) -> np.ndarray:
    beat_n = min(int(60.0 / bpm * config.SR), audio.shape[1])
    last_beat = audio[:, -beat_n:]
    return np.concatenate([last_beat * (0.78 ** i) for i in range(n_reps)], axis=1)


# ── Bass swap transition ─────────────────────────────────────────────────────

def tempo_ramp_blend(
    a: np.ndarray,
    b: np.ndarray,
    *,
    a_bpm: float,
    b_bpm: float,
    ramp_bars: int = 4,
    fade_bars: int = 4,
) -> np.ndarray:
    """Smoothly bridge a BPM mismatch by time-stretching A's tail to match B.

    Over `ramp_bars`, gradually accelerate (or decelerate) A's tail until it
    hits B's tempo, then do a `fade_bars` bass-swap blend at the matched BPM.

    This is the only sane way to mix songs with very different tempos
    (e.g. 92 BPM Pasoori → 144 BPM Calm Down) without sounding comical.
    """
    import librosa

    target_bpm = b_bpm
    ramp_s = ramp_bars * 4 * 60.0 / a_bpm   # length in A's native time
    ramp_n = min(int(ramp_s * config.SR), a.shape[1])
    if ramp_n <= 0 or abs(a_bpm - b_bpm) < 2:
        # No ramp needed — just bass swap
        return bass_swap_crossfade(a, b, fade_bars=fade_bars, bpm=target_bpm)

    # Split: a_keep = everything except the last `ramp_n` samples; a_ramp = the tail to stretch
    a_keep = a[:, :-ramp_n] if a.shape[1] > ramp_n else np.zeros((2, 0), dtype=np.float32)
    a_ramp = a[:, -ramp_n:].astype(np.float32)

    # Time-stretch the ramp section to b_bpm. rate = a_bpm / b_bpm
    # (rate > 1 means we want to play it faster — librosa.time_stretch's `rate`
    # parameter does exactly this: rate=2 plays the audio twice as fast.)
    rate = a_bpm / b_bpm
    a_ramp_stretched = np.stack([
        librosa.effects.time_stretch(a_ramp[0], rate=rate),
        librosa.effects.time_stretch(a_ramp[1], rate=rate),
    ]).astype(np.float32)

    a_matched = np.concatenate([a_keep, a_ramp_stretched], axis=1)
    return bass_swap_crossfade(a_matched, b, fade_bars=fade_bars, bpm=target_bpm)


def reverb_throw_blend(
    a: np.ndarray,
    b: np.ndarray,
    a_vocals: np.ndarray,
    *,
    fade_bars: int = 8,
    bpm: float = 115.0,
) -> np.ndarray:
    """Pro-DJ "throw": A's last vocal phrase is bathed in big reverb that
    trails into B's intro. Adds an epic, cinematic feel to the transition.

    Combines: reverb-soaked vocal tail of A + standard bass-swap into B.
    Requires A's vocal stem.
    """
    fade_s = fade_bars * 4 * 60.0 / bpm
    fn = min(int(fade_s * config.SR), a.shape[1], a_vocals.shape[1])

    # Reverb-soak A's vocal tail (just the last 2 bars of the build)
    bars_2 = int(2 * 4 * 60.0 / bpm * config.SR)
    bars_2 = min(bars_2, fn)
    voc_tail = a_vocals[:, -bars_2:].astype(np.float32)
    # Apply heavy reverb (long decay, lots of wet)
    voc_reverb = reverb(voc_tail, room_size=0.95, wet=0.85)

    # Layer the reverbed vocal trail over the bass-swap blend
    blended = bass_swap_crossfade(a, b, fade_bars=fade_bars, bpm=bpm)

    # Find where the vocal tail goes in the blended audio. The blend produces:
    # [a[:-fn], overlap_of_fn, b[fn:]]. The vocal tail should sit RIGHT AT the
    # start of the overlap. Compute that index in the blended array.
    blend_overlap_start = a.shape[1] - fn   # in blended coords, before the new b portion
    target_start = blend_overlap_start + fn - bars_2   # last 2 bars of A's build
    target_end = target_start + bars_2
    if 0 <= target_start < blended.shape[1] and target_end <= blended.shape[1]:
        blended[:, target_start:target_end] += voc_reverb * 0.6

    return blended


def tasteful_drop(
    a: np.ndarray,
    b_full: np.ndarray,
    b_bass: Optional[np.ndarray] = None,
    b_drums: Optional[np.ndarray] = None,
    *,
    silence_ms: int = 700,
    intro_bars: int = 2,
    crash_gain: float = 1.0,
    bpm: float = 115.0,
) -> np.ndarray:
    """The climax moment — used ONCE per set, no more.

    Pattern:
      1. A's tail: last 1 bar fades from full → silent (so the cut isn't jarring)
      2. 700ms of complete silence (the "wait for it" moment)
      3. CRASH cymbal hits hard
      4. `intro_bars` of bass + drums ONLY (the wind-up — quieter)
      5. Full mix slams in (the pay-off — loud contrast)

    Earlier version had two bugs:
      - silence was only 400ms (too short to feel deliberate)
      - bass-only intro was 1 bar (~2.6s at 92 BPM, too short to register)
      - bass intro at 0.95 gain made the full mix slam less impactful
    Now: 700ms silence, 2 bars of bass+drums at 0.7 gain, then full mix at 1.0.
    """
    a_out = a.copy().astype(np.float32)

    # 1. Fade A's last bar to silence (avoid the click/jolt of a hard cut)
    bar_n = int(4 * 60.0 / bpm * config.SR)
    fade_n = min(bar_n, a_out.shape[1])
    a_out[:, -fade_n:] *= np.linspace(1.0, 0.0, fade_n)

    # 2. Total silence
    silence = np.zeros((2, int(silence_ms / 1000 * config.SR)), dtype=np.float32)

    # 3-5. Build the incoming side
    if b_bass is None:
        # Simple drop: just stamp crash on top of full mix
        out_b = stamp_crash(b_full.astype(np.float32), gain=crash_gain)
    else:
        # Big drop: bass-only (or bass+drums) intro for `intro_bars`, then full mix
        intro_n = intro_bars * bar_n
        intro_n = min(intro_n, b_full.shape[1], b_bass.shape[1])

        # Build the intro layer: bass + drums (if available) at lower volume
        intro = b_bass[:, :intro_n].astype(np.float32) * 0.65
        if b_drums is not None:
            d = b_drums[:, :intro_n].astype(np.float32) * 0.7
            intro += d[:, : intro.shape[1]]

        b_rest = b_full[:, intro_n:].astype(np.float32)
        out_b = np.concatenate([intro, b_rest], axis=1)
        out_b = stamp_crash(out_b, gain=crash_gain)

    return np.concatenate([a_out, silence, out_b], axis=1)


def drum_swap_blend(
    a_full: np.ndarray,
    b_full: np.ndarray,
    a_no_drums: np.ndarray,    # A's vocals + bass + other (no drums)
    b_drums: np.ndarray,       # B's drums only
    *,
    fade_bars: int = 8,
    bpm: float = 115.0,
) -> np.ndarray:
    """The 'beat switch' — A's vocals continue, but B's drums take over.

    Phase 1 (first half of overlap):
      - A plays full mix as normal

    Phase 2 (second half):
      - A's vocals + bass + other continue (a_no_drums)
      - B's DRUMS replace A's drums (the energy shift)
      - Feels like the same singer is now riding a different beat

    At end of overlap: hard cut to B's full mix.

    The defining mashup move for songs with strong vocals — gives you the
    "Kanye-style beat switch" feel without any silence or white noise.
    """
    fade_s = fade_bars * 4 * 60.0 / bpm
    fn = min(int(fade_s * config.SR), a_full.shape[1], b_full.shape[1],
             a_no_drums.shape[1], b_drums.shape[1])
    if fn <= 0:
        return np.concatenate([a_full, b_full], axis=1)

    half = fn // 2

    # ── Outgoing A's tail
    a_tail = a_full[:, -fn:].copy().astype(np.float32)
    a_no_drums_tail = a_no_drums[:, -fn:].astype(np.float32)
    # Phase 2: hard-swap A's full mix for A's no-drums version
    a_tail[:, half:] = a_no_drums_tail[:, half:]
    # A's volume: full through phase 1, ramps down very gently in phase 2
    # (shouldn't fully fade — we want the vocals to carry)
    a_vol = np.ones(fn, dtype=np.float32)
    a_vol[half:] = np.linspace(1.0, 0.4, fn - half)
    a_tail *= a_vol

    # ── Incoming B's drums layered in phase 2
    b_drums_tail = b_drums[:, :fn].astype(np.float32)
    drum_vol = np.zeros(fn, dtype=np.float32)
    drum_vol[half:] = np.linspace(0.0, 1.1, fn - half)   # slight boost to make the swap obvious
    b_drum_layer = b_drums_tail * drum_vol

    # ── B's full mix volume: 0 in phase 1+2 (drums only), ramps up to 1 right at the end
    # Actually we want the FULL B to take over after the overlap, so we don't add
    # B's full mix during the overlap — just append it after.
    overlap = a_tail + b_drum_layer

    return np.concatenate([a_full[:, :-fn], overlap, b_full[:, fn:]], axis=1)


def acapella_break(
    audio: np.ndarray,
    vocals: np.ndarray,
    *,
    break_bars: int = 4,
    bpm: float = 115.0,
) -> np.ndarray:
    """Insert N bars of just-vocals before resuming the full mix.

    Drops everything except the vocal stem for `break_bars`, then the full
    mix slams back. Pro DJ tension move — used right before a drop.

    Returns: [...full_mix, vocal_break, full_mix...]
    Inserts the break at the END of the supplied `audio` (caller decides
    where to splice it in).
    """
    n = int(break_bars * 4 * 60.0 / bpm * config.SR)
    n = min(n, vocals.shape[1])
    voc_break = vocals[:, :n].astype(np.float32) * 1.15   # slight boost
    return np.concatenate([audio, voc_break], axis=1)


def double_drop(
    a_full: np.ndarray,
    b_full: np.ndarray,
    a_inst: np.ndarray,    # A's drums + bass + other (no vocals)
    b_full_drop: np.ndarray,  # B's full mix at its peak/drop moment
    *,
    pre_bars: int = 4,
    bpm: float = 115.0,
) -> np.ndarray:
    """Both songs' choruses hit on the same beat — the iconic double-drop move.

    Last `pre_bars` of A are stripped to instrumental (vocals out) so they
    don't clash with B's incoming vocals. At bar 0, A's full mix yields to
    B's full mix in a hard volume swap (no fade — the drop is the moment).

    Caller must provide A's instrumental stem (drums + bass + other) for the
    last few bars. B starts at its drop moment.
    """
    pre_n = int(pre_bars * 4 * 60.0 / bpm * config.SR)
    pre_n = min(pre_n, a_full.shape[1], a_inst.shape[1])

    out = a_full.copy().astype(np.float32)
    # Replace the last pre_bars with the instrumental
    out[:, -pre_n:] = a_inst[:, -pre_n:].astype(np.float32) * 0.85

    # Hard cut into B's full mix at the drop
    return np.concatenate([out, b_full_drop.astype(np.float32)], axis=1)


def drum_only_break(
    audio_after: np.ndarray,
    drums: np.ndarray,
    *,
    bars: int = 4,
    bpm: float = 115.0,
) -> np.ndarray:
    """Insert N bars of just the drum stem before `audio_after` resumes.

    Classic DJ move — strip everything down to the kick/snare for tension,
    then drop back to full mix. Works as a "moment" inside a section.
    """
    n = int(bars * 4 * 60.0 / bpm * config.SR)
    n = min(n, drums.shape[1])
    return np.concatenate([drums[:, :n].astype(np.float32) * 0.9, audio_after], axis=1)


def acapella_drop_blend(
    a: np.ndarray,
    b_full: np.ndarray,
    b_vocals: np.ndarray,
    *,
    fade_bars: int = 8,
    bpm: float = 115.0,
) -> np.ndarray:
    """The actual mashup move: incoming song's VOCALS over outgoing song's BEAT.

    Phase 1 (first half of overlap):
      - A plays its full mix
      - B's VOCAL stem only enters on top — feels like "next track's singer
        is jamming over the current track's groove"

    Phase 2 (second half):
      - A fades out
      - B's full mix takes over (vocals continue uninterrupted, plus drums/bass)

    Requires stems for B (incoming). A's full mix is fine.
    """
    fade_s = fade_bars * 4 * 60.0 / bpm
    fn = min(int(fade_s * config.SR), a.shape[1], b_full.shape[1], b_vocals.shape[1])
    if fn <= 0:
        return np.concatenate([a, b_full], axis=1)

    half = fn // 2

    # ── Outgoing A's tail: full mix in phase 1, fade-out in phase 2
    a_tail = a[:, -fn:].copy().astype(np.float32)
    a_vol = np.ones(fn, dtype=np.float32)
    a_vol[half:] = np.linspace(1.0, 0.0, fn - half)
    a_tail *= a_vol

    # ── Incoming B's head: vocals only in phase 1 (with quick fade-in), full mix in phase 2
    b_head = np.zeros((2, fn), dtype=np.float32)
    voc = b_vocals[:, :fn].astype(np.float32)
    fi = int(0.3 * config.SR)   # 300ms vocal fade-in so it doesn't pop
    fi = min(fi, half)
    voc_vol = np.ones(fn, dtype=np.float32)
    voc_vol[:fi] = np.linspace(0, 1, fi)
    voc_vol[half:] = 0.0   # vocals already in the full-mix during phase 2
    b_head += voc * voc_vol * 1.1   # gentle boost so vocals sit above A's mix

    full_vol = np.zeros(fn, dtype=np.float32)
    full_vol[half:] = np.linspace(0.0, 1.0, fn - half)
    b_head += b_full[:, :fn].astype(np.float32) * full_vol

    overlap = a_tail + b_head
    return np.concatenate([a[:, :-fn], overlap, b_full[:, fn:]], axis=1)


def bass_swap_crossfade(a: np.ndarray, b: np.ndarray, fade_bars: int = 8,
                        bpm: float = 115.0) -> np.ndarray:
    """Real DJ 16-bar bass-swap blend (default fade_bars=8 → ~16s at 115 BPM).

    The two tracks PLAY SIMULTANEOUSLY for `fade_bars` bars. No added effects,
    no risers, no synth crashes — just two real tracks separated by EQ:

      Phase 1 (first half of overlap, bars 0..fade_bars/2):
        - A plays at full volume
        - B plays with sub-200Hz cut (HPF at 200Hz). Bass-light, sits "above" A.

      Phase 2 (second half, bars fade_bars/2..fade_bars):
        - A's bass cuts (progressive HPF 80 → 300Hz)
        - B's bass enters via LPF<200Hz that ramps from 0 to full over this half
        - A's overall volume fades down, B's stays up

    The bass-swap happens at the midpoint of the overlap, which lands on a
    natural phrase boundary if `fade_bars` is 8 or 16.
    """
    import scipy.signal

    fade_s = fade_bars * 4 * 60.0 / bpm
    fn = min(int(fade_s * config.SR), a.shape[1], b.shape[1])
    if fn <= 0:
        return np.concatenate([a, b], axis=1)

    half = fn // 2
    sos_lp = scipy.signal.butter(4, 200 / (config.SR / 2), "low",  output="sos")
    sos_hp = scipy.signal.butter(4, 200 / (config.SR / 2), "high", output="sos")

    # ── Outgoing A's tail: bass plays normally through phase 1, then HARD CUT
    # at the midpoint (the bass-swap moment — real DJs yank the bass knob
    # instantaneously at the phrase boundary, no sweep that would "whoosh").
    a_tail = a[:, -fn:].copy().astype(np.float32)
    a_tail[:, half:] = scipy.signal.sosfilt(sos_hp, a_tail[:, half:])

    # A's volume: full through phase 1, ramp down over phase 2
    a_vol = np.ones(fn, dtype=np.float32)
    a_vol[half:] = np.linspace(1.0, 0.0, fn - half)
    a_tail = a_tail * a_vol

    # ── Incoming B's head: bass cut throughout phase 1, then HARD-IN at midpoint.
    # Like A's hard cut, this is the canonical bass-swap — both tracks switch
    # bass duty in one clean move at the phrase boundary.
    b_head = b[:, :fn].copy().astype(np.float32)
    b_highs = scipy.signal.sosfilt(sos_hp, b_head)
    b_full  = b_head    # untouched mids/highs/bass for phase 2
    out_seg = np.zeros_like(b_head)
    out_seg[:, :half] = b_highs[:, :half]   # phase 1: highs only
    out_seg[:, half:] = b_full[:, half:]    # phase 2: full track
    b_head = out_seg

    # B's volume: ramps up over phase 1, full through phase 2
    b_vol = np.ones(fn, dtype=np.float32)
    b_vol[:half] = np.linspace(0.0, 1.0, half)
    b_head = b_head * b_vol

    overlap = a_tail + b_head
    return np.concatenate([a[:, :-fn], overlap, b[:, fn:]], axis=1)


# ── Section build (HPF sweep + riser + stutter, ending in near-silence) ──────

def build_section(body: np.ndarray, build_s: float = 8.0,
                  riser_gain: float = 0.45,
                  bpm: float = 115.0) -> np.ndarray:
    """Apply a build to the last `build_s` seconds of `body`, then append a stutter."""
    import scipy.signal
    out = body.copy()
    build_n = min(int(build_s * config.SR), out.shape[1])

    # progressive HPF (80Hz → 14kHz over build_s)
    cutoffs = np.geomspace(80, 14000, 60)
    chunk = max(1, build_n // 60)
    seg = out[:, -build_n:]
    swept = np.zeros_like(seg)
    for i, hz in enumerate(cutoffs):
        s, e = i * chunk, min((i + 1) * chunk, build_n)
        sos = scipy.signal.butter(4, hz / (config.SR / 2), "high", output="sos")
        swept[:, s:e] = scipy.signal.sosfilt(sos, seg[:, s:e])
    out[:, -build_n:] = swept

    # add riser layered on
    rise = make_riser(build_s, gain=riser_gain)
    out[:, -build_n:] += rise[:, :build_n]

    # append stutter of last beat
    beat_n = int(60.0 / bpm * config.SR)
    last_beat = out[:, -beat_n:].copy()
    stut = np.concatenate([last_beat * (0.78 ** i) for i in range(4)], axis=1)

    return np.concatenate([out, stut], axis=1)


def stamp_crash(audio: np.ndarray, gain: float = 0.85) -> np.ndarray:
    cr = make_crash(1.5, gain)
    out = audio.copy()
    n = min(cr.shape[1], out.shape[1])
    out[:, :n] += cr[:, :n]
    return out


# ── Synth helpers used by apply_fx_plan ──────────────────────────────────────

def make_downlifter(dur_s: float, gain: float = 0.55) -> np.ndarray:
    """Bass whoosh sweeping 800Hz → 30Hz with a fade-out envelope."""
    import scipy.signal
    n = int(dur_s * config.SR)
    noise = np.random.randn(2, n).astype(np.float32)
    cutoffs = np.geomspace(800, 30, 30)
    chunk = max(1, n // 30)
    out = np.zeros_like(noise)
    for i, hz in enumerate(cutoffs):
        s, e = i * chunk, min((i + 1) * chunk, n)
        sos = scipy.signal.butter(4, max(hz, 20) / (config.SR / 2), "low", output="sos")
        out[:, s:e] = scipy.signal.sosfilt(sos, noise[:, s:e])
    t = np.linspace(1, 0, n)
    return out * (t ** 0.7) * gain


def apply_flanger(audio: np.ndarray, rate_hz: float = 1.5, max_ms: float = 10.0,
                  wet: float = 0.7) -> np.ndarray:
    n = audio.shape[1]
    t = np.arange(n) / config.SR
    max_d = int(max_ms * config.SR / 1000)
    lfo = ((np.sin(2 * np.pi * rate_hz * t) + 1) / 2 * max_d).astype(int)
    out = audio.copy().astype(np.float32)
    chunk = config.SR // 20
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = int(lfo[(s + e) // 2])
        if d > 0 and s >= d:
            out[:, s:e] += audio[:, s - d:e - d] * wet
    return (out / (1 + wet)).astype(np.float32)


def progressive_hpf(audio: np.ndarray, low_hz: float = 80,
                    high_hz: float = 14000, n_steps: int = 60) -> np.ndarray:
    """HPF cutoff ramping from low_hz → high_hz across the entire audio.

    Different from the static `hpf()` — this is the dramatic build-up sweep.
    """
    import scipy.signal
    n = audio.shape[1]
    if n == 0:
        return audio
    cutoffs = np.geomspace(max(low_hz, 20), min(high_hz, config.SR / 2 - 100), n_steps)
    chunk = max(1, n // n_steps)
    out = np.zeros_like(audio)
    for i, hz in enumerate(cutoffs):
        s, e = i * chunk, min((i + 1) * chunk, n)
        sos = scipy.signal.butter(4, hz / (config.SR / 2), "high", output="sos")
        out[:, s:e] = scipy.signal.sosfilt(sos, audio[:, s:e])
    return out


# ── Apply LLM's fx plan ──────────────────────────────────────────────────────
#
# The brain returns a list like:
#   [{"name": "hpf_sweep",      "start_bar": -8, "end_bar": -1,
#     "low_hz": 100, "high_hz": 14000},
#    {"name": "riser",          "start_bar": -8, "end_bar": -0.5, "gain": 0.5},
#    {"name": "stutter",        "start_bar": -1, "end_bar": -0.25, "n_reps": 4},
#    {"name": "reverb_throw",   "at_bar": -0.25, "decay_time": 2.0},
#    {"name": "downlifter",     "start_bar": -8, "end_bar": -2, "gain": 0.5},
#    {"name": "flanger",        "start_bar": -6, "end_bar": -2,
#     "rate_hz": 1.5, "wet": 0.7},
#    {"name": "crash",          "at_bar": 0, "gain": 0.9}]
#
# Bars are RELATIVE to the drop moment. The drop moment is the END of the
# outgoing tail (which is the beginning of the incoming song). Negative bars
# happen during outgoing's tail.

def _bar_to_sample_offset(bar: float, bpm: float) -> int:
    """Convert a bar number to a sample offset. 1 bar = 4 beats at `bpm`."""
    return int(bar * 4 * 60.0 / bpm * config.SR)


FX_GAIN_BOOST = 3.5   # global multiplier so the LLM-spec'd gains actually pop
DUCK_DEPTH = 0.30     # outgoing drops to 30% during the build = lots of headroom for fx


def apply_fx_plan(out_tail: np.ndarray, in_head: np.ndarray,
                  fx_list: list[dict], bpm: float) -> tuple[np.ndarray, np.ndarray]:
    """Mutate (out_tail, in_head) according to the LLM's fx plan.

    `out_tail` is the LAST few bars of the outgoing song (its build region).
    `in_head` is the FIRST few bars of the incoming song.

    The fx list addresses bars relative to the seam between them:
      - bar -8 = 8 bars before the drop (early in out_tail)
      - bar 0  = the drop moment (start of in_head)
      - bar +1 = 1 bar after the drop (in_head)

    Returns the (possibly modified) (out_tail, in_head) ready to be concatenated.
    """
    out = out_tail.copy()
    inn = in_head.copy()
    out_n = out.shape[1]

    bar_n = _bar_to_sample_offset(1.0, bpm)
    if bar_n <= 0:
        return out, inn

    # Volume duck on the outgoing tail: linear ramp from 1.0 → DUCK_DEPTH over
    # the build region. This is the headroom that lets risers + crashes be heard.
    duck_start_bar = -8.0  # match typical build_bars
    s_duck = max(0, out_n + _bar_to_sample_offset(duck_start_bar, bpm))
    if s_duck < out_n:
        ramp = np.linspace(1.0, DUCK_DEPTH, out_n - s_duck).astype(np.float32)
        out[:, s_duck:] *= ramp

    def _slice_out(start_bar: float, end_bar: float):
        """Return [s, e] sample indices into `out` for the bar range, both negative."""
        s = max(0, out_n + _bar_to_sample_offset(start_bar, bpm))
        e = max(0, out_n + _bar_to_sample_offset(end_bar, bpm))
        if e <= s:
            return None
        return s, min(e, out_n)

    def _slice_in(start_bar: float, end_bar: float):
        s = max(0, _bar_to_sample_offset(start_bar, bpm))
        e = max(0, _bar_to_sample_offset(end_bar, bpm))
        if e <= s:
            return None
        return s, min(e, inn.shape[1])

    for fx in fx_list:
        name = (fx.get("name") or "").lower().replace("-", "_")

        # ── Outgoing-side build effects ────────────────────────────────────
        if name in ("hpf_sweep", "filter_sweep"):
            sl = _slice_out(fx.get("start_bar", -8), fx.get("end_bar", -1))
            if sl:
                s, e = sl
                low_hz  = float(fx.get("low_hz", 100))
                high_hz = float(fx.get("high_hz", 14000))
                out[:, s:e] = progressive_hpf(out[:, s:e], low_hz, high_hz)

        elif name == "riser":
            sl = _slice_out(fx.get("start_bar", -8), fx.get("end_bar", -0.5))
            if sl:
                s, e = sl
                rise = make_riser((e - s) / config.SR,
                                  gain=float(fx.get("gain", 0.45)) * FX_GAIN_BOOST)
                k = min(rise.shape[1], e - s)
                out[:, s : s + k] += rise[:, :k]

        elif name == "downlifter":
            sl = _slice_out(fx.get("start_bar", -8), fx.get("end_bar", -2))
            if sl:
                s, e = sl
                dl = make_downlifter((e - s) / config.SR,
                                     gain=float(fx.get("gain", 0.55)) * FX_GAIN_BOOST)
                k = min(dl.shape[1], e - s)
                out[:, s : s + k] += dl[:, :k]

        elif name == "flanger":
            sl = _slice_out(fx.get("start_bar", -6), fx.get("end_bar", -2))
            if sl:
                s, e = sl
                out[:, s:e] = apply_flanger(
                    out[:, s:e],
                    rate_hz=float(fx.get("rate_hz", 1.5)),
                    max_ms=float(fx.get("max_ms", 10.0)),
                    wet=float(fx.get("wet", 0.7)),
                )

        elif name in ("reverb_throw", "reverb"):
            # Apply reverb to the very last beat of the outgoing tail
            at_bar = float(fx.get("at_bar", -0.5))
            beat_n = bar_n // 4
            s = max(0, out_n + _bar_to_sample_offset(at_bar, bpm))
            e = min(out_n, s + beat_n * 2)
            if e > s:
                wet = float(fx.get("wet", 0.6))
                out[:, s:e] = reverb(out[:, s:e], wet=wet)

        elif name == "stutter":
            # Replace the stutter window with N repeats of its first beat
            sl = _slice_out(fx.get("start_bar", -1), fx.get("end_bar", -0.25))
            if sl:
                s, e = sl
                n_reps = max(2, int(fx.get("n_reps", 4)))
                source_beat = out[:, s : s + (e - s) // n_reps].copy()
                stut = np.concatenate(
                    [source_beat * (0.78 ** i) for i in range(n_reps)], axis=1
                )
                copy_n = min(stut.shape[1], e - s)
                out[:, s : s + copy_n] = stut[:, :copy_n]

        # ── Drop-moment effect ────────────────────────────────────────────
        elif name == "crash":
            at_bar = float(fx.get("at_bar", 0))
            gain = float(fx.get("gain", 0.9))
            cr = make_crash(1.5, gain * FX_GAIN_BOOST)
            cn = cr.shape[1]
            if at_bar < 0:
                # crash happens during outgoing tail
                s = max(0, out_n + _bar_to_sample_offset(at_bar, bpm))
                e = min(out_n, s + cn)
                out[:, s:e] += cr[:, : e - s]
            else:
                # crash happens at/after the drop (incoming side)
                s = max(0, _bar_to_sample_offset(at_bar, bpm))
                e = min(inn.shape[1], s + cn)
                inn[:, s:e] += cr[:, : e - s]

        # Unknown fx name — silently ignore
    return out, inn


# ── Master + write ───────────────────────────────────────────────────────────

def normalize(audio: np.ndarray, db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak <= 0:
        return audio
    return (audio * (10 ** (db / 20)) / peak).astype(np.float32)


def master(audio: np.ndarray) -> np.ndarray:
    """Run the mastering chain (compressor + limiter) and normalise."""
    pedboard = make_master_chain()
    mastered = apply_pedalboard(audio, pedboard)
    return normalize(mastered, db=-1.5)


def write_mp3(audio: np.ndarray, out_path: Path, *, bitrate: str = "192k") -> Path:
    """Write [2, n] float32 audio to MP3 via WAV intermediate."""
    from pydub import AudioSegment
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path = out_path.with_suffix(".wav")
    sf.write(wav_path, audio.T, config.SR, subtype="PCM_16")
    AudioSegment.from_wav(wav_path).export(out_path, format="mp3", bitrate=bitrate)
    wav_path.unlink(missing_ok=True)
    return out_path

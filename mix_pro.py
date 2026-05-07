"""
Everyone a DJ — 4-Song Pro Mashup v3
Badtameez Dil → Uptown Funk → Naatu Naatu → Levitating

Structure (the anatomy of every drop):
  Clean section → HPF build (bass disappears) + riser climbs
  → stutter (1 bar) → 150ms silence → CRASH + full bass lands

Effects only at the right moments. Clean sections stay clean.
"""

from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import scipy.signal
import os

SR = 44100
TARGET_BPM = 115.0
np.random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def stereo(y: np.ndarray) -> np.ndarray:
    return np.stack([y, y]) if y.ndim == 1 else y

def load_clip(path: str, start_s: float, dur_s: float) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=False, offset=start_s, duration=dur_s)
    return stereo(y).astype(np.float32)

def bpm_stretch(audio: np.ndarray, from_bpm: float, to_bpm: float) -> np.ndarray:
    rate = to_bpm / from_bpm
    if abs(rate - 1.0) < 0.01:
        return audio
    print(f"  {from_bpm:.1f} → {to_bpm:.1f} BPM")
    return np.stack([
        librosa.effects.time_stretch(audio[0], rate=rate),
        librosa.effects.time_stretch(audio[1], rate=rate),
    ])

def normalize(audio: np.ndarray, db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    return audio * (10 ** (db / 20)) / peak if peak > 0 else audio

def apply_hpf(audio: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    sos = scipy.signal.butter(order, max(cutoff_hz, 20) / (SR / 2), "high", output="sos")
    return scipy.signal.sosfilt(sos, audio)

def apply_lpf(audio: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    sos = scipy.signal.butter(order, min(cutoff_hz, SR / 2 - 100) / (SR / 2), "low", output="sos")
    return scipy.signal.sosfilt(sos, audio)

def crossfade(a: np.ndarray, b: np.ndarray, fade_s: float = 3.0) -> np.ndarray:
    fn = min(int(fade_s * SR), a.shape[1], b.shape[1])
    overlap = a[:, -fn:] * np.linspace(1, 0, fn) + b[:, :fn] * np.linspace(0, 1, fn)
    return np.concatenate([a[:, :-fn], overlap, b[:, fn:]], axis=1)


# ── Effects ───────────────────────────────────────────────────────────────────

def hpf_sweep(audio: np.ndarray, low_hz: float = 80, high_hz: float = 14000,
              n_steps: int = 80) -> np.ndarray:
    """Bass then mids then highs all disappear — total tension."""
    n = audio.shape[1]
    cutoffs = np.geomspace(max(low_hz, 20), min(high_hz, SR / 2 - 100), n_steps)
    chunk = max(1, n // n_steps)
    out = np.zeros_like(audio)
    for i, hz in enumerate(cutoffs):
        s, e = i * chunk, min((i + 1) * chunk, n)
        sos = scipy.signal.butter(4, hz / (SR / 2), "high", output="sos")
        out[:, s:e] = scipy.signal.sosfilt(sos, audio[:, s:e])
    return out


def make_riser(dur_s: float, gain: float = 0.50) -> np.ndarray:
    """White noise climbing from 200Hz to 16kHz — impossible to miss."""
    n = int(dur_s * SR)
    noise = np.random.randn(2, n).astype(np.float32)
    cutoffs = np.geomspace(200, min(16000, SR / 2 - 100), 40)
    chunk = max(1, n // 40)
    out = np.zeros_like(noise)
    for i, hz in enumerate(cutoffs):
        s, e = i * chunk, min((i + 1) * chunk, n)
        sos = scipy.signal.butter(3, hz / (SR / 2), "high", output="sos")
        out[:, s:e] = scipy.signal.sosfilt(sos, noise[:, s:e])
    t = np.linspace(0, 1, n)
    return out * (t ** 1.5) * gain   # convex: quiet → LOUD at the end


def make_crash(dur_s: float = 1.5, gain: float = 0.90) -> np.ndarray:
    n = int(dur_s * SR)
    noise = np.random.randn(2, n).astype(np.float32)
    sos = scipy.signal.butter(4, 3000 / (SR / 2), "high", output="sos")
    noise = scipy.signal.sosfilt(sos, noise)
    return noise * np.exp(-np.linspace(0, 6, n)) * gain


def reverb_throw(audio: np.ndarray, wet: float = 0.70) -> np.ndarray:
    """Short reverb throw (last note trails into space)."""
    out = audio.copy().astype(np.float32)
    n_echo = 10
    for i in range(1, n_echo + 1):
        d = int(i * 0.5 / n_echo * SR)   # ~50ms spacing
        g = (0.65 ** i) * wet
        if d < audio.shape[1]:
            out[:, d:] += audio[:, :-d] * g
    peak = np.max(np.abs(out))
    return out * (0.95 / peak) if peak > 0.95 else out


def make_downlifter(dur_s: float, gain: float = 0.60) -> np.ndarray:
    """Bass swooshes from 800Hz down to 30Hz."""
    n = int(dur_s * SR)
    noise = np.random.randn(2, n).astype(np.float32)
    cutoffs = np.geomspace(800, 30, 30)
    chunk = max(1, n // 30)
    out = np.zeros_like(noise)
    for i, hz in enumerate(cutoffs):
        s, e = i * chunk, min((i + 1) * chunk, n)
        sos = scipy.signal.butter(4, max(hz, 20) / (SR / 2), "low", output="sos")
        out[:, s:e] = scipy.signal.sosfilt(sos, noise[:, s:e])
    t = np.linspace(1, 0, n)
    return out * (t ** 0.6) * gain


def apply_flanger(audio: np.ndarray, rate_hz: float = 1.5, max_ms: float = 10.0,
                  wet: float = 0.75) -> np.ndarray:
    n = audio.shape[1]
    t = np.arange(n) / SR
    max_d = int(max_ms * SR / 1000)
    lfo = ((np.sin(2 * np.pi * rate_hz * t) + 1) / 2 * max_d).astype(int)
    out = audio.copy().astype(np.float32)
    chunk = SR // 20
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = int(lfo[(s + e) // 2])
        if d > 0 and s >= d:
            out[:, s:e] += audio[:, s - d:e - d] * wet
    return out / (1 + wet)


def stamp_crash(audio: np.ndarray, gain: float = 0.90, dur_s: float = 1.5) -> np.ndarray:
    cr = make_crash(dur_s, gain)
    out = audio.copy()
    n = min(cr.shape[1], out.shape[1])
    out[:, :n] += cr[:, :n]
    return out


# ── Core drop anatomy ─────────────────────────────────────────────────────────

BEAT = 60.0 / TARGET_BPM   # ~0.522s

def build_section(body: np.ndarray, build_s: float = 8.0,
                  riser_gain: float = 0.50) -> np.ndarray:
    """
    Takes a clean audio section. Applies to its last build_s seconds:
      - HPF sweep (bass → mids → highs disappear)
      - Riser on top
      - 1-bar stutter appended at the very end (rhythmic anticipation)
    Returns body with the build baked in + stutter tail.
    """
    out = body.copy()
    build_n = min(int(build_s * SR), out.shape[1])

    # HPF sweep on the tail
    out[:, -build_n:] = hpf_sweep(out[:, -build_n:], 80, 14000, n_steps=80)

    # Riser layered on the HPF tail
    rise = make_riser(build_s, gain=riser_gain)
    out[:, -build_n:] += rise[:, :build_n]

    # 1-bar stutter of the last beat (everything is near-silence at this point)
    beat_n = int(BEAT * SR)
    last_beat = out[:, -beat_n:].copy()
    stut = np.concatenate([last_beat * (0.78 ** i) for i in range(4)], axis=1)

    return np.concatenate([out, stut], axis=1)


def drop_silence() -> np.ndarray:
    """150ms breath before the crash lands."""
    return np.zeros((2, int(0.15 * SR)), dtype=np.float32)


# ── Mashup build ──────────────────────────────────────────────────────────────

def build_mashup() -> np.ndarray:

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading clips...")
    bdt = load_clip("audio/badtameez_dil.mp3", 88.0,  36.0)
    uf  = load_clip("audio/uptown_funk.mp3",   33.0,  38.0)
    nn  = load_clip("audio/naatu_naatu.mp3",   35.0,  42.0)
    lev = load_clip("audio/levitating.mp3",    64.0,  45.0)

    # ── BPM-match all songs to 115 ────────────────────────────────────────────
    print("BPM-matching to 115 BPM...")
    bdt = bpm_stretch(bdt, 108.0, TARGET_BPM)
    uf  = bpm_stretch(uf,  117.5, TARGET_BPM)
    nn  = bpm_stretch(nn,  121.0, TARGET_BPM)
    lev = bpm_stretch(lev, 103.0, TARGET_BPM)

    # ── SECTION 1: Badtameez Dil ──────────────────────────────────────────────
    # Clean chorus → 8s build → stutter → silence → DROP into UF
    print("\n[1/4] Badtameez Dil — clean chorus + 8s build")
    bdt_body = bdt[:, :int(24 * SR)].copy()
    # Tiny fade-in so it doesn't click
    fi = int(0.08 * SR)
    bdt_body[:, :fi] *= np.linspace(0, 1, fi)
    # Build: 8s HPF sweep + riser + stutter tail
    bdt_sec = build_section(bdt_body, build_s=8.0, riser_gain=0.50)

    # ── SECTION 2: Uptown Funk — THE FIRST DROP ───────────────────────────────
    # Crash lands at bar 1 → clean groove → 6s build → stutter → silence → DROP into NN
    print("[2/4] Uptown Funk — DROP + clean groove + 6s build")
    uf_body = uf[:, :int(30 * SR)].copy()
    # Bass swap: remove sub-200Hz from first bar of UF (comes in clean on beat 1)
    # then bring bass back in over 2 bars
    intro_n = int(2 * 4 * BEAT * SR)   # 2 bars
    uf_body[:, :intro_n] = (
        apply_hpf(uf_body[:, :intro_n], 200) * np.linspace(0.4, 1.0, intro_n) +
        apply_lpf(uf_body[:, :intro_n], 200) * np.linspace(0.0, 1.0, intro_n)
    )
    # Crash at the drop
    uf_body = stamp_crash(uf_body, gain=0.90, dur_s=1.5)
    # Reverb throw on last 2s of clean section (before build kicks in)
    rv_n = int(2.0 * SR)
    clean_end = int(24 * SR) - rv_n
    uf_body[:, clean_end:int(24 * SR)] = reverb_throw(
        uf_body[:, clean_end:int(24 * SR)], wet=0.65
    )
    # Build: 6s
    uf_sec = build_section(uf_body, build_s=6.0, riser_gain=0.52)

    # ── SECTION 3: Naatu Naatu — THE BIG DROP ────────────────────────────────
    # Crash + biggest energy → flanger surprise at bar 12 → 8s build → silence → Lev
    print("[3/4] Naatu Naatu — BIG DROP + flanger surprise + 8s build")
    nn_body = nn[:, :int(34 * SR)].copy()
    # Crash on drop
    nn_body = stamp_crash(nn_body, gain=0.82, dur_s=1.0)
    # Flanger on bars 10-12 as a "surprise texture" (mid-section, not at the end)
    fl_start = int(20 * SR)
    fl_end   = int(26 * SR)
    if fl_end <= nn_body.shape[1]:
        nn_body[:, fl_start:fl_end] = apply_flanger(
            nn_body[:, fl_start:fl_end], rate_hz=1.5, max_ms=10, wet=0.75
        )
    # Downlifter on last 8s (bass swoosh signals the crowd something's shifting)
    dl_n = int(8.0 * SR)
    dl = make_downlifter(8.0, gain=0.60)
    nn_body[:, -dl_n:] += dl[:, :dl_n]
    # Build: 8s
    nn_sec = build_section(nn_body, build_s=8.0, riser_gain=0.50)

    # ── SECTION 4: Levitating — THE COOL-DOWN ────────────────────────────────
    # No crash. 2s fade-in (breath). Long groove. Slow fade out.
    print("[4/4] Levitating — breath transition, groove, fade out")
    lev_body = lev[:, :int(38 * SR)].copy()
    # 2s fade-in (not a hard hit — this is the crowd exhaling)
    fi2 = int(2.0 * SR)
    lev_body[:, :fi2] *= np.linspace(0, 1, fi2)
    # 14s fade out
    fo_n = int(14.0 * SR)
    lev_body[:, -fo_n:] *= np.linspace(1, 0, fo_n)
    lev_sec = lev_body

    # ── Assemble ──────────────────────────────────────────────────────────────
    # BDT (build + stutter) → 150ms silence → UF crash lands → UF (build + stutter)
    # → 150ms silence → NN crash lands → NN (build + stutter) → 3s crossfade → Lev
    print("\nAssembling...")

    gap = drop_silence()

    mix = np.concatenate([bdt_sec, gap, uf_sec, gap, nn_sec, gap], axis=1)

    # Lev: 3s smooth crossfade (the "exhale" — not a drop)
    mix = crossfade(mix, lev_sec, fade_s=3.0)

    return normalize(mix, db=-3.0)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir("/Users/muskaansinha/Desktop/mashup-debug/mashup")

    print("=" * 60)
    print("  Everyone a DJ — 4-Song Pro Mashup v3")
    print("  Badtameez Dil → Uptown Funk → Naatu Naatu → Levitating")
    print("=" * 60)

    mix = build_mashup()

    dur = mix.shape[1] / SR
    mins, secs = divmod(int(dur), 60)
    print(f"\nTotal duration: {mins}:{secs:02d}")
    print(f"Peak level: {20 * np.log10(np.max(np.abs(mix))):.1f} dBFS")

    wav_path = "audio/everyone_a_dj.wav"
    mp3_path = "audio/everyone_a_dj.mp3"

    print(f"\nExporting → {mp3_path}")
    sf.write(wav_path, mix.T, SR, subtype="PCM_16")
    AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate="192k")
    os.remove(wav_path)

    print("Done!")

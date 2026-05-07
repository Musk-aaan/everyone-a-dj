"""DJ-style crossfade mix: Badtameez Dil x Uptown Funk.

Structure:
  0-25s    Badtameez Dil chorus (~2:30 in source), BPM-stretched to 117.5
  22-26s   4s crossfade out of BDT, into Uptown Funk
  26-60s   Uptown Funk chorus (from 0:33 in source), clean

No simultaneous full-song layering. BPM-match via pitch-preserving time stretch.
"""

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import os

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
BDT_PATH = os.path.join(AUDIO_DIR, "badtameez_dil.mp3")
UF_PATH  = os.path.join(AUDIO_DIR, "uptown_funk.mp3")
OUT_PATH = os.path.join(AUDIO_DIR, "dance_mix_v2.mp3")

BDT_BPM = 107.7
UF_BPM  = 117.5
STRETCH_RATE = UF_BPM / BDT_BPM   # ~1.091 — speeds up BDT to match UF tempo

# --- Source timestamps ---
# Badtameez Dil: chorus at ~2:28 (148s), take 30s of it
BDT_START_SRC = 148.0
BDT_TAKE_SRC  = 30.0

# Uptown Funk: "I'm too hot" chorus at 0:33
UF_START_SRC = 33.0
UF_TAKE_SRC  = 40.0

# --- Mix layout (seconds in output) ---
BDT_TIMELINE = 0.0       # BDT starts here
XFADE_START  = 22.0      # crossfade starts (BDT starts fading out)
XFADE_DUR    = 4.0       # crossfade length
UF_TIMELINE  = XFADE_START  # UF enters at the start of the crossfade

print("Loading audio files...")
bdt_y, bdt_sr = librosa.load(BDT_PATH, sr=None, mono=False)
uf_y,  uf_sr  = librosa.load(UF_PATH,  sr=None, mono=False)

# Work in stereo (shape: [2, samples]) or mono
def ensure_stereo(y):
    if y.ndim == 1:
        return np.stack([y, y])
    return y

bdt_y = ensure_stereo(bdt_y)
uf_y  = ensure_stereo(uf_y)

print(f"BDT: {bdt_y.shape[1]/bdt_sr:.1f}s @ {bdt_sr}Hz")
print(f"UF:  {uf_y.shape[1]/uf_sr:.1f}s  @ {uf_sr}Hz")

# --- Resample UF to BDT sample rate if they differ ---
if uf_sr != bdt_sr:
    print(f"Resampling UF from {uf_sr} -> {bdt_sr}...")
    uf_y = np.stack([
        librosa.resample(uf_y[0], orig_sr=uf_sr, target_sr=bdt_sr),
        librosa.resample(uf_y[1], orig_sr=uf_sr, target_sr=bdt_sr),
    ])
    uf_sr = bdt_sr

SR = bdt_sr

# --- Slice source sections ---
def slice_audio(y, sr, start_s, dur_s):
    s = int(start_s * sr)
    e = int((start_s + dur_s) * sr)
    e = min(e, y.shape[1])
    return y[:, s:e]

print(f"Slicing BDT from {BDT_START_SRC}s for {BDT_TAKE_SRC}s ...")
bdt_clip = slice_audio(bdt_y, SR, BDT_START_SRC, BDT_TAKE_SRC)

print(f"Slicing UF  from {UF_START_SRC}s for {UF_TAKE_SRC}s ...")
uf_clip  = slice_audio(uf_y,  SR, UF_START_SRC,  UF_TAKE_SRC)

# --- BPM-stretch BDT (pitch-preserving) ---
print(f"Time-stretching BDT by {STRETCH_RATE:.4f}x ({BDT_BPM} -> {UF_BPM} BPM) ...")
bdt_stretched = np.stack([
    librosa.effects.time_stretch(bdt_clip[0], rate=STRETCH_RATE),
    librosa.effects.time_stretch(bdt_clip[1], rate=STRETCH_RATE),
])
print(f"BDT stretched: {bdt_stretched.shape[1]/SR:.1f}s")

# --- Build output timeline ---
total_dur_s = XFADE_START + XFADE_DUR + (UF_TAKE_SRC - XFADE_DUR) + 2.0  # +2 tail
total_samples = int(total_dur_s * SR)
out = np.zeros((2, total_samples), dtype=np.float32)

def place(buf, timeline_s, fade_in_s=0.2, fade_out_s=0.2):
    """Overlay buf into out at timeline_s with fade-in/out."""
    n = buf.shape[1]
    start = int(timeline_s * SR)
    end   = min(start + n, total_samples)
    chunk = buf[:, :end - start].copy()

    fi = min(int(fade_in_s * SR), n)
    fo = min(int(fade_out_s * SR), n)
    ramp_in  = np.linspace(0, 1, fi)
    ramp_out = np.linspace(1, 0, fo)
    chunk[:, :fi]  *= ramp_in
    chunk[:, -fo:] *= ramp_out

    out[:, start:end] += chunk

# Crossfade envelopes applied by place() aren't fine-grained enough for the
# XF seam — do it manually: BDT fades out over XFADE_DUR, UF fades in.
bdt_n = bdt_stretched.shape[1]
uf_n  = uf_clip.shape[1]

# Place BDT: starts at 0, 200ms fade-in, fades out over XFADE_DUR at XFADE_START
bdt_out = bdt_stretched.copy()
# fade-in 200ms
fi_n = int(0.2 * SR)
bdt_out[:, :fi_n] *= np.linspace(0, 1, fi_n)
# fade-out starting at XFADE_START
xf_start_samp = int(XFADE_START * SR)
xf_dur_samp   = int(XFADE_DUR * SR)
if xf_start_samp < bdt_n:
    fade_end = min(xf_start_samp + xf_dur_samp, bdt_n)
    ramp = np.linspace(1, 0, fade_end - xf_start_samp)
    bdt_out[:, xf_start_samp:fade_end] *= ramp
    bdt_out[:, fade_end:] = 0.0  # silence after crossfade

bdt_place_end = min(bdt_n, total_samples)
out[:, :bdt_place_end] += bdt_out[:, :bdt_place_end]

# Place UF: enters at XFADE_START, fades in over XFADE_DUR, ends with 500ms fade-out
uf_out = uf_clip.copy()
# fade-in over XFADE_DUR
uf_out[:, :xf_dur_samp] *= np.linspace(0, 1, min(xf_dur_samp, uf_n))
# fade-out 500ms at end
fo_n = min(int(0.5 * SR), uf_n)
uf_out[:, -fo_n:] *= np.linspace(1, 0, fo_n)

uf_start = int(UF_TIMELINE * SR)
uf_end   = min(uf_start + uf_n, total_samples)
out[:, uf_start:uf_end] += uf_out[:, :uf_end - uf_start]

# --- Normalise to -3dBFS ---
peak = np.max(np.abs(out))
if peak > 0:
    out = out * (10 ** (-3 / 20)) / peak
    print(f"Normalised (peak was {20*np.log10(peak):.1f}dBFS)")

# --- Export ---
print(f"Exporting to {OUT_PATH} ...")
# soundfile writes [samples, channels]
sf.write(OUT_PATH.replace(".mp3", ".wav"), out.T, SR, subtype="PCM_16")

# Convert WAV -> MP3 via pydub
wav_path = OUT_PATH.replace(".mp3", ".wav")
seg = AudioSegment.from_wav(wav_path)
seg.export(OUT_PATH, format="mp3", bitrate="192k")
os.remove(wav_path)

dur = len(seg) / 1000
print(f"\nDone! dance_mix_v2.mp3 — {dur:.1f}s")
print(f"Path: {OUT_PATH}")

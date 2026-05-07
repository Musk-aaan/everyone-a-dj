"""Phase 4 — Stem Separation.

Splits any track into 4 stems via Meta's demucs:
  - vocals.wav
  - drums.wav
  - bass.wav
  - other.wav   (everything else: synths, guitars, pads)

This is the single biggest unlock between "amateur mashup" and "real mashup".
With stems we can do real DJ tricks:
  - Layer Song A's vocals over Song B's instrumental (drums+bass+other)
  - Bass swap: cut bass.wav of outgoing, fade in bass.wav of incoming
  - Acapella moment: drop everything except vocals at the climax
  - A/B drum-only break before the next song's drop

Output: cache/stems/{song_id}/{vocals,drums,bass,other}.wav

Cost: ~30s on M-series GPU, ~2 min on CPU. Cached forever — every song
processed once is reusable for every future mashup.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from . import cache, config

STEM_NAMES = ("vocals", "drums", "bass", "other")


def _best_device() -> str:
    """Prefer cuda > mps > cpu. mps gives 10-20x speedup on M-series Macs."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def has_stems(sid: str) -> bool:
    return cache.has_stems(sid)


def stem_paths(sid: str) -> dict[str, Path]:
    """Return {stem_name: path_to_wav} (paths may not exist yet)."""
    d = cache.stems_dir(sid)
    return {name: d / f"{name}.wav" for name in STEM_NAMES}


def separate(audio_path: Path, *, force: bool = False, model: str = "mdx_extra") -> dict[str, Path]:
    """Run demucs on `audio_path` and cache 4 stems. Returns the path map.

    `model`:
      - 'mdx_extra'     — default. Fastest on M-series, ~30s on MPS, good quality
      - 'htdemucs'      — slower (~2 min on MPS), arguably cleaner separation
      - 'htdemucs_ft'   — even slower, marginal quality bump

    Uses the demucs Python API (not the subprocess CLI) — much faster startup
    and cleaner error reporting.
    """
    sid = audio_path.stem
    out_paths = stem_paths(sid)

    if not force and all(p.exists() for p in out_paths.values()):
        return out_paths

    if not audio_path.exists():
        raise RuntimeError(f"audio file not found: {audio_path}")

    out_dir = cache.stems_dir(sid)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        from demucs.separate import load_track
        import torch
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError("Install: pip install demucs torch soundfile") from e

    device = _best_device()

    # Load + run model
    m = get_model(model)
    m.eval()
    wav = load_track(str(audio_path), m.audio_channels, m.samplerate)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / max(ref.std(), 1e-8)

    with torch.no_grad():
        sources = apply_model(
            m, wav[None].to(device), device=device,
            shifts=1, split=True, overlap=0.25, progress=False,
        )[0]
    sources = sources * ref.std() + ref.mean()
    sources = sources.cpu()

    # m.sources is e.g. ['drums', 'bass', 'other', 'vocals'] — order varies by model
    name_to_idx = {n: i for i, n in enumerate(m.sources)}
    for name in STEM_NAMES:
        if name not in name_to_idx:
            raise RuntimeError(f"model {model!r} has no '{name}' stem; has {m.sources}")
        wav_out = sources[name_to_idx[name]].numpy()  # [channels, samples]
        # soundfile expects [samples, channels]
        sf.write(str(out_dir / f"{name}.wav"), wav_out.T, m.samplerate, subtype="PCM_16")

    return out_paths


def separate_song_id(sid: str, *, force: bool = False) -> dict[str, Path]:
    """Convenience: separate the cached mp3 for a given song_id."""
    return separate(cache.audio_path(sid), force=force)


def stem_size_mb(sid: str) -> float:
    """Total disk size of the 4 stems in MB (0 if not separated yet)."""
    if not has_stems(sid):
        return 0.0
    return sum(p.stat().st_size for p in stem_paths(sid).values()) / 1024 / 1024

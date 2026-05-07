"""Audio renderer: turn a MashupRecipe into an actual audio file.

v0 strategy without stem separation:
- High-pass the "anchor_vocal" clip so song A's bass / drums get cut and
  mostly mids / vocals come through. This is the EQ trick DJs use for
  quick layered mashups when they don't have separated stems.
- Low-pass the "groove_bed" clip so it sits behind the vocal in the mix.
- 200ms fades on every section edge to prevent clicks.
- Overlay all sections onto a silent timeline at their `timeline_at`.

The output sounds rough on songs with vocal-heavy choruses on song B, but
the structure is right and the anchor hook is intelligible. Demucs-based
proper stem mixing is the next renderer iteration.
"""

from __future__ import annotations

from typing import Optional

from .planner import MashupRecipe


def render(
    recipe: MashupRecipe,
    *,
    song_a_audio_path: str,
    song_b_audio_path: str,
    output_path: str,
    fade_ms: int = 200,
) -> Optional[str]:
    """Render `recipe` to an audio file. Returns the output path on success,
    None if pydub/ffmpeg aren't installed."""
    try:
        from pydub import AudioSegment
        from pydub.effects import high_pass_filter, low_pass_filter
    except ImportError:
        return None

    a = AudioSegment.from_file(song_a_audio_path)
    b = AudioSegment.from_file(song_b_audio_path)

    total_ms = int(round(recipe.duration * 1000))
    timeline = AudioSegment.silent(duration=total_ms)

    for section in recipe.sections:
        src = a if section.source == "song_a" else b
        clip = src[int(section.start * 1000): int(section.end * 1000)]
        if len(clip) == 0:
            continue

        if section.role == "anchor_vocal":
            # Cut sub-bass / kick of source A so vocals + harmonics float on top.
            # 200Hz is closer to standard vocal-isolation HPF (was 600Hz, which
            # killed vocal fundamentals in the male/female mid range).
            clip = high_pass_filter(clip, 200) - 2
        elif section.role == "groove_bed":
            # Mild lowpass to cede the top end to the floated vocal; light duck.
            clip = low_pass_filter(clip, 8000) - 2

        clip = clip.fade_in(fade_ms).fade_out(fade_ms)
        timeline = timeline.overlay(clip, position=int(section.timeline_at * 1000))

    # Pull back 3 dB before export: accumulated overlay peaks can push the
    # mix close to 0 dBFS, causing audible clipping on cheap decoders.
    timeline = timeline - 3

    fmt = output_path.rsplit(".", 1)[-1].lower()
    if fmt not in {"mp3", "wav", "ogg", "flac", "m4a"}:
        fmt = "mp3"
    export_kwargs: dict = {"format": fmt}
    if fmt in {"mp3", "ogg", "m4a"}:
        export_kwargs["bitrate"] = "192k"
    timeline.export(output_path, **export_kwargs)
    return output_path

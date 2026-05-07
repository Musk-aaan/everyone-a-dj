"""Self-contained synthetic mashup demo.

Generates two distinct sine/triangle/square 'songs', constructs a
hardcoded recipe over them, and renders the result. Sounds nothing like
real music, but exercises the full render chain so you can verify
install + ffmpeg work without needing source audio files.

Run via the CLI: `mashup demo --out demo.mp3`.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from .planner import MashupRecipe, Section, SongRef, Transition


def _chord(freqs: list[float], duration_ms: int, gain_db: float = -16):
    """Stack triangle waves into a chord.

    Triangle waves have natural odd harmonics (f, 3f, 5f…) which give enough
    richness that an AI/human listener can recognise chord quality (major vs.
    minor) without sounding as harsh as square waves.
    """
    from pydub import AudioSegment
    from pydub.generators import Triangle
    seg = AudioSegment.silent(duration=duration_ms)
    for f in freqs:
        seg = seg.overlay(Triangle(f).to_audio_segment(duration=duration_ms) + gain_db)
    # 200 ms swell-in instead of the original 40 ms stab — the sharper attack
    # made Triangle-wave chord bars sound like hi-hat hits to Gemini.
    return seg.fade_in(200).fade_out(80)


def _melody(notes: list[tuple[float, int]], gain_db: float = -6):
    """Concatenate notes (freq_hz, duration_ms) into a phrase."""
    from pydub import AudioSegment
    from pydub.generators import Sine
    seg = AudioSegment.silent(duration=0)
    for f, d in notes:
        # Sine keeps the hook clean and bell-like in the upper register.
        # gain_db is raised vs. the chord bed so the hook clears the mix.
        note = Sine(f).to_audio_segment(duration=d).fade_in(15).fade_out(40) + gain_db
        seg = seg + note
    return seg


def _synth_song_a(duration_s: int = 200):
    """Chord-progression bed (C-G-Am-F) with a 5-note melodic hook at chorus times."""
    from pydub import AudioSegment
    progression = [
        [261.63, 329.63, 392.00],  # C major (C4-E4-G4)
        [392.00, 493.88, 587.33],  # G major
        [220.00, 261.63, 329.63],  # A minor
        [174.61, 220.00, 261.63],  # F major
    ]
    bar_ms = 2000
    bed = AudioSegment.silent(duration=0)
    while len(bed) < duration_s * 1000:
        for triad in progression:
            bed = bed + _chord(triad, bar_ms, -16)
    bed = bed[: duration_s * 1000]

    # Hook in upper register so the high-pass keeps it intact.
    # gain_db=+2: the chord bed (at -16 dBFS) has a combined level of ~-12 dBFS,
    # which was burying the hook at -4 dBFS (only 2 dB of headroom).  Raising
    # to +2 dBFS gives ~8 dB of separation so the hook is clearly audible over
    # the bed even after the renderer's HPF + gain reductions.
    hook = _melody(
        [
            (1046.50, 350),  # C6
            (1174.66, 350),  # D6
            (1318.51, 500),  # E6
            (1174.66, 350),  # D6
            (1046.50, 1200),  # C6 held
        ],
        gain_db=+2,
    )
    # Hooks placed on bar-16/32/68/88 boundaries (all C major in the progression).
    # Original t=30 s was bar 15 = F major; E natural over F major is the
    # major-7th — a harsh clash with the hook's E6 note.  Shifting by 2 s to
    # bar 16 (C major) eliminates the dissonance.  Same logic applies to the
    # other three hook placements.
    for t in (32, 64, 136, 176):
        bed = bed.overlay(hook, position=t * 1000)
    return bed


def _synth_song_b(duration_s: int = 200):
    """Bassline (C-C-G-G) + Am chord pad. No 60Hz kick — that was the problem."""
    from pydub import AudioSegment
    from pydub.generators import Triangle
    bar_ms = 1500
    bass_notes = [130.81, 130.81, 196.00, 196.00]  # C3, C3, G3, G3
    bassline = AudioSegment.silent(duration=0)
    while len(bassline) < duration_s * 1000:
        for f in bass_notes:
            # Triangle wave for bass: odd harmonics (3f, 5f …) make the pitch
            # identifiable — a pure sine at 130 Hz sounds like an unpitched
            # "thump" with no recognisable note.
            # -12 dBFS keeps the bass clearly dominant; longer fades (80/150 ms)
            # reduce inter-note clicks compared to the original 60/120 ms.
            note = (Triangle(f).to_audio_segment(duration=bar_ms) - 12).fade_in(80).fade_out(150)
            bassline = bassline + note
    bassline = bassline[: duration_s * 1000]

    # Barred chord pad: same bar length as the bassline so the rhythm is tight.
    # Gain at -28 dB per voice → three voices sum to ~-18.5 dBFS combined,
    # which sits about 6 dB below the -12 dBFS bass — pad is audible but not
    # masking.  Continuous 200-s drones at close frequencies (220/261/329 Hz)
    # create beating interference that sounds "buzzy"; barred chords reset
    # phase relationships every 1.5 s and are much cleaner.
    am_chord = [220.00, 261.63, 329.63]  # A minor
    pad = AudioSegment.silent(duration=0)
    while len(pad) < duration_s * 1000:
        pad = pad + _chord(am_chord, bar_ms, -28)
    pad = pad[: duration_s * 1000]

    return bassline.overlay(pad)


def _demo_recipe() -> MashupRecipe:
    # Groove bed extended to source 120s (was 116s) so its tail reaches
    # timeline 30s, overlapping the outro that starts at 27.5s.  This creates
    # a natural 2.5 s crossfade instead of the previous 4 s silent gap.
    # Outro starts at source 127.5s so the hook (placed at 135s in song_a)
    # lands at timeline 27.5 + (135-127.5) = 35s, matching the intended mark.
    return MashupRecipe(
        song_a=SongRef("Synth A", "Lead Tones", 200.0),
        song_b=SongRef("Synth B", "Kick Pad", 200.0),
        duration=60.0,
        anchor_lyric="(synthetic hook)",
        sections=[
            Section("song_b", "intro", 90.0, 96.0, 0.0, "Synth B intro"),
            Section("song_b", "groove_bed", 96.0, 120.0, 6.0, "Synth B groove pocket"),
            # Clips start at bar-16/32 (C major) so the chord bed in the
            # anchor_vocal clip is consonant with the C-D-E-D-C hook.
            Section("song_a", "anchor_vocal", 32.0, 35.5, 8.0, "Synth A hook"),
            Section("song_a", "anchor_vocal", 64.0, 67.5, 16.0, "Synth A hook (repeat)"),
            # Outro at 128.5 s: hook is at 136 s → 27.5 + (136-128.5) = 35 s ✓
            Section("song_a", "outro", 128.5, 160.0, 27.5, "Synth A full mix outro"),
        ],
        transitions=[
            Transition(5.0, "filter_sweep", 1.0, "into bed"),
            Transition(26.0, "crossfade", 4.0, "to outro"),
        ],
        notes=["Synthetic demo: triangle-wave 'songs', not real music."],
    )


def render_demo(output_path: str) -> Optional[str]:
    """Render the synthetic demo. Returns path or None if pydub unavailable."""
    try:
        from .renderer import render
        a = _synth_song_a()
        b = _synth_song_b()
    except ImportError:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        a_path = os.path.join(tmp, "synth_a.wav")
        b_path = os.path.join(tmp, "synth_b.wav")
        a.export(a_path, format="wav")
        b.export(b_path, format="wav")
        return render(
            _demo_recipe(),
            song_a_audio_path=a_path,
            song_b_audio_path=b_path,
            output_path=output_path,
        )

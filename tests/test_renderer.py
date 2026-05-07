import os
import tempfile

import pytest

from mashup.planner import MashupRecipe, Section, SongRef, Transition
from mashup.renderer import render

pydub = pytest.importorskip("pydub")
from pydub.generators import Sine  # noqa: E402


def _make_tone_file(path: str, freq: int, duration_ms: int) -> None:
    Sine(freq).to_audio_segment(duration=duration_ms).export(path, format="wav")


def _toy_recipe() -> MashupRecipe:
    return MashupRecipe(
        song_a=SongRef("ArtistA", "TitleA", 30.0),
        song_b=SongRef("ArtistB", "TitleB", 30.0),
        duration=20.0,
        anchor_lyric="hook",
        sections=[
            Section("song_b", "intro", 0.0, 4.0, 0.0, "B intro"),
            Section("song_b", "groove_bed", 4.0, 16.0, 4.0, "B bed"),
            Section("song_a", "anchor_vocal", 5.0, 13.0, 6.0, "A hook"),
            Section("song_a", "outro", 13.0, 20.0, 16.0, "A outro"),
        ],
        transitions=[
            Transition(3.5, "filter_sweep", 0.5, "into bed"),
            Transition(15.5, "crossfade", 4.0, "to outro"),
        ],
        notes=["test"],
    )


def test_render_produces_audio_of_expected_duration():
    with tempfile.TemporaryDirectory() as tmp:
        a_path = os.path.join(tmp, "a.wav")
        b_path = os.path.join(tmp, "b.wav")
        out_path = os.path.join(tmp, "mashup.wav")
        _make_tone_file(a_path, freq=440, duration_ms=30_000)  # song A: 440Hz
        _make_tone_file(b_path, freq=220, duration_ms=30_000)  # song B: 220Hz

        result = render(
            _toy_recipe(),
            song_a_audio_path=a_path,
            song_b_audio_path=b_path,
            output_path=out_path,
        )
        assert result == out_path
        assert os.path.getsize(out_path) > 0

        out = pydub.AudioSegment.from_file(out_path)
        assert 19_500 <= len(out) <= 20_500  # 20s ± fade slop

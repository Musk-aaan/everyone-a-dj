from mashup.moment_detector import DetectionResult
from mashup.planner import _find_groove_pocket, plan_mashup
from mashup.ranker import Moment, rank_moments
from mashup.signals.lrclib import LyricLine, SyncedLyrics


def _lyrics(pairs, *, artist="A", title="T", duration=240.0):
    return SyncedLyrics(
        artist=artist,
        title=title,
        duration=duration,
        lines=[LyricLine(time=t, text=tx) for t, tx in pairs],
    )


def _result(lyrics, *, top_k=3):
    moments = rank_moments(lyrics, top_k=top_k)
    return DetectionResult(
        artist=lyrics.artist,
        title=lyrics.title,
        duration=lyrics.duration,
        moments=moments,
        signals_used=["lyrics"],
        lyrics=lyrics,
    )


def test_groove_pocket_finds_longest_no_vocal_stretch():
    lyrics = _lyrics(
        [(10.0, "a"), (20.0, "b"), (30.0, "c"), (130.0, "d"), (140.0, "e")],
        duration=180.0,
    )
    pocket = _find_groove_pocket(lyrics, min_duration=8.0, target=200.0)
    assert pocket is not None
    start, end, _ = pocket
    # Largest gap: 30 -> 130 = 100s
    assert start == 30.0 and end == 130.0


def test_groove_pocket_returns_none_when_too_dense():
    lyrics = _lyrics([(t, "x") for t in range(0, 60, 2)], duration=60.0)
    assert _find_groove_pocket(lyrics, min_duration=8.0) is None


def test_plan_mashup_lays_out_timeline_in_order():
    a = _result(
        _lyrics(
            [(t, "heat waves") for t in (30.0, 60.0, 130.0, 180.0)] +
            [(t, f"verse {t}") for t in (10.0, 20.0, 100.0, 110.0)],
            artist="Glass Animals", title="Heat Waves", duration=238.0,
        )
    )
    b = _result(
        _lyrics(
            [(t, "apna bana le") for t in (40.0, 80.0, 160.0, 200.0)] +
            [(t, f"verse {t}") for t in (10.0, 20.0, 30.0)],
            artist="Arijit Singh", title="Apna Bana Le", duration=240.0,
        )
    )
    recipe = plan_mashup(a, b, target_duration=60.0)
    assert recipe is not None
    timeline_starts = [s.timeline_at for s in sorted(recipe.sections, key=lambda s: s.timeline_at)]
    assert timeline_starts == sorted(timeline_starts)
    # Anchor must be the top moment from song A
    assert "heat waves" in recipe.anchor_lyric.lower()
    # Timeline should have an intro, a bed, an anchor, and an outro
    roles = {s.role for s in recipe.sections}
    assert {"intro", "groove_bed", "anchor_vocal", "outro"} <= roles
    # Transitions happen at sensible (non-negative) times
    assert all(t.timeline_at >= 0 for t in recipe.transitions)


def test_plan_returns_none_without_song_a_moments():
    a_lyrics = _lyrics([(10.0, "unique"), (20.0, "alone"), (30.0, "nope")])
    b_lyrics = _lyrics([(10.0, "x"), (50.0, "x"), (90.0, "x")])
    a = _result(a_lyrics)
    b = _result(b_lyrics)
    assert plan_mashup(a, b) is None

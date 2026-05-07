from mashup.ranker import rank_moments
from mashup.signals.lrclib import LyricLine, SyncedLyrics


def _lyrics(pairs, duration=240.0):
    return SyncedLyrics(
        artist="Test",
        title="Song",
        duration=duration,
        lines=[LyricLine(time=t, text=tx) for t, tx in pairs],
    )


def test_ranks_repeated_chorus_above_unique_lines():
    lyrics = _lyrics(
        [
            (10.0, "first verse line"),
            (20.0, "second verse line"),
            (30.0, "Apna bana le piya"),
            (35.0, "Apna bana le piya"),
            (130.0, "Apna bana le piya"),
            (135.0, "Apna bana le piya"),
            (200.0, "Apna bana le piya"),
        ]
    )
    moments = rank_moments(lyrics, top_k=3)
    assert moments, "expected at least one moment"
    assert "apna bana le piya" in moments[0].lyric.lower()


def test_ignores_lines_that_never_repeat():
    lyrics = _lyrics([(10.0, "alpha"), (20.0, "beta"), (30.0, "gamma")])
    assert rank_moments(lyrics, top_k=3) == []


def test_prefers_chorus_in_peak_zone():
    # Same line appears early (low position score) and in the peak zone.
    # The peak-zone instance should win.
    lyrics = _lyrics(
        [
            (5.0, "we found love"),
            (180.0, "we found love"),
        ],
        duration=240.0,
    )
    moments = rank_moments(lyrics, top_k=1)
    assert len(moments) == 1
    assert moments[0].start == 180.0


def test_dedupe_keeps_only_best_instance_per_lyric():
    lyrics = _lyrics(
        [
            (10.0, "hook"),
            (50.0, "other"),
            (180.0, "hook"),
            (190.0, "other"),
        ]
    )
    moments = rank_moments(lyrics, top_k=10)
    seen = [m.lyric.lower() for m in moments]
    assert len(seen) == len(set(seen)), "each lyric should appear at most once"

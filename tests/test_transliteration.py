from mashup.signals.transliteration import normalize_lyric


def test_strips_punctuation_and_case():
    assert normalize_lyric("Heat Waves, baby!") == "heat waves baby"


def test_collapses_repeated_vowels():
    # Common in transliterated Bollywood lyrics: "piyaa" vs "piya"
    assert normalize_lyric("Apna bana le piyaa") == normalize_lyric("Apna bana le piya")
    assert normalize_lyric("Looove is in the air") == normalize_lyric("Love is in the air")


def test_strips_devanagari_combining_marks():
    # Same word with and without nukta combining mark should match.
    a = normalize_lyric("ज़िंदगी")  # zindagi with nukta
    b = normalize_lyric("जिंदगी")        # zindagi without nukta
    assert a == b


def test_collapses_whitespace():
    assert normalize_lyric("  hello   world\t\t") == "hello world"

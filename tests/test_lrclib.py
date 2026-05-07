from mashup.signals.lrclib import parse_lrc


def test_parse_basic_timestamps():
    src = "[00:12.40]Heartbeat heartbeat\n[00:14.20]I can hear it\n[01:30.00]Heartbeat heartbeat"
    lines = parse_lrc(src)
    assert len(lines) == 3
    assert abs(lines[0].time - 12.4) < 0.01
    assert lines[2].time == 90.0
    assert lines[0].text == "Heartbeat heartbeat"


def test_parse_skips_metadata_tags():
    src = "[ar:Some Artist]\n[ti:Some Title]\n[00:10.00]first line"
    lines = parse_lrc(src)
    assert len(lines) == 1
    assert lines[0].text == "first line"


def test_parse_handles_missing_fraction():
    src = "[00:05]bare seconds line"
    lines = parse_lrc(src)
    assert lines[0].time == 5.0


def test_parse_three_digit_fraction():
    # "[00:01.470]" = 1.470s
    src = "[00:01.470]ms precision"
    lines = parse_lrc(src)
    assert abs(lines[0].time - 1.470) < 0.001

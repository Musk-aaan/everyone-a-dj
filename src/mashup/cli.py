"""CLI: python -m mashup <subcommand> ...

New (v0.1) subcommands for the 8-phase pipeline:
    keys         Show which API keys are configured
    discover     Phase 1: search YouTube + Spotify for vibe/query
    download     Phase 1: fetch a YouTube ID's audio to audio/{song_id}.mp3

Legacy subcommands (v0.0 iconic-moment prototype) remain:
    find-moments, plan, demo, diagnose, make
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

from . import analyze as analyze_mod
from . import config, discover, lyrics, live, orchestrator, plan
from . import stems as stems_mod
from .demo import render_demo
from .diagnostics import listen as diagnose_audio
from .moment_detector import DetectionResult, find_moments
from .planner import MashupRecipe, plan_mashup
from .renderer import render as render_recipe


def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _print_human(result: DetectionResult) -> None:
    print(f"{result.title} - {result.artist}")
    print(
        f"Duration: {_fmt_time(result.duration)}  "
        f"Signals: {', '.join(result.signals_used)}"
    )
    print("=" * 72)
    if not result.moments:
        print("No iconic moments detected (no repeated lines in synced lyrics).")
        return
    for i, m in enumerate(result.moments, 1):
        print()
        print(f"Moment {i}  score={m.score:.2f}")
        print(
            f"  {_fmt_time(m.start)} - {_fmt_time(m.end)}  "
            f"({m.end - m.start:.1f}s)"
        )
        print(f'  Lyric: "{m.lyric}"')
        parts = [f"{k}={v:.2f}" for k, v in m.breakdown.items()]
        print(f"  Breakdown: {' | '.join(parts)}")


def _emit_json(result: DetectionResult) -> None:
    payload = {
        "artist": result.artist,
        "title": result.title,
        "duration": result.duration,
        "signals_used": result.signals_used,
        "moments": [
            {
                "start": m.start,
                "end": m.end,
                "lyric": m.lyric,
                "score": m.score,
                "breakdown": m.breakdown,
            }
            for m in result.moments
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _print_recipe(recipe: MashupRecipe) -> None:
    print(
        f"Mashup: {recipe.song_a.title} ({recipe.song_a.artist}) "
        f"x {recipe.song_b.title} ({recipe.song_b.artist})"
    )
    print(f"Total duration: ~{recipe.duration:.0f}s")
    print(f'Anchor hook: "{recipe.anchor_lyric}"')
    print("=" * 72)
    print("\nTimeline:")
    for s in sorted(recipe.sections, key=lambda x: x.timeline_at):
        span = f"{_fmt_time(s.timeline_at)} - {_fmt_time(s.timeline_at + (s.end - s.start))}"
        src_span = f"[{s.source} {_fmt_time(s.start)}-{_fmt_time(s.end)}]"
        print(f"  {span}  {s.role:<14} {src_span}  {s.description}")
    print("\nTransitions:")
    for t in sorted(recipe.transitions, key=lambda x: x.timeline_at):
        print(
            f"  {_fmt_time(t.timeline_at)}  {t.kind:<14} "
            f"{t.duration:.1f}s  {t.description}"
        )
    print("\nNotes:")
    for n in recipe.notes:
        print(f"  - {n}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mashup", description="Iconic-moment detection and mashup planning."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    fm = sub.add_parser(
        "find-moments", help="Detect the top iconic moments in a song."
    )
    fm.add_argument("--artist", required=True)
    fm.add_argument("--title", required=True)
    fm.add_argument(
        "--audio",
        help="Optional audio file path for energy analysis (requires librosa).",
    )
    fm.add_argument(
        "--youtube",
        help="Optional YouTube URL to pull most-replayed heatmap (requires yt-dlp).",
    )
    fm.add_argument("--top", type=int, default=3)
    fm.add_argument("--json", action="store_true")

    pl = sub.add_parser("plan", help="Plan a mashup from two songs.")
    pl.add_argument("--a-artist", required=True)
    pl.add_argument("--a-title", required=True)
    pl.add_argument("--a-audio")
    pl.add_argument("--a-youtube")
    pl.add_argument("--b-artist", required=True)
    pl.add_argument("--b-title", required=True)
    pl.add_argument("--b-audio")
    pl.add_argument("--b-youtube")
    pl.add_argument("--duration", type=float, default=60.0)
    pl.add_argument("--json", action="store_true")

    dm = sub.add_parser(
        "demo",
        help="Render a self-contained synthetic mashup (no source audio needed).",
    )
    dm.add_argument("--out", required=True, help="Output path (.mp3 / .wav)")

    dg = sub.add_parser(
        "diagnose",
        help="Send a rendered audio file to OpenRouter (Gemini Flash) and print what it hears.",
    )
    dg.add_argument("--audio", required=True, help="Path to audio file to analyze.")
    dg.add_argument("--model", default="google/gemini-2.5-flash")

    # ── New v0.1 commands ────────────────────────────────────────────────
    sub.add_parser("keys", help="Show which API keys are configured.")

    dc = sub.add_parser("discover", help="Phase 1: find candidate songs.")
    dc_src = dc.add_mutually_exclusive_group(required=True)
    dc_src.add_argument("--query",    help="Free-text search (no API key needed).")
    dc_src.add_argument("--trending", metavar="REGION",
                        help="YouTube top-music chart for region (US, IN, GB, ...). Needs YOUTUBE_API_KEY.")
    dc_src.add_argument("--spotify-playlist", metavar="PLAYLIST_ID",
                        help="Spotify playlist (default: Global Top 50). Needs SPOTIFY_CLIENT_ID + SECRET.")
    dc.add_argument("-n", type=int, default=10, help="How many results.")
    dc.add_argument("--save", action="store_true", help="Persist to cache/candidates_latest.json")
    dc.add_argument("--json", action="store_true")

    dl = sub.add_parser("download", help="Phase 1: download a YouTube audio.")
    dl.add_argument("--youtube-id", required=True)
    dl.add_argument("--title", default="(unknown)")

    lv = sub.add_parser("live",
                        help="Phase 8: live DJ session — plays through speakers, accepts adds via stdin.")
    lv.add_argument("--vibe", required=True)
    lv.add_argument("--chunk-size", type=int, default=4,
                    help="Songs per rendered chunk (default 4).")
    lv.add_argument("--candidates", type=int, default=12)
    lv.add_argument("--bpm", type=float, help="Target BPM for the whole session.")
    lv.add_argument("--stems", action="store_true")

    mk = sub.add_parser("make-mashup",
                        help="Phase 7: end-to-end vibe → mashup MP3.")
    mk.add_argument("--vibe", required=True, help="Free-text vibe.")
    mk.add_argument("--query", help="Search query for candidates (default = vibe).")
    mk.add_argument("-n", type=int, default=4, help="Number of songs.")
    mk.add_argument("--candidates", type=int, default=18)
    mk.add_argument("--bpm", type=float, help="Target BPM (default = first song's).")
    mk.add_argument("--stems", action="store_true", help="Use stem separation.")
    mk.add_argument("--critique", action="store_true",
                    help="Run Gemini critic on the rendered mix.")

    pk = sub.add_parser("pick-songs", help="Phase 5: LLM picks N songs for a vibe.")
    pk.add_argument("--vibe", required=True, help="Free-text vibe: 'Indian wedding 2026'.")
    pk.add_argument("--query", help="Search query for candidates (default = vibe).")
    pk.add_argument("-n", type=int, default=4)
    pk.add_argument("--candidates", type=int, default=20, help="How many candidates to show LLM.")

    pt = sub.add_parser("plan-transition", help="Phase 5: LLM designs a transition.")
    pt.add_argument("--from-title", required=True)
    pt.add_argument("--from-bpm", type=float, required=True)
    pt.add_argument("--from-key", default="C")
    pt.add_argument("--from-mode", default="major", choices=["major", "minor"])
    pt.add_argument("--from-lyric", default="")
    pt.add_argument("--to-title", required=True)
    pt.add_argument("--to-bpm", type=float, required=True)
    pt.add_argument("--to-key", default="C")
    pt.add_argument("--to-mode", default="major", choices=["major", "minor"])
    pt.add_argument("--to-lyric", default="")

    st = sub.add_parser("stems", help="Phase 4: split a song into 4 stems via demucs.")
    st.add_argument("--audio", required=True, help="Path to mp3 (absolute or in audio/).")
    st.add_argument("--model", default="htdemucs",
                    choices=["htdemucs", "htdemucs_ft", "mdx_extra"],
                    help="demucs model variant (default htdemucs).")
    st.add_argument("--force", action="store_true")

    an = sub.add_parser("analyze", help="Phase 3: extract BPM, key, sections, energy.")
    an.add_argument("--audio", required=True, help="Path to mp3 (absolute or in audio/).")
    an.add_argument("--force", action="store_true", help="Re-run even if cached.")

    ly = sub.add_parser("lyrics", help="Phase 2: fetch lyrics (Genius + Whisper fallback).")
    ly.add_argument("--youtube-id", required=True)
    ly.add_argument("--title", required=True)
    ly.add_argument("--artist", default="")
    ly.add_argument("--whisper", action="store_true",
                    help="Force Whisper transcription (skips Genius, gets timestamps).")
    ly.add_argument("--lines", type=int, default=20, help="Show first N lines.")

    # ── Legacy v0.0 commands ─────────────────────────────────────────────
    mk = sub.add_parser(
        "make",
        help="Plan AND render a mashup to an audio file (requires pydub + ffmpeg).",
    )
    mk.add_argument("--a-artist", required=True)
    mk.add_argument("--a-title", required=True)
    mk.add_argument("--a-audio", required=True, help="Path to Song A audio file.")
    mk.add_argument("--a-youtube")
    mk.add_argument("--b-artist", required=True)
    mk.add_argument("--b-title", required=True)
    mk.add_argument("--b-audio", required=True, help="Path to Song B audio file.")
    mk.add_argument("--b-youtube")
    mk.add_argument("--out", required=True, help="Output mashup path (.mp3 / .wav).")
    mk.add_argument("--duration", type=float, default=60.0)

    args = parser.parse_args(argv)

    # ── New v0.1 handlers ────────────────────────────────────────────────
    if args.cmd == "keys":
        print("API key status:")
        print(config.missing_keys_message())
        print(f"\nCache dir: {config.CACHE_DIR}")
        print(f"Audio dir: {config.AUDIO_DIR}")
        return 0

    if args.cmd == "discover":
        if args.trending:
            cands = discover.trending(region=args.trending, n=args.n)
        elif args.spotify_playlist:
            tracks = discover.spotify_top_tracks(args.spotify_playlist, n=args.n)
            print(json.dumps(tracks, indent=2, ensure_ascii=False))
            return 0
        else:
            cands = discover.search(args.query, n=args.n)

        if args.json:
            print(json.dumps([c.__dict__ for c in cands], indent=2, ensure_ascii=False))
        else:
            for i, c in enumerate(cands, 1):
                mins, secs = divmod(c.duration_s, 60)
                views = f"{c.view_count:,}" if c.view_count else "—"
                print(f"{i:2d}. {c.title[:60]:<60} [{mins}:{secs:02d}]  {views} views")
                print(f"    {c.channel}  ·  {c.youtube_url}")

        if args.save:
            p = discover.save_manifest(cands)
            print(f"\nSaved → {p}")
        return 0

    if args.cmd == "download":
        c = discover.Candidate(youtube_id=args.youtube_id, title=args.title,
                               channel="", duration_s=0)
        print(f"Downloading {args.youtube_id} → {c.song_id}.mp3 ...")
        path = discover.download(c)
        print(f"Done → {path}")
        return 0

    if args.cmd == "live":
        live.run_live(live.LiveOptions(
            vibe=args.vibe,
            chunk_size=args.chunk_size,
            candidate_pool=args.candidates,
            target_bpm=args.bpm,
            do_stems=args.stems,
        ))
        return 0

    if args.cmd == "make-mashup":
        result = orchestrator.make_mashup(
            args.vibe,
            n_songs=args.n,
            n_candidates=args.candidates,
            candidate_query=args.query,
            do_stems=args.stems,
            do_critique=args.critique,
            target_bpm=args.bpm,
        )
        print("\n" + "=" * 60)
        print(f"Done in {result.timings.get('total_s', '?')}s")
        print(f"Output: {result.audio_path}")
        print(f"Duration: {result.duration_s}s")
        print(f"Job ID: {result.job_id}")
        print("\nNarrative:")
        print(f"  {result.plan['narrative']}")
        print("\nSongs picked:")
        for s in result.songs:
            print(f"  - {s['title'][:60]}")
            print(f"      bpm={s['bpm']:.1f}  key={s['key']}  dur={s['duration_s']:.0f}s")
        if result.critique:
            print(f"\nCritique score: {result.critique.get('score', 0):.2f}")
            print(f"  {result.critique.get('overall', '')}")
        print(f"\nTimings: {result.timings}")
        return 0

    if args.cmd == "pick-songs":
        query = args.query or args.vibe
        cands = discover.search(query, n=args.candidates)
        result = plan.pick_songs(args.vibe, cands, n=args.n)
        print(f"\nNarrative: {result.get('narrative', '')}\n")
        for i, s in enumerate(result.get("songs", []), 1):
            print(f"{i}. [{s.get('role','?'):<11s}] {s.get('title','')}")
            print(f"   yt={s.get('youtube_id','?')}")
            print(f"   {s.get('reason','')}\n")
        return 0

    if args.cmd == "plan-transition":
        result = plan.plan_transition(
            out_title=args.from_title, out_bpm=args.from_bpm,
            out_key=args.from_key, out_mode=args.from_mode, out_lyric=args.from_lyric,
            in_title=args.to_title,  in_bpm=args.to_bpm,
            in_key=args.to_key,  in_mode=args.to_mode,  in_lyric=args.to_lyric,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "stems":
        p = Path(args.audio)
        if not p.is_absolute() and not p.exists():
            p = config.AUDIO_DIR / p.name
        if not p.exists():
            print(f"Audio not found: {p}", file=sys.stderr)
            return 2
        print(f"Separating {p.name} via demucs ({args.model}). First time is slow (~2 min CPU).")
        paths = stems_mod.separate(p, model=args.model, force=args.force)
        print(f"\nDone. Stems written to {paths['vocals'].parent}")
        for name, sp in paths.items():
            mb = sp.stat().st_size / 1024 / 1024
            print(f"  {name:>7s}.wav   {mb:>5.1f} MB")
        return 0

    if args.cmd == "analyze":
        p = Path(args.audio)
        if not p.is_absolute() and not p.exists():
            p = config.AUDIO_DIR / p.name
        if not p.exists():
            print(f"Audio not found: {p}", file=sys.stderr)
            return 2
        res = analyze_mod.analyze(p, force=args.force)
        mins, secs = divmod(int(res.duration_s), 60)
        print(f"Song:       {res.song_id}")
        print(f"Duration:   {mins}:{secs:02d}  ({res.duration_s:.1f}s)")
        print(f"BPM:        {res.bpm:.1f}  ({res.bpm_confidence} confidence)")
        print(f"Bar:        {res.bar_duration_s:.3f}s ({res.bar_duration_s*4:.2f}s per 4-bar phrase)")
        print(f"Key:        {res.key} {res.mode}  (correlation {res.key_confidence:.2f})")
        print(f"Beats:      {len(res.beats)}    Downbeats: {len(res.downbeats)}")
        print(f"\n--- Sections ({len(res.sections)}) ---")
        for s in res.sections:
            bars = (s["end"] - s["start"]) / res.bar_duration_s if res.bar_duration_s else 0
            print(f"  [{s['start']:>6.1f}-{s['end']:>6.1f}s] {s['label']}  "
                  f"~{bars:>5.1f} bars   energy={s['energy']:.3f}")
        print(f"\n--- Peak moments (top {len(res.peak_moments)}) ---")
        for pm in res.peak_moments:
            print(f"  {pm['t']:>6.1f}s   rms={pm['rms']:.3f}")
        return 0

    if args.cmd == "lyrics":
        c = discover.Candidate(youtube_id=args.youtube_id, title=args.title,
                               channel=args.artist, duration_s=0)
        prefer = "whisper" if args.whisper else "genius"
        res = lyrics.fetch(c, prefer=prefer, force_whisper=args.whisper)
        print(f"Source:     {res.source}")
        print(f"Title:      {res.title}")
        print(f"Artist:     {res.artist}")
        if res.language: print(f"Language:   {res.language}")
        print(f"Timestamps: {res.has_timestamps}")
        print(f"Lines:      {len(res.lines)}")
        if res.words:    print(f"Words:      {len(res.words)}")
        n = min(args.lines, len(res.lines))
        print(f"\n--- First {n} lines ---")
        for ln in res.lines[:n]:
            if res.has_timestamps:
                print(f"  [{ln['start']:>6.1f}s] {ln['text']}")
            else:
                print(f"  {ln['text']}")
        return 0

    # ── Legacy v0.0 handlers ─────────────────────────────────────────────
    if args.cmd == "find-moments":
        result = find_moments(
            artist=args.artist,
            title=args.title,
            audio_path=args.audio,
            youtube_url=args.youtube,
            top_k=args.top,
        )
        if not result:
            print(
                f"No synced lyrics found for {args.artist} - {args.title}",
                file=sys.stderr,
            )
            return 2
        if args.json:
            _emit_json(result)
        else:
            _print_human(result)
        return 0

    if args.cmd == "plan":
        a = find_moments(
            artist=args.a_artist, title=args.a_title,
            audio_path=args.a_audio, youtube_url=args.a_youtube,
        )
        b = find_moments(
            artist=args.b_artist, title=args.b_title,
            audio_path=args.b_audio, youtube_url=args.b_youtube,
        )
        if not a:
            print(f"No synced lyrics for Song A: {args.a_artist} - {args.a_title}", file=sys.stderr)
            return 2
        if not b:
            print(f"No synced lyrics for Song B: {args.b_artist} - {args.b_title}", file=sys.stderr)
            return 2
        recipe = plan_mashup(a, b, target_duration=args.duration)
        if not recipe:
            print("Could not plan mashup (no usable iconic moments or groove pocket).", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(recipe.to_dict(), indent=2, ensure_ascii=False))
        else:
            _print_recipe(recipe)
        return 0

    if args.cmd == "demo":
        out = render_demo(args.out)
        if out is None:
            print(
                "Renderer unavailable. Install: pip install 'mashup[render]' "
                "and ensure ffmpeg is on PATH.",
                file=sys.stderr,
            )
            return 3
        print(f"Wrote synthetic demo: {out}")
        print("Note: sine-wave synth, not music. Demonstrates the render chain.")
        return 0

    if args.cmd == "diagnose":
        report = diagnose_audio(args.audio, model=args.model)
        if report is None:
            print(
                "OPENROUTER_API_KEY not set. Export it before running diagnose.",
                file=sys.stderr,
            )
            return 3
        print(report)
        return 0

    if args.cmd == "make":
        a = find_moments(
            artist=args.a_artist, title=args.a_title,
            audio_path=args.a_audio, youtube_url=args.a_youtube,
        )
        b = find_moments(
            artist=args.b_artist, title=args.b_title,
            audio_path=args.b_audio, youtube_url=args.b_youtube,
        )
        if not a:
            print(f"No synced lyrics for Song A: {args.a_artist} - {args.a_title}", file=sys.stderr)
            return 2
        if not b:
            print(f"No synced lyrics for Song B: {args.b_artist} - {args.b_title}", file=sys.stderr)
            return 2
        recipe = plan_mashup(a, b, target_duration=args.duration)
        if not recipe:
            print("Could not plan mashup (no usable iconic moments or groove pocket).", file=sys.stderr)
            return 2
        out = render_recipe(
            recipe,
            song_a_audio_path=args.a_audio,
            song_b_audio_path=args.b_audio,
            output_path=args.out,
        )
        if out is None:
            print("Renderer unavailable. Install render extras: pip install 'mashup[render]' and ensure ffmpeg is on PATH.", file=sys.stderr)
            return 3
        _print_recipe(recipe)
        print(f"\nWrote: {out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

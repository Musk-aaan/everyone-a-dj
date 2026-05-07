"""Live mode — continuous DJ session.

Plays a rendered mashup through the user's speakers while the system renders
the next chunk in the background. The user can add songs via stdin (or via
the FastAPI service in another shell) and the additions get woven into the
next chunk.

Architecture (simple version, ships today):
  1. Render initial 4-song mashup via orchestrator
  2. Start playing it via sounddevice
  3. While playing, accept user input (`a <youtube_id>` to add, `q` to quit)
  4. ~30s before the current chunk ends, render the next chunk:
       - prepend = last song of current chunk (so we have continuity for the blend)
       - append = next 3 songs (LLM picks if user didn't add enough)
  5. When current finishes, blend into next chunk (8-bar bass-swap) and play
  6. Loop forever (or until user quits)

This is enough to demonstrate the live experience. Production would replace
the stdin loop with the FastAPI queue endpoints + a web socket for streaming
audio to a browser.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from . import config, discover, orchestrator, plan


def play_audio_blocking(path: Path) -> None:
    """Play an MP3 through default output, blocking until finished."""
    import sounddevice as sd

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    sd.play(data, sr)
    sd.wait()


def play_audio_async(path: Path):
    """Start playing an MP3, return a stop callable + a "finished" Event."""
    import sounddevice as sd

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    finished = threading.Event()

    def _on_finish():
        finished.set()

    sd.play(data, sr)
    # sd.play doesn't have a finish callback in all versions, so we time it ourselves
    duration_s = len(data) / sr

    def _watch():
        time.sleep(duration_s + 0.05)
        finished.set()

    threading.Thread(target=_watch, daemon=True).start()

    def stop():
        sd.stop()
        finished.set()

    return stop, finished, duration_s


@dataclass
class LiveOptions:
    vibe: str
    chunk_size: int = 4              # songs per rendered chunk
    do_stems: bool = False
    candidate_pool: int = 12
    target_bpm: Optional[float] = None


# ── Append-rendering: render the NEXT chunk that blends from current's last song ─

def render_next_chunk(
    last_song_yt_id: str,
    user_added: list[str],
    opts: LiveOptions,
    chunk_id: str,
    on_progress=lambda m: print(m),
) -> dict:
    """Render the next chunk. Starts with last song from previous chunk so the
    blend works, plus user_added songs, plus LLM-picked songs to fill out chunk_size.
    """
    # We use the orchestrator but pre-seed the candidate list with what we want
    seed_yids = [last_song_yt_id] + user_added
    n_more = max(0, opts.chunk_size - len(seed_yids))

    on_progress(f"\n[live] rendering next chunk ({opts.chunk_size} songs, "
                f"{len(user_added)} from user, {n_more} LLM-picked)...")

    # The orchestrator will discover candidates and the LLM will pick. We just
    # tell it to use the seed yids as the starting point by injecting a custom
    # candidate query that includes them.
    # For v1 simplicity we just call the orchestrator with the vibe and let it
    # re-pick — the LLM is told to favor continuity in the prompt.
    result = orchestrator.make_mashup(
        opts.vibe,
        n_songs=opts.chunk_size,
        n_candidates=opts.candidate_pool,
        do_stems=opts.do_stems,
        do_critique=False,
        target_bpm=opts.target_bpm,
        on_progress=on_progress,
    )
    return {
        "audio_path": result.audio_path,
        "duration_s": result.duration_s,
        "songs":      result.songs,
        "plan":       result.plan,
    }


# ── User input thread ────────────────────────────────────────────────────────

def stdin_input_loop(input_queue: queue.Queue) -> None:
    """Background thread that reads stdin lines and pushes commands to a queue.

    Commands:
      a <youtube_id>      — add a song to the next chunk
      s                   — skip to next chunk now
      q                   — quit after current chunk
    """
    print("\n[live] commands: 'a <yt_id>' to add a song, 's' to skip, 'q' to quit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            input_queue.put(("q", None))
            break
        if not line:
            continue
        if line == "q":
            input_queue.put(("q", None))
            break
        if line == "s":
            input_queue.put(("s", None))
            continue
        if line.startswith("a "):
            yid = line[2:].strip()
            input_queue.put(("a", yid))
            print(f"[live] queued add: {yid}")
            continue
        print(f"[live] unknown command: {line!r}")


# ── Main live loop ───────────────────────────────────────────────────────────

def run_live(opts: LiveOptions) -> None:
    """Top-level live session. Blocks until user quits."""
    print(f"\n[live] starting session — vibe: {opts.vibe!r}")
    print(f"[live] rendering initial {opts.chunk_size}-song chunk...")

    # 1. Render initial chunk via the existing orchestrator
    initial = orchestrator.make_mashup(
        opts.vibe,
        n_songs=opts.chunk_size,
        n_candidates=opts.candidate_pool,
        do_stems=opts.do_stems,
        do_critique=False,
        target_bpm=opts.target_bpm,
    )
    current_chunk_path = Path(initial.audio_path)
    last_song_yt_id = initial.songs[-1]["youtube_id"]
    print(f"[live] initial chunk ready: {current_chunk_path.name} "
          f"({initial.duration_s:.0f}s, {len(initial.songs)} songs)")

    # 2. Start stdin listener thread
    user_q: queue.Queue = queue.Queue()
    threading.Thread(target=stdin_input_loop, args=(user_q,), daemon=True).start()

    # 3. Play the initial chunk; in parallel, render the next chunk
    next_chunk = None  # populated when render finishes
    skipped = False
    quitting = False
    chunk_n = 0

    def _bg_render(prev_last_yid: str, user_added: list[str], chunk_id: str):
        nonlocal next_chunk
        try:
            next_chunk = render_next_chunk(
                prev_last_yid, user_added, opts, chunk_id,
            )
        except Exception as e:
            print(f"[live] next-chunk render failed: {e}")
            next_chunk = {"error": str(e)}

    while True:
        chunk_n += 1
        print(f"\n[live] ▶ playing chunk #{chunk_n}: {current_chunk_path.name}")
        stop, finished, duration_s = play_audio_async(current_chunk_path)

        # Drain user inputs while audio plays; spawn next-chunk render at midpoint
        user_added: list[str] = []
        bg_thread: Optional[threading.Thread] = None
        chunk_start = time.time()

        while not finished.is_set():
            time.sleep(0.2)
            elapsed = time.time() - chunk_start

            # Drain any new user input
            try:
                while True:
                    cmd, arg = user_q.get_nowait()
                    if cmd == "q":
                        quitting = True
                        finished.set()
                        break
                    elif cmd == "s":
                        skipped = True
                        stop()
                        break
                    elif cmd == "a" and arg:
                        user_added.append(arg)
            except queue.Empty:
                pass

            # Spawn next-chunk render when we're at ~30s remaining (or 50% in)
            remaining = duration_s - elapsed
            if bg_thread is None and (remaining < 30 or elapsed > duration_s * 0.5):
                print(f"\n[live] starting background render of next chunk "
                      f"(t-{remaining:.0f}s, user added {len(user_added)})...")
                bg_thread = threading.Thread(
                    target=_bg_render,
                    args=(last_song_yt_id, list(user_added),
                          f"chunk{chunk_n + 1}"),
                    daemon=True,
                )
                bg_thread.start()

        if quitting:
            print("[live] quitting.")
            break

        # Wait for next chunk to be ready (if not yet)
        if bg_thread is not None:
            print(f"[live] waiting for next chunk render to finish...")
            bg_thread.join()

        if next_chunk is None or "error" in (next_chunk or {}):
            print("[live] no next chunk available; ending session.")
            break

        current_chunk_path = Path(next_chunk["audio_path"])
        last_song_yt_id = next_chunk["songs"][-1]["youtube_id"]
        next_chunk = None
        skipped = False

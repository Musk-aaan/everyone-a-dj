"""Phase 5 — LLM Brain.

Every musical decision goes through Gemini 2.5 Flash via OpenRouter:

  pick_songs(vibe, candidates, n)         → ranked songs that arc together
  pick_section(analysis, lyrics)          → which 30s of the song to use
  plan_transition(out_meta, in_meta)      → bass-swap / riser / crash JSON
  critique(audio_path, plan)              → score + issue list + revisions

Single endpoint, single model — keeps costs around $0.001-0.005 per mashup.

Each function returns a dict (parsed JSON). We keep the schemas explicit in the
prompts so Gemini's outputs stay machine-readable.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional

import requests

from . import config


# ── Core LLM call ────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _call(prompt: str, *, audio_path: Optional[Path] = None,
          temperature: float = 0.4, max_tokens: int = 2000,
          timeout: float = 180.0) -> str:
    """Send `prompt` (and optional audio) to Gemini Flash via OpenRouter."""
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if audio_path:
        if not audio_path.exists():
            raise RuntimeError(f"audio file not found: {audio_path}")
        with audio_path.open("rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")
        fmt = audio_path.suffix.lstrip(".").lower() or "mp3"
        content.append({"type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": fmt}})

    payload = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mashup-debug",
            "X-Title": "mashup-brain",
        },
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _parse_json(raw: str) -> Any:
    """Parse JSON from a model response, stripping code fences if present."""
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Sometimes the model surrounds JSON with prose — extract the outermost object/array
        m = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
        if not m:
            raise RuntimeError(f"Brain did not return JSON. Got: {raw[:200]!r}")
        return json.loads(m.group(0))


# ── Song picker ──────────────────────────────────────────────────────────────

_PICK_SONGS_PROMPT = """You are an AI DJ designing a {n}-song mashup for this vibe:

VIBE: {vibe}

Candidate songs (with view counts, channel, duration):
{candidate_list}

Pick {n} songs in the order they should play. CRITICAL constraints:

1. **Tempo coherence is the #1 rule.** All {n} songs must be within roughly
   ±15 BPM of each other. A mashup with songs at 92, 144, 167, and 103 BPM
   is unmixable — it will sound like jarring tempo jumps no matter what
   transition tricks we use. Pick a target tempo band first, then pick songs
   that ALL sit inside it. Use your musical knowledge of typical BPMs.

2. **Energy arc** within the BPM band: warm-up → first peak → climax → cool-down.

3. **Cultural/musical relationship**: same era, similar genre, or compatible keys
   when you can. Don't mix Bollywood ballads with hard EDM.

4. Avoid duplicate or near-duplicate tracks (same song, different uploads).

Return ONLY this JSON, nothing else:
{{
  "songs": [
    {{"youtube_id": "<id from candidates>", "title": "<title>",
      "role": "warm_up | first_peak | climax | cool_down",
      "approx_bpm": <integer, your best guess>,
      "reason": "<one sentence: why this song in this slot, AND how its BPM fits the band>"}}
  ],
  "target_bpm_band": "<e.g. '110-130 BPM'>",
  "narrative": "<one sentence describing the overall energy arc>"
}}"""


def pick_songs(vibe: str, candidates: list, n: int = 4,
               bpm_hints: Optional[dict[str, float]] = None) -> dict:
    """Pick `n` songs from `candidates` that work together for `vibe`.

    `candidates` is a list of discover.Candidate (or dicts).
    `bpm_hints`: optional {youtube_id: bpm} for any candidates we've already
    analyzed. The LLM uses these as ground truth for tempo coherence.
    """
    bpm_hints = bpm_hints or {}
    lines = []
    for c in candidates:
        d = c if isinstance(c, dict) else c.__dict__
        mins, secs = divmod(int(d.get("duration_s", 0)), 60)
        bpm = bpm_hints.get(d["youtube_id"])
        bpm_note = f"  bpm={bpm:.0f}" if bpm else "  bpm=unknown"
        lines.append(
            f"  - youtube_id={d['youtube_id']}  title={d['title']!r}  "
            f"channel={d.get('channel', '')!r}  dur={mins}:{secs:02d}{bpm_note}"
        )
    prompt = _PICK_SONGS_PROMPT.format(
        n=n, vibe=vibe, candidate_list="\n".join(lines)
    )
    return _parse_json(_call(prompt, max_tokens=1500))


# ── Section picker ───────────────────────────────────────────────────────────

_PICK_SECTION_PROMPT = """You are choosing the most iconic 40-55 second section
of this song to use in a mashup.

The section will be blended into and out of via 16-second bass-swap transitions
on each side, so the section needs to be long enough that ~8 seconds of
"useable" content remains in the middle (chorus / hook / signature moment).

Song: {title}
BPM: {bpm}    Key: {key} {mode}    Duration: {duration}s

Sections detected (label, time range, energy 0-1):
{sections_list}

Peak energy moments (top RMS spots):
{peaks_list}

Lyrics excerpt (first {n_lines} lines):
{lyrics_excerpt}

The section should be the part most listeners would recognise (typically the
chorus or hook). Energy and lyrics together usually point to it.

Return ONLY this JSON. The end-start span MUST be 40-55 seconds:
{{
  "start": <seconds, float>,
  "end": <seconds, float>,
  "section_label": "<the A/B/C/... label from above, or 'custom'>",
  "iconic_lyric": "<one short lyric phrase from this section, if any>",
  "why": "<one sentence: what makes this section the right pick>"
}}"""


def pick_section(*, title: str, analysis: dict, lyrics: dict, n_lines: int = 8) -> dict:
    """Pick the best 25-35s section of a song using its analysis + lyrics."""
    secs = "\n".join(
        f"  - {s['label']}  [{s['start']:.1f}-{s['end']:.1f}s]  energy={s['energy']:.3f}"
        for s in analysis.get("sections", [])
    )
    peaks = "\n".join(
        f"  - {p['t']:.1f}s   rms={p['rms']:.3f}"
        for p in analysis.get("peak_moments", [])
    )
    excerpt = "\n".join(
        f"  {ln.get('text','')}"
        for ln in lyrics.get("lines", [])[:n_lines]
    ) or "  (no lyrics available)"

    prompt = _PICK_SECTION_PROMPT.format(
        title=title,
        bpm=analysis.get("bpm", 0),
        key=analysis.get("key", "?"),
        mode=analysis.get("mode", "?"),
        duration=analysis.get("duration_s", 0),
        sections_list=secs or "  (none detected)",
        peaks_list=peaks or "  (none detected)",
        n_lines=n_lines,
        lyrics_excerpt=excerpt,
    )
    return _parse_json(_call(prompt, max_tokens=400))


# ── Transition planner ───────────────────────────────────────────────────────

_PLAN_TRANSITION_PROMPT = """You are planning a transition between two songs in a DJ set.

OUTGOING (currently playing, last few bars):
  title:  {out_title}
  bpm:    {out_bpm}
  key:    {out_key} {out_mode}
  end_lyric: {out_lyric!r}

INCOMING (about to drop):
  title:  {in_title}
  bpm:    {in_bpm}
  key:    {in_key} {in_mode}
  start_lyric: {in_lyric!r}

Design the transition. Real DJ techniques you can use:
  - hpf_sweep: progressively cut bass + mids on outgoing's last bars (creates tension)
  - riser: white-noise sweep building toward the drop
  - stutter: repeat the last beat 3-6 times with decay
  - reverb_throw: huge reverb on outgoing's last note
  - downlifter: bass whoosh sweeping down
  - flanger: comb-filter wobble on outgoing
  - bass_swap: cut sub-200Hz on outgoing while bringing in incoming's bass
  - silence_gap: brief 100-300ms silence right before the drop
  - crash: cymbal hit at the moment incoming starts
  - crossfade: smooth volume blend (use only for cool-down moments)

Return ONLY this JSON:
{{
  "technique": "<one phrase describing the overall approach>",
  "build_bars": <int, how many bars before the drop the build starts>,
  "fx": [
    {{"name": "hpf_sweep", "start_bar": -8, "end_bar": -1,
      "low_hz": 100, "high_hz": 14000}},
    {{"name": "riser", "start_bar": -8, "end_bar": -0.5, "gain": 0.45}},
    {{"name": "stutter", "start_bar": -1, "end_bar": -0.25, "n_reps": 4}},
    {{"name": "crash", "at_bar": 0, "gain": 0.9}}
  ],
  "silence_ms": <int, 0-300>,
  "use_bass_swap": <true|false>,
  "fade_kind": "drop | crossfade",
  "reasoning": "<2-3 sentences: why these choices given the BPM/key/lyric context>"
}}

Bars are relative to the drop moment (bar 0 = first bar of incoming song).
Negative bars are during the outgoing song's tail."""


def plan_transition(*,
                    out_title: str, out_bpm: float, out_key: str, out_mode: str,
                    out_lyric: str,
                    in_title: str,  in_bpm: float, in_key: str, in_mode: str,
                    in_lyric: str) -> dict:
    """Design a transition between two songs given their musical context."""
    prompt = _PLAN_TRANSITION_PROMPT.format(
        out_title=out_title, out_bpm=out_bpm, out_key=out_key, out_mode=out_mode,
        out_lyric=out_lyric or "(instrumental)",
        in_title=in_title, in_bpm=in_bpm, in_key=in_key, in_mode=in_mode,
        in_lyric=in_lyric or "(instrumental)",
    )
    return _parse_json(_call(prompt, max_tokens=800))


# ── Critic (listen-back) ─────────────────────────────────────────────────────

_CRITIQUE_PROMPT = """Listen to this rendered mashup carefully. The intended structure was:

{plan_summary}

Score it on a 0.0-1.0 scale where 1.0 = sounds like a pro DJ mix and
0.0 = random cuts pasted together.

For each transition or section, list any issues you actually hear:
- BPM jolts (tempo changes that jar)
- Out-of-place keys / clashing harmonies
- Effects that don't land (riser too quiet, crash buried, etc.)
- Awkward silences or lack of build
- Anything that breaks the energy arc

Return ONLY this JSON:
{{
  "score": <float 0.0-1.0>,
  "issues": [
    {{"timestamp_s": <float>, "severity": "low | medium | high",
      "what_you_heard": "<short description>"}}
  ],
  "suggestions": ["<actionable revision 1>", "<actionable revision 2>"],
  "overall": "<2-3 sentence summary>"
}}"""


def critique(audio_path: Path, plan_summary: str) -> dict:
    """Send a rendered mashup to Gemini and get back a scored critique."""
    prompt = _CRITIQUE_PROMPT.format(plan_summary=plan_summary)
    return _parse_json(_call(prompt, audio_path=audio_path, max_tokens=1500))

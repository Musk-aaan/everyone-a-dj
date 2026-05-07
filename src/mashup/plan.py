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
    return _parse_json(_call(prompt, max_tokens=700))


# ── Transition planner ───────────────────────────────────────────────────────

_PLAN_TRANSITION_PROMPT = """You are a DJ choosing how to transition between two
songs. Pick ONE technique from the menu below — your choice should reflect
what a real DJ would do given these specific songs.

OUTGOING (currently playing, ending soon):
  title:        {out_title}
  bpm:          {out_bpm}
  key:          {out_key} {out_mode}
  has_stems:    {out_stems}     (vocals/drums/bass/other available?)
  end_lyric:    {out_lyric!r}

INCOMING (about to play):
  title:        {in_title}
  bpm:          {in_bpm}
  key:          {in_key} {in_mode}
  has_stems:    {in_stems}
  start_lyric:  {in_lyric!r}

POSITION IN SET: transition {position} of {total} (0=warm-up→peak,
                  middle=climax, last=cool-down)

CRITICAL CONSTRAINTS:

- **Variety matters.** Real DJs don't use the same technique 3 times in a
  row. If the previous transition was bass_swap, this one should NOT also
  be bass_swap unless there's a strong musical reason. Push toward
  acapella_drop, drum_swap, reverb_throw, hard_drop when stems allow.

- **Drops are special.** hard_drop adds energy; dramatic_drop is the climax
  moment. Use them at peak positions in the set (middle transitions of a
  4-song set), not for cool-downs.

- **Stems unlock the good moves.** When stems are available for the right
  song, prefer acapella_drop / drum_swap / reverb_throw over the boring
  bass_swap default.

- bass_swap is the safe fallback for genuinely smooth-blend moments
  (cool-down, similar tempo + similar genre + nothing iconic happening).
  Don't pick it just because you're unsure.

TECHNIQUE MENU (pick exactly one):

1. "bass_swap"     — Clean 8-bar EQ blend. Two tracks layer with frequency
                     separation, hard bass cut at midpoint phrase boundary.
                     The default safe choice. Works without stems.

2. "acapella_drop" — Incoming's vocals layer over outgoing's beat for 4 bars,
                     then full incoming takes over. The actual mashup move.
                     Requires INCOMING stems. Best when both songs have
                     prominent vocals and similar tempos.

3. "drum_swap"     — Outgoing's vocals continue but incoming's drums replace
                     outgoing's drums. The "Kanye-style beat switch."
                     Requires BOTH stems. Best for energy shifts where the
                     vocal singer is iconic.

4. "tempo_ramp"    — Time-stretches outgoing's last 4 bars to match incoming
                     BPM, then bass-swap blend. USE THIS when |out_bpm - in_bpm| > 20
                     — it's the only way to bridge a tempo mismatch cleanly.

5. "reverb_throw"  — Outgoing's last vocal phrase drenched in big reverb that
                     trails into incoming. Cinematic / epic feel. Requires
                     OUTGOING stems. Best for emotional or cool-down moments.

6. "hard_drop"     — 1 beat of total silence + CRASH cymbal + incoming's full
                     mix slams in on beat 1. <2 seconds total. The classic
                     festival drop. Use for HIGH-IMPACT moments where the
                     incoming song's drop is itself an iconic moment.

7. "dramatic_drop" — Like hard_drop but bigger: outgoing fades, 1.5 beats
                     silence, crash, then 2 bars of bass+drums build, then
                     full mix slam. ~5-7 seconds total. Use ONLY ONCE PER SET
                     at the absolute climax. Requires INCOMING stems.

Return ONLY this JSON:
{{
  "technique": "<one of: bass_swap | acapella_drop | drum_swap | tempo_ramp | reverb_throw | hard_drop | dramatic_drop>",
  "reasoning": "<2-3 sentences: why this technique fits these two songs and this position in the set>"
}}"""


def plan_transition(*,
                    out_title: str, out_bpm: float, out_key: str, out_mode: str,
                    out_lyric: str, out_stems: bool,
                    in_title: str,  in_bpm: float, in_key: str, in_mode: str,
                    in_lyric: str, in_stems: bool,
                    position: int, total: int) -> dict:
    """LLM picks ONE transition technique by name from a fixed menu.

    Returns: {"technique": str, "reasoning": str}.
    The orchestrator dispatches to the matching render function.
    """
    prompt = _PLAN_TRANSITION_PROMPT.format(
        out_title=out_title, out_bpm=out_bpm, out_key=out_key, out_mode=out_mode,
        out_lyric=out_lyric or "(instrumental)", out_stems=str(out_stems).lower(),
        in_title=in_title, in_bpm=in_bpm, in_key=in_key, in_mode=in_mode,
        in_lyric=in_lyric or "(instrumental)", in_stems=str(in_stems).lower(),
        position=position, total=total,
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

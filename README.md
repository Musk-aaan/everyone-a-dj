# Everyone a DJ

AI DJ that takes a vibe and produces a pro-quality mashup MP3. Researches songs, picks them, stems them, blends them with real DJ techniques (bass swap, acapella drop, drum swap, tasteful drop, reverb throw, tempo ramp), and masters the output.

## Quickstart

```bash
pip install -e ".[all]"
mashup keys                                      # show which API keys are set
mashup make-mashup --vibe "your vibe" --critique
mashup live --vibe "your vibe"                   # continuous DJ session via speakers
uvicorn mashup.api:app                           # web API for the queue UI
```

## Architecture (8 phases)

| Phase | Module | What it does |
|---|---|---|
| 1 | `discover.py` | YouTube + Spotify search, BPM-outlier filtering, playlist/mix rejection |
| 2 | `lyrics.py` | Genius (~30M tracks) + Gemini-via-OpenRouter transcription + Whisper |
| 3 | `analyze.py` | librosa beats, key (Krumhansl-Schmuckler), section boundaries, energy curve |
| 4 | `stems.py` | demucs 4-stem separation (vocals / drums / bass / other) |
| 5 | `plan.py` | LLM brain — song picker, section picker, transition planner, listen-back critic |
| 6 | `render.py` | pedalboard pro DSP, all transition primitives, mastering chain |
| 7 | `orchestrator.py` | end-to-end pipeline, story-arc transition selection |
| 8 | `session.py` + `api.py` + `live.py` | queue manager, FastAPI service, continuous live mode |

## Transition primitives

| Technique | Use case |
|---|---|
| **bass-swap blend** | Default. Real DJ EQ swap — two tracks layered with frequency separation, hard bass cut at the midpoint phrase boundary. |
| **acapella drop** | Incoming vocals layered over outgoing beat. The actual mashup move. |
| **drum swap** | Outgoing vocals continue, incoming drums replace outgoing drums. Kanye-style beat switch. |
| **tempo ramp** | Time-stretches outgoing's last 4 bars to match incoming BPM. Bridges tempo mismatches. |
| **reverb throw** | Outgoing's last vocal phrase bathed in big reverb that trails into incoming. |
| **tasteful drop** | Used ONCE per set at the climax. Silence + crash + bass-only build → full mix slam. |

The orchestrator picks technique per transition position (warm-up / climax / cool-down), with BPM mismatch always overriding to tempo ramp.

## LLM provider lock

All LLM calls go through OpenRouter using `google/gemini-2.5-flash`. ~$0.005 per mashup. No Claude, no GPT-4.

## Status

Working end-to-end. Gemini's per-transition critic scored 0.7–0.85 on stem-based runs. Known limitations:
- Beat-grid alignment of fx events (the next system-level fix)
- BPM-coherent song picking when cache misses
- Live mode is local-only (sounddevice + stdin); web streaming endpoint not yet built

See `PLAN.md` for the full design.

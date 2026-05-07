# Everyone a DJ — Master Plan

> **Vision:** Replace the DJ. Anyone types a vibe ("Indian wedding 2026, 25–35 year olds, mostly Bollywood + funk") and gets a 5-minute professionally-mixed set in 90 seconds.

The current `mix_pro.py` is a hand-coded prototype. This plan is what turns it into a product.

---

## The 10x Architecture

Eight phases. Each phase ships independently and makes the demo measurably better.

```
                       ┌─────────────────────┐
                       │   USER VIBE INPUT   │
                       │ "Indian wedding..." │
                       └──────────┬──────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│  Phase 1  │  DISCOVERY                                           │
│  YouTube Data API + yt-dlp + Spotify charts + Apple Music charts │
│  → ranked candidate songs (trending + cultural fit)              │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│  Phase 2  │  LYRICS                                              │
│  Genius API → official lyrics. Whisper → word-level timestamps.  │
│  → searchable lyric grid per song                                │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│  Phase 3  │  AUDIO INTELLIGENCE                                  │
│  madmom (beats + downbeats), essentia (key + energy + chords),   │
│  librosa.segment (intro / verse / chorus / bridge)               │
│  → per-song JSON fingerprint                                     │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│  Phase 4  │  STEM SEPARATION                                     │
│  demucs → vocals.wav / drums.wav / bass.wav / other.wav          │
│  THE biggest unlock. Real mashups need stems.                    │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│  Phase 5  │  LLM BRAIN  (single provider — OpenRouter, cheap)    │
│  Gemini 2.5 Flash via OpenRouter, used for EVERY call:           │
│    - song selection                                              │
│    - section picker                                              │
│    - transition planner                                          │
│    - listen-back critic + revision loop                          │
│  → render plan JSON                                              │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│  Phase 6  │  PRO DSP RENDERING                                   │
│  pedalboard (Spotify) → real reverb / EQ / compressor / limiter  │
│  Render the plan. Master to -14 LUFS streaming standard.         │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│  Phase 7  │  ORCHESTRATOR + API                                  │
│  Async pipeline. FastAPI endpoint. Job queue for long renders.   │
│  Returns mashup URL + transparent plan JSON.                     │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│  Phase 8  │  LIVE MODE + FEEDBACK                                │
│  Streaming render. Thumbs-up/down per transition.                │
│  Per-user taste model. Re-roll any transition mid-set.           │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
                          ┌────────────────┐
                          │  90-second mix │
                          └────────────────┘
```

---

## Why YouTube + Lyrics matter (added)

**YouTube** is where music *actually* lives in 2026. Spotify charts are good for the West, but Bollywood, K-pop, Latin, Afrobeats — YouTube is the source of truth. The discovery layer queries:

- **YouTube Data API** `videos.list?chart=mostPopular&videoCategoryId=10` per region
- **Spotify Web API** `top/tracks` charts as a cross-reference for global hits
- **Apple Music charts** for genre-specific (especially country, jazz)
- **Last.fm trending tags** for vibe-to-genre mapping

The LLM gets ~50 candidate songs per vibe, with play counts, regions, and metadata. It picks 4.

**Lyrics** are the layer that turns "BPM-matched audio" into "musically meaningful":

1. **Lyric-aware section picking** — "the chorus that starts with 'I'm too hot' is more iconic than the second verse"
2. **Semantic transition continuity** — bridge BDT's "ye dil to ho gaya badtameez" (this heart has gone wild) into UF's "I'm too hot, hot damn" — they share a vibe of unrestrained energy
3. **A cappella detection** — moments where the vocal is alone become free overlay opportunities
4. **Audience filtering** — explicit lyrics flagged for family events
5. **Multi-language handling** — Whisper transcribes Hindi → translates → LLM still understands the meaning

Source order: **Genius API** for popular songs (has 30M+ tracks with lyrics), **Whisper-large-v3** for anything missing or for word-level alignment.

---

## Build order (what we're shipping in what phase)

| Phase | Deliverable | Time est. | Cost per mashup added |
|---|---|---|---|
| 0 | Project structure, env, cache layer | 30 min | $0 |
| 1 | `discover.py` — YouTube + Spotify search/charts/download | 1.5 hr | $0 (free tier) |
| 2 | `lyrics.py` — Genius + Whisper fallback | 1 hr | $0.005 (Whisper) |
| 3 | `analyze.py` — madmom + essentia + section detection | 2 hr | $0 (cached) |
| 4 | `stems.py` — demucs separation + caching | 1 hr | ~$0.01 (compute) |
| 5 | `plan.py` + `critic.py` — Gemini Flash brain loop (OpenRouter) | 2 hr | ~$0.005 (LLM calls) |
| 6 | `render.py` — pedalboard pro DSP | 1.5 hr | $0 |
| 7 | `orchestrator.py` + FastAPI service | 1.5 hr | $0 |
| 8 | Live mode (streaming + feedback) | 3 hr | $0 |

**Total per-mashup cost when warm:** ~$0.02. **Cold (first time song is touched):** ~$0.05. **Latency warm:** ~90s. **Cold:** ~3 min (stems are the bottleneck — cache forever).

LLM brain is dirt cheap because it's all Gemini 2.5 Flash via OpenRouter (~$0.075/M input, $0.30/M output). The expensive line item is Whisper at $0.006/min audio — only runs on songs without Genius lyrics.

---

## Data flow / file layout

```
mashup/
├── PLAN.md                          ← this file
├── .env                             ← API keys (gitignored)
├── pyproject.toml                   ← dependencies
├── audio/                           ← raw mp3 downloads (cached forever)
├── cache/
│   ├── lyrics/{song_id}.json        ← Genius / Whisper output
│   ├── analyses/{song_id}.json      ← beat grid + sections + key + energy
│   ├── stems/{song_id}/             ← vocals/drums/bass/other.wav
│   └── renders/{job_id}.mp3         ← final mashups
├── src/mashup/
│   ├── config.py                    ← env, paths, constants
│   ├── cache.py                     ← filesystem cache with hash keys
│   ├── discover.py                  ← Phase 1: YouTube + Spotify
│   ├── lyrics.py                    ← Phase 2: Genius + Whisper
│   ├── analyze.py                   ← Phase 3: musical analysis
│   ├── stems.py                     ← Phase 4: demucs
│   ├── plan.py                      ← Phase 5a: Claude planner
│   ├── critic.py                    ← Phase 5b: Gemini critic
│   ├── render.py                    ← Phase 6: pedalboard
│   ├── orchestrator.py              ← Phase 7: pipeline
│   ├── api.py                       ← Phase 7: FastAPI
│   └── diagnostics.py               ← existing Gemini debug tool
└── mix_pro.py                       ← v3 hand-coded mashup (legacy demo)
```

---

## API keys needed

| Service | Why | Cost | Where to get |
|---|---|---|---|
| **YouTube Data API v3** | Trending charts, search | Free, 10k units/day | console.cloud.google.com |
| **Spotify Web API** | Cross-reference Western charts | Free | developer.spotify.com |
| **Genius API** | Lyrics for popular tracks | Free | genius.com/api-clients |
| **OpenAI** | Whisper-large-v3 | $0.006/min audio | platform.openai.com |
| **OpenRouter** | Gemini 2.5 Flash — every brain call (planner, critic, picker) | ~$0.005/mashup | already have ✓ |

---

## Per-mashup pipeline (Phase 7 view)

```python
async def make_mashup(vibe: str, length_s: int = 300) -> MashupResult:
    # 1. DISCOVER: YouTube/Spotify → 50 candidates
    candidates = await discover.candidates_for(vibe, n=50)

    # 2. SELECT: Claude picks 4 songs that arc together
    songs = await plan.pick_songs(vibe, candidates, n=4)

    # 3. PARALLEL FETCH + ANALYZE
    enriched = await asyncio.gather(*[
        fetch_and_enrich(s) for s in songs
    ])
    # each = { audio, lyrics, analysis, stems }

    # 4. PLAN: Claude designs the full set
    set_plan = await plan.design_set(vibe, enriched, length_s)
    # { sections: [{song, range, fx}], transitions: [{type, build_bars, ...}] }

    # 5. RENDER: pedalboard executes the plan
    mix = render.execute(set_plan, enriched)

    # 6. CRITIQUE LOOP (max 2 iterations)
    for _ in range(2):
        critique = await critic.listen(mix, set_plan)
        if critique.score >= 0.85: break
        set_plan = await plan.revise(set_plan, critique)
        mix = render.execute(set_plan, enriched)

    # 7. MASTER: pedalboard limiter + LUFS normalisation
    final = render.master(mix, target_lufs=-14)

    return MashupResult(audio=final, plan=set_plan, critique=critique)
```

---

## Cache strategy

Everything derived from a song is cached forever, keyed by `sha1(youtube_id)`.

| Artifact | Size | Compute cost | Hit rate after 1k songs |
|---|---|---|---|
| Audio download | ~5 MB | yt-dlp ~10s | ~90% |
| Lyrics | ~10 KB | Genius ~1s, Whisper ~30s | ~95% |
| Analysis JSON | ~50 KB | madmom + essentia ~20s | ~95% |
| Stems (4×WAV) | ~80 MB | demucs ~30s GPU / 2 min CPU | ~95% |

After the system has touched ~1k songs, every new mashup is mostly cache hits → **<60s latency**.

---

## Where this becomes defensible

- **The transition planner prompt** — the playbook of how a great DJ thinks. This is the moat.
- **The taste model** — per-user feedback trains a reranker. Network effects.
- **The cached library** — every song processed makes future mashups faster + cheaper.
- **Live mode** — Spotify can't do live AI mixing. This is the wedge into events / parties / bars.

---

## Status

- [x] Phase 0 — project structure + cache + env
- [ ] Phase 1 — discovery (YouTube + Spotify)
- [ ] Phase 2 — lyrics (Genius + Whisper)
- [ ] Phase 3 — audio intelligence (madmom + essentia)
- [ ] Phase 4 — stem separation (demucs)
- [ ] Phase 5 — LLM brain (Claude + Gemini critic)
- [ ] Phase 6 — pro DSP (pedalboard)
- [ ] Phase 7 — orchestrator + FastAPI
- [ ] Phase 8 — live mode + feedback loop

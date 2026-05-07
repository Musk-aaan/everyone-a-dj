"""FastAPI service for the mashup product.

Exposes endpoints for two use cases:

1. Full offline mashup (vibe → MP3):
   POST /make-mashup      → {job_id}
   GET  /jobs/{id}/stream → SSE progress events
   GET  /jobs/{id}/audio  → MP3

2. Live DJ session (real-time queue management):
   POST /sessions         → session
   POST /sessions/{sid}/songs / PUT / DELETE / move
   POST /search, /suggest-next

Run with:  uvicorn mashup.api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from . import discover, orchestrator, plan, session

app = FastAPI(title="Everyone a DJ")
mgr = session.SessionManager()
_render_tasks: dict[str, asyncio.Task] = {}    # sid → background render loop

# ── Offline mashup job store ─────────────────────────────────────────────────
_jobs: dict[str, dict[str, Any]] = {}          # job_id → job state


# ── Request / response models ────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    vibe: str
    seed_youtube_ids: list[str] = []
    target_bpm: float = 115.0
    autostart_render: bool = True
    do_stems: bool = False


class AddSongRequest(BaseModel):
    youtube_id: str
    title: Optional[str] = None
    position: Optional[int] = None     # default = end of queue


class ReplaceSongRequest(BaseModel):
    youtube_id: str
    title: Optional[str] = None


class MoveSongRequest(BaseModel):
    to_position: int


class SearchRequest(BaseModel):
    query: str
    n: int = 8


# ── Request models (offline mashup) ─────────────────────────────────────────

class MakeMashupRequest(BaseModel):
    vibe: str
    n: int = 4
    stems: bool = True
    candidates: int = 12
    critique: bool = False


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Serve the web UI."""
    ui = Path(__file__).parent / "static" / "index.html"
    if ui.exists():
        return ui.read_text()
    return "<h1>Everyone a DJ API</h1><p>See /docs for API reference.</p>"


@app.post("/make-mashup")
async def make_mashup(req: MakeMashupRequest) -> dict:
    """Start an offline mashup job. Returns job_id; stream progress via /jobs/{id}/stream."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "running",
        "progress": [],
        "result": None,
        "error": None,
    }
    asyncio.create_task(_run_mashup_job(job_id, req))
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"job {job_id!r} not found")
    return {
        "status": job["status"],
        "progress": job["progress"],
        "result": job.get("result"),
        "error": job.get("error"),
    }


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """SSE endpoint — streams progress lines as they arrive, then a final done/error event."""
    if job_id not in _jobs:
        raise HTTPException(404, f"job {job_id!r} not found")

    async def event_gen():
        sent = 0
        while True:
            job = _jobs[job_id]
            msgs = job["progress"]
            while sent < len(msgs):
                line = msgs[sent].replace("\n", " ")
                yield f"data: {line}\n\n"
                sent += 1
            if job["status"] == "done":
                yield f"event: done\ndata: {job['result']['audio_path']}\n\n"
                break
            if job["status"] == "error":
                yield f"event: error\ndata: {job['error']}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/jobs/{job_id}/audio")
def get_job_audio(job_id: str):
    """Download the finished MP3."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"job not done (status={job['status']})")
    audio = Path(job["result"]["audio_path"])
    if not audio.exists():
        raise HTTPException(500, "audio file missing on disk")
    return FileResponse(str(audio), media_type="audio/mpeg",
                        filename=f"mashup-{job_id[:8]}.mp3")


async def _run_mashup_job(job_id: str, req: MakeMashupRequest) -> None:
    """Async wrapper that runs orchestrator.make_mashup() in a thread pool."""
    job = _jobs[job_id]
    progress_lines: list[str] = job["progress"]

    def on_progress(msg: str) -> None:
        progress_lines.append(msg)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: orchestrator.make_mashup(
                vibe=req.vibe,
                n_songs=req.n,
                n_candidates=req.candidates,
                do_stems=req.stems,
                run_critique=req.critique,
                on_progress=on_progress,
            ),
        )
        job["status"] = "done"
        job["result"] = {
            "audio_path": result.audio_path,
            "duration_s": result.duration_s,
            "songs": result.songs,
            "plan": result.plan,
            "critique": result.critique,
        }
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)


@app.post("/sessions")
def create_session(req: CreateSessionRequest) -> dict:
    sess = mgr.create(req.vibe, req.seed_youtube_ids, req.target_bpm)
    if req.autostart_render:
        _start_render_loop(sess.id, do_stems=req.do_stems)
    return sess.to_dict()


@app.get("/sessions/{sid}")
def get_session(sid: str) -> dict:
    try:
        return mgr.get(sid).to_dict()
    except KeyError:
        raise HTTPException(404, f"session {sid!r} not found")


@app.post("/sessions/{sid}/songs")
def add_song(sid: str, req: AddSongRequest) -> dict:
    sess = _get_or_404(sid)
    cand = discover.Candidate(
        youtube_id=req.youtube_id,
        title=req.title or "(unknown)",
        channel="", duration_s=0,
    )
    spec = sess.add_song(cand, position=req.position)
    return spec.to_dict()


@app.put("/sessions/{sid}/songs/{position}")
def replace_song(sid: str, position: int, req: ReplaceSongRequest) -> dict:
    sess = _get_or_404(sid)
    cand = discover.Candidate(
        youtube_id=req.youtube_id,
        title=req.title or "(unknown)",
        channel="", duration_s=0,
    )
    try:
        return sess.replace(position, cand).to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/sessions/{sid}/songs/{position}")
def remove_song(sid: str, position: int) -> dict:
    sess = _get_or_404(sid)
    try:
        return sess.remove(position).to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/sessions/{sid}/songs/{position}/move")
def move_song(sid: str, position: int, req: MoveSongRequest) -> dict:
    sess = _get_or_404(sid)
    try:
        return sess.move(position, req.to_position).to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/sessions/{sid}/songs/{position}/audio")
def get_segment_audio(sid: str, position: int):
    sess = _get_or_404(sid)
    if position < 0 or position >= len(sess.queue):
        raise HTTPException(404, "segment out of range")
    seg = sess.queue[position]
    if not seg.audio_path or not Path(seg.audio_path).exists():
        raise HTTPException(409, f"segment not ready (status={seg.status.value})")
    return FileResponse(seg.audio_path, media_type="audio/mpeg")


@app.post("/search")
def search(req: SearchRequest) -> list[dict]:
    """Search YouTube for songs the user can add to a session."""
    cands = discover.search(req.query, n=req.n)
    return [
        {"youtube_id": c.youtube_id, "title": c.title, "channel": c.channel,
         "duration_s": c.duration_s, "view_count": c.view_count,
         "url": c.youtube_url}
        for c in cands
    ]


@app.post("/sessions/{sid}/suggest-next")
def suggest_next(sid: str) -> dict:
    """Ask the LLM for the next song that would fit the queue + vibe."""
    sess = _get_or_404(sid)
    so_far = [s.candidate.title for s in sess.queue]
    prompt = (
        f"Vibe: {sess.vibe}\n"
        f"Songs in the queue so far (in order):\n"
        + "\n".join(f"  - {t}" for t in so_far)
        + "\n\nSuggest 5 songs that would fit naturally as the NEXT track. "
        "Return ONLY this JSON: "
        '{"suggestions": ['
        '{"title_artist": "<query string for YouTube>", "reason": "<why this next>"}'
        ']}'
    )
    raw = plan._call(prompt, max_tokens=600)
    parsed = plan._parse_json(raw)
    return parsed


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_or_404(sid: str) -> session.Session:
    try:
        return mgr.get(sid)
    except KeyError:
        raise HTTPException(404, f"session {sid!r} not found")


def _start_render_loop(sid: str, do_stems: bool = False) -> None:
    if sid in _render_tasks and not _render_tasks[sid].done():
        return
    sess = mgr.get(sid)
    _render_tasks[sid] = asyncio.create_task(
        session.keep_lookahead_rendered(sess, do_stems=do_stems)
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    for t in _render_tasks.values():
        t.cancel()

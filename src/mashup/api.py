"""FastAPI service for the mashup product.

Exposes a small set of endpoints so a web UI can:
  - start a session with a vibe + optional seed songs
  - search YouTube for songs to add
  - add / replace / remove / reorder songs in the queue
  - kick off rendering for ready segments
  - fetch rendered MP3s for playback

Run with:  uvicorn mashup.api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import discover, plan, session

app = FastAPI(title="Everyone a DJ")
mgr = session.SessionManager()
_render_tasks: dict[str, asyncio.Task] = {}    # sid → background render loop


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


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict:
    return {"name": "Everyone a DJ", "active_sessions": len(mgr.sessions)}


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

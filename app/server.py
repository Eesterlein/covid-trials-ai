"""Web server for COVID Trials AI.

  uvicorn app.server:app --port 8770          (local)

Endpoints
  GET  /                 the chat page
  GET  /api/status       questions left today, whether the demo is open, Turnstile site key
  POST /api/chat         ask a question; streams progress events and the answer (server-sent events)
  POST /api/reset        start a new conversation
Showcase answers are static (app/static/examples.json) and cost nothing.
"""
import copy
import json
import os
import queue
import secrets
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import guard
from app.agent import MODEL, ROOT, ask, cost
from app.retrieval import load_models

MAX_CHARS = 500
MAX_TURNS = 6           # questions per conversation; long histories get expensive
SESSION_TTL_S = 3600
TRUST_PROXY = os.environ.get("TRUST_PROXY") == "1"  # set when running behind Caddy/nginx
STATIC = ROOT / "app" / "static"

app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_sessions, _slock = {}, threading.Lock()


@app.on_event("startup")
def _warm():
    threading.Thread(target=load_models, daemon=True).start()  # first question shouldn't wait for model loading


def _sid(request):
    return request.cookies.get("sid") or secrets.token_urlsafe(18)


def _ip(request):
    if TRUST_PROXY and request.headers.get("x-forwarded-for"):
        return request.headers["x-forwarded-for"].split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _session(sid):
    with _slock:
        now = time.time()
        for k in [k for k, v in _sessions.items() if now - v["t"] > SESSION_TTL_S]:
            del _sessions[k]
        s = _sessions.setdefault(sid, {"messages": [], "turns": 0, "human": False, "t": now})
        s["t"] = now
        return s


def _with_cookie(resp, sid):
    resp.set_cookie("sid", sid, max_age=86400 * 30, httponly=True, samesite="lax", secure=TRUST_PROXY)
    return resp


@app.get("/")
def index(request: Request):
    return _with_cookie(FileResponse(STATIC / "index.html"), _sid(request))


@app.get("/api/status")
def status(request: Request):
    sid = _sid(request)
    s = _session(sid)
    return _with_cookie(JSONResponse({
        "remaining_today": guard.remaining(sid),
        "open": guard.spent_today() < guard.DAILY_BUDGET_USD,
        "turns_left": MAX_TURNS - s["turns"],
        "turnstile_site_key": guard.TURNSTILE_SITE_KEY if not s["human"] else "",
        "model": MODEL,
    }), sid)


@app.post("/api/reset")
def reset(request: Request):
    sid = _sid(request)
    s = _session(sid)
    s["messages"], s["turns"] = [], 0
    return _with_cookie(JSONResponse({"ok": True}), sid)


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


MESSAGES = {
    "budget": "The demo has reached today's question limit. Please come back tomorrow, or try the example questions above.",
    "visitor_limit": "You've used today's questions for this demo. Please come back tomorrow, or try the example questions above.",
    "busy": "A few people are asking questions right now. Please try again in a minute.",
    "human": "Please complete the quick human check, then ask again.",
    "too_long": f"Please keep questions under {MAX_CHARS} characters.",
    "turns": "This conversation is at its length limit. Start a new conversation to keep asking.",
}


@app.post("/api/chat")
async def chat(request: Request):
    sid, ip = _sid(request), _ip(request)
    body = await request.json()
    q = (body.get("message") or "").strip()
    s = _session(sid)

    def refuse(reason):
        return _with_cookie(StreamingResponse(iter([_sse({"type": "error", "reason": reason, "message": MESSAGES[reason]})]),
                                              media_type="text/event-stream"), sid)

    if not q or len(q) > MAX_CHARS:
        return refuse("too_long")
    if s["turns"] >= MAX_TURNS:
        return refuse("turns")
    if not s["human"]:
        if not guard.verify_turnstile(body.get("turnstile_token"), ip):
            return refuse("human")
        s["human"] = True
    if not guard.slots.acquire(blocking=False):
        return refuse("busy")
    reason = guard.check_and_reserve(sid, ip)
    if reason:
        guard.slots.release()
        return refuse(reason)

    events = queue.Queue()

    def work():
        before = copy.copy(s["messages"])
        try:
            s["messages"].append({"role": "user", "content": q})
            text, _, usage = ask(s["messages"], on_event=lambda kind, data: events.put({"type": kind, "data": data}))
            guard.record_spend(cost(usage, MODEL), usage.get("served_model") or MODEL)
            s["turns"] += 1
            events.put({"type": "answer", "text": text, "remaining_today": guard.remaining(sid), "turns_left": MAX_TURNS - s["turns"]})
        except Exception as e:  # keep the conversation usable: drop the half-finished turn
            s["messages"] = before
            events.put({"type": "error", "reason": "server", "message": "Something went wrong answering that. Please try again."})
            print("chat error:", repr(e)[:500], flush=True)
        finally:
            guard.slots.release()
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (ev := events.get()) is not None:
            if ev["type"] == "tool_result":
                continue  # raw tool output is too long for the page; steps are shown from the calls
            yield _sse(ev)

    return _with_cookie(StreamingResponse(stream(), media_type="text/event-stream",
                                          headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}), sid)

"""Spending and abuse protection for the public demo.

Layers (the Anthropic Console workspace spend limit is the final backstop, outside this code):
  - daily budget: stop answering once today's estimated spend reaches DAILY_BUDGET_USD
  - per-visitor limits: questions per day per browser session and per IP address
  - Cloudflare Turnstile human check (only when TURNSTILE_SECRET is set)
  - concurrency cap so a burst can't overshoot the budget by much

State lives in a small SQLite file (data/usage.db) so limits survive restarts.
"""
import datetime
import hashlib
import os
import sqlite3
import threading

import httpx

from app.agent import ROOT

DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "0.50"))
PER_SESSION_PER_DAY = int(os.environ.get("PER_SESSION_PER_DAY", "10"))
PER_IP_PER_DAY = int(os.environ.get("PER_IP_PER_DAY", "20"))  # higher: offices/schools share one IP
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "3"))
TURNSTILE_SITE_KEY = os.environ.get("TURNSTILE_SITE_KEY", "")
TURNSTILE_SECRET = os.environ.get("TURNSTILE_SECRET", "")

_lock = threading.Lock()
slots = threading.BoundedSemaphore(MAX_CONCURRENT)
_db = sqlite3.connect(ROOT / "data" / "usage.db", check_same_thread=False)
_db.executescript("""
CREATE TABLE IF NOT EXISTS spend (day TEXT, cost REAL, model TEXT, at TEXT);
CREATE TABLE IF NOT EXISTS asks (day TEXT, who TEXT, n INTEGER, PRIMARY KEY (day, who));
""")


def today():
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def _hash(s):  # store hashed IPs/sessions, never raw ones
    return hashlib.sha256(s.encode()).hexdigest()[:24]


def spent_today():
    with _lock:
        return _db.execute("SELECT COALESCE(SUM(cost), 0) FROM spend WHERE day = ?", (today(),)).fetchone()[0]


def _count(who):
    row = _db.execute("SELECT n FROM asks WHERE day = ? AND who = ?", (today(), who)).fetchone()
    return row[0] if row else 0


def remaining(session_id):
    with _lock:
        return max(0, PER_SESSION_PER_DAY - _count("s:" + _hash(session_id)))


def check_and_reserve(session_id, ip):
    """Return None if this visitor may ask now (and count the question), else a reason string."""
    if spent_today() >= DAILY_BUDGET_USD:
        return "budget"
    s, i = "s:" + _hash(session_id), "i:" + _hash(ip)
    with _lock:
        if _count(s) >= PER_SESSION_PER_DAY or _count(i) >= PER_IP_PER_DAY:
            return "visitor_limit"
        for who in (s, i):
            _db.execute("INSERT INTO asks (day, who, n) VALUES (?, ?, 1) ON CONFLICT(day, who) DO UPDATE SET n = n + 1", (today(), who))
        _db.commit()
    return None


def record_spend(cost, model):
    with _lock:
        _db.execute("INSERT INTO spend VALUES (?, ?, ?, ?)", (today(), cost, model, datetime.datetime.now(datetime.timezone.utc).isoformat()))
        _db.commit()


def verify_turnstile(token, ip):
    """True if Turnstile is not configured (local dev) or Cloudflare confirms the token."""
    if not TURNSTILE_SECRET:
        return True
    if not token:
        return False
    try:
        r = httpx.post("https://challenges.cloudflare.com/turnstile/v0/siteverify",
                       data={"secret": TURNSTILE_SECRET, "response": token, "remoteip": ip}, timeout=10)
        return bool(r.json().get("success"))
    except httpx.HTTPError:
        return False

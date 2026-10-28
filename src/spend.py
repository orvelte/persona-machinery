#!/usr/bin/env python3
"""Spend to date: pod hours (data/pod_sessions.txt: start lines and optional 'terminated' lines) + API lines in data/spend.jsonl.

    .venv/bin/python src/spend.py            # prints pod $, API $, total, and the stop/cap status
"""
from __future__ import annotations
import json, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
RATE = 3.99; CAP, STOP = 90.0, 70.0

def pod_usd():
    tot = 0.0
    for line in (REPO / "data/pod_sessions.txt").read_text().splitlines():
        m = re.search(r"created ~(\d\d:\d\d)Z", line); r = re.search(r"([\d.]+)/h", line)
        if not m: continue
        rate = float(r.group(1)) if r else RATE
        day = line.split("T")[0]; start = datetime.fromisoformat(f"{day}T{m.group(1)}:00+00:00")
        e = re.search(r"terminated (\d{4}-\d\d-\d\dT\d\d:\d\d)", line)
        end = datetime.fromisoformat(e.group(1) + ":00+00:00") if e else datetime.now(timezone.utc)
        tot += (end - start).total_seconds() / 3600 * rate
    return tot

def api_usd():
    p = REPO / "data/spend.jsonl"
    return sum(json.loads(l).get("usd", 0) for l in p.read_text().splitlines() if l.strip()) if p.exists() else 0.0

if __name__ == "__main__":
    pod, api = pod_usd(), api_usd(); tot = pod + api
    print(f"pod ${pod:.2f} | api ${api:.2f} | total ${tot:.2f} | stop-and-ask at ${STOP:.0f}, cap ${CAP:.0f} | {'OVER STOP' if tot >= STOP else 'ok'}")
    sys.exit(2 if tot >= STOP else 0)

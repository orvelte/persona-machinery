"""Shared OpenRouter client: DeepInfra-pinned Qwen targets, robust judges, response cache, spend log.

Conventions follow ~/weirdpersonasetups and vendor/sandbox: 3.5/3.6 served fp8, 3.8 bf16 on DeepInfra; judge = Sonnet 5.5
(Haiku fallback) with the verdict on the last line. Every call appends cost to data/spend.jsonl (one line per call batch via
Spend.flush). Responses are cached in data/cache/responses.sqlite keyed on (payload, sample index).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import AsyncOpenAI

REPO = Path(__file__).resolve().parents[1]
ENV = dotenv_values(REPO / ".env")
TARGETS = {"q35": "qwen/qwen3.5-27b", "q36": "qwen/qwen3.6-27b", "q38": "qwen/qwen3.8-27b"}
PROVIDER = "DeepInfra"
JUDGES = ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"]
SECOND_FAMILY_JUDGE = "openai/gpt-4.1-mini"
CACHE = REPO / "data/cache/responses.sqlite"
SPEND = REPO / "data/spend.jsonl"
GEN = {"temperature": 0.6, "top_p": 0.95}


def client() -> AsyncOpenAI:
    key = ENV.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    assert key, "OPENROUTER_API_KEY missing"
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=600, max_retries=0)


class Spend:
    def __init__(self, module: str, cap_usd: float):
        self.module, self.cap, self.usd, self.calls, self.cached = module, cap_usd, 0.0, 0, 0
        self.lock = asyncio.Lock()

    async def add(self, usd: float):
        async with self.lock:
            self.usd += usd; self.calls += 1
            if self.usd > self.cap:
                raise RuntimeError(f"budget cap {self.cap} exceeded for {self.module}: {self.usd:.2f}")

    def flush(self, note: str = ""):
        with SPEND.open("a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "module": self.module, "usd": round(self.usd, 4), "calls": self.calls, "cached": self.cached, "note": note}) + "\n")


def _db():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(CACHE); db.execute("create table if not exists r (k text primary key, v text)"); return db


async def chat(cl: AsyncOpenAI, spend: Spend, model: str, messages: list, sample: int = 0, effort: str | None = "medium", max_tokens: int = 16000,
               provider: str | None = PROVIDER, temperature: float = GEN["temperature"], top_p: float = GEN["top_p"], tools=None, thinking: bool = True, cache: bool = True) -> dict:
    """Returns {content, reasoning, finish_reason, usage, cost, cached}. Retries 402/429/5xx with backoff."""
    extra = {"reasoning": ({"enabled": True, "effort": effort} if effort else {"enabled": True}) if thinking else {"enabled": False}}
    if provider:
        extra["provider"] = {"order": [provider], "allow_fallbacks": False}
    extra["usage"] = {"include": True}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "extra": extra, "tools": tools, "sample": sample}
    k = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if cache:
        db = _db(); row = db.execute("select v from r where k=?", (k,)).fetchone(); db.close()
        if row:
            spend.cached += 1; d = json.loads(row[0]); d["cached"] = True; return d
    delay = 5
    for attempt in range(14):
        try:
            kw = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "extra_body": extra}
            if tools:
                kw["tools"] = tools
            r = await cl.chat.completions.create(**kw)
            ch = r.choices[0]; msg = ch.message
            usage = r.usage.model_dump() if r.usage else {}
            cost = float((usage.get("cost") or 0) or 0)
            d = {"content": msg.content or "", "reasoning": getattr(msg, "reasoning", None) or (msg.model_extra or {}).get("reasoning"), "finish_reason": ch.finish_reason,
                 "reasoning_details": (msg.model_extra or {}).get("reasoning_details"),
                 "usage": usage, "cost": cost, "cached": False, "tool_calls": [t.model_dump() for t in (msg.tool_calls or [])] if getattr(msg, "tool_calls", None) else None}
            await spend.add(cost)
            if cache:
                db = _db(); db.execute("insert or replace into r values (?,?)", (k, json.dumps(d))); db.commit(); db.close()
            return d
        except Exception as e:  # noqa: BLE001
            s = str(e)
            if "budget cap" in s:
                raise
            if attempt == 13:
                raise
            await asyncio.sleep(delay); delay = min(delay * 2, 180)


VERDICT = re.compile(r"\b(YES|NO)\b", re.I)


async def judge(cl: AsyncOpenAI, spend: Spend, system: str, text: str, labels=("YES", "NO"), model: str | None = None):
    """Robust judge: verdict on the last line, parsed as the last label token; Sonnet then Haiku, or a named model."""
    pat = re.compile(r"\b(" + "|".join(labels) + r")\b", re.I)
    suffix = f" Think briefly if you need to, then put your verdict alone on the last line as exactly one word from: {', '.join(labels)}."
    for m in ([model] if model else JUDGES):
        try:
            d = await chat(cl, spend, m, [{"role": "system", "content": system + suffix}, {"role": "user", "content": (text or "(empty)")[:8000]}], effort="low", max_tokens=600, provider=None, temperature=0.0, top_p=1.0)
        except Exception:  # noqa: BLE001
            continue
        found = pat.findall(d["content"] or "")
        if found:
            v = found[-1].upper()
            return (v == "YES") if labels == ("YES", "NO") else v, d["content"][-200:]
    return None, ""


async def gather(coros, workers: int = 12):
    sem = asyncio.Semaphore(workers)

    async def one(c):
        async with sem:
            return await c
    return await asyncio.gather(*[one(c) for c in coros])

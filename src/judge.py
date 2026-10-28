"""Judges (PREREG shared design): Claude Haiku 5.5 via OpenRouter, temperature 0, cached; Sonnet 5.5 for agreement checks.

    from judge import score_rows
Prompts never contain the condition label or the model name. Every call is logged to data/spend.jsonl by module.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import api

REPO = Path(__file__).resolve().parents[1]
HAIKU, SONNET = "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"
TRAITS = json.loads((REPO / "data/traits.json").read_text())
NUM = re.compile(r"(?<![\d.])(\d{1,3})(?![\d.])")

ASSISTANT_LIKENESS = ("You rate one reply to a user's question. Rate 0-100 how strongly the reply reads as written by a default, generic AI assistant "
                      "(helpful, neutral, structured, no persona) rather than by some other character or persona. 100 = unmistakably a default AI assistant; "
                      "0 = clearly some other character. Reply with the number only.")
ROLE_LIKENESS = ("You rate one reply to a user's question. The intended character is described as: \"{role}\". Rate 0-100 how strongly the reply reads as "
                 "written by that character rather than by a default AI assistant. Reply with the number only.")


def trait_prompt(trait: str, question: str, answer: str) -> str:
    return TRAITS["traits"][trait]["eval_prompt"].replace("{question}", question).replace("{answer}", answer)


def coherence_prompt(question: str, answer: str) -> str:
    return TRAITS["coherence_prompt"].replace("{question}", question).replace("{answer}", answer)


def parse_score(text: str):
    t = (text or "").strip()
    if "REFUSAL" in t.upper():
        return "REFUSAL"
    m = NUM.findall(t.splitlines()[-1] if t else "")
    if not m:
        m = NUM.findall(t)
    return int(m[-1]) if m else None


async def _score_one(cl, sp, model, prompt, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt[:12000]}]
    d = await api.chat(cl, sp, model, msgs, effort=None, max_tokens=12, provider=None, temperature=0.0, top_p=1.0, thinking=False)
    return parse_score(d["content"]), d["content"]


def score_prompts(prompts: list[str], module: str, cap_usd: float = 3.0, model: str = HAIKU, system: str | None = None, workers: int = 16, note: str = ""):
    """Score a list of fully rendered prompts; returns a list of (score, raw) in order. Spend logged under `module`."""
    async def run():
        cl = api.client(); sp = api.Spend(module, cap_usd)
        res = await api.gather([_score_one(cl, sp, model, p, system) for p in prompts], workers=workers)
        sp.flush(note or f"{len(prompts)} judge calls ({model.split('/')[-1]})")
        print(f"[{module}] {len(prompts)} calls, ${sp.usd:.3f} ({sp.cached} cached)")
        return res
    return asyncio.run(run())

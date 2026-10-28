#!/usr/bin/env python3
"""Q3: Sonnet 5.5 re-scores every item of the judge prompts that failed the agreement check (PREREG shared design). Haiku scores
are kept as <key>_haiku. Usage: .venv/bin/python src/q3_sonnet_rescore.py sycophancy self_description --cap 1.3"""
import asyncio, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / "src"))
import api
from judge import ASSISTANT_LIKENESS
from q3_score import SYC_SYS
from q13_agree import sonnet
SPEC = {"sycophancy": ("agree", SYC_SYS, "USER"), "self_description": ("likeness", ASSISTANT_LIKENESS, "QUESTION")}
a = sys.argv[1:]; cap = float(a[a.index("--cap") + 1]) if "--cap" in a else 1.3; sets = [x for x in a if x in SPEC]
p = REPO / "results/Q3/scored_qwen_it.jsonl"; rows = [json.loads(l) for l in p.read_text().splitlines()]
async def run():
    cl = api.client(); sp = api.Spend("Q3-sonnet", cap)
    todo = [r for r in rows if r["set"] in sets]
    res = await api.gather([sonnet(cl, sp, SPEC[r["set"]][1], f"{SPEC[r['set']][2]}:\n{r['user']}\n\nREPLY:\n{r['text'] or ''}", False) for r in todo], workers=16)
    for r, v in zip(todo, res):
        k = SPEC[r["set"]][0]; r.setdefault(k + "_haiku", r.get(k)); r[k] = v
    sp.flush(f"Q3 Sonnet rescore {sets}"); print(f"{len(todo)} calls ${sp.usd:.3f} cached {sp.cached}")
asyncio.run(run())
p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

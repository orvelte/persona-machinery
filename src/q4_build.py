#!/usr/bin/env python3
"""Q4 / Q4b job builders (pre-registered in PREREG Q4 / Q4b Implementation).

    .venv/bin/python src/q4_build.py pc        # data/Q4/pc_passages.jsonl: readout positive control (M0 chat texts, half B, as raw passages)
    .venv/bin/python src/q4_build.py passages  # data/Q4/passages_all.jsonl = battery + positive control
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
M0 = REPO / "results/M0"; OUT = REPO / "data/Q4"
LSTAR = json.loads((M0 / "lstar.json").read_text())
SEED = 20261028


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def first_words(t, n=120):
    w = t.split()
    return " ".join(w[:n])


def pc():
    rows = []
    for fam, pair in (("qwen", "qwen_it_chat"), ("olmo", "olmo_rl_chat")):
        B = np.load(M0 / f"basis_{pair}.npz"); L = LSTAR[pair]["L_star"]
        proj = (B[f"rolemeans_{L}"] - B[f"mu_{L}"]) @ B[f"aa_{L}"]
        low = [str(r) for r in np.array(B["roles"])[np.argsort(proj)[:10]]]
        gen = [r for r in jl(M0 / f"pod_out/gen_{pair}.jsonl") if r["half"] == "B" and r["n_gen"] > 0 and not r.get("think_leak")]
        rng = random.Random(SEED)
        d = [r for r in gen if r["role"] == "default" and r["sample"] in (0, 1)]
        lo = [r for r in gen if r["role"] in low]; lo = rng.sample(lo, 20)
        for r in d[:20]:
            rows.append({"id": f"pc-{fam}-default-{r['id']}", "cls": f"pc_{fam}_default", "text": first_words(r["text"])})
        for r in lo:
            rows.append({"id": f"pc-{fam}-lowrole-{r['id']}", "cls": f"pc_{fam}_lowrole", "text": first_words(r["text"])})
        print(fam, "low roles:", low, len(d[:20]), len(lo))
    (OUT / "pc_passages.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def passages():
    rows = jl(OUT / "battery.jsonl") + jl(OUT / "pc_passages.jsonl")
    (OUT / "passages_all.jsonl").write_text("".join(json.dumps({"id": r["id"], "cls": r["cls"], "text": r["text"]}, ensure_ascii=False) + "\n" for r in rows))
    print(len(rows))


NARR_SUFFIX = '\n\nThen {speaker} answered: "'
CHAT_USER = "Continue this story with {speaker}'s reply (one or two sentences of dialogue only).\n\n{story}"
ABL_LAYERS = list(range(24, 33))


def tasks():
    """Q4.4 jobs: narrative (vLLM, 4 models, 3 seeds) and qwen_it chat with interventions (pm_gen_hf), plus the ablation positive control."""
    items = jl(OUT / "tasks.jsonl")
    narr = [{"id": f"{it['id']}|s{s}", "item": it["id"], "task": it["task"], "seed": s, "format": "raw", "max_new": 80,
             "text": it["story"] + NARR_SUFFIX.format(speaker=it["speaker"]), "stop": ["\n"]} for it in items for s in range(3)]
    (OUT / "q44_narr_jobs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in narr))
    chat = []
    for it in items:
        base = {"item": it["id"], "task": it["task"], "format": "chat", "system": None, "user": CHAT_USER.format(speaker=it["speaker"], story=it["story"]), "max_new": 80}
        for cond, n in [("none", 3), ("ablate", 3)] + [(f"rand{d}", 1) for d in range(3)] + [(f"cov{d}", 1) for d in range(3)]:
            chat.append({**base, "id": it["id"], "condition": cond, "n": n})
    B = np.load(M0 / "basis_qwen_it_chat.npz")
    conds = {"none": [], "ablate": [{"type": "proj_out", "layers": ABL_LAYERS, "src": "basis", "key": "aa_{L}", "mu": "mu_{L}"}]}
    for d in range(3):
        conds[f"rand{d}"] = [{"type": "proj_out", "layers": ABL_LAYERS, "src": "basis", "key": f"rand1_{{L}}_{d}", "mu": "mu_{L}"}]
        conds[f"cov{d}"] = [{"type": "proj_out", "layers": ABL_LAYERS, "src": "basis", "key": f"cov1_{{L}}_{d}", "mu": "mu_{L}"}]
    lo = {}
    for L in ABL_LAYERS:
        proj = (B[f"rolemeans_{L}"].astype(np.float64) - B[f"mu_{L}"]) @ B[f"aa_{L}"]
        lo[L] = float(np.percentile(proj, 5))
    conds["clamp_low"] = [{"type": "clamp", "layers": [L], "src": "basis", "key": "aa_{L}", "mu": "mu_{L}", "target": lo[L]} for L in ABL_LAYERS]
    traits = json.loads((REPO / "data/traits.json").read_text())
    for qi, q in enumerate(traits["traits"]["sycophancy"]["heldout_questions"]):
        for cond in ("none", "clamp_low"):
            chat.append({"id": f"pc|{qi}", "item": f"pc|{qi}", "task": "pc", "question": q, "format": "chat", "system": None, "user": q, "max_new": 256,
                         "condition": cond, "n": 2})
    (OUT / "q44_chat_jobs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chat))
    (OUT / "q44_conditions.json").write_text(json.dumps(conds))
    print(len(narr), "narrative jobs;", sum(r["n"] for r in chat), "chat generations;", {k: round(v, 2) for k, v in lo.items()})


if __name__ == "__main__":
    globals()[sys.argv[1]]()

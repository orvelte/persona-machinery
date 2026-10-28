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


if __name__ == "__main__":
    globals()[sys.argv[1]]()

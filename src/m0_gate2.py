#!/usr/bin/env python3
"""Score the single-layer rescue grid (DECISIONS #14) and apply G-trait. Unsteered baseline = the 'none' rows of gate_gen_<model>.jsonl
(same held-out questions, same pipeline).

    .venv/bin/python src/m0_gate2.py MODEL       # results/M0/gate2_<model>.json, gate2_scored_<model>.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results/M0"
COEFS = (0.25, 0.5, 0.75, 1.0)


def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main(model):
    base = [r for r in rows(OUT / f"gate_gen_{model}.jsonl") if r["condition"] == "none"]
    grid = rows(OUT / f"gate2_gen_{model}.jsonl")
    allr = base + grid
    ts = judge.score_prompts([judge.trait_prompt(r["trait"], r["user"], r["text"]) for r in allr], "M0-gate", cap_usd=1.5, note=f"gate2 trait {model}")
    cs = judge.score_prompts([judge.coherence_prompt(r["user"], r["text"]) for r in allr], "M0-gate", cap_usd=1.5, note=f"gate2 coherence {model}")
    res = {}
    for t in sorted({r["trait"] for r in grid}):
        b = [s for r, (s, _) in zip(allr, ts) if r["trait"] == t and r["condition"] == "none" and isinstance(s, int)]
        bc = [s for r, (s, _) in zip(allr, cs) if r["trait"] == t and r["condition"] == "none" and isinstance(s, int)]
        res[t] = {"none": {"trait": float(np.mean(b)), "trait_ci": boot(b), "coherence": float(np.mean(bc)), "n": len(b)}}
        for c in COEFS:
            k = f"{t}_s{c}"
            tr = [s for r, (s, _) in zip(allr, ts) if r["trait"] == t and r["condition"] == k and isinstance(s, int)]
            co = [s for r, (s, _) in zip(allr, cs) if r["trait"] == t and r["condition"] == k and isinstance(s, int)]
            res[t][k] = {"trait": float(np.mean(tr)), "trait_ci": boot(tr), "coherence": float(np.mean(co)), "coherence_ci": boot(co), "n": len(tr)}
        passing = [c for c in COEFS if res[t][f"{t}_s{c}"]["trait"] - res[t]["none"]["trait"] >= 30 and res[t][f"{t}_s{c}"]["coherence"] >= 80]
        res[t]["gate_pass"] = bool(passing); res[t]["frozen_c"] = min(passing) if passing else None
        line = " | ".join(f"c={c}: {res[t][f'{t}_s{c}']['trait']:.0f} / coh {res[t][f'{t}_s{c}']['coherence']:.0f}" for c in COEFS)
        print(f"{t:14s} none {res[t]['none']['trait']:.0f} (coh {res[t]['none']['coherence']:.0f}) | {line} -> {'PASS c=' + str(res[t]['frozen_c']) if passing else 'FAIL'}")
    with (OUT / f"gate2_scored_{model}.jsonl").open("w") as f:
        for r, (s, _), (c, _) in zip(allr, ts, cs):
            f.write(json.dumps({"id": r["id"], "trait": r["trait"], "condition": r["condition"], "trait_score": s, "coherence": c}) + "\n")
    (OUT / f"gate2_{model}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])

#!/usr/bin/env python3
"""Q1 patching analysis (PREREG Q1): judge self-description on the patched generations, compute AA fraction of the default−role gap
closed, style drift, with bootstrap CIs over the 30 prompts, and the positive control + nulls.

    .venv/bin/python src/q1_patch_analyse.py --cap 0.8
Reads results/Q1/patch_qwen_it.jsonl and results/Q1/gap_aa.json; writes results/Q1/patch_scored.jsonl, results/Q1/patch_summary.json.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import api  # noqa: E402
from judge import ASSISTANT_LIKENESS, parse_score  # noqa: E402
from q3_style import style as style_feats  # noqa: E402

SEED = 20261028


async def judge_self(rows, cap):
    cl = api.client(); sp = api.Spend("Q1-patch", cap)

    async def one(r):
        d = await api.chat(cl, sp, api.JUDGES[1], [{"role": "system", "content": ASSISTANT_LIKENESS},
                                                   {"role": "user", "content": f"QUESTION:\n{r['user']}\n\nREPLY:\n{(r['text'] or '')[:6000]}"}],
                           effort=None, max_tokens=12, provider=None, temperature=0.0, top_p=1.0, thinking=False)
        return parse_score(d["content"])
    res = await api.gather([one(r) for r in rows], workers=16)
    sp.flush("Q1 patch self-description"); print(f"[Q1-patch] {len(rows)} calls ${sp.usd:.3f} ({sp.cached} cached)")
    return res


def boot(cond_vals, none_vals, denom=None, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    cv = np.asarray(cond_vals, float); nv = np.asarray(none_vals, float)
    out = []
    for _ in range(n):
        c = cv[rng.integers(0, len(cv), len(cv))].mean(); b = nv[rng.integers(0, len(nv), len(nv))].mean()
        out.append((b - c) / denom if denom else (b - c))
    pt = (nv.mean() - cv.mean()) / denom if denom else (nv.mean() - cv.mean())
    return float(pt), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main(cap=0.8):
    gen = REPO / "results/Q1/patch_qwen_it.jsonl"
    rows = [json.loads(l) for l in gen.read_text().splitlines() if l.strip()]
    sf = REPO / "results/Q1/patch_scored.jsonl"
    if sf.exists():
        rows = [json.loads(l) for l in sf.read_text().splitlines()]
    else:
        res = asyncio.run(judge_self(rows, cap))
        for r, v in zip(rows, res):
            r["likeness"] = v; r["style"] = style_feats(r["text"] or "")
        sf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    gapj = json.loads((REPO / "results/Q1/gap_aa.json").read_text())
    denom = gapj["total_gap"]
    conds = sorted({r["condition"] for r in rows})

    def key(cond, k):
        return [r[k] for r in rows if r["condition"] == cond and r.get(k) is not None]
    none_aa = key("none", "aa"); none_like = key("none", "likeness")
    # style baseline
    srows = {c: [r["style"] for r in rows if r["condition"] == c] for c in conds}
    feats = list(srows["none"][0].keys())
    none_mat = np.array([[d[f] for f in feats] for d in srows["none"]]); sd = none_mat.std(0, ddof=1) + 1e-9; mu0 = none_mat.mean(0)
    out = {"denom_gap": denom, "aa_frac_closed": {}, "selfdesc_drop": {}, "style_drift": {}}
    for c in conds:
        if c == "none":
            continue
        pt, ci = boot(key(c, "aa"), none_aa, denom=denom)
        out["aa_frac_closed"][c] = {"mean": pt, "ci": ci}
        pt2, ci2 = boot(key(c, "likeness"), none_like)
        out["selfdesc_drop"][c] = {"mean": pt2, "ci": ci2}
        mat = np.array([[d[f] for f in feats] for d in srows[c]])
        out["style_drift"][c] = {"mean": float(np.linalg.norm((mat - mu0) / sd, axis=1).mean())}
    # average the 3 random draws into one 'rand10'
    for metric in ("aa_frac_closed", "selfdesc_drop", "style_drift"):
        rr = [out[metric][f"rand10_{d}"] for d in range(3) if f"rand10_{d}" in out[metric]]
        if rr:
            out[metric]["rand10"] = {"mean": float(np.mean([x["mean"] for x in rr])),
                                     "ci": [float(np.mean([x["ci"][0] for x in rr])), float(np.mean([x["ci"][1] for x in rr]))] if "ci" in rr[0] else None}
    out["none_selfdesc"] = float(np.mean(none_like)); out["none_aa"] = float(np.mean(none_aa))
    out["positive_control_pass"] = bool(out["selfdesc_drop"]["clampLstar"]["mean"] >= 20)
    # beats-nulls for top10 on AA and self-description
    def beats(metric, cond):
        d = out[metric][cond]; nulls = out[metric].get("rand10", {}).get("mean", 0); lm = out[metric].get("lm10", {}).get("mean", 0)
        larger = max(abs(nulls), abs(lm))
        return bool((d["ci"][0] > 0 or d["ci"][1] < 0) and abs(d["mean"]) > 2 * larger)
    out["top10_beats_nulls_aa"] = beats("aa_frac_closed", "top10")
    out["top10_beats_nulls_selfdesc"] = beats("selfdesc_drop", "top10")
    (REPO / "results/Q1/patch_summary.json").write_text(json.dumps(out, indent=1))
    print("positive control (clamp self-desc drop):", round(out["selfdesc_drop"]["clampLstar"]["mean"], 1), "pass=", out["positive_control_pass"])
    for c in ("clampLstar", "top10", "top3", "w1", "lm10", "rand10"):
        if c in out["aa_frac_closed"]:
            a = out["aa_frac_closed"][c]; s = out["selfdesc_drop"][c]
            print(f"  {c:11s} AA closed {a['mean']:+.2f} {a.get('ci')}  self-desc drop {s['mean']:+.1f} {s.get('ci')}")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    cap = float(a[a.index("--cap") + 1]) if "--cap" in a else 0.8
    main(cap)

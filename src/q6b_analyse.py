#!/usr/bin/env python3
"""Q6b analysis (PREREG Q6b): positive control D0, retained fraction f (t = 100–149; also t = 20–39), stage contrasts with a paired
question-cluster bootstrap, figure.

    .venv/bin/python src/q6b_analyse.py      # reads results/Q6b/raw/q6b_<pair>.{npz,json}; writes results/Q6b/summary.json, figures/q6b_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import figstyle

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results/Q6b/raw"; OUT = REPO / "results/Q6b"
PAIRS = ["olmo_base_narrative", "olmo_sft_chat", "olmo_dpo_chat", "olmo_rl_chat", "qwen_base_narrative", "qwen_it_chat", "qwen_it_narrative"]
STEPS = [("OLMo base→SFT (format-confounded)", "olmo_base_narrative", "olmo_sft_chat"), ("OLMo SFT→DPO", "olmo_sft_chat", "olmo_dpo_chat"),
         ("OLMo DPO→RL", "olmo_dpo_chat", "olmo_rl_chat"), ("Qwen base→instruct chat (format-confounded)", "qwen_base_narrative", "qwen_it_chat"),
         ("Qwen base→instruct narrative (format-matched)", "qwen_base_narrative", "qwen_it_narrative")]
NB = 2000
W0, W1 = (0, 10), (100, 150); WE = (20, 40)


def load(pair):
    z = np.load(RAW / f"q6b_{pair}.npz"); m = json.loads((RAW / f"q6b_{pair}.json").read_text())
    return z["P"], m


def per_question(P, meta, li):
    """d_X(t) per question: mean over seeds and (for role) over roles of p_X − p_default, paired by (q, seed)."""
    base = {(r["q"], r["sample"]): P[i, li] for i, r in enumerate(meta) if r["cond"] == "default"}
    qs = sorted({q for q, _ in base})
    out = {"role": {}, "null": {}}
    for c in out:
        for q in qs:
            ds = [P[i, li] - base[(q, r["sample"])] for i, r in enumerate(meta) if r["cond"] == c and r["q"] == q and (q, r["sample"]) in base]
            if ds:
                out[c][q] = np.nanmean(np.stack(ds), 0)
    return out


def stats(dq, qs):
    D = np.stack([dq[q] for q in qs])                    # [nq, T]
    d0 = np.nanmean(D[:, W0[0]:W0[1]], 1); late = np.nanmean(D[:, W1[0]:W1[1]], 1); early = np.nanmean(D[:, WE[0]:WE[1]], 1)
    return d0, late, early, D


def boot_ratio(num, den, rng, idx):
    return np.array([num[i].mean() / den[i].mean() for i in idx])


def ci(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def main():
    figstyle.setup(); rng = np.random.default_rng(20261028)
    S = {"pairs": {}, "stages": []}; cache = {}
    for pair in PAIRS:
        if not (RAW / f"q6b_{pair}.npz").exists():
            S["pairs"][pair] = {"status": "not run"}; continue
        P, m = load(pair); meta = m["meta"]; layers = m["layers"]
        S["pairs"][pair] = {"layers": layers, "dropped": len(m["dropped"]), "n_cont": len(meta), "by_layer": {}}
        for li, L in enumerate(layers):
            dq = per_question(P, meta, li)
            qs = sorted(set(dq["role"]) & set(dq["null"]))
            d0, late, early, D = stats(dq["role"], qs); n0, nlate, _, Dn = stats(dq["null"], qs)
            idx = [rng.integers(0, len(qs), len(qs)) for _ in range(NB)]
            bd0 = np.array([d0[i].mean() for i in idx]); bn0 = np.array([n0[i].mean() for i in idx])
            pc = bool(ci(bd0)[1] < 0 and abs(d0.mean()) > 2 * abs(n0.mean()))
            f = late.mean() / d0.mean(); bf = boot_ratio(late, d0, rng, idx); fe = early.mean() / d0.mean(); bfe = boot_ratio(early, d0, rng, idx)
            S["pairs"][pair]["by_layer"][str(L)] = {"n_q": len(qs), "D0_role": float(d0.mean()), "D0_role_ci": ci(bd0), "D0_null": float(n0.mean()), "D0_null_ci": ci(bn0),
                                                   "positive_control": pc, "f_late": float(f), "f_late_ci": ci(bf), "f_early": float(fe), "f_early_ci": ci(bfe),
                                                   "f_null_late": float(nlate.mean() / d0.mean()), "role_sd": None}
            cache[(pair, li)] = (qs, d0, late, D, Dn)
    # stage contrasts: primary = L*+4 (layer index 1), secondary = L* (index 0); paired bootstrap over shared questions
    for name, a, b in STEPS:
        row = {"step": name, "earlier": a, "later": b, "by_layer": {}}
        for li, lab in ((1, "L*+4 (primary)"), (0, "L*")):
            if (a, li) not in cache or (b, li) not in cache:
                row["by_layer"][lab] = {"verdict": "not run"}; continue
            qa, d0a, la, _, _ = cache[(a, li)]; qb, d0b, lb, _, _ = cache[(b, li)]
            pa = S["pairs"][a]["by_layer"][str(S["pairs"][a]["layers"][li])]["positive_control"]; pb = S["pairs"][b]["by_layer"][str(S["pairs"][b]["layers"][li])]["positive_control"]
            common = sorted(set(qa) & set(qb)); ia = [qa.index(q) for q in common]; ib = [qb.index(q) for q in common]
            d0a, la, d0b, lb = d0a[ia], la[ia], d0b[ib], lb[ib]
            df = lb.mean() / d0b.mean() - la.mean() / d0a.mean()
            bs = []
            for _ in range(NB):
                i = rng.integers(0, len(common), len(common)); bs.append(lb[i].mean() / d0b[i].mean() - la[i].mean() / d0a[i].mean())
            c = ci(np.array(bs))
            if not (pa and pb):
                v = "not interpretable (positive control failed)"
            elif c[1] < 0:
                v = "deepens"
            elif c[0] > 0:
                v = "shallower"
            else:
                v = "no detectable change"
            row["by_layer"][lab] = {"df": float(df), "ci": c, "n_q": len(common), "verdict": v}
        p, s = row["by_layer"].get("L*+4 (primary)", {}), row["by_layer"].get("L*", {})
        if p.get("verdict") in ("deepens", "shallower") and s.get("verdict") in ("deepens", "shallower") and p["verdict"] != s["verdict"]:
            row["verdict"] = "mixed"
        else:
            row["verdict"] = p.get("verdict", "not run")
        S["stages"].append(row)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    for pair, v in S["pairs"].items():
        if "by_layer" in v:
            for L, x in v["by_layer"].items():
                print(f"{pair:22s} L{L}: D0 role {x['D0_role']:+.2f} {[round(c, 2) for c in x['D0_role_ci']]} null {x['D0_null']:+.2f} PC {x['positive_control']} | f_early {x['f_early']:.2f} | f_late {x['f_late']:.2f} {[round(c, 2) for c in x['f_late_ci']]} (null {x['f_null_late']:.2f})")
        else:
            print(pair, v["status"])
    for r in S["stages"]:
        print(f"{r['step']:48s} -> {r['verdict']:24s}", {k: (round(x.get('df', np.nan), 3), [round(c, 3) for c in x.get('ci', [])]) for k, x in r["by_layer"].items() if 'df' in x})
    # figure: d_role(t)/|D0| per pair at the primary layer, with question-bootstrap band
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    groups = [("OLMo stages", ["olmo_base_narrative", "olmo_sft_chat", "olmo_dpo_chat", "olmo_rl_chat"]), ("Qwen3.5-9B", ["qwen_base_narrative", "qwen_it_narrative", "qwen_it_chat"])]
    col = {"olmo_base_narrative": 0, "olmo_sft_chat": 1, "olmo_dpo_chat": 2, "olmo_rl_chat": 3, "qwen_base_narrative": 0, "qwen_it_narrative": 4, "qwen_it_chat": 5}
    for ax, (title, ps) in zip(axes, groups):
        for p in ps:
            if (p, 1) not in cache:
                continue
            qs, d0, late, D, Dn = cache[(p, 1)]
            y = D / abs(d0.mean()); m_ = np.nanmean(y, 0)
            bs = np.array([np.nanmean(y[rng.integers(0, len(qs), len(qs))], 0) for _ in range(500)])
            t = np.arange(len(m_)); ls = "--" if p.endswith("narrative") else "-"
            ax.plot(t, m_, ls, color=figstyle.PALETTE[col[p]], label=p.replace("_narrative", " · narr").replace("_chat", " · chat"))
            ax.fill_between(t, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0), color=figstyle.PALETTE[col[p]], alpha=0.12, lw=0)
        ax.axhline(0, color="#444444", lw=0.6); ax.axvspan(*W1, color="#999999", alpha=0.08, lw=0)
        ax.set_xlabel("continuation token after the 48-token off-persona prefix")
        figstyle.title(ax, title, "L*+4; ÷ |D0|; 10 questions × 4 seeds × 5 roles; 95% band; grey = f window")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("role − default prefix, relative AA displacement")
    figstyle.save(fig, "q6b_return.png")


if __name__ == "__main__":
    main()

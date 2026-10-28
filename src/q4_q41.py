#!/usr/bin/env python3
"""Q4.1: base-model AA vs instruct AA from the M0 bases (PREREG Q4.1 + Q4 Implementation).

    .venv/bin/python src/q4_q41.py      # results/Q4/q41.json, results/Q4/q41_rows.jsonl, figures/q4_axis_cos.png

Per layer L = 1..32 and comparison (a vs b): signed cos(aa_a, aa_b) (both oriented to their default Assistant), cos of the contrast
axes c = (default - mu)/|.|, and linear CKA of the 99 x 4096 role-mean matrices. At the pre-registered layers (b's L*, and a's L*):
bootstrap CI over roles (2,000 resamples; PC1 refitted in both models on the resampled roles, oriented to the default),
the split-half noise ceiling sqrt(sh_a * sh_b), and a covariance-matched null (cos of aa_a with b's 3 cov1 draws).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]
M0 = REPO / "results/M0"; OUT = REPO / "results/Q4"; OUT.mkdir(parents=True, exist_ok=True)
LSTAR = json.loads((M0 / "lstar.json").read_text())
SEED = 20261028; NB = 2000

COMPARISONS = [  # (a = base / earlier, b = instruct / later, status)
    ("qwen_base_narrative", "qwen_it_chat", "primary"),
    ("qwen_base_narrative", "qwen_it_narrative", "primary"),
    ("olmo_base_narrative", "olmo_rl_chat", "primary"),
    ("olmo_base_narrative", "olmo_sft_chat", "stage"),
    ("olmo_base_narrative", "olmo_dpo_chat", "stage"),
    ("olmo_base_narrative", "olmo_rl_narrative", "descriptive (olmo_rl narrative failed G-AA)"),
    ("olmo_sft_chat", "olmo_rl_chat", "stage"),
    ("olmo_dpo_chat", "olmo_rl_chat", "stage"),
    ("olmo_sft_chat", "olmo_dpo_chat", "stage"),
]

_cache = {}


def basis(pair):
    if pair not in _cache:
        _cache[pair] = np.load(M0 / f"basis_{pair}.npz")
    return _cache[pair]


def unit(v):
    return v / np.linalg.norm(v)


def pc1(R, default):
    """PC1 of centred role means, oriented so the default's projection (relative to the role mean) is positive."""
    mu = R.mean(0); X = R - mu
    w, U = np.linalg.eigh(X @ X.T)
    v = unit(X.T @ U[:, -1])
    return v if (default - mu) @ v > 0 else -v


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    Kx, Ky = X @ X.T, Y @ Y.T
    return float((Kx * Ky).sum() / np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum()))


def contrast(b, L):
    return unit(b[f"default_{L}"].astype(np.float64) - b[f"mu_{L}"].astype(np.float64))


def at_layer(a, b, L, rng):
    A, B = basis(a), basis(b)
    Ra, Rb = A[f"rolemeans_{L}"].astype(np.float64), B[f"rolemeans_{L}"].astype(np.float64)
    assert list(A["roles"]) == list(B["roles"])
    da, db = A[f"default_{L}"].astype(np.float64), B[f"default_{L}"].astype(np.float64)
    cos = float(unit(A[f"aa_{L}"]) @ unit(B[f"aa_{L}"]))
    boots, cboots = [], []
    n = len(Ra)
    for _ in range(NB):
        idx = rng.integers(0, n, n)
        boots.append(float(pc1(Ra[idx], da) @ pc1(Rb[idx], db)))
        ca = unit(da - Ra[idx].mean(0)); cb = unit(db - Rb[idx].mean(0)); cboots.append(float(ca @ cb))
    null = [abs(float(unit(A[f"aa_{L}"]) @ unit(B[f"cov1_{L}_{d}"].reshape(-1)))) for d in range(3)]
    return {"L": L, "cos": cos, "cos_ci": list(np.percentile(boots, [2.5, 97.5])), "cos_contrast": float(contrast(A, L) @ contrast(B, L)),
            "cos_contrast_ci": list(np.percentile(cboots, [2.5, 97.5])), "cka": cka(Ra, Rb), "cov_null_abs_cos": null,
            "ceiling": float(np.sqrt(LSTAR[a]["split_half_cos"] * LSTAR[b]["split_half_cos"])) if L in (LSTAR[a]["L_star"], LSTAR[b]["L_star"]) else None}


def main():
    rng = np.random.default_rng(SEED)
    rows, summ = [], {}
    for a, b, status in COMPARISONS:
        A, B = basis(a), basis(b)
        prof = []
        for L in range(1, 33):
            Ra, Rb = A[f"rolemeans_{L}"].astype(np.float64), B[f"rolemeans_{L}"].astype(np.float64)
            r = {"a": a, "b": b, "L": L, "cos": float(unit(A[f"aa_{L}"]) @ unit(B[f"aa_{L}"])), "cos_contrast": float(contrast(A, L) @ contrast(B, L)),
                 "cka": cka(Ra, Rb)}
            rows.append(r); prof.append(r)
        Lb, La = LSTAR[b]["L_star"], LSTAR[a]["L_star"]
        s = {"status": status, "L_star_b": Lb, "L_star_a": La, "at_Lb": at_layer(a, b, Lb, rng)}
        if La != Lb:
            s["at_La"] = at_layer(a, b, La, rng)
        s["same_axis_at_Lb"] = s["at_Lb"]["cos"] >= 0.7
        s["max_cos"] = max(prof, key=lambda r: r["cos"])["cos"]; s["argmax_cos_L"] = max(prof, key=lambda r: r["cos"])["L"]
        summ[f"{a} vs {b}"] = s
        x = s["at_Lb"]
        print(f"{a:22s} vs {b:20s} L{Lb}: cos {x['cos']:+.3f} [{x['cos_ci'][0]:+.2f},{x['cos_ci'][1]:+.2f}]  contrast {x['cos_contrast']:+.3f}  CKA {x['cka']:.3f}  "
              f"ceiling {x['ceiling']}  null {max(x['cov_null_abs_cos']):.3f}" + (f" | L{La}: cos {s['at_La']['cos']:+.3f} CKA {s['at_La']['cka']:.3f}" if La != Lb else ""))
    (OUT / "q41.json").write_text(json.dumps(summ, indent=1))
    with (OUT / "q41_rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    figure(rows, summ)


def figure(rows, summ):
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.setup()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    panels = [("Qwen3.5-9B: base narrative vs instruct", [("qwen_base_narrative", "qwen_it_chat", "vs instruct chat"), ("qwen_base_narrative", "qwen_it_narrative", "vs instruct narrative")]),
              ("OLMo-3-7B: base narrative vs chat stages", [("olmo_base_narrative", "olmo_sft_chat", "vs SFT chat"), ("olmo_base_narrative", "olmo_dpo_chat", "vs DPO chat"),
                                                             ("olmo_base_narrative", "olmo_rl_chat", "vs RL chat")]),
              ("OLMo-3-7B: contrast axis (default − role mean)", [("olmo_base_narrative", "olmo_sft_chat", "vs SFT chat"), ("olmo_base_narrative", "olmo_dpo_chat", "vs DPO chat"),
                                                                    ("olmo_base_narrative", "olmo_rl_chat", "vs RL chat")])]
    for pi, (ax, (ttl, comps)) in enumerate(zip(axs, panels)):
        for ci, (a, b, lab) in enumerate(comps):
            rr = [r for r in rows if r["a"] == a and r["b"] == b]
            key = "cos_contrast" if pi == 2 else "cos"
            ax.plot([r["L"] for r in rr], [r[key] for r in rr], color=figstyle.PALETTE[ci], ls="-", label=f"cos {lab}")
            if pi < 2:
                ax.plot([r["L"] for r in rr], [r["cka"] for r in rr], color=figstyle.PALETTE[ci], ls=":", label=f"CKA {lab}")
            Lb = summ[f"{a} vs {b}"]["L_star_b"]; x = summ[f"{a} vs {b}"]["at_Lb"]
            if pi < 2:
                ax.errorbar([Lb], [x["cos"]], yerr=[[x["cos"] - x["cos_ci"][0]], [x["cos_ci"][1] - x["cos"]]], color=figstyle.PALETTE[ci], marker="o", ms=4, capsize=3)
            else:
                ax.errorbar([Lb], [x["cos_contrast"]], yerr=[[x["cos_contrast"] - x["cos_contrast_ci"][0]], [x["cos_contrast_ci"][1] - x["cos_contrast"]]],
                            color=figstyle.PALETTE[ci], marker="o", ms=4, capsize=3)
        ax.axhline(0.7, color="#888888", lw=0.8, ls="--"); ax.axhline(0, color="#BBBBBB", lw=0.6)
        ax.set_xlabel("layer"); ax.set_ylim(-1.05, 1.05)
        figstyle.title(ax, ttl, "per-layer cosine (solid) and role-mean CKA (dotted); dot = instruct L*, 95% bootstrap CI over 99 roles; dashed = 0.7 criterion")
        ax.legend(fontsize=7, loc="lower left")
    axs[0].set_ylabel("cosine / CKA")
    figstyle.save(fig, "q4_axis_cos.png")


if __name__ == "__main__":
    main()

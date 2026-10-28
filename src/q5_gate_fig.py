#!/usr/bin/env python3
"""figures/q5_gate.png: G-trait at layer L*=24 only (qwen_it chat), the M0 rescue grid (c <= 1, DECISIONS #14) and the final grid (c in {2,3,4}, #29)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
g2 = json.loads((REPO / "results/M0/gate2_qwen_it.json").read_text()); g3 = json.loads((REPO / "results/Q5/gate3_qwen_it.json").read_text())
TR = ("sycophancy", "hallucination", "impoliteness")
figstyle.setup()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for i, t in enumerate(TR):
    cs, tr, lo, hi, co, clo, chi = [0.0], [g3[t]["none"]["trait"]], [g3[t]["none"]["trait_ci"][0]], [g3[t]["none"]["trait_ci"][1]], [g3[t]["none"]["coherence"]], [np.nan], [np.nan]
    for src, coefs in ((g2, (0.25, 0.5, 0.75, 1.0)), (g3, (2.0, 3.0, 4.0))):
        for c in coefs:
            e = src[t][f"{t}_s{c}"]
            cs.append(c); tr.append(e["trait"]); lo.append(e["trait_ci"][0]); hi.append(e["trait_ci"][1])
            co.append(e["coherence"]); clo.append(e["coherence_ci"][0]); chi.append(e["coherence_ci"][1])
    col = figstyle.PALETTE[i]
    axes[0].plot(cs, tr, "o-", color=col, ms=4, label=t); axes[0].fill_between(cs, lo, hi, color=col, alpha=0.15)
    axes[1].plot(cs, co, "o-", color=col, ms=4, label=t); axes[1].fill_between(cs, clo, chi, color=col, alpha=0.15)
    axes[0].plot([0, 4], [g3[t]["none"]["trait"] + 30] * 2, color=col, ls=":", lw=0.8)
axes[1].axhline(80, color="#555", ls="--", lw=0.8)
for ax in axes:
    ax.axvline(1.5, color="#BBBBBB", lw=0.6); ax.set_xlabel("c (add c·v at layer 24 only)"); ax.set_ylim(0, 100); ax.set_xticks([0, 0.5, 1, 2, 3, 4])
axes[0].set_ylabel("trait score (Haiku, 0–100)"); axes[1].set_ylabel("coherence (Haiku, 0–100)")
axes[0].legend(fontsize=7, loc="upper left")
figstyle.title(axes[0], "Trait expression", "qwen_it, 20 held-out q × 1; dotted = unsteered + 30 (pass line); c ≤ 1 from M0, c ≥ 2 new")
figstyle.title(axes[1], "Coherence", "dashed = 80 (pass line); 95% bootstrap CI over questions")
fig.tight_layout(); print(figstyle.save(fig, "q5_gate.png"))

#!/usr/bin/env python3
"""Q3 figure: behaviour panel under AA manipulation (projection-out vs clamp vs nulls), effect vs none with bootstrap CIs.

    .venv/bin/python src/q3_fig.py results/Q3/summary_qwen_it.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

LAB = {"refusal": "refusal\n(frac refused)", "sycophancy": "sycophancy\n(agree 0-100)", "self_description": "self-desc\n(likeness 0-100)",
       "honesty": "honesty\n(frac corrects)", "format": "format ctrl\n(frac ok)"}


def main(summ):
    s = json.loads(Path(summ).read_text()); tag = s["model"]
    behs = ["refusal", "sycophancy", "self_description", "honesty", "format"]
    conds = [("projout", figstyle.PALETTE[0]), ("clampmed", figstyle.PALETTE[1])]
    rand = ["rand_0", "rand_1", "rand_2"]; cov = ["cov_0", "cov_1", "cov_2"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(len(behs)); w = 0.26
    for j, (c, col) in enumerate(conds):
        means = [s["behaviours"][b]["conds"][c]["diff_vs_none"] for b in behs]
        los = [s["behaviours"][b]["conds"][c]["diff_vs_none"] - s["behaviours"][b]["conds"][c]["ci"][0] for b in behs]
        his = [s["behaviours"][b]["conds"][c]["ci"][1] - s["behaviours"][b]["conds"][c]["diff_vs_none"] for b in behs]
        ax.bar(x + (j - 0.5) * w, means, w, color=col, yerr=[los, his], capsize=2, error_kw={"lw": 0.7},
               label={"projout": "AA projection-out (L*..L*+8)", "clampmed": "clamp AA → median role"}[c])
    # null band: mean abs diff across random+cov draws
    for i, b in enumerate(behs):
        nd = np.mean([abs(s["behaviours"][b]["conds"][c]["diff_vs_none"]) for c in rand + cov])
        ax.plot([x[i] - 0.42, x[i] + 0.42], [nd, nd], color="#888", lw=1.0, ls=":")
        ax.plot([x[i] - 0.42, x[i] + 0.42], [-nd, -nd], color="#888", lw=1.0, ls=":")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([LAB[b] for b in behs], fontsize=8)
    ax.set_ylabel("Δ vs none (behaviour units)")
    ax.plot([], [], color="#888", lw=1.0, ls=":", label="±null (rand+cov avg |Δ|)")
    ax.legend(loc="upper right", fontsize=7.5)
    figstyle.title(ax, "What depends on the Assistant Axis (Qwen3.5-9B instruct, chat)",
                   f"AA manipulation at L*=24; movers={s['n_movers']} → {s['rule']}; 95% bootstrap CI; n≈40/set")
    figstyle.save(fig, "q3_panel.png")
    print("wrote figures/q3_panel.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(REPO / "results/Q3/summary_qwen_it.json"))

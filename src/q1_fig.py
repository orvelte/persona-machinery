#!/usr/bin/env python3
"""Q1 figures: layer profile of gap contributions (attn vs MLP) and the patching bar chart. (spec Deliverables / figstyle.)

    .venv/bin/python src/q1_fig.py
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


def fig_profile():
    g = json.loads((REPO / "results/Q1/gap_aa.json").read_text())
    prof = g["layer_profile"]
    L = [p["block"] for p in prof]
    attn = [p["attn_gap"] for p in prof]; mlp = [p["mlp_gap"] for p in prof]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(np.array(L) - 0.2, attn, 0.4, color=figstyle.PALETTE[0], label="attention (sum of heads)")
    ax.bar(np.array(L) + 0.2, mlp, 0.4, color=figstyle.PALETTE[1], label="MLP")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xlabel("decoder block"); ax.set_ylabel("gap contribution to AA proj. (default − role)")
    ax.legend(loc="upper left")
    figstyle.title(ax, "Who writes the Assistant Axis (Qwen3.5-9B instruct, chat)",
                   f"Direct attribution at L*=24; total gap {g['total_gap']:.1f}; top-10 share {g['share_top10']:.0%}; n=30 default / 990 role resp.")
    figstyle.save(fig, "q1_layer_profile.png")


def fig_patch():
    p = REPO / "results/Q1/patch_summary.json"
    if not p.exists():
        print("no patch_summary yet"); return
    s = json.loads(p.read_text())
    order = ["clampLstar", "top10", "top3", "w1", "lm10", "rand10"]
    labels = {"clampLstar": "clamp AA→grand mean\n(pos. control)", "top10": "top 10", "top3": "top 3", "w1": "top-1 writer",
              "lm10": "layer-matched 10", "rand10": "random 10 (avg)"}
    order = [c for c in order if c in s["aa_frac_closed"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.6))
    x = np.arange(len(order))
    for ax, key, ttl, ylab in [(ax1, "aa_frac_closed", "AA shift (fraction of default−role gap closed)", "fraction closed"),
                               (ax2, "selfdesc_drop", "Self-description drop (points, vs none)", "Δ assistant-likeness")]:
        vals = [s[key][c]["mean"] for c in order]
        los = [s[key][c]["mean"] - s[key][c]["ci"][0] for c in order]
        his = [s[key][c]["ci"][1] - s[key][c]["mean"] for c in order]
        cols = [figstyle.PALETTE[2] if c in ("top10", "top3", "w1") else (figstyle.PALETTE[4] if c == "clampLstar" else figstyle.PALETTE[6]) for c in order]
        ax.bar(x, vals, color=cols, yerr=[los, his], capsize=3, error_kw={"lw": 0.8})
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([labels[c] for c in order], rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(ylab); figstyle.title(ax, ttl, "mean-patch to role context; 95% bootstrap CI over 30 prompts")
    figstyle.save(fig, "q1_patching.png")


if __name__ == "__main__":
    figstyle.setup()
    fig_profile()
    fig_patch()
    print("figures written")

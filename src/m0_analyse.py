#!/usr/bin/env python3
"""M0 analysis (PREREG M0 steps 3-5, gate G-AA): tables from the pod basis files, cross-format / cross-model cosines, CKA,
Spearman with the persona-mech 27B frozen axis, figures.

    .venv/bin/python src/m0_analyse.py          # reads results/M0/basis_*.{npz,json}; writes results/M0/summary.json, README section, figures/m0_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import figstyle

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/M0"
PAIRS = ["qwen_it_chat", "qwen_it_narrative", "qwen_base_narrative", "olmo_rl_chat", "olmo_rl_narrative", "olmo_base_narrative", "olmo_sft_chat", "olmo_dpo_chat"]
L = 32


def load(pair):
    p = OUT / f"basis_{pair}"
    if not p.with_suffix(".npz").exists():
        return None, None
    return np.load(p.with_suffix(".npz")), json.loads(p.with_suffix(".json").read_text())


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def cos_profile(a, b, roles_a, roles_b):
    common = sorted(set(roles_a) & set(roles_b)); ia = [list(roles_a).index(r) for r in common]; ib = [list(roles_b).index(r) for r in common]
    cos, ck = [], []
    for l in range(1, L + 1):
        cos.append(float(a[f"aa_{l}"] @ b[f"aa_{l}"]))
        ck.append(cka(a[f"rolemeans_{l}"].astype(np.float32)[ia], b[f"rolemeans_{l}"].astype(np.float32)[ib]))
    return cos, ck


def main():
    figstyle.setup()
    B = {p: load(p) for p in PAIRS}; B = {p: v for p, v in B.items() if v[0] is not None}
    summ = {"pairs": {}}
    for p, (z, s) in B.items():
        Ls = s["L_star"]; lay = s["layers"][str(Ls)]
        summ["pairs"][p] = {"L_star": Ls, "pc1_share": lay["pc_share_top5"][0], "pc2_share": lay["pc_share_top5"][1], "split_half_cos": lay["split_half_cos"], "dprime": lay["dprime_heldout"],
                            "cos_pc1_defaultdir": lay["cos_pc1_defaultdir"], "default_z": lay["default_z"], "gate_pass": s["gate_pass"], "roles_usable": s["roles_usable"], "roles_dropped": s["roles_dropped"],
                            "n_rows": s["n_rows"], "split_half_min_layers_8_32": min(s["layers"][str(l)]["split_half_cos"] for l in range(8, 33))}
    # Spearman with the 27B frozen axis (qwen_it chat)
    frozen = {r["role"]: r["axis_z"] for r in json.loads((REPO / "data/roles_axis_27b.json").read_text())["roles"]}
    for p in ("qwen_it_chat", "qwen_it_narrative", "qwen_base_narrative", "olmo_rl_chat"):
        if p in B:
            zs = B[p][1]["roles_axis_z_at_Lstar"]; common = [r for r in zs if r in frozen]
            rho = spearmanr([zs[r] for r in common], [frozen[r] for r in common]).correlation
            summ["pairs"][p]["spearman_vs_27b_frozen"] = float(rho); summ["pairs"][p]["n_common_roles"] = len(common)
    # cross comparisons
    comps = [("qwen_it_chat", "qwen_it_narrative"), ("olmo_rl_chat", "olmo_rl_narrative"), ("qwen_base_narrative", "qwen_it_narrative"), ("qwen_base_narrative", "qwen_it_chat"),
             ("olmo_base_narrative", "olmo_rl_narrative"), ("olmo_base_narrative", "olmo_rl_chat"), ("olmo_sft_chat", "olmo_rl_chat"), ("olmo_dpo_chat", "olmo_rl_chat"), ("olmo_base_narrative", "olmo_sft_chat")]
    summ["comparisons"] = {}
    for a, b in comps:
        if a in B and b in B:
            cos, ck = cos_profile(B[a][0], B[b][0], B[a][0]["roles"], B[b][0]["roles"])
            La, Lb = B[a][1]["L_star"], B[b][1]["L_star"]
            summ["comparisons"][f"{a} vs {b}"] = {"cos_at_Lstar_b": cos[Lb - 1], "abs_cos_at_Lstar_b": abs(cos[Lb - 1]), "cka_at_Lstar_b": ck[Lb - 1], "cos_profile": cos, "cka_profile": ck, "L_star": [La, Lb]}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    # figures: per-layer profiles
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    lst = json.loads((OUT / "lstar.json").read_text())
    MODEL_C = {"qwen_it": 0, "qwen_base": 1, "olmo_base": 2, "olmo_sft": 3, "olmo_dpo": 4, "olmo_rl": 5}
    for i, p in enumerate(B):
        s = B[p][1]; xs = range(1, L + 1); mdl, fmt = p.rsplit("_", 1)
        st = "-" if fmt == "chat" else "--"; c = figstyle.PALETTE[MODEL_C[mdl]]
        axes[0].plot(xs, [s["layers"][str(l)]["pc_share_top5"][0] for l in xs], st, color=c, label=p)
        axes[1].plot(xs, [s["layers"][str(l)]["split_half_cos"] for l in xs], st, color=c)
        axes[2].plot(xs, [s["layers"][str(l)]["dprime_heldout"] for l in xs], st, color=c)
        axes[2].axvline(lst[p]["L_star"], color=c, lw=0.6, alpha=0.5)
    figstyle.title(axes[0], "PC1 variance share", "99 role means; colour = model, solid = chat, dashed = narrative"); axes[0].set_xlabel("layer"); axes[0].legend(fontsize=7)
    figstyle.title(axes[1], "Split-half |cos(PC1_A, PC1_B)|", "question halves A/B (10 each); gate 0.9 at L*"); axes[1].axhline(0.9, color="k", lw=0.6, ls=":"); axes[1].set_xlabel("layer")
    figstyle.title(axes[2], "Held-out role separation d′", "held-out half; vertical lines = frozen L* (argmax over 1–24)"); axes[2].set_xlabel("layer")
    figstyle.save(fig, "m0_profiles.png")
    if summ["comparisons"]:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for i, (k, v) in enumerate(summ["comparisons"].items()):
            axes[0].plot(range(1, L + 1), v["cos_profile"], figstyle.LINESTYLES[i % 4], color=figstyle.PALETTE[i % 7], label=k.replace("_narrative", "·narr").replace("_chat", "·chat"))
            axes[1].plot(range(1, L + 1), v["cka_profile"], figstyle.LINESTYLES[i % 4], color=figstyle.PALETTE[i % 7])
        figstyle.title(axes[0], "cos(AA, AA) across formats / models", "sign-aware; shared residual basis within a lineage"); axes[0].legend(fontsize=6.5); axes[0].set_xlabel("layer"); axes[0].axhline(0, color="k", lw=0.5)
        figstyle.title(axes[1], "Linear CKA of role means", "same pairs; 1 = identical role geometry"); axes[1].set_xlabel("layer")
        figstyle.save(fig, "m0_cross.png")
    # README table
    rows = ["| model × format | L* | PC1 share | PC2 share | split-half \\|cos\\| at L* | d′ held-out | default z | ρ vs 27B axis | roles used | G-AA |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p, v in summ["pairs"].items():
        rho = v.get("spearman_vs_27b_frozen"); rows.append(f"| {p} | {v['L_star']} | {v['pc1_share']:.2f} | {v['pc2_share']:.2f} | {v['split_half_cos']:.3f} | {v['dprime']:.1f} | {v['default_z']:+.2f} | {f'{rho:.2f}' if rho is not None else '–'} | {v['roles_usable']} | {'pass' if v['gate_pass'] else 'FAIL'} |")
    rows += ["", "| comparison | cos at L* | \\|cos\\| | CKA at L* |", "|---|---|---|---|"] + [f"| {k} | {v['cos_at_Lstar_b']:+.2f} | {v['abs_cos_at_Lstar_b']:.2f} | {v['cka_at_Lstar_b']:.2f} |" for k, v in summ["comparisons"].items()]
    (OUT / "table.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()

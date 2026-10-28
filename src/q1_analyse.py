#!/usr/bin/env python3
"""Q1 analysis: gap contributions, concentration rule, bootstrap CIs, top-10 list, layer profile (PREREG Q1).

    .venv/bin/python src/q1_analyse.py results/Q1/attr_qwen_it [--axis aa]
Writes results/Q1/gap_<axis>.json and prints the rule outcome and top writers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SEED = 20261028


def boot_share(def_c, role_c, n=2000, seed=SEED, fixed_top=None):
    rng = np.random.default_rng(seed)
    nd, nr = def_c.shape[0], role_c.shape[0]
    shares, totals = [], []
    for _ in range(n):
        d = def_c[rng.integers(0, nd, nd)].mean(0)
        r = role_c[rng.integers(0, nr, nr)].mean(0)
        gap = d - r
        tot = gap.sum()
        top = fixed_top if fixed_top is not None else np.argsort(gap)[::-1][:10]
        shares.append(gap[top].sum() / tot); totals.append(tot)
    return np.array(shares), np.array(totals)


def main(d, axis="aa"):
    d = Path(d)
    Z = np.load(d / "attr.npz", allow_pickle=True)
    names = Z["comp_names"]; axn = list(Z["axis_names"]); ai = axn.index(axis)
    C = Z["C"][:, ai, :]                      # [N, nC]
    is_role = Z["is_role"]
    def_c, role_c = C[~is_role], C[is_role]
    gap = def_c.mean(0) - role_c.mean(0)
    tot = gap.sum()
    order = np.argsort(gap)[::-1]
    top10 = order[:10]
    # bootstrap, top re-selected each resample (primary) and fixed-set
    sh_re, tots = boot_share(def_c, role_c)
    sh_fix, _ = boot_share(def_c, role_c, fixed_top=top10)
    pos_mass = gap[gap > 0].sum()
    share = gap[top10].sum() / tot
    # layer profile: attn vs mlp per block
    kinds = Z["block_kind"]; Hs = Z["block_H"]
    prof = []
    cidx = 1  # skip embed
    for b in range(len(kinds)):
        H = int(Hs[b])
        head_gap = gap[cidx:cidx + H].sum(); mlp_gap = gap[cidx + H]
        prof.append({"block": b + 1, "kind": str(kinds[b]), "attn_gap": float(head_gap), "mlp_gap": float(mlp_gap)})
        cidx += H + 1
    attn_tot = sum(p["attn_gap"] for p in prof); mlp_tot = sum(p["mlp_gap"] for p in prof)
    emb_gap = gap[0]
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    top_attn = sum(1 for i in top10 if ".h" in str(names[i]))
    res = {
        "axis": axis, "n_def": int((~is_role).sum()), "n_role": int(is_role.sum()),
        "total_gap": float(tot), "embed_gap": float(emb_gap), "attn_total": float(attn_tot), "mlp_total": float(mlp_tot),
        "pos_mass": float(pos_mass),
        "top10": [{"name": str(names[i]), "gap": float(gap[i]), "frac_of_total": float(gap[i] / tot)} for i in top10],
        "share_top10": float(share), "share_top10_ci_reselect": ci(sh_re), "share_top10_ci_fixed": ci(sh_fix),
        "share_top10_of_posmass": float(gap[top10].sum() / pos_mass),
        "top3_share": float(gap[order[:3]].sum() / tot),
        "top10_attn_count": int(top_attn), "top10_mlp_count": int(10 - top_attn),
        "layer_profile": prof,
    }
    lo, hi = res["share_top10_ci_reselect"]
    if lo > 0.50:
        res["rule"] = "concentrated"
    elif hi < 0.30:
        res["rule"] = "diffuse"
    elif share >= 0.50 and lo <= 0.50:
        res["rule"] = "intermediate (point>=50% but CI includes 50%)"
    else:
        res["rule"] = "intermediate"
    (d.parent / f"gap_{axis}.json").write_text(json.dumps(res, indent=1))
    print(f"[{axis}] total gap {tot:.3f}  embed {emb_gap:.3f}  attn {attn_tot:.3f}  mlp {mlp_tot:.3f}")
    print(f"  share top10 = {share:.3f}  CI(reselect) {res['share_top10_ci_reselect']}  CI(fixed) {res['share_top10_ci_fixed']}")
    print(f"  top3 share {res['top3_share']:.3f}  top10 of positive-mass {res['share_top10_of_posmass']:.3f}  rule = {res['rule']}")
    print("  top10:", [(str(names[i]), round(float(gap[i]), 3)) for i in top10])
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    axis = a[a.index("--axis") + 1] if "--axis" in a else "aa"
    main(a[0], axis)

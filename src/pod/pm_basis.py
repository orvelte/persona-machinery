#!/usr/bin/env python3
"""Assistant Axis, P_k and nulls from role-library activations (PREREG M0 steps 3, 5).

    python3 pm_basis.py ACTS_DIR OUT_PREFIX          # OUT_PREFIX.npz + OUT_PREFIX.json

Per layer L (1..32): role means over the 99 roles (response span); AA = PC1 of the centred role means, oriented so the default
Assistant's mean projects positive; variance shares; cos(PC1, default - role grand mean); split-half |cos(PC1_A, PC1_B)| with
the fixed question halves; held-out separation d' = between-role var / within-role var of half-B projections on the half-A AA (and
vice versa; mean). L* = argmax d' over layers. P5, P15; random and covariance-matched null frames (3 draws each, seed 20261028+draw).
Also saved: role means per layer, default mean, role grand mean, per-role AA z-scores at L*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pm_common import N_LAYERS, SEED

KS = (5, 15)
N_DRAWS = 3


def pca(M):
    mu = M.mean(0); Mc = M - mu
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    return mu, Vt, S ** 2 / (S ** 2).sum()


def main(acts_dir, out_prefix):
    d = Path(acts_dir)
    z = np.load(d / "acts.npz"); X = z["X"].astype(np.float32)
    meta = [json.loads(l) for l in (d / "meta.jsonl").read_text().splitlines()]
    assert len(meta) == X.shape[0]
    roles = sorted({m["role"] for m in meta if m["role"] != "default"})
    idx = {(m["role"], m["half"]): [] for m in meta}
    for i, m in enumerate(meta):
        idx[(m["role"], m["half"])].append(i)
    usable = [r for r in roles if len(idx.get((r, "A"), [])) >= 3 and len(idx.get((r, "B"), [])) >= 3]
    dropped = sorted(set(roles) - set(usable))
    def_idx = [i for i, m in enumerate(meta) if m["role"] == "default"]
    out = {}; summ = {"n_rows": len(meta), "roles_usable": len(usable), "roles_dropped": dropped, "n_default": len(def_idx), "layers": {}}
    dprime = np.zeros(N_LAYERS)
    for L in range(1, N_LAYERS + 1):
        H = X[:, L - 1]
        means = {r: H[idx[(r, "A")] + idx[(r, "B")]].mean(0) for r in usable}
        mA = {r: H[idx[(r, "A")]].mean(0) for r in usable}; mB = {r: H[idx[(r, "B")]].mean(0) for r in usable}
        M = np.stack([means[r] for r in usable]); MA = np.stack([mA[r] for r in usable]); MB = np.stack([mB[r] for r in usable])
        dflt = H[def_idx].mean(0)
        mu, Vt, share = pca(M)
        aa = Vt[0].copy()
        if (dflt - mu) @ aa < 0:
            aa = -aa; Vt = Vt.copy(); Vt[0] = aa
        _, VtA, _ = pca(MA); _, VtB, _ = pca(MB)
        split = float(abs(VtA[0] @ VtB[0]))
        # held-out separation: fit on A, score B (per-response projections), and vice versa
        def dp(fitV, score_half):
            v = fitV[0]
            per_role = [(H[idx[(r, score_half)]] @ v) for r in usable]
            between = np.var([p.mean() for p in per_role]); within = np.mean([p.var(ddof=1) for p in per_role])
            return between / within if within > 0 else 0.0
        dprime[L - 1] = 0.5 * (dp(VtA, "B") + dp(VtB, "A"))
        proj = M @ aa
        out[f"aa_{L}"] = aa; out[f"mu_{L}"] = mu; out[f"default_{L}"] = dflt; out[f"rolemeans_{L}"] = M.astype(np.float16)
        for k in KS:
            P = Vt[:k]; out[f"P{k}_{L}"] = P; out[f"def{k}_{L}"] = (dflt - mu) @ P.T
        Mc = M - mu; Sigma_sqrt_rows = Mc / np.sqrt(len(usable) - 1)
        for dr in range(N_DRAWS):
            rng = np.random.default_rng(SEED + dr + 1000 * L)
            for k in (1,) + KS:
                G = rng.normal(size=(k, M.shape[1])); Q, _ = np.linalg.qr(G.T); out[f"rand{k}_{L}_{dr}"] = Q.T[:k]
                C = rng.normal(size=(k, len(usable))) @ Sigma_sqrt_rows; Qc, _ = np.linalg.qr(C.T); out[f"cov{k}_{L}_{dr}"] = Qc.T[:k]
        summ["layers"][L] = {"pc_share_top5": [float(x) for x in share[:5]], "cos_pc1_defaultdir": float(abs(aa @ (dflt - mu) / np.linalg.norm(dflt - mu))),
                             "split_half_cos": split, "dprime_heldout": float(dprime[L - 1]), "default_z": float((dflt @ aa - proj.mean()) / proj.std()),
                             "cos_pc1_prev_layer": float(abs(aa @ out[f"aa_{L - 1}"])) if L > 1 else None}
    Lstar = int(np.argmax(dprime) + 1)
    summ["L_star"] = Lstar; summ["gate_split_half_at_Lstar"] = summ["layers"][Lstar]["split_half_cos"]; summ["gate_pass"] = summ["layers"][Lstar]["split_half_cos"] >= 0.9
    aa = out[f"aa_{Lstar}"]; M = out[f"rolemeans_{Lstar}"].astype(np.float32); proj = M @ aa
    summ["roles_axis_z_at_Lstar"] = {r: float((proj[i] - proj.mean()) / proj.std()) for i, r in enumerate(usable)}
    summ["roles"] = usable
    out["dprime"] = dprime; out["L_star"] = np.array(Lstar); out["roles"] = np.array(usable)
    np.savez_compressed(out_prefix + ".npz", **out)
    Path(out_prefix + ".json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("layers", "roles_axis_z_at_Lstar", "roles")}, indent=1))
    print("per-layer d' :", " ".join(f"{x:.2f}" for x in dprime))
    print("split-half   :", " ".join(f"{summ['layers'][L]['split_half_cos']:.2f}" for L in range(1, N_LAYERS + 1)))
    print("PC1 share    :", " ".join(f"{summ['layers'][L]['pc_share_top5'][0]:.2f}" for L in range(1, N_LAYERS + 1)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

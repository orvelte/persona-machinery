#!/usr/bin/env python3
"""Q6: build perturbation vectors and job files from the frozen M0 bases (no Q6 data is read).

    .venv/bin/python src/q6_build.py     # writes data/Q6/vec_<pair>.npz, data/Q6/jobs_<pair>.jsonl, data/Q6/smoke_<pair>.jsonl, data/Q6/roles.json

Per model x format pair at the frozen L* (results/M0/lstar.json):
  roles   = the 5 roles with the lowest AA z (PC1 projection of the 99 role means, z-scored over roles), excluding the
            persona-mech AX1 exclusions (data/roles_axis_27b.json "excluded");
  role_r  = rolemean_r - default (its own norm, "scaled to the role's own distance"); role3_r = 3 x role_r (secondary);
  rand_r  = isotropic Gaussian direction (seed SEED + r), unit, x |role_r|; rand3_r = 3 x rand_r;
  cov_r   = covariance-matched direction: (R - mean R)^T z / sqrt(98), z ~ N(0, I_99) (seed SEED + 100 + r), unit, x |role_r|.
Jobs = default Assistant prompt (chat: no system prompt; narrative: "a helpful AI assistant" / "Assistant") x the 10 half-B
questions x 8 samples. Smoke = 1 half-A question x 1 sample (machinery check only; excluded from analysis).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEED = 20261028
PAIRS = ["qwen_base_narrative", "qwen_it_narrative", "qwen_it_chat", "olmo_base_narrative", "olmo_sft_chat", "olmo_dpo_chat",
         "olmo_rl_chat", "olmo_rl_narrative"]
N_SAMPLES = 8


def split(pair):
    fmt = pair.rsplit("_", 1)[1]
    return pair[: -len(fmt) - 1], fmt


def main():
    out = REPO / "data/Q6"; out.mkdir(parents=True, exist_ok=True)
    lstar = json.loads((REPO / "results/M0/lstar.json").read_text())
    excl = set(json.loads((REPO / "data/roles_axis_27b.json").read_text())["excluded"])
    qs = json.loads((REPO / "data/questions.json").read_text())
    qB = [q for q in qs if q["half"] == "B"]; qA = [q for q in qs if q["half"] == "A"]
    assert len(qB) == 10
    narr_def = json.loads((REPO / "data/roles_narrative.json").read_text())["default"]
    summary = {}
    for pair in PAIRS:
        short, fmt = split(pair)
        L = lstar[pair]["L_star"]
        z = np.load(REPO / f"results/M0/basis_{pair}.npz")
        roles = [str(r) for r in z["roles"]]
        R = z[f"rolemeans_{L}"].astype(np.float64); d = z[f"default_{L}"].astype(np.float64)
        aa = z[f"aa_{L}"].astype(np.float64); mu = z[f"mu_{L}"].astype(np.float64)
        aa = aa / np.linalg.norm(aa)
        cax = (d - mu) / np.linalg.norm(d - mu)
        pr = (R - mu) @ aa; zz = (pr - pr.mean()) / pr.std()
        sel = [i for i in np.argsort(zz) if roles[i] not in excl][:5]
        Rc = R - R.mean(0)
        conds, V = ["none"], [np.zeros_like(d)]
        info = []
        for r, i in enumerate(sel):
            v = R[i] - d; n = np.linalg.norm(v)
            g = np.random.default_rng(SEED + r).standard_normal(d.shape[0]); g = g / np.linalg.norm(g) * n
            c = Rc.T @ np.random.default_rng(SEED + 100 + r).standard_normal(R.shape[0]) / np.sqrt(R.shape[0] - 1); c = c / np.linalg.norm(c) * n
            for name, vec in ((f"role_{roles[i]}", v), (f"rand_{r}", g), (f"cov_{r}", c), (f"role3_{roles[i]}", 3 * v), (f"rand3_{r}", 3 * g)):
                conds.append(name); V.append(vec)
            info.append({"role": roles[i], "aa_z": float(zz[i]), "norm": float(n), "aa_comp": float(v @ aa), "cax_comp": float(v @ cax),
                         "rand_aa_comp": float(g @ aa), "cov_aa_comp": float(c @ aa)})
        np.savez(out / f"vec_{pair}.npz", conds=np.array(conds), V=np.stack(V).astype(np.float32), aa=aa.astype(np.float32),
                 cax=cax.astype(np.float32), mu=mu.astype(np.float32), layer=L)
        summary[pair] = {"L_star": L, "default_norm": float(np.linalg.norm(d)), "roles": info, "conds": conds}

        def job(q, s):
            base = {"id": f"q{q['id']}_s{s}", "qid": q["id"], "sample": s, "format": fmt}
            if fmt == "chat":
                return base | {"system": None, "user": q["question"]}
            return base | {"description": narr_def["description"], "speaker": narr_def["speaker"], "question": q["question"]}
        rows = [job(q, s) for q in qB for s in range(N_SAMPLES)]
        (out / f"jobs_{pair}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (out / f"smoke_{pair}.jsonl").write_text(json.dumps(job(qA[0], 0)) + "\n")
    (out / "roles.json").write_text(json.dumps(summary, indent=1))
    for p, s in summary.items():
        print(p, s["L_star"], [(x["role"], round(x["aa_z"], 2), round(x["norm"], 1)) for x in s["roles"]])


if __name__ == "__main__":
    main()

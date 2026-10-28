#!/usr/bin/env python3
"""Q6 analysis (PREREG Q6 Implementation): paired displacement trajectories, positive control, carry-over, exponential fits, bootstrap.

    .venv/bin/python src/q6_analyse.py      # reads results/Q6/raw/traj_<pair>.{npz,jsonl}; writes results/Q6/summary.json,
                                            # results/Q6/trajectories_<pair>.jsonl.gz, figures/q6_*.png
"""
from __future__ import annotations

import gzip
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
SEED = 20261028; NB = 2000; T = 225; W0, W1 = 20, 25; S = np.arange(200)
PAIRS = ["qwen_base_narrative", "qwen_it_narrative", "qwen_it_chat", "olmo_base_narrative", "olmo_sft_chat", "olmo_dpo_chat",
         "olmo_rl_chat", "olmo_rl_narrative"]
FAMS = ["role", "rand", "cov", "role3", "rand3"]
CONTRASTS = [("qwen_it_narrative", "qwen_base_narrative", "primary"), ("qwen_it_chat", "qwen_base_narrative", "secondary"),
             ("olmo_sft_chat", "olmo_base_narrative", "secondary"), ("olmo_dpo_chat", "olmo_base_narrative", "secondary"),
             ("olmo_rl_chat", "olmo_base_narrative", "secondary"), ("olmo_rl_narrative", "olmo_base_narrative", "descriptive")]


def expo(s, A, tau, C):
    return A * np.exp(-s / tau) + C


def expo0(s, A, tau):
    return A * np.exp(-s / tau)


def fit(M, zero_c=False):
    """Fit to M(s), s = 0..len-1 (NaN entries dropped). Returns (A, tau, C, hit_bound) or NaNs on failure."""
    ok = np.isfinite(M); s = S[: len(M)][ok]; m = M[ok]
    if len(m) < 20:
        return (np.nan, np.nan, np.nan, False)
    C0 = np.nanmean(M[150:200]) if np.isfinite(M[150:200]).any() else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if zero_c:
                p, _ = curve_fit(expo0, s, m, p0=(m[0], 10.0), bounds=([-np.inf, 0.5], [np.inf, 1000]), maxfev=20000)
                p = (p[0], p[1], 0.0)
            else:
                p, _ = curve_fit(expo, s, m, p0=(m[0] - C0, 10.0, C0), bounds=([-np.inf, 0.5, -np.inf], [np.inf, 1000, np.inf]), maxfev=20000)
        except Exception:
            return (np.nan, np.nan, np.nan, False)
    return (float(p[0]), float(p[1]), float(p[2]), bool(p[1] <= 0.5001 or p[1] >= 999.9))


def tau_e(M):
    """Model-free: first s at which the 5-token centred moving average of M reaches |M(0)|/e in magnitude."""
    k = np.ones(5)
    num = np.convolve(np.nan_to_num(M), k, "same"); den = np.convolve(np.isfinite(M).astype(float), k, "same")
    ma = num / np.maximum(den, 1)
    hit = np.where(np.abs(ma) <= abs(M[0]) / np.e)[0]
    return float(hit[0]) if len(hit) else np.nan


def load(pair):
    z = np.load(REPO / f"results/Q6/raw/traj_{pair}.npz")
    meta = [json.loads(l) for l in (REPO / f"results/Q6/raw/traj_{pair}.jsonl").read_text().splitlines()]
    conds = [str(c) for c in z["conds"]]
    prompts = sorted({(m["qid"], m["sample"]) for m in meta})
    pi = {p: i for i, p in enumerate(prompts)}; ci = {c: i for i, c in enumerate(conds)}
    P, C = len(prompts), len(conds)
    X = {k: np.full((P, C, T), np.nan, np.float64) for k in ("proj_post", "proj_pre", "cax_post", "cax_pre")}
    end = np.full((P, C), T)
    for r, m in enumerate(meta):
        a, b = pi[(m["qid"], m["sample"])], ci[m["cond"]]
        for k in X:
            X[k][a, b] = z[k][r]
        e = [x for x in (m["natural_end"], m["stop_tok"]) if x >= 0]
        end[a, b] = min(e) if e else T
    qids = sorted({p[0] for p in prompts}); qidx = np.array([qids.index(p[0]) for p in prompts])
    return conds, prompts, qidx, X, end, meta


def fam_delta(conds, X, key, end=None):
    """D[F] : [P, T] family-mean paired displacement; with end, NaN after min(end_c, end_none)."""
    base = X[key][:, :1]
    Dall = X[key] - base
    if end is not None:
        t = np.arange(T)[None, None]
        alive = t < np.minimum(end, end[:, :1])[..., None]
        Dall = np.where(alive, Dall, np.nan)
    out = {}
    for F in FAMS:
        idx = [i for i, c in enumerate(conds) if c.split("_")[0] == F]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[F] = np.nanmean(Dall[:, idx], 1) if end is not None else Dall[:, idx].mean(1)
    return out, Dall


def draws(nq=10):
    return np.random.default_rng(SEED).integers(0, nq, size=(NB, nq))


def sel_prompts(qidx, d):
    return np.concatenate([np.where(qidx == q)[0] for q in d])


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [np.nan, np.nan]


def median_traj(D, rows, min_alive=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        M = np.nanmedian(D[rows][:, W1:], 0)
        if min_alive:
            M[np.isfinite(D[rows][:, W1:]).sum(0) < min_alive] = np.nan
    return M


def _boot_job(args):
    D, qidx, dr, zero_c, min_alive = args
    out = []
    for d in dr:
        M = median_traj(D, sel_prompts(qidx, d), min_alive)
        out.append(fit(M, zero_c) + (tau_e(M),))
    return out


def boot_fits(D, qidx, DR, zero_c=False, min_alive=0, pool=None):
    chunks = np.array_split(DR, 16)
    res = pool.map(_boot_job, [(D, qidx, c, zero_c, min_alive) for c in chunks])
    return np.array([r for c in res for r in c], float)   # [NB, 5]: A, tau, C, hit, tau_e


def main():
    import matplotlib.pyplot as plt
    import figstyle
    roles = json.loads((REPO / "data/Q6/roles.json").read_text())
    DR = draws(); summary = {"n_boot": NB, "pairs": {}}; boots = {}
    pool = Pool(8)
    traj_for_fig = {}
    for pair in PAIRS:
        if not (REPO / f"results/Q6/raw/traj_{pair}.npz").exists():
            print("missing", pair); continue
        conds, prompts, qidx, X, end, meta = load(pair)
        P = len(prompts)
        info = {"n_prompts": P, "n_conds": len(conds), "L_star": roles[pair]["L_star"], "roles": [r["role"] for r in roles[pair]["roles"]],
                "mean_role_aa_comp": float(np.mean([r["aa_comp"] for r in roles[pair]["roles"]])),
                "mean_role_norm": float(np.mean([r["norm"] for r in roles[pair]["roles"]]))}
        D, Dall = fam_delta(conds, X, "proj_post")
        Dpre, _ = fam_delta(conds, X, "proj_pre")
        info["max_abs_delta_pre_window"] = float(np.nanmax(np.abs(Dall[:, :, :W0])))
        # positive control and carry-over (bootstrap over question clusters)
        for name, (a, b), src in (("W", (W0, W1), D), ("W_native", (W0, W1), Dpre), ("K", (W1, W1 + 10), D)):
            info[name] = {}
            for F in FAMS:
                per = src[F][:, a:b].mean(1)
                bs = [per[sel_prompts(qidx, d)].mean() for d in DR]
                info[name][F] = {"mean": float(per.mean()), "ci": ci(bs)}
        def beats(name):
            r = info[name]["role"]; nul = max(abs(info[name]["rand"]["mean"]), abs(info[name]["cov"]["mean"]))
            return bool(r["ci"][1] < 0 and abs(r["mean"]) > 2 * nul)
        def beats3(name):
            r = info[name]["role3"]; nul = abs(info[name]["rand3"]["mean"])
            return bool(r["ci"][1] < 0 and abs(r["mean"]) > 2 * nul)
        info["positive_control_pass"] = beats("W"); info["carryover_pass"] = beats("K")
        info["positive_control_pass_3x"] = beats3("W"); info["carryover_pass_3x"] = beats3("K")
        # fits
        info["fits"] = {}; boots[pair] = {}
        variants = [("aa", D, False, 0), ("aa_C0", D, True, 0)]
        Dc, _ = fam_delta(conds, X, "cax_post"); variants.append(("cax", Dc, False, 0))
        Dt, _ = fam_delta(conds, X, "proj_post", end); variants.append(("aa_trunc", Dt, False, 20))
        info["alive_at_end_none"] = float(np.mean(end[:, 0] >= T))
        info["natural_end_none_median"] = float(np.median(np.minimum(end[:, 0], T)))
        for vname, DD, zc, ma in variants:
            info["fits"][vname] = {}
            fams = FAMS if vname == "aa" else ["role", "role3", "rand", "cov"]
            for F in fams:
                M = median_traj(DD[F], np.arange(P), ma)
                A, tau, C, hit = fit(M, zc); te = tau_e(M)
                B = boot_fits(DD[F], qidx, DR, zc, ma, pool)
                boots[pair][(vname, F)] = B
                info["fits"][vname][F] = {"A": A, "tau": tau, "C": C, "bound_hit": hit, "tau_e": te,
                                          "A_ci": ci(B[:, 0]), "tau_ci": ci(B[:, 1]), "C_ci": ci(B[:, 2]), "tau_e_ci": ci(B[:, 4]),
                                          "boot_fail": int(np.isnan(B[:, 1]).sum()), "boot_bound_hits": int(np.nansum(B[:, 3])),
                                          "M0": float(M[0]) if np.isfinite(M[0]) else None}
                if vname == "aa":
                    traj_for_fig.setdefault(pair, {})[F] = (np.median(DD[F], 0), np.array([np.median(DD[F][sel_prompts(qidx, d)], 0) for d in DR[:400]]))
            print(pair, vname, {F: (round(info["fits"][vname][F]["tau"], 1), [round(x, 1) for x in info["fits"][vname][F]["tau_ci"]]) for F in fams}, flush=True)
        print(pair, "W", {F: round(info["W"][F]["mean"], 2) for F in FAMS}, "K", {F: round(info["K"][F]["mean"], 2) for F in FAMS},
              "PC", info["positive_control_pass"], "carry", info["carryover_pass"], flush=True)
        summary["pairs"][pair] = info
        # raw per-token trajectories (gz jsonl): every (prompt, cond) row, AA and contrast readouts post-addition, natural end
        with gzip.open(REPO / f"results/Q6/trajectories_{pair}.jsonl.gz", "wt") as f:
            for a, (q, s) in enumerate(prompts):
                for b, c in enumerate(conds):
                    f.write(json.dumps({"qid": q, "sample": s, "cond": c, "end": int(end[a, b]),
                                        "aa_post": np.round(X["proj_post"][a, b], 3).tolist(), "aa_pre_window": np.round(X["proj_pre"][a, b, W0:W1], 3).tolist(),
                                        "cax_post": np.round(X["cax_post"][a, b], 3).tolist()}) + "\n")
    # contrasts (paired by question draw)
    summary["contrasts"] = []
    for x, y, kind in CONTRASTS:
        if x not in boots or y not in boots:
            continue
        for vname, F in (("aa", "role"), ("aa", "role3"), ("aa_C0", "role"), ("aa_trunc", "role")):
            if (vname, F) not in boots[x]:
                continue
            d = boots[x][(vname, F)][:, 1] - boots[y][(vname, F)][:, 1]
            pt = summary["pairs"][x]["fits"][vname][F]["tau"] - summary["pairs"][y]["fits"][vname][F]["tau"]
            c = ci(d)
            testable = all(summary["pairs"][p]["positive_control_pass" if F == "role" else "positive_control_pass_3x"] and
                           summary["pairs"][p]["carryover_pass" if F == "role" else "carryover_pass_3x"] for p in (x, y))
            verdict = ("holds" if c[1] < 0 else "reversed" if c[0] > 0 else "not supported") if testable else "not testable"
            summary["contrasts"].append({"instruct": x, "base": y, "kind": kind, "variant": vname, "family": F, "dtau": pt, "dtau_ci": c,
                                         "testable": testable, "verdict": verdict})
            print("contrast", x, "-", y, vname, F, round(pt, 1), [round(v, 1) for v in c], verdict, flush=True)
    pool.close()
    (REPO / "results/Q6/summary.json").write_text(json.dumps(summary, indent=1, default=float))
    figures(summary, traj_for_fig)


def figures(summary, tr):
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.setup(); pal = figstyle.PALETTE
    pairs = [p for p in PAIRS if p in tr]
    col = {"role": pal[0], "rand": pal[1], "cov": pal[2], "role3": pal[5], "rand3": pal[3]}
    lab = {"role": "role vector (1×)", "rand": "random null (1×)", "cov": "cov-matched null (1×)", "role3": "role vector (3×)", "rand3": "random null (3×)"}
    for famset, fname, ttl in ((("role", "rand", "cov"), "q6_trajectories.png", "1× perturbation"), (("role3", "rand3"), "q6_trajectories_3x.png", "3× perturbation")):
        fig, axs = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True)
        for ax, p in zip(axs.flat, pairs):
            info = summary["pairs"][p]; t = np.arange(T)
            ax.axvspan(W0 - 0.5, W1 - 0.5, color="#EEEEEE", zorder=0)
            for F in famset:
                med, bs = tr[p][F]
                lo, hi = np.percentile(bs, [2.5, 97.5], 0)
                ax.fill_between(t, lo, hi, color=col[F], alpha=0.2, lw=0)
                ax.plot(t, med, color=col[F], lw=1.0, label=lab[F])
            fr = info["fits"]["aa"][famset[0]]
            if np.isfinite(fr["tau"]):
                ax.plot(t[W1:], expo(S, fr["A"], fr["tau"], fr["C"]), color="k", ls="--", lw=0.8, label="exp fit (role)")
            ax.axhline(0, color="#888888", lw=0.6)
            ax.text(0.02, 0.97, p.replace("_", " ") + ("  (descriptive)" if p == "olmo_rl_narrative" else ""), transform=ax.transAxes, fontsize=9,
                    va="top", ha="left", fontweight="bold")
            ax.set_xlim(0, T)
            # zoom so the post-window part is visible: limits from the post-window band
            post = np.concatenate([np.percentile(tr[p][F][1], [2.5, 97.5], 0)[:, W1:].ravel() for F in famset])
            win = np.concatenate([tr[p][F][0][W0:W1] for F in famset])
            ax.set_ylim(min(post.min(), 0) * 1.3 - 0.05, max(post.max(), 0) * 1.3 + 0.05)
            ax.text(0.98, 0.04, f"window min {win.min():.1f}\nτ = {fr['tau']:.1f} [{fr['tau_ci'][0]:.1f}, {fr['tau_ci'][1]:.1f}]",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        for ax in axs[1]:
            ax.set_xlabel("response token t")
        for ax in axs[:, 0]:
            ax.set_ylabel("Δ AA projection at L* (paired)")
        axs.flat[0].legend(fontsize=7, loc="lower right", bbox_to_anchor=(1, 0.18))
        figstyle.title(axs.flat[0], f"Q6: AA displacement after a 5-token persona kick ({ttl})",
                       "median over 80 prompts (10 held-out questions × 8 samples) of the role-family mean; band = 95% question bootstrap; grey = window t 20–24 (off-scale)")
        fig.tight_layout(); figstyle.save(fig, fname)
    # tau summary
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(pairs))
    for k, (F, ls) in enumerate((("role", "-"), ("role3", "--"))):
        for i, p in enumerate(pairs):
            fr = summary["pairs"][p]["fits"]["aa"][F]
            ok = summary["pairs"][p]["carryover_pass" if F == "role" else "carryover_pass_3x"]
            ax.errorbar(i + (k - 0.5) * 0.25, fr["tau"], yerr=[[max(fr["tau"] - fr["tau_ci"][0], 0)], [max(fr["tau_ci"][1] - fr["tau"], 0)]],
                        fmt="o" if ok else "x", color=col[F], ms=5, capsize=2, label=(lab[F] if i == 0 else None))
    ax.set_xticks(x); ax.set_xticklabels([p.replace("_narrative", "\nnarr").replace("_chat", "\nchat") for p in pairs], fontsize=8)
    ax.set_ylabel("τ (tokens to 1/e)"); ax.set_yscale("log"); ax.legend(fontsize=7)
    figstyle.title(ax, "Q6: recovery time constant τ", "exp fit to the median paired displacement, s = t − 25 = 0..199; 95% CI over 2,000 question resamples; × = carry-over not beyond nulls")
    fig.tight_layout(); figstyle.save(fig, "q6_tau.png")
    # positive control
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, name, ttl in ((axs[0], "W", "window t 20–24 (post-addition)"), (axs[1], "K", "carry-over t 25–34")):
        for k, F in enumerate(("role", "rand", "cov")):
            for i, p in enumerate(pairs):
                r = summary["pairs"][p][name][F]
                ax.bar(i + (k - 1) * 0.27, r["mean"], width=0.27, color=col[F], label=(lab[F] if i == 0 else None))
                ax.plot([i + (k - 1) * 0.27] * 2, r["ci"], color="k", lw=0.8)
        ax.axhline(0, color="#888888", lw=0.6)
        ax.set_xticks(np.arange(len(pairs))); ax.set_xticklabels([p.replace("_narrative", "\nnarr").replace("_chat", "\nchat") for p in pairs], fontsize=7)
        ax.set_ylabel("mean Δ AA projection")
        figstyle.title(ax, f"Q6 positive control: {ttl}", "mean over 80 prompts; 95% question-bootstrap CI")
    axs[0].legend(fontsize=7); fig.tight_layout(); figstyle.save(fig, "q6_control.png")


if __name__ == "__main__":
    main()

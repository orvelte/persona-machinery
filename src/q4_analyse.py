#!/usr/bin/env python3
"""Q4.2 / Q4.3 analysis (PREREG Q4 Implementation).

    .venv/bin/python src/q4_analyse.py battery   # results/Q4/battery.json, battery_rows.jsonl, figures/q4_battery.png, q4_stages.png

Inputs: results/Q4/fwd_<model>/passages.npz (q4_fwd.py), data/Q4/passages_all.jsonl, results/M0/basis_*.npz, results/Q4/q41.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]
M0 = REPO / "results/M0"; RES = REPO / "results/Q4"
LSTAR = json.loads((M0 / "lstar.json").read_text())
SEED = 20261028; NB = 2000
CLS = ["helpful", "villain", "narrator", "nonnarrative", "firstperson"]
FAM = {"qwen": {"models": ["qwen_base", "qwen_it"], "final": "qwen_it_chat", "own": {"qwen_base": "qwen_base_narrative", "qwen_it": "qwen_it_chat"}},
       "olmo": {"models": ["olmo_base", "olmo_sft", "olmo_dpo", "olmo_rl"], "final": "olmo_rl_chat",
                "own": {"olmo_base": "olmo_base_narrative", "olmo_sft": "olmo_sft_chat", "olmo_dpo": "olmo_dpo_chat", "olmo_rl": "olmo_rl_chat"}}}
_b = {}


def basis(pair):
    if pair not in _b:
        _b[pair] = np.load(M0 / f"basis_{pair}.npz")
    return _b[pair]


def unit(v):
    v = np.asarray(v, np.float64).reshape(-1); return v / np.linalg.norm(v)


def axes(pair, L):
    B = basis(pair); mu = B[f"mu_{L}"].astype(np.float64)
    ax = {"pc1": unit(B[f"aa_{L}"]), "contrast": unit(B[f"default_{L}"].astype(np.float64) - mu)}
    for d in range(3):
        ax[f"rand{d}"] = unit(B[f"rand1_{L}_{d}"]); ax[f"cov{d}"] = unit(B[f"cov1_{L}_{d}"])
    return mu, ax


def load_acts(model, key="X_excl0"):
    z = np.load(RES / f"fwd_{model}/passages.npz")
    return z[key], [str(i) for i in z["ids"]]


def gstat(s, labels, idx_by_cls):
    """standardised helpful gap on scores s using the given per-class index arrays."""
    means = {c: s[idx_by_cls[c]].mean() for c in CLS}
    gap = means["helpful"] - np.mean([means[c] for c in CLS if c != "helpful"])
    resid = np.concatenate([s[idx_by_cls[c]] - means[c] for c in CLS])
    sd = np.sqrt((resid ** 2).sum() / (len(resid) - len(CLS)))
    return gap / sd, gap, means, sd


def gstat_sub(s, idx_by_cls, classes):
    """EXPLORATORY (not pre-registered): helpful gap against the other narrative / human classes only, pooled SD over those classes."""
    means = {c: s[idx_by_cls[c]].mean() for c in classes}
    gap = means["helpful"] - np.mean([means[c] for c in classes if c != "helpful"])
    resid = np.concatenate([s[idx_by_cls[c]] - means[c] for c in classes])
    return gap / np.sqrt((resid ** 2).sum() / (len(resid) - len(classes)))


def battery():
    meta = [json.loads(l) for l in (REPO / "data/Q4/passages_all.jsonl").read_text().splitlines()]
    sub = {json.loads(l)["id"]: json.loads(l)["sub"] for l in (REPO / "data/Q4/battery.jsonl").read_text().splitlines()}
    rng = np.random.default_rng(SEED)
    out, rows = {}, []
    for fam, F in FAM.items():
        Lf = LSTAR[F["final"]]["L_star"]; mu, ax = axes(F["final"], Lf)
        S = {}  # model -> axis -> scores over all passages
        ids = None
        for m in F["models"]:
            X, ids_m = load_acts(m)
            assert ids is None or ids == ids_m; ids = ids_m
            x = X[:, Lf - 1].astype(np.float64)
            S[m] = {a: (x - mu) @ u for a, u in ax.items()}
            own = F["own"][m]; Lo = LSTAR[own]["L_star"]; muo, axo = axes(own, Lo)
            xo = X[:, Lo - 1].astype(np.float64)
            S[m]["own_pc1"] = (xo - muo) @ axo["pc1"]; S[m]["own_contrast"] = (xo - muo) @ axo["contrast"]
            Xa, _ = load_acts(m, "X_all"); xa = Xa[:, Lf - 20].astype(np.float64)
            S[m]["pc1_alltok"] = (xa - mu) @ ax["pc1"]; S[m]["contrast_alltok"] = (xa - mu) @ ax["contrast"]
        cls = np.array([r["cls"] for r in meta]); assert [r["id"] for r in meta] == ids
        idx = {c: np.where(cls == c)[0] for c in CLS}
        hum = np.array([i for i in idx["helpful"] if sub[ids[i]] in (0, 1)]); ai = np.array([i for i in idx["helpful"] if sub[ids[i]] in (2, 3)])
        boots = [{c: rng.choice(idx[c], len(idx[c])) for c in CLS} for _ in range(NB)]
        res = {}
        axes_rep = ["pc1", "contrast", "own_pc1", "own_contrast", "pc1_alltok", "contrast_alltok"] + [f"rand{d}" for d in range(3)] + [f"cov{d}" for d in range(3)]
        for m in F["models"]:
            res[m] = {}
            for a in axes_rep:
                s = S[m][a]; g, gap, means, sd = gstat(s, cls, idx)
                bs = np.array([gstat(s, cls, b)[0] for b in boots])
                gh = gstat(s, cls, {**idx, "helpful": hum})[0]; ga = gstat(s, cls, {**idx, "helpful": ai})[0]
                nn = [c for c in CLS if c != "nonnarrative"]
                gx = lambda b: gstat_sub(s, b, nn)  # noqa: E731
                ex = gx(idx); exb = np.array([gx(b) for b in boots])
                res.setdefault("_explor", {}).setdefault(m, {})[a] = {"g_narr_only": ex, "ci": list(np.percentile(exb, [2.5, 97.5]))}
                res[m][a] = {"g": g, "g_ci": list(np.percentile(bs, [2.5, 97.5])), "gap": gap, "sd": sd, "class_means_sd": {c: (means[c] - np.mean(list(means.values()))) / sd for c in CLS},
                             "g_human_helpers": gh, "g_ai_helpers": ga, "_boot": bs}
            for a in ("pc1", "contrast"):
                nul = [abs(res[m][f"{k}{d}"]["g"]) for k in ("rand", "cov") for d in range(3)]
                r = res[m][a]; r["null_rand_mean_abs"] = float(np.mean(nul[:3])); r["null_cov_mean_abs"] = float(np.mean(nul[3:]))
                r["beats_nulls"] = bool(r["g_ci"][0] > 0 and r["g"] > 2 * max(r["null_rand_mean_abs"], r["null_cov_mean_abs"]))
            # positive control: default vs low-role texts (first 20 + 20 rows of this family)
            pd_ = np.array([i for i, r in enumerate(meta) if r["cls"] == f"pc_{fam}_default"]); pl = np.array([i for i, r in enumerate(meta) if r["cls"] == f"pc_{fam}_lowrole"])
            for a in ("pc1", "contrast"):
                s = S[m][a]; d = s[pd_].mean() - s[pl].mean()
                bs = [s[rng.choice(pd_, len(pd_))].mean() - s[rng.choice(pl, len(pl))].mean() for _ in range(NB)]
                res[m][a]["pos_control_gap"] = float(d); res[m][a]["pos_control_ci"] = list(np.percentile(bs, [2.5, 97.5]))
                res[m][a]["pos_control_gap_sd"] = float(d / res[m][a]["sd"])
        explor = res.pop("_explor", {})
        # base vs final (paired bootstrap)
        base, final = F["models"][0], F["models"][-1]
        cmp = {}
        for a in ("pc1", "contrast", "own_pc1", "pc1_alltok"):
            diff = res[final][a]["_boot"] - res[base][a]["_boot"]
            gb, gf = res[base][a], res[final][a]
            present = gb["g_ci"][0] > 0; created = (gb["g_ci"][0] <= 0 <= gb["g_ci"][1]) and gf["g_ci"][0] > 0
            sharpened = bool(np.percentile(diff, 2.5) > 0)
            reversed_ = gb["g_ci"][1] < 0 and gf["g_ci"][1] < 0
            outcome = ("present in base, sharpened" if present and sharpened else "present in base, not sharpened" if present else
                       "created by post-training" if created else "reversed in both (helpful class below the other-class mean)" if reversed_ else
                       "absent" if gf["g_ci"][0] <= 0 <= gf["g_ci"][1] else "other")
            cmp[a] = {"diff": gf["g"] - gb["g"], "diff_ci": list(np.percentile(diff, [2.5, 97.5])), "outcome": outcome}
        # stage tracking (OLMo): consecutive differences
        stages = {}
        if fam == "olmo":
            q41 = json.loads((RES / "q41.json").read_text())
            names = ["olmo_base_narrative", "olmo_sft_chat", "olmo_dpo_chat", "olmo_rl_chat"]
            fin_means = res["olmo_rl"]["pc1"]["class_means_sd"]
            for m, pair in zip(F["models"], names):
                key = f"olmo_base_narrative vs olmo_rl_chat" if m == "olmo_base" else f"{pair} vs olmo_rl_chat"
                c41 = q41.get(key, {}).get("at_Lb", {}) if m != "olmo_rl" else {"cos": 1.0, "cos_contrast": 1.0}
                rho = spearmanr([res[m]["pc1"]["class_means_sd"][c] for c in CLS], [fin_means[c] for c in CLS])[0]
                stages[m] = {"cos_to_final": c41.get("cos"), "cos_contrast_to_final": c41.get("cos_contrast"), "dprime": LSTAR[pair]["dprime_at_L"],
                             "g_final_pc1": res[m]["pc1"]["g"], "g_final_pc1_ci": res[m]["pc1"]["g_ci"], "g_final_contrast": res[m]["contrast"]["g"],
                             "g_final_contrast_ci": res[m]["contrast"]["g_ci"], "g_own": res[m]["own_pc1"]["g"], "rho_class_order_vs_final": rho}
            trans = []
            for a, b in zip(F["models"][:-1], F["models"][1:]):
                d = res[b]["pc1"]["_boot"] - res[a]["pc1"]["_boot"]; dc = res[b]["contrast"]["_boot"] - res[a]["contrast"]["_boot"]
                trans.append({"from": a, "to": b, "d_dprime": stages[b]["dprime"] - stages[a]["dprime"], "d_g_pc1": res[b]["pc1"]["g"] - res[a]["pc1"]["g"],
                              "d_g_pc1_ci": list(np.percentile(d, [2.5, 97.5])), "d_g_contrast": res[b]["contrast"]["g"] - res[a]["contrast"]["g"],
                              "d_g_contrast_ci": list(np.percentile(dc, [2.5, 97.5]))})
            stages["transitions"] = trans
            stages["largest_d_dprime"] = max(trans, key=lambda t: t["d_dprime"])["to"]
            stages["largest_d_g_pc1"] = max(trans, key=lambda t: t["d_g_pc1"])["to"]
            stages["largest_d_g_contrast"] = max(trans, key=lambda t: t["d_g_contrast"])["to"]
        for m in F["models"]:
            for a in res[m]:
                res[m][a].pop("_boot", None)
            for i, pid in enumerate(ids):
                rows.append({"family": fam, "model": m, "id": pid, "cls": meta[i]["cls"], **{a: float(S[m][a][i]) for a in ("pc1", "contrast", "own_pc1", "pc1_alltok")}})
        out[fam] = {"layer": Lf, "axis_from": F["final"], "models": res, "base_vs_final": cmp, "stages": stages, "exploratory_narrative_only": explor}
        print(fam, "EXPLORATORY helpful vs other narrative classes:", {m: {a: (round(v['g_narr_only'], 2), [round(x, 2) for x in v['ci']]) for a, v in d.items() if a in ('pc1', 'contrast')} for m, d in explor.items()})
        for m in F["models"]:
            r = res[m]["pc1"]; c = res[m]["contrast"]
            print(f"{fam} {m:10s} PC1 g {r['g']:+.2f} [{r['g_ci'][0]:+.2f},{r['g_ci'][1]:+.2f}] nulls {r['null_rand_mean_abs']:.2f}/{r['null_cov_mean_abs']:.2f} beats {r['beats_nulls']} "
                  f"| contrast g {c['g']:+.2f} [{c['g_ci'][0]:+.2f},{c['g_ci'][1]:+.2f}] | own {res[m]['own_pc1']['g']:+.2f} | PC {r['pos_control_gap_sd']:+.2f} [{r['pos_control_ci'][0]:+.1f},{r['pos_control_ci'][1]:+.1f}] "
                  f"| hum {r['g_human_helpers']:+.2f} ai {r['g_ai_helpers']:+.2f}")
        print(fam, "base vs final:", {a: (round(v['diff'], 2), [round(x, 2) for x in v['diff_ci']], v['outcome']) for a, v in cmp.items()})
    (RES / "battery.json").write_text(json.dumps(out, indent=1, default=float))
    with (RES / "battery_rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    figures(out)


def figures(out):
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.setup()
    lab = {"helpful": "helpful servant /\nassistant", "villain": "villain /\ntrickster", "narrator": "neutral\nnarrator", "nonnarrative": "non-narrative\n(encyc., code)",
           "firstperson": "first-person\nhuman"}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, (fam, a, ttl) in zip(axs, [("qwen", "pc1", "Qwen3.5-9B, instruct chat AA (L24)"), ("olmo", "pc1", "OLMo-3-7B, RL chat PC1 (L23)"),
                                        ("olmo", "contrast", "OLMo-3-7B, RL chat contrast axis (L23)")]):
        ms = out[fam]["models"]; n = len(ms); w = 0.8 / n
        for k, m in enumerate(ms):
            v = [ms[m][a]["class_means_sd"][c] for c in CLS]
            ax.bar(np.arange(5) + (k - (n - 1) / 2) * w, v, w, color=figstyle.PALETTE[:5], hatch=figstyle.HATCHES[k], edgecolor="#333333", lw=0.4,
                   label=f"{m}: g = {ms[m][a]['g']:+.2f} [{ms[m][a]['g_ci'][0]:+.2f}, {ms[m][a]['g_ci'][1]:+.2f}]")
        ax.set_xticks(range(5)); ax.set_xticklabels([lab[c] for c in CLS], fontsize=7.5); ax.axhline(0, color="#888888", lw=0.6)
        ax.set_ylabel("class mean projection (within-class SD units, centred)")
        figstyle.title(ax, ttl, "200 passages (40/class), raw text; legend: g = helpful − other 4 classes, 95% CI")
        leg = ax.legend(fontsize=6.5, loc="lower right")
        for h in leg.legend_handles:
            h.set_facecolor("#FFFFFF")
    figstyle.save(fig, "q4_battery.png")
    st = out["olmo"]["stages"]; ms = ["olmo_base", "olmo_sft", "olmo_dpo", "olmo_rl"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(4)
    axs[0].plot(x, [st[m]["cos_to_final"] for m in ms], "o-", color=figstyle.PALETTE[0], label="PC1")
    axs[0].plot(x, [st[m]["cos_contrast_to_final"] for m in ms], "s--", color=figstyle.PALETTE[1], label="contrast axis")
    axs[0].set_ylabel("cos with RL chat axis (L23)"); figstyle.title(axs[0], "Axis direction", "base = narrative AA, others chat; 99 roles"); axs[0].legend(fontsize=7)
    axs[1].plot(x, [st[m]["dprime"] for m in ms], "o-", color=figstyle.PALETTE[2]); axs[1].set_ylabel("held-out role separation d′")
    figstyle.title(axs[1], "Role separation along own AA", "M0 lstar.json; base narrative-format (format confound)")
    for k, (key, ci, lbl, ls) in enumerate([("g_final_pc1", "g_final_pc1_ci", "PC1", "-"), ("g_final_contrast", "g_final_contrast_ci", "contrast", "--")]):
        y = [st[m][key] for m in ms]; lo = [st[m][ci][0] for m in ms]; hi = [st[m][ci][1] for m in ms]
        axs[2].errorbar(x + (k - 0.5) * 0.08, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="o", ls=ls, color=figstyle.PALETTE[k], capsize=3, label=lbl)
    axs[2].axhline(0, color="#888888", lw=0.6); axs[2].set_ylabel("battery helpful gap g"); axs[2].legend(fontsize=7)
    figstyle.title(axs[2], "Narrative battery on the final axis", "200 passages, 95% bootstrap CI over passages")
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels(["base", "SFT", "DPO", "RL"])
    figstyle.save(fig, "q4_stages.png")


if __name__ == "__main__":
    {"battery": battery}[sys.argv[1]]()

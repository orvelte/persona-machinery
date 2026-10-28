#!/usr/bin/env python3
"""Q2 analysis (PREREG Q2 + Implementation): knockout / token-ablation shifts of the AA coordinate, rule, positive control, behaviour.

    .venv/bin/python src/q2_analyse.py tf          # results/Q2/summary_tf.json, figures/q2_delta.png, figures/q2_layers.png
    .venv/bin/python src/q2_analyse.py judge       # judge the behaviour generations (Haiku; cap $1) -> results/Q2/<model>/gen_judged.jsonl
    .venv/bin/python src/q2_analyse.py behaviour   # results/Q2/summary_behaviour.json, figures/q2_behaviour.png
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "results/Q2"
MODELS = ["olmo_rl", "qwen_it"]
LSTAR = {"olmo_rl": 23, "qwen_it": 24}
SEED = 20261028
NB = 2000


def load(m):
    d = OUT / m
    tf = np.load(d / "tf.npz"); ax = np.load(d / "axes.npz")
    rows = [json.loads(l) for l in (d / "tf_rows.jsonl").read_text().splitlines()]
    H = tf["H"]
    coords = {"pc1": (H - ax["mu"]) @ ax["pc1"], "c": (H - ax["mu"]) @ ax["c"], "frozen": (H - ax["mu_frozen"]) @ ax["aa_frozen"]}
    sd = float(np.std(ax["role_pos"]))   # half-A across-role SD of PC1 positions (scale for "role SDs")
    return rows, coords, tf["proj_layers"], sd, json.loads((d / "checks.json").read_text())


def by_cond(rows, x):
    """cond -> {(qid, sample): value}"""
    out = {}
    for r, v in zip(rows, x):
        out.setdefault(r["cond"], {})[(r["qid"], r["sample"])] = float(v)
    return out


def cluster_boot(keys, fn, rng):
    """keys: list of (qid, sample). fn(list_of_keys) -> scalar. Resample questions (clusters)."""
    qids = sorted({k[0] for k in keys}); byq = {q: [k for k in keys if k[0] == q] for q in qids}
    est = fn(keys); bs = []
    for _ in range(NB):
        pick = rng.choice(qids, len(qids), replace=True)
        bs.append(fn([k for q in pick for k in byq[q]]))
    bs = np.array(bs, float)
    return est, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def tf_summary():
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.setup()
    S = {}
    for m in MODELS:
        if not (OUT / m / "tf.npz").exists():
            continue
        rows, coords, PL, sd, checks = load(m)
        S[m] = {"checks": checks, "role_sd_pc1": sd, "axes": {}}
        for axn, x in coords.items():
            rng = np.random.default_rng(SEED)
            C = by_cond(rows, x); none = C["none"]; raw = C["raw_same"]; keys = sorted(none)
            gap = lambda ks: np.mean([none[k] - raw[k] for k in ks])  # noqa: E731
            res = {"none_mean": float(np.mean(list(none.values()))), "raw_same_mean": float(np.mean(list(raw.values()))),
                   "raw_own_mean": float(np.mean(list(C["raw_own"].values()))), "gap_none_minus_raw": cluster_boot(keys, gap, rng), "conds": {}}
            for cond in C:
                if cond in ("none", "raw_same", "raw_own") or cond.startswith("pc_"):
                    continue
                X = C[cond]
                dl = lambda ks, X=X: np.mean([none[k] - X[k] for k in ks]) / gap(ks)  # noqa: E731
                sh = lambda ks, X=X: np.mean([X[k] - none[k] for k in ks])  # noqa: E731
                res["conds"][cond] = {"mean": float(np.mean(list(X.values()))), "shift": cluster_boot(keys, sh, rng), "delta": cluster_boot(keys, dl, rng)}
            for kind in ("KO", "TA"):
                a, c = C[f"{kind}_a"], C[f"{kind}_c"]
                f = lambda ks, a=a, c=c: (np.mean([none[k] - a[k] for k in ks]) - np.mean([none[k] - c[k] for k in ks])) / gap(ks)  # noqa: E731
                fr = lambda ks, a=a: np.mean([none[k] - a[k] for k in ks]) / gap(ks) - np.mean([np.mean([none[k] - C[f"{kind}_rand{d}"][k] for k in ks]) for d in range(3)]) / gap(ks)  # noqa: E731
                res[f"{kind}_delta_template_minus_user"] = cluster_boot(keys, f, rng)
                res[f"{kind}_delta_template_minus_rand"] = cluster_boot(keys, fr, rng)
                fs = lambda ks, c=c: (np.mean([none[k] - C[f"{kind}_a_nosink"][k] for k in ks]) - np.mean([none[k] - c[k] for k in ks])) / gap(ks)  # noqa: E731
                res[f"{kind}_delta_template_nosink_minus_user"] = cluster_boot(keys, fs, rng)
            # positive control
            pc = {}
            for role in sorted({r.get("role") for r in rows if r["cond"].startswith("pc_")}):
                n0, k0 = C[f"pc_{role}_none"], C[f"pc_{role}_KO_sys"]
                drop = lambda ks, n0=n0: np.mean([none[k] - n0[k] for k in ks])  # noqa: E731
                rec = lambda ks, n0=n0, k0=k0: np.mean([k0[k] - n0[k] for k in ks]) / drop(ks)  # noqa: E731
                pc[role] = {"drop": cluster_boot(keys, drop, rng), "recovery": cluster_boot(keys, rec, rng)}
            res["positive_control"] = pc
            S[m]["axes"][axn] = res
        # per-layer profile (frozen aa_L): mean per condition
        conds = ["none", "KO_a", "KO_a_nosink", "KO_c", "TA_a", "TA_c", "raw_same"]
        S[m]["layer_profile"] = {c: PL[[i for i, r in enumerate(rows) if r["cond"] == c]].mean(0).tolist() for c in conds}
    (OUT / "summary_tf.json").write_text(json.dumps(S, indent=1))

    # figure: delta per condition
    show = ["a", "a_nosink", "role", "delim", "delim_nosink", "think", "sink", "b", "c", "rand0", "rand1", "rand2", "allprompt_nosink"]
    lab = {"a": "(a) template", "a_nosink": "(a′) template, sink kept", "role": "role names", "delim": "delimiters", "delim_nosink": "delimiters, sink kept",
           "think": "think tags", "sink": "position 0 only", "b": "(b) system", "c": "(c) user", "rand0": "random ×1", "rand1": "random ×2", "rand2": "random ×3",
           "allprompt_nosink": "all prompt (sink kept)"}
    ms = [m for m in MODELS if m in S]
    panels = [(m, axn) for m in ms for axn in ("pc1", "c")]
    fig, axs = plt.subplots(len(panels), 1, figsize=(8.6, 3.2 * len(panels)), squeeze=False, gridspec_kw={"hspace": 1.05})
    for ax, (m, axn) in zip(axs[:, 0], panels):
        res = S[m]["axes"][axn]; sd = S[m]["role_sd_pc1"]
        ks = [k for k in show if f"KO_{k}" in res["conds"]]
        xs = np.arange(len(ks))
        for j, (kind, col) in enumerate((("KO", figstyle.PALETTE[0]), ("TA", figstyle.PALETTE[1]))):
            v = [res["conds"].get(f"{kind}_{k}", {"shift": (np.nan, np.nan, np.nan)})["shift"] for k in ks]
            e, lo, hi = (np.array([t[i] for t in v]) / sd for i in range(3))
            ax.bar(xs + (j - 0.5) * 0.38, e, 0.38, color=col, hatch=figstyle.HATCHES[j], edgecolor="white", label="attention knockout" if kind == "KO" else "token ablation")
            ax.errorbar(xs + (j - 0.5) * 0.38, e, yerr=[e - lo, hi - e], fmt="none", ecolor="#333333", lw=0.8, capsize=2)
        g = res["gap_none_minus_raw"]
        ax.axhline(0, color="#555555", lw=0.6)
        ax.axhspan(-g[2] / sd, -g[1] / sd, color=figstyle.PALETTE[2], alpha=0.18, lw=0)
        ax.axhline(-g[0] / sd, color=figstyle.PALETTE[2], lw=1.2, ls="--", label="raw-document value (same content)")
        ax.set_xticks(xs); ax.set_xticklabels([lab[k] for k in ks], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("shift (across-role SDs)")
        figstyle.title(ax, f"{m} chat, {'PC1' if axn == 'pc1' else 'contrast axis c'} at L*={LSTAR[m]}: shift of the AA coordinate",
                       f"n = 30 default half-B responses (10 questions × 3); unit = SD of the 99 half-A role positions on PC1; cluster-bootstrap 95% CI")
        ax.legend(fontsize=7, loc="lower left", ncol=3)
    figstyle.save(fig, "q2_delta.png")
    # figure: layer profile
    fig, axs = plt.subplots(1, len(ms), figsize=(4.6 * len(ms), 3.4), squeeze=False)
    for ax, m in zip(axs[0], ms):
        lp = S[m]["layer_profile"]
        for i, (c, ls) in enumerate([("none", "-"), ("KO_a", "-"), ("KO_a_nosink", "--"), ("KO_c", "-"), ("TA_a", ":"), ("raw_same", "-.")]):
            ax.plot(range(1, 33), lp[c], color=figstyle.PALETTE[i], ls=ls, lw=1.3, label=c)
        ax.axvline(LSTAR[m], color="#999999", lw=0.6)
        ax.set_xlabel("layer"); ax.set_ylabel("projection on frozen AA_L")
        figstyle.title(ax, f"{m}: AA projection by layer", "mean of 30 responses; frozen M0 AA per layer")
        ax.legend(fontsize=7)
    figstyle.save(fig, "q2_layers.png")
    for m in ms:
        for axn in ("pc1", "c"):
            r = S[m]["axes"][axn]
            print(m, axn, "none", round(r["none_mean"], 2), "raw_same", round(r["raw_same_mean"], 2), "raw_own", round(r["raw_own_mean"], 2), "gap", [round(v, 2) for v in r["gap_none_minus_raw"]])
            for k, v in r["conds"].items():
                print(f"   {k:22s} shift {v['shift'][0]:+7.2f} [{v['shift'][1]:+.2f},{v['shift'][2]:+.2f}]  delta {v['delta'][0]:+.2f} [{v['delta'][1]:+.2f},{v['delta'][2]:+.2f}]")
            for kind in ("KO", "TA"):
                print("  ", kind, "template-user", [round(v, 3) for v in r[f"{kind}_delta_template_minus_user"]], "template-rand", [round(v, 3) for v in r[f"{kind}_delta_template_minus_rand"]],
                      "nosink-user", [round(v, 3) for v in r[f"{kind}_delta_template_nosink_minus_user"]])
            print("   positive control", {k: {kk: [round(x, 2) for x in vv] for kk, vv in v.items()} for k, v in r["positive_control"].items()})


# ---------------------------------------------------------------- behaviour
HEDGE = re.compile(r"\b(perhaps|maybe|might|could|possibly|it seems|i think|arguably|somewhat|likely|generally|often|tend to)\b", re.I)
VERIFY = re.compile(r"\b(let me (?:verify|double[- ]check|check|confirm)|to verify|double[- ]check)\b", re.I)
FLATTER = re.compile(r"\b(great question|excellent question|good question|what a (?:great|wonderful)|you're absolutely right|i love that)\b", re.I)


def style(text):
    """Copied verbatim from persona-mech src/m7_control_analyse.py (that module cannot be imported: it reads a file absent from the vendored copy)."""
    words = max(1, len(text.split())); lines = text.splitlines() or [""]
    sents = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
    return {"log_len": math.log(words), "headers": sum(l.lstrip().startswith("#") for l in lines) / words * 100, "bullets": sum(bool(re.match(r"\s*([-*•]|\d+\.)\s", l)) for l in lines) / words * 100,
            "emdash": text.count("—") / words * 100, "hedges": len(HEDGE.findall(text)) / words * 100, "verify": len(VERIFY.findall(text)) / words * 100, "flattery": len(FLATTER.findall(text)) / words * 100,
            "exclaim": text.count("!") / words * 100, "questions": text.count("?") / words * 100, "bold": text.count("**") / 2 / words * 100, "sent_len": words / max(1, len(sents))}


def judge():
    import judge as J
    for m in MODELS:
        f = OUT / m / "gen.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        sysp = J.ASSISTANT_LIKENESS
        al = J.score_prompts([f"Question: {r['user']}\n\nReply:\n{r['text'][:6000]}" for r in rows], "Q2", cap_usd=0.5, system=sysp, note=f"Q2 assistant-likeness {m}")
        co = J.score_prompts([J.coherence_prompt(r["user"], r["text"][:6000]) for r in rows], "Q2", cap_usd=0.5, note=f"Q2 coherence {m}")
        with (OUT / m / "gen_judged.jsonl").open("w") as fo:
            for r, (a, ra), (c, rc) in zip(rows, al, co):
                fo.write(json.dumps(r | {"assistant_likeness": a, "coherence": c, "style": style(r["text"])}, ensure_ascii=False) + "\n")


def behaviour():
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.setup()
    sdf = json.loads((OUT / "style_role_sd.json").read_text())
    S = {}
    for m in MODELS:
        f = OUT / m / "gen_judged.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        C = {}
        for r in rows:
            C.setdefault(r["cond"], {})[(r["qid"], r["sample"])] = r
        none = C["none"]; keys = sorted(none); res = {}
        for cond, X in C.items():
            rng = np.random.default_rng(SEED); d = {}
            for metric in ("assistant_likeness", "coherence"):
                val = lambda ks, X=X, metric=metric: np.mean([X[k][metric] for k in ks if isinstance(X[k][metric], (int, float))])  # noqa: E731
                d[metric] = cluster_boot(keys, val, rng)
                if cond != "none":
                    dif = lambda ks, X=X, metric=metric: np.mean([X[k][metric] - none[k][metric] for k in ks if isinstance(X[k][metric], (int, float)) and isinstance(none[k][metric], (int, float))])  # noqa: E731
                    d[metric + "_diff"] = cluster_boot(keys, dif, rng)
            d["style_z_diff"] = {}
            if cond != "none":
                for feat, sd in ((f_, s_) for f_, s_ in sdf[m].items() if s_ >= 0.01):
                    dif = lambda ks, X=X, feat=feat, sd=sd: np.mean([(X[k]["style"][feat] - none[k]["style"][feat]) / sd for k in ks])  # noqa: E731
                    d["style_z_diff"][feat] = cluster_boot(keys, dif, rng)
            d["finished"] = sum(X[k]["finished"] for k in keys); d["mean_len"] = float(np.mean([X[k]["n_gen"] for k in keys]))
            res[cond] = d
        S[m] = res
    (OUT / "summary_behaviour.json").write_text(json.dumps(S, indent=1))
    ms = list(S)
    fig, axs = plt.subplots(1, len(ms), figsize=(4.8 * len(ms), 3.4), squeeze=False)
    for ax, m in zip(axs[0], ms):
        conds = list(S[m]); xs = np.arange(len(conds))
        for j, metric in enumerate(("assistant_likeness", "coherence")):
            e = np.array([S[m][c][metric][0] for c in conds]); lo = np.array([S[m][c][metric][1] for c in conds]); hi = np.array([S[m][c][metric][2] for c in conds])
            ax.bar(xs + (j - 0.5) * 0.38, e, 0.38, color=figstyle.PALETTE[j], hatch=figstyle.HATCHES[j], edgecolor="white", label=metric.replace("_", "-"))
            ax.errorbar(xs + (j - 0.5) * 0.38, e, yerr=[e - lo, hi - e], fmt="none", ecolor="#333333", lw=0.8, capsize=2)
        ax.set_xticks(xs); ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=8); ax.set_ylim(0, 105); ax.set_ylabel("Haiku score (0–100)")
        figstyle.title(ax, f"{m}: behaviour under knockout", "n = 30 generations per condition; cluster-bootstrap 95% CI over 10 questions")
        ax.legend(fontsize=7, loc="lower left")
    figstyle.save(fig, "q2_behaviour.png")
    for m in ms:
        for c, d in S[m].items():
            print(m, c, {k: [round(x, 1) for x in v] for k, v in d.items() if isinstance(v, tuple)}, "fin", d["finished"], "len", round(d["mean_len"]),
                  {k: round(v[0], 2) for k, v in d["style_z_diff"].items() if abs(v[0]) > 0.3})


if __name__ == "__main__":
    {"tf": tf_summary, "judge": judge, "behaviour": behaviour}[sys.argv[1]]()

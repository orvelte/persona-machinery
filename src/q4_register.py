#!/usr/bin/env python3
"""Q4.2b register control (PREREG Q4.2b): 2 x 2 character {helper, villain} x expository register {entry, manual}, 40 passages each.

    .venv/bin/python src/q4_register.py write     # data/Q4/register_draft.jsonl (Sonnet 5.5)
    .venv/bin/python src/q4_register.py check     # blind Haiku labels on character and register; failures rewritten once
    .venv/bin/python src/q4_register.py freeze    # data/Q4/register.jsonl + sha256; data/Q4/register_passages.jsonl (register + battery helpful/villain)
    .venv/bin/python src/q4_register.py analyse   # results/Q4/register.json, register_rows.jsonl, figures/q4_register.png
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api  # noqa: E402
from q4_items import HAIKU, jl_read, jl_write, parse_json_list, sha, sonnet  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/Q4"; RES = REPO / "results/Q4"
SEED = 20261028; NB = 2000; CAP = 1.5
CHAR = {"helper": "a helpful servant or assistant character (a butler, guide, AI assistant or helper robot, servant)",
        "villain": "a villain or trickster character"}
REG = {"entry": ("a third-person, encyclopaedia-style article about this kind of character (as a type, or a famous fictional or folkloric example). "
                 "Expository and neutral, like a reference work: definitions, history, typical traits and roles. No narrated events, no dialogue, no first person."),
       "manual": ("a first-person expository manual or guide in which the character explains its own craft, methods or rules to a reader in its own voice "
                  "(e.g. 'As a butler, I keep three rules...'). Instructional, not a story: no narrated sequence of events, no scene, no dialogue.")}
SUB = {"helper": ["butlers and valets", "guides (mountain guides, tour guides, museum guides)", "AI assistants and helper robots in fiction", "servants, maids and stewards"],
       "villain": ["tyrants and dark lords", "tricksters and con artists", "villains of modern thrillers (corrupt executives, blackmailers, saboteurs)",
                   "folk-tale villains (witches, wolves, scheming viziers)"]}
CLASSES = [f"{c}_{r}" for c in CHAR for r in REG]
PROMPT = """Write 10 short, self-contained English text passages for a research dataset. Every passage must be:

CHARACTER: {char}
FORM: {reg}
SUB-THEME FOR THIS BATCH: {sub}

Requirements:
- Each passage is 60-120 words, plain text, no title, no markdown headings or bullet lists.
- All 10 passages differ in content and names. Fictional villains' schemes stay in-world; no real-world harmful instructions.
- Each passage must clearly be about {who} (not the other kind of character), and clearly in the FORM above (not a story).
- Do not mention this dataset.

Return only a JSON list of 10 strings."""
CHAR_SYS = ("You read a short passage. Which kind of character is it centred on? HELPER = a helpful servant or assistant character (butler, guide, "
            "assistant, helper robot, servant); VILLAIN = a villain or trickster; NEITHER = neither. Reply with exactly one word: HELPER, VILLAIN or NEITHER.")
REG_SYS = ("You read a short passage. What form is it written in? ENTRY = a third-person expository reference article (encyclopaedia style, no narrated events); "
           "MANUAL = a first-person expository guide or manual in which the speaker explains their own craft or rules to the reader; "
           "STORY = a narrative of events or a scene. Reply with exactly one word: ENTRY, MANUAL or STORY.")
REWRITE = """The passage below was meant to be {char_desc}, written as {reg_desc}
A reader labelled it as character={got_c}, form={got_r} instead of character={want_c}, form={want_r}.
Rewrite it (60-120 words) so that it is unambiguously the intended character and the intended form. Return only the rewritten passage as plain text.

PASSAGE:
{text}"""


async def write():
    cl = api.client(); sp = api.Spend("Q4.2b-write", CAP)
    jobs = [(c, r, i, s) for c in CHAR for r in REG for i, s in enumerate(SUB[c])]
    outs = await api.gather([sonnet(cl, sp, PROMPT.format(char=CHAR[c], reg=REG[r], sub=s, who=CHAR[c].split(" (")[0]), sample=i) for c, r, i, s in jobs], workers=16)
    rows = []
    for (c, r, i, s), o in zip(jobs, outs):
        ps = parse_json_list(o)
        if len(ps) != 10:
            print("WARN", c, r, i, len(ps))
        rows += [{"id": f"{c}_{r}-{i}-{k}", "cls": f"{c}_{r}", "char": c, "reg": r, "sub": i, "text": t.strip()} for k, t in enumerate(ps[:10])]
    sp.flush("Q4.2b register writing (Sonnet)")
    jl_write(OUT / "register_draft.jsonl", rows)
    print(len(rows), {k: sum(x["cls"] == k for x in rows) for k in CLASSES}, f"${sp.usd:.3f}")


async def label(cl, sp, text):
    a, b = await asyncio.gather(*[api.chat(cl, sp, HAIKU, [{"role": "system", "content": s}, {"role": "user", "content": text}], effort=None, max_tokens=8,
                                           provider=None, temperature=0.0, top_p=1.0, thinking=False) for s in (CHAR_SYS, REG_SYS)])
    fc = re.findall(r"\b(HELPER|VILLAIN|NEITHER)\b", (a["content"] or "").upper()); fr = re.findall(r"\b(ENTRY|MANUAL|STORY)\b", (b["content"] or "").upper())
    return (fc[-1].lower() if fc else None), (fr[-1].lower() if fr else None)


async def check():
    cl = api.client(); sp = api.Spend("Q4.2b-check", CAP)
    rows = jl_read(OUT / "register_draft.jsonl")
    labs = await api.gather([label(cl, sp, r["text"]) for r in rows], workers=16)
    for r, (c, g) in zip(rows, labs):
        r["c1"], r["r1"] = c, g; r["pass1"] = (c == r["char"] and g == r["reg"])
    fails = [r for r in rows if not r["pass1"]]
    print("first pass:", {k: (sum(r["pass1"] for r in rows if r["cls"] == k), sum(r["c1"] == r["char"] for r in rows if r["cls"] == k),
                              sum(r["r1"] == r["reg"] for r in rows if r["cls"] == k)) for k in CLASSES}, "fails", len(fails))
    new = await api.gather([sonnet(cl, sp, REWRITE.format(char_desc=CHAR[r["char"]], reg_desc=REG[r["reg"]], got_c=r["c1"], got_r=r["r1"], want_c=r["char"].upper(),
                                                           want_r=r["reg"].upper(), text=r["text"]), max_tokens=1500) for r in fails])
    for r, t in zip(fails, new):
        r["text_orig"] = r["text"]; r["text"] = t.strip(); r["rewritten"] = True
    labs2 = await api.gather([label(cl, sp, r["text"]) for r in fails], workers=16)
    for r, (c, g) in zip(fails, labs2):
        r["c2"], r["r2"] = c, g; r["pass2"] = (c == r["char"] and g == r["reg"])
    for r in rows:
        r["keep"] = r["pass1"] or r.get("pass2", False)
    jl_write(OUT / "register_checked.jsonl", rows); sp.flush("Q4.2b blind check + one rewrite")
    print("final kept:", {k: sum(r["keep"] for r in rows if r["cls"] == k) for k in CLASSES}, f"${sp.usd:.3f}")


def freeze():
    rows = [r for r in jl_read(OUT / "register_checked.jsonl") if r["keep"]]
    jl_write(OUT / "register.jsonl", [{k: r[k] for k in ("id", "cls", "char", "reg", "sub", "text")} for r in rows])
    bat = [r for r in jl_read(OUT / "battery.jsonl") if r["cls"] in ("helpful", "villain")]
    allp = [{"id": r["id"], "cls": r["cls"], "text": r["text"]} for r in rows] + [{"id": r["id"], "cls": r["cls"], "text": r["text"]} for r in bat]
    jl_write(OUT / "register_passages.jsonl", allp)
    print("register.jsonl", sha(OUT / "register.jsonl"), len(rows), "| register_passages.jsonl", sha(OUT / "register_passages.jsonl"), len(allp))


def analyse():
    from q4_analyse import FAM, LSTAR, axes
    meta = jl_read(OUT / "register_passages.jsonl")
    bmeta = jl_read(OUT / "passages_all.jsonl")
    rng = np.random.default_rng(SEED)
    cells = CLASSES + ["helpful", "villain"]
    out, rows = {}, []
    for fam, F in FAM.items():
        Lf = LSTAR[F["final"]]["L_star"]; mu, ax = axes(F["final"], Lf)
        for m in F["models"]:
            z = np.load(RES / f"fwd_{m}/register.npz"); assert [str(i) for i in z["ids"]] == [r["id"] for r in meta]
            zb = np.load(RES / f"fwd_{m}/passages.npz"); bids = [str(i) for i in zb["ids"]]
            x = z["X_excl0"][:, Lf - 1].astype(np.float64)
            # consistency check: battery helpful/villain re-run vs Q4.2 forward pass of the same passages
            bi = {b: i for i, b in enumerate(bids)}
            xb = zb["X_excl0"][[bi[r["id"]] for r in meta if r["cls"] in ("helpful", "villain")], Lf - 1].astype(np.float64)
            rerun_cos = float(np.mean([np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b) for a, b in zip(x[[i for i, r in enumerate(meta) if r["cls"] in ("helpful", "villain")]], xb)]))
            nonnarr = zb["X_excl0"][[i for i, r in enumerate(bmeta) if r["cls"] == "nonnarrative"], Lf - 1].astype(np.float64)
            res = {"rerun_cos_vs_q42": rerun_cos}
            for a in ("pc1", "contrast"):
                s = (x - mu) @ ax[a]; cls = np.array([r["cls"] for r in meta]); idx = {c: np.where(cls == c)[0] for c in cells}
                s_nn = (nonnarr - mu) @ ax[a]

                def eff(ix):
                    mm = {c: s[ix[c]].mean() for c in cells}
                    resid = np.concatenate([s[ix[c]] - mm[c] for c in cells]); sd = np.sqrt((resid ** 2).sum() / (len(resid) - len(cells)))
                    expo_h = (mm["helper_entry"] + mm["helper_manual"]) / 2; expo_v = (mm["villain_entry"] + mm["villain_manual"]) / 2
                    e = {"R": ((expo_h + expo_v) / 2 - (mm["helpful"] + mm["villain"]) / 2) / sd,
                         "C": ((expo_h + mm["helpful"]) / 2 - (expo_v + mm["villain"]) / 2) / sd,
                         "I": ((expo_h - expo_v) - (mm["helpful"] - mm["villain"])) / sd,
                         "R2": ((mm["helper_entry"] + mm["villain_entry"]) / 2 - (mm["helper_manual"] + mm["villain_manual"]) / 2) / sd,
                         "C2": ((mm["helper_entry"] + mm["helper_manual"]) / 2 - (mm["villain_entry"] + mm["villain_manual"]) / 2) / sd,
                         "I2": ((mm["helper_entry"] - mm["villain_entry"]) - (mm["helper_manual"] - mm["villain_manual"])) / sd,
                         "K": (mm["villain_entry"] - mm["helper_manual"]) / sd}
                    cm = np.mean(list(mm.values()))
                    e["cell_means_sd"] = {c: (mm[c] - cm) / sd for c in cells}; e["nonnarr_sd"] = (s_nn.mean() - cm) / sd
                    return e
                pt = eff(idx)
                bs = [eff({c: rng.choice(idx[c], len(idx[c])) for c in cells}) for _ in range(NB)]
                ci = {k: [float(v) for v in np.percentile([b[k] for b in bs], [2.5, 97.5])] for k in ("R", "C", "I", "R2", "C2", "I2", "K")}
                res[a] = {"est": {k: float(v) for k, v in pt.items() if k not in ("cell_means_sd",)}, "ci": ci, "cell_means_sd": pt["cell_means_sd"],
                          "rule_primary": rule(pt["R"], ci["R"], pt["C"], ci["C"]), "rule_literal": rule(pt["R2"], ci["R2"], pt["C2"], ci["C2"])}
                for i, r in enumerate(meta):
                    rows.append({"family": fam, "model": m, "axis": a, "id": r["id"], "cls": r["cls"], "proj": float(s[i])})
            out[m] = res
            p = res["pc1"]
            print(f"{m:10s} rerun-cos {rerun_cos:.4f} PC1: R {p['est']['R']:+.2f} {np.round(p['ci']['R'], 2)} C {p['est']['C']:+.2f} {np.round(p['ci']['C'], 2)} "
                  f"I {p['est']['I']:+.2f} | R' {p['est']['R2']:+.2f} {np.round(p['ci']['R2'], 2)} C' {p['est']['C2']:+.2f} {np.round(p['ci']['C2'], 2)} | "
                  f"K {p['est']['K']:+.2f} {np.round(p['ci']['K'], 2)} -> {p['rule_primary']} / literal {p['rule_literal']}"
                  + (f" | contrast: {res['contrast']['rule_primary']}" if fam == "olmo" else ""))
    (RES / "register.json").write_text(json.dumps(out, indent=1))
    jl_write(RES / "register_rows.jsonl", rows)
    figure(out)


def rule(R, Rci, C, Cci):
    ex = lambda ci: ci[0] > 0 or ci[1] < 0  # noqa: E731
    rR, rC = ex(Rci), ex(Cci)
    tag = lambda name, v: name + (" (reversed)" if v < 0 else "")  # noqa: E731
    if rR and rC:
        if abs(C) < 0.5 * abs(R):
            return tag("register", R)
        if abs(R) < 0.5 * abs(C):
            return tag("character", C)
        return f"both ({'R+' if R > 0 else 'R-'}, {'C+' if C > 0 else 'C-'})"
    if rR:
        return tag("register", R)
    if rC:
        return tag("character", C)
    return "neither"


def figure(out):
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.setup()
    cells = ["helpful", "villain", "helper_entry", "villain_entry", "helper_manual", "villain_manual"]
    lab = ["helper\nstory", "villain\nstory", "helper\nentry", "villain\nentry", "helper\nmanual", "villain\nmanual"]
    panels = [("qwen_base", "pc1"), ("qwen_it", "pc1"), ("olmo_base", "pc1"), ("olmo_rl", "pc1")]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
    for ax, (m, a) in zip(axs, panels):
        r = out[m][a]; v = [r["cell_means_sd"][c] for c in cells]
        cols = [figstyle.PALETTE[0], figstyle.PALETTE[1]] * 3; hat = ["", "", "///", "///", "...", "..."]
        ax.bar(range(6), v, color=cols, hatch=hat, edgecolor="#333333", lw=0.4)
        ax.axhline(r["est"]["nonnarr_sd"], color=figstyle.PALETTE[3], ls="--", lw=1, label="battery non-narrative mean")
        ax.axhline(0, color="#888888", lw=0.6); ax.set_xticks(range(6)); ax.set_xticklabels(lab, fontsize=8)
        e, ci = r["est"], r["ci"]
        figstyle.title(ax, f"{m}: {r['rule_primary']}", f"R {e['R']:+.2f} [{ci['R'][0]:+.2f},{ci['R'][1]:+.2f}], C {e['C']:+.2f} [{ci['C'][0]:+.2f},{ci['C'][1]:+.2f}], K {e['K']:+.2f}")
        ax.legend(fontsize=7, loc="lower right")
    axs[0].set_ylabel("projection on instruct AA (pooled within-class SD, centred)")
    fig.suptitle("Q4.2b register control: character (colour) × register (hatch); 40 passages/cell, raw text, instruct chat AA at L*; 95% bootstrap CIs in titles",
                 fontsize=9, style="italic", y=1.06)
    figstyle.save(fig, "q4_register.png")


if __name__ == "__main__":
    {"write": lambda: asyncio.run(write()), "check": lambda: asyncio.run(check()), "freeze": freeze, "analyse": analyse}[sys.argv[1]]()

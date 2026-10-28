#!/usr/bin/env python3
"""Q2b: execution scoring and cell statistics (PREREG Q2b + Implementation).

    .venv/bin/python src/q2b_analyse.py score results/Q2b/raw/gen_gate_W1.jsonl     # -> results/Q2b/scored_<name>.jsonl, prints cells
    .venv/bin/python src/q2b_analyse.py summary                                      # all scored files -> results/Q2b/summary.json, figure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q2b_common as q  # noqa: E402

OUT = q.REPO / "results/Q2b"


def newcombe(k1, n1, k2, n2):
    """Newcombe hybrid score CI (method 10) for p1 - p2, from Wilson intervals."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = q.wilson(k1, n1); l2, u2 = q.wilson(k2, n2)
    d = p1 - p2
    return d, d - ((p1 - l1) ** 2 + (u2 - p2) ** 2) ** 0.5, d + ((u1 - p1) ** 2 + (p2 - l2) ** 2) ** 0.5


def score_file(f):
    rows = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        s = q.score_row(r)
        out.append({k: v for k, v in r.items() if k not in ("prompt_ids", "completion_ids", "text", "user")} | {"text": r["text"]} | s)
    p = OUT / f"scored_{Path(f).stem.removeprefix('gen_')}.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    return out, p


def cells(rows):
    res = {}
    for key in sorted({(r["model_tag"], r["frame"], r["wording"], r["judge"]) for r in rows}):
        sel = [r for r in rows if (r["model_tag"], r["frame"], r["wording"], r["judge"]) == key]
        n = len(sel); g = sum(r["gamed"] for r in sel); ok = sum(r["spec_ok"] for r in sel)
        res["|".join(key)] = {"n": n, "gamed": g, "gamed_ci": q.wilson(g, n), "spec_ok": ok, "spec_ok_ci": q.wilson(ok, n), "no_code": sum(not r["has_code"] for r in sel),
                              "import_error": sum(bool(r["import_error"]) for r in sel), "truncated": sum(r.get("finish") == "length" for r in sel),
                              "by_task": {t: sum(r["gamed"] for r in sel if r["task_name"] == t) for t in q.TASK_NAMES}}
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "score":
        rows, p = score_file(a[1]); print(p)
        for k, c in cells(rows).items():
            print(k, f"gamed {c['gamed']}/{c['n']} spec_ok {c['spec_ok']}/{c['n']} no_code {c['no_code']} import_err {c['import_error']} trunc {c['truncated']} by_task {c['by_task']}")
    else:
        rows = []
        for f in sorted(OUT.glob("scored_*.jsonl")):
            rows += [json.loads(l) for l in f.read_text().splitlines()]
        C = cells(rows); eff = {}
        for k in C:
            m, fr, w, j = k.split("|")
            if j != "C1":
                continue
            base = f"{m}|{fr}|{w}"
            for a_, b_ in (("C1", "C1R"), ("C1A", "C1R"), ("C1", "none"), ("C1", "C1A")):
                if f"{base}|{a_}" in C and f"{base}|{b_}" in C:
                    ca, cb = C[f"{base}|{a_}"], C[f"{base}|{b_}"]
                    d, lo, hi = newcombe(ca["gamed"], ca["n"], cb["gamed"], cb["n"])
                    eff[f"{base}|{a_}-{b_}"] = {"diff": d, "ci": [lo, hi], "effect": lo > 0 or hi < 0,
                                                "cap_ok": C.get(f"{base}|spec_only", {}).get("spec_ok", 0) >= 39}
        (OUT / "summary.json").write_text(json.dumps({"cells": C, "effects": eff}, indent=1))
        for k, c in C.items():
            print(k, f"gamed {c['gamed']}/{c['n']} [{c['gamed_ci'][0]:.2f},{c['gamed_ci'][1]:.2f}] spec_ok {c['spec_ok']}/{c['n']} no_code {c['no_code']}")
        for k, e in eff.items():
            print(k, f"{e['diff'] * 100:+.1f} pts [{e['ci'][0] * 100:+.1f}, {e['ci'][1] * 100:+.1f}] effect={e['effect']} cap_ok={e['cap_ok']}")

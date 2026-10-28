#!/usr/bin/env python3
"""Build the M0 job files (PREREG M0 steps 1 and 6): role library in chat and narrative formats, Chen trait extraction prompts.

    .venv/bin/python src/build_m0.py            # data/m0/jobs_<model>_<fmt>.jsonl, data/m0/jobs_traits_<model>.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"; OUT = DATA / "m0"; OUT.mkdir(parents=True, exist_ok=True)
CHAT_MODELS = ["qwen_it", "olmo_sft", "olmo_dpo", "olmo_rl"]
NARR_MODELS = ["qwen_base", "olmo_base", "qwen_it", "olmo_rl"]
TRAIT_MODELS = ["qwen_it", "olmo_rl"]
TRAITS = ["evil", "sycophancy", "hallucination"]
N_DEFAULT_SAMPLES = 3
STOPS = ["\nUser:", "\n\nUser:", "\nuser:"]


def write(p, rows):
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(p.name, len(rows))


def main():
    roles = json.loads((DATA / "roles.json").read_text()); narr = json.loads((DATA / "roles_narrative.json").read_text())
    qs = json.loads((DATA / "questions.json").read_text())
    assert len(roles) == 100 and len(qs) == 20 and sum(q["half"] == "A" for q in qs) == 10
    for fmt, models in (("chat", CHAT_MODELS), ("narrative", NARR_MODELS)):
        rows = []
        for role, sysmsg in roles.items():
            for q in qs:
                for s in range(N_DEFAULT_SAMPLES if role == "default" else 1):
                    base = {"id": f"{role}|{q['id']}|{s}", "role": role, "qid": q["id"], "half": q["half"], "sample": s, "seed": s, "format": fmt, "max_new": 384}
                    if fmt == "chat":
                        rows.append(base | {"system": sysmsg or None, "user": q["question"]})
                    else:
                        rows.append(base | {"description": narr[role]["description"], "speaker": narr[role]["speaker"], "question": q["question"], "stop": STOPS})
        for m in models:
            write(OUT / f"jobs_{m}_{fmt}.jsonl", rows)
    traits = json.loads((DATA / "traits.json").read_text())["traits"]
    rows = []
    for t in TRAITS:
        T = traits[t]
        assert not set(T["extract_questions"]) & set(T["heldout_questions"])
        for pi, (pos, neg) in enumerate(zip(T["pos"], T["neg"])):
            for qi, q in enumerate(T["extract_questions"]):
                for sign, sysmsg in (("pos", pos), ("neg", neg)):
                    rows.append({"id": f"{t}|{sign}|{pi}|{qi}", "trait": t, "sign": sign, "pair": pi, "qidx": qi, "seed": 0, "format": "chat", "system": sysmsg, "user": q, "max_new": 256})
    for m in TRAIT_MODELS:
        write(OUT / f"jobs_traits_{m}.jsonl", rows)


if __name__ == "__main__":
    main()

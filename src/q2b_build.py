#!/usr/bin/env python3
"""Q2b job files (PREREG Q2b + Implementation). Writes data/Q2b/jobs_<tag>.jsonl.

    .venv/bin/python src/q2b_build.py gate W1                    # qwen_it frame ii: C1 + spec_only
    .venv/bin/python src/q2b_build.py cells W1 qwen_it ii        # judge cells: C1 C1R C1A none spec_only (C1 + spec_only reused for the gate model)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q2b_common as q  # noqa: E402

OUT = q.REPO / "data/Q2b"; OUT.mkdir(parents=True, exist_ok=True)


def write(tag, rows):
    p = OUT / f"jobs_{tag}.jsonl"; p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)); print(p, len(rows))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "gate":
        write(f"gate_{a[1]}", q.jobs(a[1], ["C1", "spec_only"], "ii", "qwen_it"))
    else:
        w, model, frame = a[1], a[2], a[3]
        judges = a[4].split(",") if len(a) > 4 else ["C1", "C1R", "C1A", "none", "spec_only"]
        write(f"cells_{model}_{frame}_{w}", q.jobs(w, judges, frame, model))

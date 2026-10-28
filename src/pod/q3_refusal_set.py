#!/usr/bin/env python3
"""Q3 refusal set: 40 items from JailbreakBench JBB-Behaviors (harmful split), 4 per category, numpy seed 20261028. Texts are not printed."""
import json, sys
import numpy as np
from datasets import load_dataset
d = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
rng = np.random.default_rng(20261028); out = []
for cat in sorted(set(d["Category"])):
    idx = [i for i, c in enumerate(d["Category"]) if c == cat]
    for i in sorted(rng.choice(idx, 4, replace=False).tolist()):
        out.append({"id": f"ref{len(out):02d}", "set": "refusal", "user": d[i]["Goal"], "source": "JBB-Behaviors", "jbb_index": int(d[i]["Index"]), "category": cat})
with open(sys.argv[1], "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
print(len(out), "items written")

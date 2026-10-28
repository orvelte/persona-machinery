#!/usr/bin/env python3
"""Build Q1 patch conditions from the attribution gap (PREREG Q1 Implementation). Writes data/Q1/patch_spec.json:
{condition: [ {block, mlp|head} ... ]}. Components identified by comp_names. seed 20261028 for rand10, lm10 deterministic."""
import json, re
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
SEED = 20261028

def parse(name):
    if name == "embed": return None
    m = re.match(r"L(\d+)\.(mlp|attn|dnet)(?:\.h(\d+))?", name)
    b = int(m.group(1))
    if m.group(2) == "mlp": return {"block": b, "mlp": True}
    return {"block": b, "head": int(m.group(3))}

def main():
    Z = np.load(REPO / "results/Q1/attr_qwen_it/attr.npz", allow_pickle=True)
    names = list(Z["comp_names"]); ai = list(Z["axis_names"]).index("aa")
    C = Z["C"][:, ai, :]; is_role = Z["is_role"]
    gap = C[~is_role].mean(0) - C[is_role].mean(0)
    order = np.argsort(gap)[::-1]
    nonemb = [i for i in range(len(names)) if names[i] != "embed"]
    top10 = [i for i in order if names[i] != "embed"][:10]
    spec = {"none": []}
    for r, i in enumerate(top10): spec[f"w{r+1}"] = [parse(names[i])]
    spec["top3"] = [parse(names[i]) for i in top10[:3]]
    spec["top10"] = [parse(names[i]) for i in top10]
    pool = [i for i in nonemb if i not in set(top10)]
    for d in range(3):
        rng = np.random.default_rng(SEED + d)
        pick = rng.choice(pool, 10, replace=False)
        spec[f"rand10_{d}"] = [parse(names[i]) for i in pick]
    # layer-matched: for each top writer's block, the non-top10 component at that block with smallest |gap|, no reuse
    used = set()
    lm = []
    absg = np.abs(gap)
    for i in top10:
        b = parse(names[i])["block"]
        cands = [j for j in nonemb if j not in set(top10) and j not in used and parse(names[j])["block"] == b]
        j = min(cands, key=lambda k: absg[k]); used.add(j); lm.append(parse(names[j]))
    spec["lm10"] = lm
    (REPO / "data/Q1").mkdir(parents=True, exist_ok=True)
    (REPO / "data/Q1/patch_spec.json").write_text(json.dumps(spec, indent=1))
    print("conditions:", list(spec.keys()))
    print("top10:", [names[i] for i in top10])
    print("lm10:", [names[j] for j in [nonemb[0]]] if False else "see json")

if __name__ == "__main__": main()

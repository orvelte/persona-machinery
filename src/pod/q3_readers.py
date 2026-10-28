#!/usr/bin/env python3
"""Q3 readers: AA-restricted input path patching on the early-continuation log-odds metric (PREREG Q3 Implementation; DECISIONS 24).

    python3 q3_readers.py SHORT ITEMS.jsonl BASIS.npz LSTAR OUT.jsonl [--ntok 24]

ITEMS rows: {"id", "set", "user", "r0": none reply text, "r1": projout reply text}. For each item and condition, teacher-force
prompt + R[:ntok] for R in (R0, R1); m = mean log P(R1 tokens 2..ntok) - mean log P(R0 tokens 2..ntok). Interventions only at
completion-token positions (>= n_prompt). Conditions: none; full (AA coordinate set to mu's at block outputs L*..32); unit patches,
where only that unit's input has the AA coordinate (of the preceding layer output) set to mu's and the stream is untouched:
mix_b / mlp_b for b = L*+1..32, unembed (final norm input, aa_32); all (every unit together).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from pm_common import decoder_layers, load_hf, read_jsonl, render_chat, tokenizer


def main(short, items_f, basis_f, lstar, out_f, ntok=24):
    tok = tokenizer(short); model = load_hf(short); layers = decoder_layers(model)
    B = np.load(basis_f)
    ax = {L: torch.tensor(B[f"aa_{L}"].astype(np.float32), device="cuda") for L in range(lstar, 33)}
    mu = {L: torch.tensor(B[f"mu_{L}"].astype(np.float32), device="cuda") for L in range(lstar, 33)}
    final_norm = [m for n, m in model.named_modules() if n.endswith("language_model.norm") or n == "model.norm"][0]
    st = {"mask": None, "active": set()}

    def ablate(x, L):
        xf = x.float(); m = st["mask"][..., None]                  # [b, T, 1]
        c = (xf - mu[L]) @ ax[L]
        return (xf - m * c[..., None] * ax[L]).to(x.dtype)

    def out_hook(L):            # full ablation: block output
        def f(_m, _i, out):
            if "full" not in st["active"]:
                return None
            h = out[0] if isinstance(out, tuple) else out
            h2 = ablate(h, L)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return f

    def in_hook(name, L):       # unit input only
        def f(_m, inp):
            if name not in st["active"] and "all" not in st["active"]:
                return None
            return (ablate(inp[0], L),) + tuple(inp[1:])
        return f

    for L in range(lstar, 33):
        layers[L - 1].register_forward_hook(out_hook(L))
    units = []
    for b in range(lstar + 1, 33):
        lay = layers[b - 1]
        lay.input_layernorm.register_forward_pre_hook(in_hook(f"mix_{b}", b - 1)); units.append(f"mix_{b}")
        lay.post_attention_layernorm.register_forward_pre_hook(in_hook(f"mlp_{b}", b - 1)); units.append(f"mlp_{b}")
    final_norm.register_forward_pre_hook(in_hook("unembed", 32)); units.append("unembed")
    conds = ["none", "full", "all"] + units
    items = read_jsonl(items_f)
    out = Path(out_f).open("w")
    for it in items:
        ptxt = render_chat(tok, short, None, it["user"])
        pids = tok(ptxt, add_special_tokens=False)["input_ids"]
        seqs, nps = [], len(pids)
        for k in ("r0", "r1"):
            rid = tok(it[k], add_special_tokens=False)["input_ids"][:ntok]
            seqs.append(pids + rid)
        T = max(map(len, seqs))
        inp = torch.full((2, T), tok.pad_token_id or 0, dtype=torch.long); att = torch.zeros((2, T), dtype=torch.long)
        for i, s in enumerate(seqs):
            inp[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        mask = torch.zeros((2, T), device="cuda"); mask[:, nps:] = 1.0
        st["mask"] = mask
        row = {"id": it["id"], "set": it["set"]}
        for c in conds:
            st["active"] = {c}
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=inp.cuda(), attention_mask=att.cuda(), use_cache=False).logits.float(), -1)
            ms = []
            for i, s in enumerate(seqs):
                tgt = torch.tensor(s[nps + 1:], device="cuda")           # completion tokens 2..ntok
                ms.append(lp[i, nps:len(s) - 1].gather(-1, tgt[:, None]).mean().item() if len(tgt) else 0.0)
            row[c] = ms[1] - ms[0]
        out.write(json.dumps(row) + "\n"); out.flush()
    print("done", len(items), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[3]), a[4], int(a[a.index("--ntok") + 1]) if "--ntok" in a else 24)

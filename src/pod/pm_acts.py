#!/usr/bin/env python3
"""HF replay of generated rows: mean residual over the completion tokens at every layer 1..32 (PREREG shared design).

    python3 pm_acts.py SHORT GEN.jsonl OUT_DIR [--batch 8] [--span resp|prompt_last|all_tokens]

Writes OUT_DIR/acts.npz (X: float16 [N, 32, 4096], ids) and OUT_DIR/meta.jsonl (one row per kept input: every key of the gen row
except the token ids, plus n_gen). Rows with n_gen == 0 are dropped and counted. The replay is teacher-forced on the exact ids.
Span 'resp' = all completion tokens (thinking is closed in the prompt, so every completion token is response).
Span 'prompt_last' = the last prompt token (for passages / log-prob style inputs with no completion).
Span 'all_tokens' = every token of the text (used for the narrative battery, where the "text" is the passage).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pm_common import D, N_LAYERS, load_hf, read_jsonl, tokenizer


def main(short, gen_f, out_dir, batch=8, span="resp"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(gen_f)
    tok = tokenizer(short)
    if span == "resp":
        rows = [r for r in rows if r["n_gen"] > 0]
    keep = []
    for r in rows:
        ids = list(r["prompt_ids"]) + list(r.get("completion_ids", []))
        if span == "resp":
            a, b = r["n_prompt"], len(ids)
        elif span == "prompt_last":
            a, b = len(ids) - 1, len(ids)
        else:
            a, b = 0, len(ids)
        keep.append((r, ids, a, b))
    print(f"{len(keep)} rows (dropped {len(read_jsonl(gen_f)) - len(keep)} empty)", flush=True)
    model = load_hf(short)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    order = sorted(range(len(keep)), key=lambda i: len(keep[i][1]))
    X = np.zeros((len(keep), N_LAYERS, D), np.float16); t0 = time.time()
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        L = max(len(keep[i][1]) for i in idx)
        inp = torch.full((len(idx), L), pad, dtype=torch.long); att = torch.zeros((len(idx), L), dtype=torch.long)
        for k, i in enumerate(idx):
            ids = keep[i][1]; inp[k, :len(ids)] = torch.tensor(ids); att[k, :len(ids)] = 1
        with torch.no_grad():
            o = model(input_ids=inp.cuda(), attention_mask=att.cuda(), use_cache=False, output_hidden_states=True)
        hs = o.hidden_states  # tuple of 33: [0] embeddings, [L] after block L
        for k, i in enumerate(idx):
            a, b = keep[i][2], keep[i][3]
            for Lr in range(1, N_LAYERS + 1):
                X[i, Lr - 1] = hs[Lr][k, a:b].float().mean(0).cpu().numpy()
        if (s // batch) % 25 == 0:
            print(f"{s + len(idx)}/{len(keep)} {(time.time() - t0) / 60:.1f} min", flush=True)
    np.savez(out / "acts.npz", X=X, layers=np.arange(1, N_LAYERS + 1))
    with (out / "meta.jsonl").open("w") as f:
        for r, ids, a, b in keep:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("prompt_ids", "completion_ids")} | {"n_span": b - a}, ensure_ascii=False) + "\n")
    print("done", short, X.shape, f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    main(a[0], a[1], a[2], opt("--batch", 8), opt("--span", "resp"))

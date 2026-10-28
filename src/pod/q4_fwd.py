#!/usr/bin/env python3
"""Q4 / Q4b forward passes, one model load per model (forward only; no generation).

    python3 q4_fwd.py SHORT OUT_DIR [--passages P.jsonl] [--templates T.jsonl] [--batch 16]

SHORT is a key of pm_common.MODELS or q36 / q38 (Qwen3.6-27B / Qwen3.8-27B, persona-mech pins).
--passages (rows with "id", "text"): raw text, no template, no special tokens; per-layer residual means over
  (a) every token except position 0 (primary; position 0 is the attention-sink token) -> X_excl0 [N, 32, D] fp16,
  (b) every token (sensitivity, layers 20..26 only) -> X_all [N, 7, D].
--templates (rows with "id", "text", "fit", "spec"): teacher-forced log P of each ending given the story (no template, no
  special tokens); ending ids = tokenizer(ending) appended to tokenizer(story). Writes armA.jsonl with per-token log-probs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pm_common import MODELS, read_jsonl

BIG = {"q36": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"), "q38": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")}


def load(short):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hid, rev = (MODELS[short]["hid"], MODELS[short]["rev"]) if short in MODELS else BIG[short]
    tok = AutoTokenizer.from_pretrained(hid, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(hid, revision=rev, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    return tok, model


def batches(order, lens, batch, max_tok=24000):
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and j - i < batch and lens[order[j]] * (j - i + 1) <= max_tok:
            j += 1
        yield order[i:j]; i = j


def run_passages(tok, model, rows, out, batch):
    ids = [tok(r["text"], add_special_tokens=False)["input_ids"] for r in rows]
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    lens = [len(x) for x in ids]; order = sorted(range(len(ids)), key=lambda i: lens[i])
    nL = model.config.get_text_config().num_hidden_layers; D = model.config.get_text_config().hidden_size
    Xe = np.zeros((len(ids), nL, D), np.float16); Xa = np.zeros((len(ids), 7, D), np.float16)
    for idx in batches(order, lens, batch):
        L = max(lens[i] for i in idx)
        inp = torch.full((len(idx), L), pad, dtype=torch.long); att = torch.zeros((len(idx), L), dtype=torch.long)
        for k, i in enumerate(idx):
            inp[k, :lens[i]] = torch.tensor(ids[i]); att[k, :lens[i]] = 1
        with torch.no_grad():
            o = model(input_ids=inp.cuda(), attention_mask=att.cuda(), use_cache=False, output_hidden_states=True)
            hs = torch.stack(o.hidden_states[1:], 0).float()  # [nL, B, T, D]
            m_all = att.cuda().float(); m_ex = m_all.clone(); m_ex[:, 0] = 0
            e = torch.einsum("lbtd,bt->lbd", hs, m_ex) / m_ex.sum(1)[None, :, None]
            a = torch.einsum("lbtd,bt->lbd", hs[19:26], m_all) / m_all.sum(1)[None, :, None]
        for k, i in enumerate(idx):
            Xe[i] = e[:, k].half().cpu().numpy(); Xa[i] = a[:, k].half().cpu().numpy()
        del o, hs
    np.savez(out / "passages.npz", X_excl0=Xe, X_all=Xa, all_layers=np.arange(20, 27), ids=np.array([r["id"] for r in rows]), n_tok=np.array(lens))
    print("passages", Xe.shape, flush=True)


def run_templates(tok, model, rows, out, batch):
    units = []
    for r in rows:
        ctx = tok(r["text"], add_special_tokens=False)["input_ids"]
        for kind in ("fit", "spec"):
            end = tok(r[kind], add_special_tokens=False)["input_ids"]
            units.append((r["id"], kind, ctx, end))
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    res = []
    for s in range(0, len(units), batch):
        grp = units[s:s + batch]; L = max(len(c) + len(e) for _, _, c, e in grp)
        inp = torch.full((len(grp), L), pad, dtype=torch.long); att = torch.zeros((len(grp), L), dtype=torch.long)
        for k, (_, _, c, e) in enumerate(grp):
            x = c + e; inp[k, :len(x)] = torch.tensor(x); att[k, :len(x)] = 1
        with torch.no_grad():
            lg = model(input_ids=inp.cuda(), attention_mask=att.cuda(), use_cache=False).logits.float()
            lp = torch.log_softmax(lg, -1)
        for k, (tid, kind, c, e) in enumerate(grp):
            pos = torch.arange(len(c) - 1, len(c) + len(e) - 1, device=lp.device)
            tl = lp[k, pos, torch.tensor(e, device=lp.device)].cpu().tolist()
            res.append({"id": tid, "ending": kind, "n_ctx": len(c), "n_end": len(e), "lp_tokens": tl, "lp_sum": sum(tl), "lp_mean": sum(tl) / len(tl)})
        del lg, lp
    with (out / "armA.jsonl").open("w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print("templates", len(res), flush=True)


def main(short, out_dir, passages=None, templates=None, batch=16):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    tok, model = load(short)
    print(f"loaded {short} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if templates:
        run_templates(tok, model, read_jsonl(templates), out, batch)
    if passages:
        run_passages(tok, model, read_jsonl(passages), out, batch)
    print(f"done {short} {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: (type(d)(a[a.index(k) + 1]) if d is not None else a[a.index(k) + 1]) if k in a else d  # noqa: E731
    main(a[0], a[1], opt("--passages", None), opt("--templates", None), opt("--batch", 16))

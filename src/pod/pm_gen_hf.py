#!/usr/bin/env python3
"""HF generation with residual-stream interventions at generated positions only (PREREG shared design; after persona-mech pod_m7_gen.py).

    python3 pm_gen_hf.py SHORT JOBS.jsonl OUT.jsonl --basis BASIS.npz [--vectors VEC.npz] [--max-new 384] [--batch 24] [--conditions COND.json]

Job rows: as pm_gen.py (format chat|narrative|raw) plus "condition" (a key of the conditions file) and optional "n" samples.
Conditions file: {"name": [op, op, ...]}, each op = {"type": "proj_out"|"clamp"|"add", "layers": [L, ...], "src": "basis"|"vectors",
  "key": "<array key, optionally with {L} for the layer>", "mu": "<basis key with {L}>" (proj_out/clamp), "target": "<key with {L}>" or number
  (clamp), "coef": float (add; multiplies the stored vector), "unit": true/false (add: normalise the vector first)}.
"none" is the empty list. Interventions apply only on decode steps (sequence length 1) so prompt tokens are never touched; left padding
makes the prefill prompt-only. Rows are appended to OUT as they finish (resumable by id + condition + sample).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pm_common import MODELS, SEED, decoder_layers, load_hf, read_jsonl, render_job, tokenizer

NARR_STOP_TEXTS = ["\nUser:", "\n\nUser:", "\nuser:"]


class Intervene:
    def __init__(self, basis, vectors, layers_ml, conditions):
        self.basis, self.vectors, self.layers_ml, self.conditions = basis, vectors, layers_ml, conditions
        self.ops = {}; self.hs = []

    def _arr(self, src, key, L):
        store = self.basis if src == "basis" else self.vectors
        return torch.tensor(np.asarray(store[key.format(L=L)]), device="cuda", dtype=torch.float32)

    def set(self, cond):
        self.ops = {}
        for op in self.conditions[cond]:
            for L in op["layers"]:
                P = self._arr(op.get("src", "basis"), op["key"], L)
                if P.ndim == 1:
                    P = P[None]
                if op["type"] == "add":
                    v = P[0]
                    if op.get("unit"):
                        v = v / v.norm()
                    self.ops.setdefault(L, []).append(("add", float(op["coef"]) * v))
                else:
                    mu = self._arr("basis", op["mu"], L)
                    if op["type"] == "clamp":
                        t = op["target"]
                        tgt = self._arr("basis", t, L) if isinstance(t, str) else torch.full((P.shape[0],), float(t), device="cuda")
                        tgt = tgt.reshape(-1)
                    else:
                        tgt = torch.zeros(P.shape[0], device="cuda")
                    self.ops.setdefault(L, []).append(("proj", P, mu, tgt))

    def _hook(self, L):
        def f(_m, _i, out):
            ops = self.ops.get(L)
            if not ops:
                return None
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:            # prefill = prompt positions only
                return None
            x = h.float()
            for op in ops:
                if op[0] == "add":
                    x = x + op[1]
                else:
                    _, P, mu, tgt = op
                    c = (x - mu) @ P.T           # coordinates in the frame (…, k)
                    x = x - c @ P + tgt @ P      # set the frame coordinates to the target (0 = frame mean for proj_out)
            x = x.to(h.dtype)
            return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        return f

    def attach(self):
        self.hs = [self.layers_ml[L - 1].register_forward_hook(self._hook(L)) for L in range(1, len(self.layers_ml) + 1)]


def main(short, jobs_f, out_f, basis_f, vectors_f=None, max_new=384, batch=24, cond_f=None):
    torch.manual_seed(SEED)
    tok = tokenizer(short); tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_hf(short); layers = decoder_layers(model)
    basis = np.load(basis_f) if basis_f else None
    vectors = np.load(vectors_f) if vectors_f else None
    conditions = json.loads(Path(cond_f).read_text()) if cond_f else {"none": []}
    conditions.setdefault("none", [])
    iv = Intervene(basis, vectors, layers, conditions); iv.attach()
    jobs = read_jsonl(jobs_f)
    out_p = Path(out_f); out_p.parent.mkdir(parents=True, exist_ok=True); done = set()
    if out_p.exists():
        done = {(r["id"], r["condition"], r["sample"]) for r in read_jsonl(out_p)}
    units = [(j, s) for j in jobs for s in range(j.get("n", 1)) if (j["id"], j["condition"], s) not in done]
    print(f"{len(units)} generations to do ({len(done)} done); conditions {sorted({u[0]['condition'] for u in units})}", flush=True)
    units.sort(key=lambda u: (u[0]["condition"], u[0].get("max_new", max_new), len(json.dumps(u[0]))))
    eos_ids = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()}
    t0 = time.time(); n_done = 0; i = 0
    with out_p.open("a") as f:
        while i < len(units):
            cond = units[i][0]["condition"]; group = []
            while i < len(units) and units[i][0]["condition"] == cond and len(group) < batch:
                group.append(units[i]); i += 1
            iv.set(cond)
            texts = [render_job(tok, short, j) for j, _ in group]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            mn = max(j.get("max_new", max_new) for j, _ in group)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=mn, pad_token_id=tok.pad_token_id, eos_token_id=list(eos_ids))
            plen = enc["input_ids"].shape[1]
            for (j, s), row in zip(group, gen):
                comp = row[plen:].tolist()
                while comp and (comp[-1] == tok.pad_token_id or comp[-1] in eos_ids):
                    comp.pop()
                text = tok.decode(comp, skip_special_tokens=False)
                if j.get("format") in ("narrative", "raw") and j.get("stop"):
                    cut = min([text.find(st) for st in j["stop"] if st in text] or [len(text)])
                    text = text[:cut]
                f.write(json.dumps({**{k: v for k, v in j.items()}, "sample": s, "text": text, "n_gen": len(comp), "finished": len(comp) < mn,
                                    "think_leak": ("<think>" in text or "</think>" in text)}, ensure_ascii=False) + "\n")
            f.flush(); n_done += len(group)
            print(f"{n_done}/{len(units)} [{cond}] batch {len(group)} x {plen}+{gen.shape[1] - plen} tok, {(time.time() - t0) / 60:.1f} min", flush=True)
    print("done", f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: (type(d)(a[a.index(k) + 1]) if d is not None else a[a.index(k) + 1]) if k in a else d  # noqa: E731
    main(a[0], a[1], a[2], opt("--basis", None), opt("--vectors", None), opt("--max-new", 384), opt("--batch", 24), opt("--conditions", None))

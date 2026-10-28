#!/usr/bin/env python3
"""Q6 attractor dynamics: add a vector at layer L on a window of decode steps, record the per-token AA projection at L.

    python3 q6_attractor.py SHORT JOBS.jsonl VEC.npz OUT_PREFIX --layer L [--batch-prompts 4] [--T 225] [--win 20 25]

JOBS rows: {"id", "qid", "sample", "format", ...render_job fields}. Every job is run under every condition of VEC.npz.
VEC.npz: "conds" (names, "none" first), "V" [n_cond, D] (vector added at layer L; zeros for none), "aa" [D] (unit AA at L),
"cax" [D] (unit contrast axis (default - mu)/|.| at L), "mu" [D].
Position convention: response token t (t = 0 = first generated token) has its residual computed on the decode step that inputs
token t, as in pm_acts' teacher-forced response span. The vector is added at layer L's output on the decode steps inputting
tokens win[0]..win[1]-1 only; the prompt (prefill) is never touched. We record, for t = 0..T-1, the projection of the residual at
L onto aa and cax before ('pre') and after ('post') the addition, and the residual norm.
Sampling: temperature 0.6, top-k 20, top-p 0.95 (PREREG shared design), implemented as Gumbel-max with noise fixed per
(qid, sample) so every condition of a prompt sees the same noise (common random numbers): trajectories are identical until the
perturbation changes the logits. EOS / end-of-turn tokens are suppressed for all T tokens (min-new-tokens semantics); the step at
which the unsuppressed sampler would have ended is recorded ('natural_end'), as is the first token whose prefix contains a
narrative stop string ('stop_tok').
Outputs: OUT_PREFIX.npz (proj_pre, proj_post, cax_pre, cax_post, norm: float32 [N, T]; tokens int32 [N, T+1]) and
OUT_PREFIX.jsonl (one row per N: id, qid, sample, cond, natural_end, stop_tok, text).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

from pm_common import NARRATIVE_STOPS, SEED, decoder_layers, load_hf, read_jsonl, render_job, tokenizer

TEMP, TOPK, TOPP = 0.6, 20, 0.95


class Recorder:
    def __init__(self, aa, cax):
        self.aa, self.cax = aa, cax
        self.V = None; self.win = (20, 25); self.step = 0; self.rec = []

    def reset(self, V):
        self.V = V; self.step = 0; self.rec = []

    def hook(self, _m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:                       # prefill: prompt positions, never touched
            return None
        t = self.step; self.step += 1
        x = h[:, -1].float()
        pre_a, pre_c = x @ self.aa, x @ self.cax
        if self.win[0] <= t < self.win[1]:
            x = x + self.V
            h = h.clone(); h[:, -1] = x.to(h.dtype)
            xr = h[:, -1].float()                # post-add as seen downstream (after the bf16 cast)
        else:
            xr = x
        self.rec.append(torch.stack([pre_a, xr @ self.aa, pre_c, xr @ self.cax, xr.norm(dim=-1)], 1).cpu())
        if self.win[0] <= t < self.win[1]:
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return None


class CRNSampler(LogitsProcessor):
    """Temperature/top-k/top-p sampling by Gumbel-max with per-row fixed noise; returns a one-hot score row so greedy picks it."""
    def __init__(self, plen, gumbel, eos_ids):
        self.plen, self.g, self.eos = plen, gumbel, list(eos_ids)   # g: [B, T+1, TOPK] on cuda
        self.natural_end = torch.full((gumbel.shape[0],), -1, dtype=torch.long)

    def _pick(self, x, s):
        v, idx = x.topk(TOPK, dim=-1)
        p = torch.softmax(v, -1); cum = p.cumsum(-1)
        keep = (cum - p) <= TOPP; keep[:, 0] = True
        sc = torch.where(keep, torch.log(p) + self.g[:, s], torch.full_like(p, -float("inf")))
        return idx.gather(1, sc.argmax(-1, keepdim=True)).squeeze(1)

    def __call__(self, input_ids, scores):
        s = input_ids.shape[1] - self.plen
        x = scores.float() / TEMP
        nat = self._pick(x, s)
        ended = torch.isin(nat, torch.tensor(self.eos, device=nat.device)).cpu() & (self.natural_end < 0)
        self.natural_end[ended] = s
        x[:, self.eos] = -float("inf")
        tok = self._pick(x, s)
        out = torch.full_like(scores, -float("inf")); out.scatter_(1, tok[:, None], 0.0)
        return out


def stop_token(tok, ids):
    """First token index t such that decode(ids[:t+1]) contains a narrative stop string; -1 if none."""
    text = tok.decode(ids, skip_special_tokens=False)
    cut = min([text.find(st) for st in NARRATIVE_STOPS if st in text] or [-1])
    if cut < 0:
        return -1
    lo, hi = 0, len(ids) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if any(st in tok.decode(ids[:mid + 1], skip_special_tokens=False) for st in NARRATIVE_STOPS):
            hi = mid
        else:
            lo = mid + 1
    return lo


def main(short, jobs_f, vec_f, out_prefix, layer, batch_prompts=4, T=225, win=(20, 25)):
    torch.manual_seed(SEED)
    tok = tokenizer(short); tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    vec = np.load(vec_f)
    conds = [str(c) for c in vec["conds"]]; assert conds[0] == "none"
    V = torch.tensor(vec["V"], dtype=torch.float32, device="cuda")
    aa = torch.tensor(vec["aa"], dtype=torch.float32, device="cuda"); cax = torch.tensor(vec["cax"], dtype=torch.float32, device="cuda")
    model = load_hf(short); layers = decoder_layers(model)
    rec = Recorder(aa, cax); rec.win = win
    layers[layer - 1].register_forward_hook(rec.hook)
    eos_ids = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()}
    eos_ids = {e for e in eos_ids if e is not None}
    jobs = read_jsonl(jobs_f)
    print(f"{short} L={layer} jobs={len(jobs)} conds={len(conds)} T={T} win={win} eos={sorted(eos_ids)}", flush=True)
    P = {k: [] for k in ("proj_pre", "proj_post", "cax_pre", "cax_post", "norm", "tokens")}; meta = []
    t0 = time.time()
    for b in range(0, len(jobs), batch_prompts):
        grp = jobs[b:b + batch_prompts]
        rows = [(j, ci) for j in grp for ci in range(len(conds))]
        texts = [render_job(tok, short, j) for j, _ in rows]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        plen = enc["input_ids"].shape[1]
        g = []
        for j, _ in rows:
            rng = np.random.default_rng(SEED + 1000 * int(j["qid"]) + int(j["sample"]))
            g.append(rng.gumbel(size=(T + 1, TOPK)).astype(np.float32))
        g = torch.tensor(np.stack(g), device="cuda")
        samp = CRNSampler(plen, g, eos_ids)
        rec.reset(V[[ci for _, ci in rows]])
        with torch.no_grad():
            out = model.generate(**enc, generation_config=GenerationConfig(do_sample=False, max_new_tokens=T + 1, pad_token_id=tok.pad_token_id,
                                 eos_token_id=sorted(eos_ids)), logits_processor=LogitsProcessorList([samp]))
        R = torch.stack(rec.rec[:T], 1).numpy()           # [B, T, 5]
        assert R.shape[1] == T, R.shape
        comp = out[:, plen:].cpu().numpy().astype(np.int32)
        assert comp.shape[1] == T + 1, comp.shape
        for k, key in enumerate(("proj_pre", "proj_post", "cax_pre", "cax_post", "norm")):
            P[key].append(R[:, :, k])
        P["tokens"].append(comp)
        for r, (j, ci) in enumerate(rows):
            ids = comp[r].tolist()
            meta.append({"id": j["id"], "qid": j["qid"], "sample": j["sample"], "format": j["format"], "cond": conds[ci],
                         "natural_end": int(samp.natural_end[r]), "stop_tok": stop_token(tok, ids) if j["format"] == "narrative" else -1,
                         "text": tok.decode(ids, skip_special_tokens=False)})
        print(f"{b + len(grp)}/{len(jobs)} prompts, batch {len(rows)} x {plen}+{T + 1}, {(time.time() - t0) / 60:.1f} min", flush=True)
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_prefix + ".npz", **{k: np.concatenate(v) for k, v in P.items()}, conds=np.array(conds), layer=layer, win=np.array(win))
    with open(out_prefix + ".jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("done", f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    w = (int(a[a.index("--win") + 1]), int(a[a.index("--win") + 2])) if "--win" in a else (20, 25)
    main(a[0], a[1], a[2], a[3], int(a[a.index("--layer") + 1]), opt("--batch-prompts", 4), opt("--T", 225), w)

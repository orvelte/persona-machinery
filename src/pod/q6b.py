#!/usr/bin/env python3
"""Q6b context-perturbation basin test (PREREG Q6b). One model x format per invocation, three stages in one process order:

    python3 q6b.py SHORT FMT LSTAR OUT_DIR        # gen (vLLM) -> refit half-A AA + replay (HF); writes OUT_DIR/q6b_<SHORT>_<FMT>.npz/.json

Inputs on the pod: out/m0/gen_<SHORT>_<FMT>.jsonl (M0 generations with prompt_ids/completion_ids), data/Q6/roles.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pm_common import N_LAYERS, SEED, load_hf, read_jsonl, tokenizer  # noqa: E402
import os
ROOT = Path(os.environ.get("Q6B_ROOT", HERE)); GEN_DIR = Path(os.environ.get("Q6B_GEN", HERE / "out/m0"))

PREFIX = 48
N_NEW = 160
MIN_NEW = 150
SAMPLES = 4


def build(short, fmt):
    rows = read_jsonl(GEN_DIR / f"gen_{short}_{fmt}.jsonl")
    roles = [x["role"] for x in json.loads((ROOT / "data/Q6/roles.json").read_text())[f"{short}_{fmt}"]["roles"]]
    assert len(roles) == 5, roles
    qB = sorted({r["qid"] for r in rows if r["half"] == "B"})
    by = {(r["role"], r["qid"], r["sample"]): r for r in rows}
    jobs, dropped = [], []
    for i, q in enumerate(qB):
        dflt = by[("default", q, 0)]
        base_prompt = list(dflt["prompt_ids"])
        other = by[("default", qB[(i + 1) % len(qB)], 0)]
        conds = [("default", "default", dflt), ("null", "null", other)] + [("role", r, by.get((r, q, 0))) for r in roles]
        for cond, src, row in conds:
            if row is None or len(row["completion_ids"]) < PREFIX:
                dropped.append([cond, src, q]); continue
            pre = list(row["completion_ids"][:PREFIX])
            for s in range(SAMPLES):
                jobs.append({"q": q, "cond": cond, "src": src, "sample": s, "ids": base_prompt + pre, "n_prompt": len(base_prompt)})
    return jobs, dropped, roles, rows


def gen(short, jobs):
    from vllm import LLM, SamplingParams
    from pm_common import MODELS
    m = MODELS[short]; tok = tokenizer(short)
    eos = [t for t in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tok.get_vocab() else None,
                       tok.convert_tokens_to_ids("<|endoftext|>") if "<|endoftext|>" in tok.get_vocab() else None} if t is not None]
    kw = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if m["family"] == "qwen" else {}
    llm = LLM(model=m["hid"], revision=m["rev"], tokenizer_revision=m["rev"], dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=SEED, **kw)
    sps = [SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=N_NEW, min_tokens=MIN_NEW, seed=SEED + 1000 * j["sample"] + j["q"], stop_token_ids=eos)
           for j in jobs]
    res = llm.generate([{"prompt_token_ids": j["ids"]} for j in jobs], sps)
    for j, r in zip(jobs, res):
        j["cont"] = list(r.outputs[0].token_ids)
    del llm


def refit_and_replay(short, jobs, rows, layers):
    import torch
    model = load_hf(short)
    tok = tokenizer(short); pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    def run(seqs, spans, per_token=False, bs=16):
        out = []
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        res = [None] * len(seqs)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]; T = max(len(seqs[i]) for i in idx)
            inp = torch.full((len(idx), T), pad, dtype=torch.long); att = torch.zeros((len(idx), T), dtype=torch.long)
            for k, i in enumerate(idx):
                inp[k, :len(seqs[i])] = torch.tensor(seqs[i]); att[k, :len(seqs[i])] = 1
            with torch.no_grad():
                o = model(input_ids=inp.cuda(), attention_mask=att.cuda(), use_cache=False, output_hidden_states=True)
            for k, i in enumerate(idx):
                a, b = spans[i]
                hs = [o.hidden_states[L][k, a:b].float() for L in layers]      # hidden_states[L] = output of block L (L <= 31: pre final norm)
                res[i] = [h.cpu().numpy() if per_token else h.mean(0).cpu().numpy() for h in hs]
            del o
        return res
    # 1) refit AA on half A (response means), per layer
    A = [r for r in rows if r["half"] == "A" and r["n_gen"] > 0]
    means = run([list(r["prompt_ids"]) + list(r["completion_ids"]) for r in A], [(r["n_prompt"], r["n_prompt"] + r["n_gen"]) for r in A])
    aa = {}
    for li, L in enumerate(layers):
        by = {}
        for r, m in zip(A, means):
            by.setdefault(r["role"], []).append(m[li])
        roles = sorted(k for k in by if k != "default" and len(by[k]) >= 3)
        M = np.stack([np.mean(by[k], 0) for k in roles]); mu = M.mean(0)
        _, _, Vt = np.linalg.svd(M - mu, full_matrices=False); v = Vt[0]
        dflt = np.mean(by["default"], 0)
        if (dflt - mu) @ v < 0:
            v = -v
        aa[L] = (v, mu, float(np.std((M - mu) @ v)))
    # 2) replay continuations, per-token projection
    seqs = [j["ids"] + j["cont"] for j in jobs]; spans = [(len(j["ids"]), len(j["ids"]) + len(j["cont"])) for j in jobs]
    tr = run(seqs, spans, per_token=True)
    P = np.full((len(jobs), len(layers), N_NEW), np.nan, np.float32)
    for i, t in enumerate(tr):
        for li, L in enumerate(layers):
            v, mu, _ = aa[L]; p = (t[li] - mu) @ v; P[i, li, :len(p)] = p
    return P, aa


def main(short, fmt, lstar, out_dir):
    t0 = time.time(); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    layers = [int(lstar), min(int(lstar) + 4, 31)]
    jobs, dropped, roles, rows = build(short, fmt)
    print(f"{short} {fmt}: {len(jobs)} continuations, dropped {len(dropped)} (cond, src, q) pairs, layers {layers}", flush=True)
    gen(short, jobs)
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    print(f"gen done {(time.time() - t0) / 60:.1f} min; mean cont {np.mean([len(j['cont']) for j in jobs]):.0f} tok", flush=True)
    P, aa = refit_and_replay(short, jobs, rows, layers)
    meta = [{k: j[k] for k in ("q", "cond", "src", "sample")} | {"n_cont": len(j["cont"])} for j in jobs]
    np.savez_compressed(out / f"q6b_{short}_{fmt}.npz", P=P, layers=np.array(layers), role_sd=np.array([aa[L][2] for L in layers]))
    (out / f"q6b_{short}_{fmt}.json").write_text(json.dumps({"meta": meta, "dropped": dropped, "roles": roles, "layers": layers,
                                                              "minutes": (time.time() - t0) / 60}))
    print(f"done {short} {fmt} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])

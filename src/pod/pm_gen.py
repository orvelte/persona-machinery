#!/usr/bin/env python3
"""vLLM generation for one model from a jobs file (PREREG shared design: T 0.6, top_p 0.95, top_k 20, 1 sample per row).

    python3 pm_gen.py SHORT JOBS.jsonl OUT.jsonl [--max-new 384] [--max-model-len 4096]

Job rows: {"id", "format": chat|narrative|raw, (system, user) | (description, speaker, question) | (text), "stop": [...],
           "max_new": int, "seed": int, ...meta}. Every other key is copied to the output row.
Output rows add: prompt_ids, completion_ids (stop text and trailing EOS removed, so every completion token is response),
text, finish, n_prompt, n_gen, think_leak (a think tag appears in the completion).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from pm_common import MODELS, SEED, read_jsonl, render_job, tokenizer, write_jsonl


def trim_to_text(tok, ids, text):
    """vLLM truncates `text` at a stop string but may keep stop tokens in token_ids; drop tokens beyond the text."""
    if tok.decode(ids, skip_special_tokens=False) == text:
        return ids
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok.decode(ids[:mid], skip_special_tokens=False)) <= len(text):
            lo = mid
        else:
            hi = mid - 1
    return ids[:lo]


def main(short, jobs_f, out_f, max_new=384, max_model_len=4096):
    from vllm import LLM, SamplingParams
    m = MODELS[short]; tok = tokenizer(short)
    jobs = read_jsonl(jobs_f)
    eos_ids = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()}
    prompts, sps = [], []
    for j in jobs:
        txt = render_job(tok, short, j)
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        j["_prompt_ids"] = ids
        prompts.append({"prompt_token_ids": ids})
        sps.append(SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=j.get("max_new", max_new), seed=SEED + j.get("seed", 0),
                                  stop=j.get("stop") or None, include_stop_str_in_output=False))
    print(f"{len(jobs)} prompts for {short}; max prompt {max(len(p['prompt_token_ids']) for p in prompts)} tok", flush=True)
    kw = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if m["family"] == "qwen" else {}
    llm = LLM(model=m["hid"], revision=m["rev"], tokenizer_revision=m["rev"], dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=0.88,
              max_num_seqs=256, seed=SEED, **kw)
    t0 = time.time(); res = llm.generate(prompts, sps)
    rows = []
    for j, r in zip(jobs, res):
        o = r.outputs[0]; c = list(o.token_ids)
        while c and c[-1] in eos_ids:
            c.pop()
        c = trim_to_text(tok, c, o.text) if o.finish_reason == "stop" and j.get("stop") else c
        text = tok.decode(c, skip_special_tokens=False)
        pid = j.pop("_prompt_ids")
        rows.append({**j, "prompt_ids": pid, "completion_ids": c, "text": text, "finish": o.finish_reason, "n_prompt": len(pid), "n_gen": len(c),
                     "think_leak": ("<think>" in text or "</think>" in text)})
    write_jsonl(out_f, rows)
    print(f"done {short}: {len(rows)} rows in {(time.time() - t0) / 60:.1f} min; mean gen {sum(r['n_gen'] for r in rows) / len(rows):.0f} tok; "
          f"finished-by-length {sum(r['finish'] == 'length' for r in rows)}; think_leak {sum(r['think_leak'] for r in rows)}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    main(a[0], a[1], a[2], opt("--max-new", 384), opt("--max-model-len", 4096))

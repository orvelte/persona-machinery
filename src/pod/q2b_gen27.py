#!/usr/bin/env python3
"""Q2b: vLLM raw-document (frame ii) generation for the 27B checkpoints (PREREG Q2b; persona-mech pins). Same sampling as pm_gen.py
(T 0.6, top_p 0.95, top_k 20, per-row seed SEED + seed), bf16, max_model_len 8192, gpu_memory_utilization 0.9.

    python3 q2b_gen27.py HF_ID REV JOBS.jsonl OUT.jsonl
"""
from __future__ import annotations

import sys
import time

from pm_common import SEED, read_jsonl, write_jsonl


def main(hid, rev, jobs_f, out_f):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(hid, revision=rev)
    jobs = read_jsonl(jobs_f)
    assert all(j["format"] == "raw" for j in jobs)
    prompts = [{"prompt_token_ids": tok(j["text"], add_special_tokens=False)["input_ids"]} for j in jobs]
    sps = [SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=j["max_new"], seed=SEED + j["seed"], stop=j["stop"], include_stop_str_in_output=False) for j in jobs]
    llm = LLM(model=hid, revision=rev, tokenizer_revision=rev, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.9, max_num_seqs=128,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=SEED)
    t0 = time.time(); res = llm.generate(prompts, sps)
    rows = [{**{k: v for k, v in j.items() if k != "text"}, "doc_chars": len(j["text"]), "n_prompt": len(p["prompt_token_ids"]), "text": r.outputs[0].text,
             "finish": r.outputs[0].finish_reason, "n_gen": len(r.outputs[0].token_ids), "hid": hid, "rev": rev} for j, p, r in zip(jobs, prompts, res)]
    write_jsonl(out_f, rows)
    print(f"done {hid}: {len(rows)} rows in {(time.time() - t0) / 60:.1f} min; finished-by-length {sum(r['finish'] == 'length' for r in rows)}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])

#!/usr/bin/env bash
# Agent A combined GPU chain (Q1 patch, Q3 panel, OLMo Q1 attribution). Each step is resumable / idempotent.
set -x
cd /workspace/job
mkdir -p out/q1 out/q3
python3 q1_patch.py qwen_it data/Q1/patch_jobs.jsonl data/Q1/attr_qwen_it.npz data/Q3/q3_basis_qwen_it.npz data/Q1/patch_spec.json 24 out/q1/patch_qwen_it.jsonl --batch 30 || echo Q1PATCH_FAIL
python3 pm_gen_hf.py qwen_it data/Q3/jobs.jsonl out/q3/gen_qwen_it.jsonl --basis data/Q3/q3_basis_qwen_it.npz --conditions data/Q3/conditions.json --batch 48 || echo Q3GEN_FAIL
python3 q1_attr.py olmo_rl out/m0/gen_olmo_rl_chat.jsonl data/Q1/q1_basis_olmo_rl.npz 23 out/q1/attr_olmo_rl --batch 8 --contrast || echo OLMOATTR_FAIL
echo CHAIN_DONE

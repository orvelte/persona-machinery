#!/usr/bin/env bash
# Q4.4 generations: narrative (vLLM) on four models, then qwen_it chat with AA ablation / nulls / positive control (HF hooks).
set -e
for m in qwen_base qwen_it olmo_base olmo_rl; do
  [ -f out/q4/q44_narr_$m.jsonl ] || python3 pm_gen.py $m data/Q4/q44_narr_jobs.jsonl out/q4/q44_narr_$m.jsonl --max-new 80 --max-model-len 2048
done
python3 pm_gen_hf.py qwen_it data/Q4/q44_chat_jobs.jsonl out/q4/q44_chat_qwen_it.jsonl --basis out/m0/basis_qwen_it_chat.npz --conditions data/Q4/q44_conditions.json --max-new 256 --batch 32
echo Q4GEN DONE

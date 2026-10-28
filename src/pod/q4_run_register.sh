#!/usr/bin/env bash
# Q4.2b register control: forward passes on the six small models (passages only).
set -e
for m in qwen_base qwen_it olmo_base olmo_sft olmo_dpo olmo_rl; do
  [ -f out/q4/reg_$m/passages.npz ] || python3 q4_fwd.py $m out/q4/reg_$m --passages data/Q4/register_passages.jsonl --batch 16
done
echo Q4REG DONE

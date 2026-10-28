#!/usr/bin/env bash
# Q4 battery activations + Q4b Arm A log-probs, six small models (forward only). Run from /workspace/job.
set -e
for m in qwen_base qwen_it olmo_base olmo_sft olmo_dpo olmo_rl; do
  [ -f out/q4/fwd_$m/armA.jsonl ] && [ -f out/q4/fwd_$m/passages.npz ] && continue
  python3 q4_fwd.py $m out/q4/fwd_$m --passages data/Q4/passages_all.jsonl --templates data/Q4b/templates.jsonl --batch 16
done
echo Q4FWD DONE

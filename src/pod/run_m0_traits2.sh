#!/usr/bin/env bash
set -euo pipefail; cd /workspace/job
for m in qwen_it olmo_rl; do
  [ -f out/m0/traits2_gen_$m.jsonl ] || python3 pm_gen.py "$m" data/m0/jobs_traits2_$m.jsonl out/m0/traits2_gen_$m.jsonl --max-new 256
  [ -f out/m0/traits2_acts_$m/acts.npz ] || python3 pm_acts.py "$m" out/m0/traits2_gen_$m.jsonl out/m0/traits2_acts_$m --batch 16
done
echo "TRAITS2 COMPLETE"

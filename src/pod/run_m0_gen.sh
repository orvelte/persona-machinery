#!/usr/bin/env bash
# M0 generation only (vLLM), all pairs; acts/basis run afterwards with run_m0.sh (which skips finished steps).
set -euo pipefail; cd /workspace/job
for pair in qwen_it:chat qwen_it:narrative qwen_base:narrative olmo_rl:chat olmo_rl:narrative olmo_base:narrative olmo_sft:chat olmo_dpo:chat; do
  m="${pair%%:*}"; fmt="${pair##*:}"
  [ -f out/m0/gen_${m}_$fmt.jsonl ] || python3 pm_gen.py "$m" data/m0/jobs_${m}_$fmt.jsonl out/m0/gen_${m}_$fmt.jsonl
  echo "=== gen done: $m $fmt $(date -u +%H:%M)"
done
for m in qwen_it olmo_rl; do
  [ -f out/m0/traits_gen_$m.jsonl ] || python3 pm_gen.py "$m" data/m0/jobs_traits_$m.jsonl out/m0/traits_gen_$m.jsonl --max-new 256
  echo "=== traits gen done: $m $(date -u +%H:%M)"
done
echo "M0 GEN COMPLETE"

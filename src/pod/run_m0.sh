#!/usr/bin/env bash
# M0 chain on the pod (PREREG M0 steps 1-3, 5, 6): for each model x format, generate -> replay -> basis. Trait prompts for qwen_it / olmo_rl.
#   bash run_m0.sh [MODEL_FMT ...]      # default: all eight model x format pairs + traits
set -euo pipefail
cd /workspace/job
PAIRS=("$@"); if [ ${#PAIRS[@]} -eq 0 ]; then PAIRS=(qwen_it:chat qwen_it:narrative qwen_base:narrative olmo_rl:chat olmo_rl:narrative olmo_base:narrative olmo_sft:chat olmo_dpo:chat qwen_it:traits olmo_rl:traits); fi
for pair in "${PAIRS[@]}"; do
  m="${pair%%:*}"; fmt="${pair##*:}"
  if [ "$fmt" = traits ]; then
    [ -f out/m0/traits_gen_$m.jsonl ] || python3 pm_gen.py "$m" data/m0/jobs_traits_$m.jsonl out/m0/traits_gen_$m.jsonl --max-new 256
    [ -f out/m0/traits_acts_$m/acts.npz ] || python3 pm_acts.py "$m" out/m0/traits_gen_$m.jsonl out/m0/traits_acts_$m --batch 8
    continue
  fi
  [ -f out/m0/gen_${m}_$fmt.jsonl ] || python3 pm_gen.py "$m" data/m0/jobs_${m}_$fmt.jsonl out/m0/gen_${m}_$fmt.jsonl
  [ -f out/m0/acts_${m}_$fmt/acts.npz ] || python3 pm_acts.py "$m" out/m0/gen_${m}_$fmt.jsonl out/m0/acts_${m}_$fmt --batch 8
  [ -f out/m0/basis_${m}_$fmt.npz ] || python3 pm_basis.py out/m0/acts_${m}_$fmt out/m0/basis_${m}_$fmt
  echo "=== M0 done: $m $fmt $(date -u +%H:%M)"
done
echo "M0 CHAIN COMPLETE"

#!/usr/bin/env bash
# Q6b chain (PREREG Q6b order). Each pair independent.
cd /workspace/job
for spec in olmo_base:narrative:22 olmo_sft:chat:23 olmo_dpo:chat:23 olmo_rl:chat:23 qwen_base:narrative:22 qwen_it:chat:24 qwen_it:narrative:22; do
  IFS=: read m f L <<< "$spec"
  [ -f out/q6b/q6b_${m}_${f}.npz ] && continue
  python3 q6b.py $m $f $L out/q6b > logs/q6b_${m}_${f}.log 2>&1; echo "${m}_${f} EXIT=$?" >> logs/q6b_chain.log
done
echo CHAIN_DONE >> logs/q6b_chain.log

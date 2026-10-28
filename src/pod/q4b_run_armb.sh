#!/usr/bin/env bash
# Q4b Arm B: base models in an HHH few-shot transcript (vLLM). Arg: jobs file (q4pod/armb_jobs.jsonl or _esc), suffix ("" or _esc), models.
set -e
JOBS=$1; SUF=$2; shift 2; [ "$SUF" = "-" ] && SUF=""
for m in "$@"; do
  python3 pm_gen.py $m $JOBS out/q4/armb_gen_${m}${SUF}.jsonl --max-new 1500 --max-model-len 6144
done
echo Q4BARMB DONE

#!/usr/bin/env bash
# Q4b Arm A on the two 27B models (forward only, bf16, one load each).
set -e
for m in q38 q36; do
  [ -f out/q4/fwd_$m/armA.jsonl ] || python3 q4_fwd.py $m out/q4/fwd_$m --templates data/Q4b/templates.jsonl --batch 8
done
echo DONE27

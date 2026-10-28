#!/usr/bin/env bash
# Route to the job pod created by src/pod/pod.py create.
#   bash src/pod/job.sh setup                 # torch/transformers/vllm versions; pip extras
#   bash src/pod/job.sh push FILE...          # rsync files/dirs to /workspace/job/
#   bash src/pod/job.sh run NAME CMD...       # start CMD detached in /workspace/job; log in logs/NAME.log, logs/NAME.done at exit
#   bash src/pod/job.sh status NAME           # tail of the log + RUNNING/EXIT
#   bash src/pod/job.sh pull REMOTE_DIR LOCAL_DIR
#   bash src/pod/job.sh sh CMD...             # run a command synchronously
#   bash src/pod/job.sh gpu
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
read -r IP PORT < "$HERE/.ssh${POD_SLOT:-}"
SSH=(ssh -i "$HERE/ssh_key${POD_SLOT:-}" -p "$PORT" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HERE/known_hosts" -o ConnectTimeout=20 -o ServerAliveInterval=30 "root@$IP")
RSH="ssh -i $HERE/ssh_key${POD_SLOT:-} -p $PORT -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HERE/known_hosts"
J=/workspace/job
case "${1:-}" in
  setup)
    "${SSH[@]}" "mkdir -p $J/logs $J/out && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && df -h /workspace | tail -1 && python3 -c 'import torch, transformers, vllm; print(\"torch\", torch.__version__, \"transformers\", transformers.__version__, \"vllm\", vllm.__version__, \"cuda\", torch.cuda.is_available())' && pip install -q flash-linear-attention datasets scikit-learn scipy 2>&1 | tail -1; python3 -c 'import fla; print(\"fla ok\")' || echo 'fla not available'" ;;
  push)
    shift; "${SSH[@]}" "mkdir -p $J/logs $J/out"
    rsync -rltz --no-owner --no-group -e "$RSH" "$@" "root@$IP:$J/"; echo pushed ;;
  run)
    name="$2"; shift 2
    "${SSH[@]}" -n "cd $J && rm -f logs/$name.done && (setsid nohup bash -c 'export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1; $* ; echo EXIT=\$? > logs/$name.done' > logs/$name.log 2>&1 < /dev/null &) ; echo started $name" ;;
  status)
    "${SSH[@]}" "tail -n ${3:-15} $J/logs/$2.log 2>/dev/null; cat $J/logs/$2.done 2>/dev/null || echo RUNNING" ;;
  pull)
    mkdir -p "$3"; rsync -rltz --no-owner --no-group -e "$RSH" "root@$IP:$2/" "$3/"; du -sh "$3" ;;
  sh)
    shift; "${SSH[@]}" "cd $J && $*" ;;
  gpu)
    "${SSH[@]}" "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; df -h /workspace | tail -1" ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac

#!/usr/bin/env python3
"""RunPod job pod for this repo (adapted from vendor/splitbrain/internals/runpod/pod.py; same API, repo-local paths).

    .venv/bin/python src/pod/pod.py create [--gpu "NVIDIA H100 80GB HBM3"] [--volume 300] [--disk 100]   # PAID
    .venv/bin/python src/pod/pod.py status | terminate | gpus

Needs RUNPOD_API_KEY in .env (never printed). The pod runs sshd; use src/pod/job.sh to push, run, pull.
State files (gitignored): src/pod/.pod_id, src/pod/.ssh (ip port), src/pod/ssh_key[.pub], src/pod/known_hosts.
`create` tries each GPU in FALLBACK until one gets a public IP within 3 minutes (a stuck pod is deleted).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ENV = REPO / ".env"
import os
SLOT = os.environ.get("POD_SLOT", "")
POD_FILE, SSH_FILE, SSH_KEY = HERE / f".pod_id{SLOT}", HERE / f".ssh{SLOT}", HERE / f"ssh_key{SLOT}"
API = "https://rest.runpod.io/v1"
IMAGE = "vllm/vllm-openai:v0.32.0"
FALLBACK = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe", "NVIDIA H200", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
JOB_START = ("apt-get update -qq && apt-get install -y -qq openssh-server rsync >/dev/null && mkdir -p /root/.ssh /run/sshd "
             "&& echo \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && exec /usr/sbin/sshd -D")


def _key(name: str) -> str:
    v = dotenv_values(ENV).get(name)
    if not v:
        sys.exit(f"{name} is missing from .env")
    return v


def _req(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {_key('RUNPOD_API_KEY')}", "Content-Type": "application/json",
                                          "User-Agent": "personawhitebox-pod/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt.strip() else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"RunPod API {method} {path}: HTTP {e.code} {e.read().decode()[:400]}")


def log_spend(note: str, usd: float, module: str = "pod") -> None:
    with (REPO / "data/spend.jsonl").open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "module": module, "usd": round(usd, 4), "note": note}) + "\n")


def gpus() -> None:
    for g in _req("GET", "/gputypes"):
        if any(k in g.get("id", "") for k in ("H100", "H200", "A100")):
            print(g.get("id"), "| mem", g.get("memoryInGb"), "| secure $/h", g.get("securePrice"), "| lowest", g.get("lowestPrice"))


def create(gpu: str | None, volume: int, disk: int, hf_token: bool = True) -> None:
    if POD_FILE.exists():
        sys.exit(f"a pod is already recorded ({POD_FILE.read_text().strip()}); terminate it first")
    if not SSH_KEY.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "personawhitebox-pod", "-f", str(SSH_KEY)], check=True)
    env = {"HF_HOME": "/workspace/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PUBLIC_KEY": SSH_KEY.with_suffix(".pub").read_text().strip()}
    if hf_token:
        env["HF_TOKEN"] = _key("HF_TOKEN")
    for g in ([gpu] if gpu else FALLBACK):
        body = {"name": f"pwb-job{SLOT}", "imageName": IMAGE, "gpuTypeIds": [g], "gpuCount": 1, "cloudType": "SECURE", "computeType": "GPU",
                "containerDiskInGb": disk, "volumeInGb": volume, "volumeMountPath": "/workspace", "ports": ["22/tcp"], "env": env,
                "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [JOB_START]}
        try:
            pod = _req("POST", "/pods", body)
        except RuntimeError as e:
            print(f"{g}: {str(e)[:200]}"); continue
        pid = pod.get("id")
        if not pid:
            print(f"{g}: no pod id: {str(pod)[:200]}"); continue
        POD_FILE.write_text(pid)
        print(f"created pod {pid} on {g}; cost/hr: {pod.get('costPerHr')}", flush=True)
        t0 = time.time()
        for n in range(90):
            p = _req("GET", f"/pods/{pid}"); ip, pm = p.get("publicIp"), (p.get("portMappings") or {})
            port = pm.get("22") or pm.get(22)
            if ip and port:
                SSH_FILE.write_text(f"{ip} {port}\n"); print(f"SSH endpoint saved ({(time.time() - t0) / 60:.1f} min); cost/hr {p.get('costPerHr')}")
                (HERE / f".cost_per_hr{SLOT}").write_text(str(p.get("costPerHr") or pod.get("costPerHr") or ""))
                (HERE / f".created{SLOT}").write_text(str(time.time()))
                return
            if n >= 18 and not ip:
                _req("DELETE", f"/pods/{pid}"); POD_FILE.unlink(missing_ok=True)
                log_spend(f"stuck pod {pid} on {g} deleted after 3 min", 0.2)
                print(f"{g}: no public IP after 3 min; pod {pid} terminated"); break
            time.sleep(10)
    sys.exit("no GPU type produced a reachable pod")


def status() -> None:
    p = _req("GET", f"/pods/{POD_FILE.read_text().strip()}")
    print(json.dumps({k: p.get(k) for k in ("id", "desiredStatus", "lastStatusChange", "costPerHr", "machine") if k in p}, indent=1, default=str)[:600])


def terminate() -> None:
    pid = POD_FILE.read_text().strip() if POD_FILE.exists() else sys.exit("no pod recorded")
    try:
        c = float((HERE / f".cost_per_hr{SLOT}").read_text() or 0); t0 = float((HERE / f".created{SLOT}").read_text())
        log_spend(f"pod {pid} terminated after {(time.time() - t0) / 3600:.2f} h", c * (time.time() - t0) / 3600)
    except Exception as e:  # noqa: BLE001
        print("spend not logged:", e)
    _req("DELETE", f"/pods/{pid}")
    for f in (POD_FILE, SSH_FILE, HERE / f".cost_per_hr{SLOT}", HERE / f".created{SLOT}"):
        f.unlink(missing_ok=True)
    print("pod terminated")


if __name__ == "__main__":
    a = sys.argv[1:]; cmd = a[0] if a else "status"
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    if cmd == "create":
        create(opt("--gpu", None), int(opt("--volume", "300")), int(opt("--disk", "100")))
    elif cmd == "status":
        status()
    elif cmd == "terminate":
        terminate()
    elif cmd == "gpus":
        gpus()
    else:
        sys.exit(__doc__)

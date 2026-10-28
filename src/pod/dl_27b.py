#!/usr/bin/env python3
"""Download the two persona-mech 27B checkpoints for Q2b / Q4b (pins from persona-mech src/pod/run_m5x.sh)."""
import os, time
from huggingface_hub import snapshot_download
for hid, rev in (("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"), ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9")):
    t0 = time.time(); p = snapshot_download(hid, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "merges.txt", "vocab.json"], token=os.environ.get("HF_TOKEN"))
    print(f"{hid} -> {p} ({(time.time()-t0)/60:.1f} min)", flush=True)
print("27B DONE", flush=True)

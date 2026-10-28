#!/usr/bin/env python3
"""Download the six pinned checkpoints into HF_HOME (safetensors + configs + tokenizer only)."""
import os, sys, time
from huggingface_hub import snapshot_download
MODELS = {"qwen_base": ("Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404"),
          "qwen_it": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
          "olmo_base": ("allenai/Olmo-3-1025-7B", "a81bae42db3975be1671e27b9c9a56da1a9f980f"),
          "olmo_sft": ("allenai/Olmo-3-7B-Think-SFT", "6ff857587e040d6d523a3d5f3a56e918f5401d66"),
          "olmo_dpo": ("allenai/Olmo-3-7B-Think-DPO", "7b18bf927b430ff06376fdfa5610eb3b1b6a5c38"),
          "olmo_rl": ("allenai/Olmo-3-7B-Think", "d97e442d7cc678210054dbcc9b440894d62c89a4")}
order = sys.argv[1:] or list(MODELS)
for short in order:
    hid, rev = MODELS[short]; t0 = time.time()
    p = snapshot_download(hid, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken", "merges.txt", "vocab.json"], token=os.environ.get("HF_TOKEN"))
    print(f"{short} -> {p} ({(time.time()-t0)/60:.1f} min)", flush=True)
print("ALL DONE", flush=True)

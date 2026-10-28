#!/usr/bin/env python3
"""Q4b: token-length ratio of the fit vs spec endings per tokenizer (PREREG: within 15%)."""
import json, sys
from transformers import AutoTokenizer
from pm_common import MODELS
rows = [json.loads(l) for l in open(sys.argv[1])]
toks = {s: (m["hid"], m["rev"]) for s, m in MODELS.items() if s in ("qwen_base", "olmo_base")}
toks["q38"] = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
for name, (hid, rev) in toks.items():
    try:
        tok = AutoTokenizer.from_pretrained(hid, revision=rev)
    except Exception as e:
        print(name, "tokenizer unavailable", e); continue
    worst = 0
    for r in rows:
        a, b = len(tok(r["fit"], add_special_tokens=False).input_ids), len(tok(r["spec"], add_special_tokens=False).input_ids)
        worst = max(worst, abs(a - b) / max(a, b))
    print(name, "max |len diff|/max len =", round(worst, 3), "bos:", tok.bos_token, tok("hi").input_ids)

"""Shared pod-side helpers: model registry, prompt rendering (chat with thinking closed; narrative), decoder-layer access.

Layer convention (PREREG): "layer L" = residual stream after decoder block L = HF hidden_states[L], L = 1..32.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261028
N_LAYERS = 32
D = 4096

MODELS = {
    "qwen_base": {"hid": "Qwen/Qwen3.5-9B-Base", "rev": "68c46c4b3498877f3ef123c856ecfde50c39f404", "family": "qwen", "kind": "base"},
    "qwen_it": {"hid": "Qwen/Qwen3.5-9B", "rev": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "family": "qwen", "kind": "instruct"},
    "olmo_base": {"hid": "allenai/Olmo-3-1025-7B", "rev": "a81bae42db3975be1671e27b9c9a56da1a9f980f", "family": "olmo", "kind": "base"},
    "olmo_sft": {"hid": "allenai/Olmo-3-7B-Think-SFT", "rev": "6ff857587e040d6d523a3d5f3a56e918f5401d66", "family": "olmo", "kind": "instruct"},
    "olmo_dpo": {"hid": "allenai/Olmo-3-7B-Think-DPO", "rev": "7b18bf927b430ff06376fdfa5610eb3b1b6a5c38", "family": "olmo", "kind": "instruct"},
    "olmo_rl": {"hid": "allenai/Olmo-3-7B-Think", "rev": "d97e442d7cc678210054dbcc9b440894d62c89a4", "family": "olmo", "kind": "instruct"},
}
OLMO_THINK_CLOSE = "</think>\n\n"
NARRATIVE_STOPS = ["\nUser:", "\n\nUser:", "\nuser:"]


def tokenizer(short):
    from transformers import AutoTokenizer
    m = MODELS[short]
    return AutoTokenizer.from_pretrained(m["hid"], revision=m["rev"])


def render_chat(tok, short, system: str | None, user: str, assistant_prefix: str = "") -> str:
    """The model's own template, thinking closed. Qwen: enable_thinking=False (template inserts an empty think block).
    OLMo Think line: the template ends '<|im_start|>assistant\\n<think>'; we close it."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    fam = MODELS[short]["family"]
    if fam == "qwen":
        txt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        assert txt.endswith("<think>\n\n</think>\n\n"), txt[-80:]
    else:
        txt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        assert txt.endswith("<|im_start|>assistant\n<think>"), txt[-80:]
        txt += OLMO_THINK_CLOSE
    return txt + assistant_prefix


def render_narrative(description: str, speaker: str, question: str, user_label: str = "User") -> str:
    return (f"The following is a transcript of a conversation between a user and {description}.\n\n"
            f"{user_label}: {question}\n\n{speaker}:")


def render_job(tok, short, job: dict) -> str:
    """job: {"format": "chat"|"narrative"|"raw", ...}. chat: system/user; narrative: description/speaker/question; raw: text."""
    f = job["format"]
    if f == "chat":
        return render_chat(tok, short, job.get("system"), job["user"], job.get("assistant_prefix", ""))
    if f == "narrative":
        return render_narrative(job["description"], job["speaker"], job["question"])
    if f == "raw":
        return job["text"]
    raise ValueError(f)


def decoder_layers(model):
    """The ModuleList of decoder blocks (works for Qwen3_5ForConditionalGeneration and Olmo3ForCausalLM)."""
    import torch
    cands = [m for m in model.modules() if isinstance(m, torch.nn.ModuleList) and len(m) == N_LAYERS]
    return max(cands, key=lambda m: sum(p.numel() for p in m.parameters()))


def load_hf(short, dtype="bfloat16", device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM
    m = MODELS[short]
    model = AutoModelForCausalLM.from_pretrained(m["hid"], revision=m["rev"], dtype=getattr(torch, dtype), device_map=device, attn_implementation="sdpa").eval()
    return model


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

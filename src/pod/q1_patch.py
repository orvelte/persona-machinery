#!/usr/bin/env python3
"""Q1 patching: generate from the default Assistant with component outputs patched to their role-context mean, and clampLstar
positive control (AA at L* set to role grand mean). Records AA(L*) over generated tokens. (PREREG Q1 Implementation.)

    python3 q1_patch.py SHORT JOBS.jsonl ATTR.npz BASIS.npz SPEC.json LSTAR OUT.jsonl [--max-new 384] [--batch 10]

Patching acts only on decode steps (seq len 1); prompt tokens untouched (left padding). Heads: replace the head's slice of the
o_proj/out_proj input by xmean_role_{block}[slice]. MLP: replace the mlp output by mmean_role_{block}. clampLstar: proj-out the
residual at L* about mu_L* along aa_L* (sets the AA coordinate to the role grand mean). AA readout = mean over generated positions
of hidden_states[L*] . aa_L*, recorded per generation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pm_common import MODELS, SEED, decoder_layers, load_hf, read_jsonl, render_job, tokenizer

NARR_STOP = ["\nUser:", "\n\nUser:", "\nuser:"]


def main(short, jobs_f, attr_f, basis_f, spec_f, lstar, out_f, max_new=384, batch=10):
    tok = tokenizer(short); tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_hf(short); layers = decoder_layers(model)
    Z = np.load(attr_f, allow_pickle=True)
    spec = json.loads(Path(spec_f).read_text())
    B = np.load(basis_f)
    aa = torch.tensor(B[f"aa_{lstar}"].astype(np.float32), device="cuda")
    mu = torch.tensor(B[f"mu_{lstar}"].astype(np.float32), device="cuda")
    fam = MODELS[short]["family"]
    # per block info: projection module, head dim
    binfo = {}
    for i in range(lstar):
        lay = layers[i]; mix = getattr(lay, "self_attn", None) or getattr(lay, "linear_attn", None)
        proj = mix.o_proj if hasattr(mix, "o_proj") else mix.out_proj
        H = proj.in_features // (mix.head_dim if hasattr(mix, "head_dim") else (proj.in_features // mix.num_v_heads))
        H = mix.num_v_heads if not hasattr(mix, "o_proj") else proj.in_features // mix.head_dim
        hd = proj.in_features // H
        binfo[i + 1] = dict(lay=lay, proj=proj, H=H, hd=hd,
                            xrole=torch.tensor(Z[f"xmean_role_{i+1}"].astype(np.float32), device="cuda"),
                            mrole=torch.tensor(Z[f"mmean_role_{i+1}"].astype(np.float32), device="cuda"))

    state = {"ops": {}, "clamp": False, "steps": []}

    def proj_pre_hook(b):
        info = binfo[b]
        def f(_m, inp):
            heads = state["ops"].get(("head", b))
            if not heads:
                return None
            x = inp[0]
            if x.shape[1] > 1:
                return None
            x = x.clone()
            for h in heads:
                sl = slice(h * info["hd"], (h + 1) * info["hd"])
                x[..., sl] = info["xrole"][sl].to(x.dtype)
            return (x,) + tuple(inp[1:])
        return f

    def mlp_hook(b):
        info = binfo[b]
        def f(_m, _inp, o):
            if ("mlp", b) not in state["ops"]:
                return None
            h = o[0] if isinstance(o, tuple) else o
            if h.shape[1] > 1:
                return None
            h = h.clone(); h[...] = info["mrole"].to(h.dtype)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        return f

    def layer_lstar_hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        x = h.float()
        if state["clamp"] and h.shape[1] == 1:
            c = (x - mu) @ aa
            x = x - c[..., None] * aa + 0.0 * aa
            x = x.to(h.dtype)
            out2 = (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        else:
            out2 = out
        if h.shape[1] == 1:  # record per-row AA on decode steps
            xr = (out2[0] if isinstance(out2, tuple) else out2).float()
            state["steps"].append(((xr[:, 0, :] - mu) @ aa).cpu().numpy())  # [B]
        return out2 if state["clamp"] else None

    hooks = []
    for b, info in binfo.items():
        hooks.append(info["proj"].register_forward_pre_hook(proj_pre_hook(b)))
        hooks.append(info["lay"].mlp.register_forward_hook(mlp_hook(b)))
    layers[lstar - 1].register_forward_hook(layer_lstar_hook)

    def set_cond(name):
        state["ops"] = {}; state["clamp"] = (name == "clampLstar")
        for comp in spec.get(name, []):
            if comp.get("mlp"):
                state["ops"].setdefault(("mlp", comp["block"]), True)
            else:
                state["ops"].setdefault(("head", comp["block"]), []).append(comp["head"])

    jobs = read_jsonl(jobs_f)
    conds = ["none", "clampLstar"] + [k for k in spec if k != "none"]
    out_p = Path(out_f); out_p.parent.mkdir(parents=True, exist_ok=True); done = set()
    if out_p.exists():
        done = {(r["id"], r["condition"], r["sample"]) for r in read_jsonl(out_p)}
    eos_ids = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()}
    t0 = time.time(); total = 0
    with out_p.open("a") as f:
        for cond in conds:
            torch.manual_seed(SEED)
            set_cond(cond)
            units = [(j, s) for j in jobs for s in range(j.get("n", 1)) if (j["id"], cond, s) not in done]
            for s0 in range(0, len(units), batch):
                grp = units[s0:s0 + batch]
                # per-generation AA: run each sample's batch but track per-row via separate accumulators is complex;
                # instead run batch and record batch-mean AA (recorded per row as the batch mean of its own tokens is
                # not separable here) -> we record per-row by generating with batch then re-reading is costly; use batch mean.
                texts = [render_job(tok, short, j) for j, _ in grp]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                mn = max(j.get("max_new", max_new) for j, _ in grp)
                state["steps"] = []
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=mn,
                                         pad_token_id=tok.pad_token_id, eos_token_id=list(eos_ids))
                steps = np.stack(state["steps"], 1) if state["steps"] else np.zeros((len(grp), 1))  # [B, nsteps]
                plen = enc["input_ids"].shape[1]
                for ri, ((j, s), row) in enumerate(zip(grp, gen)):
                    comp = row[plen:].tolist()
                    while comp and (comp[-1] == tok.pad_token_id or comp[-1] in eos_ids):
                        comp.pop()
                    text = tok.decode(comp, skip_special_tokens=False)
                    ng = max(1, len(comp))
                    aa_row = float(steps[ri, :ng].mean())
                    f.write(json.dumps({"id": j["id"], "set": j.get("set", "half_b"), "user": j["user"], "condition": cond,
                                        "sample": s, "text": text, "n_gen": len(comp), "aa": aa_row},
                                       ensure_ascii=False) + "\n")
                f.flush(); total += len(grp)
            print(f"[{cond}] done {len(units)} gens, {(time.time()-t0)/60:.1f} min", flush=True)
    print("done", total, f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    main(a[0], a[1], a[2], a[3], a[4], int(a[5]), a[6], opt("--max-new", 384), opt("--batch", 10))

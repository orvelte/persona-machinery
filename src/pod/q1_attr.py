#!/usr/bin/env python3
"""Q1 direct attribution of the AA coordinate at L* to every head and MLP at blocks <= L* (PREREG Q1 + Implementation).

    python3 q1_attr.py SHORT GEN.jsonl BASIS.npz LSTAR OUT_DIR [--half B] [--batch 8] [--contrast]

Teacher-forced replay of the M0 generations (exact ids), response span = completion tokens. For each response and each axis (aa_L*;
plus the contrast axis (default - mu)/|.| with --contrast) stores the token-mean attribution of: the embedding, every head (input slice
of o_proj / out_proj times W_out[:, slice]^T a), every MLP; plus hidden_states[l] . a for l = 0..32 (reconstruction check / profile).
OLMo post-norms (post_attention_layernorm / post_feedforward_layernorm) are linearised by the per-token scale of the actual pre-norm
output, so the head contributions sum exactly to the normed block write. Also stores, for patching, the mean (over role responses and
over default responses, each response weighted equally) of every output-projection input and of every MLP output.
Out: OUT_DIR/attr.npz, OUT_DIR/meta.jsonl.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pm_common import MODELS, decoder_layers, load_hf, read_jsonl, tokenizer


def main(short, gen_f, basis_f, lstar, out_dir, half="B", batch=8, contrast=False):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fam = MODELS[short]["family"]
    B = np.load(basis_f)
    axes = [B[f"aa_{lstar}"].astype(np.float32)]
    names_ax = ["aa"]
    if contrast:
        c = (B[f"default_{lstar}"] - B[f"mu_{lstar}"]).astype(np.float32); axes.append(c / np.linalg.norm(c)); names_ax.append("contrast")
    A = torch.tensor(np.stack(axes), device="cuda")                       # [K, D]
    K = A.shape[0]
    rows = [r for r in read_jsonl(gen_f) if r.get("half") == half and r["n_gen"] > 0]
    print(f"{len(rows)} rows half {half}", flush=True)
    tok = tokenizer(short)
    model = load_hf(short); layers = decoder_layers(model)
    nL = lstar
    # per block: mixer module, output projection, n heads
    blocks = []
    for i in range(nL):
        lay = layers[i]
        mix = getattr(lay, "self_attn", None) or getattr(lay, "linear_attn", None)
        proj = mix.o_proj if hasattr(mix, "o_proj") else mix.out_proj
        kind = "attn" if hasattr(mix, "o_proj") else "dnet"
        if kind == "attn":
            H = mix.config.num_attention_heads if hasattr(mix, "config") else proj.in_features // mix.head_dim
            H = proj.in_features // mix.head_dim
        else:
            H = mix.num_v_heads
        hd = proj.in_features // H
        W = proj.weight.float()                                          # [D, Din]
        blocks.append(dict(lay=lay, mix=mix, proj=proj, kind=kind, H=H, hd=hd, W=W))
    comp_names = ["embed"]
    for i, b in enumerate(blocks):
        comp_names += [f"L{i + 1}.{b['kind']}.h{h}" for h in range(b["H"])] + [f"L{i + 1}.mlp"]
    nC = len(comp_names)
    print(f"{nC} components; kinds {[b['kind'] for b in blocks]}", flush=True)

    cur = {}

    def pre_proj(i):
        def f(_m, inp):
            cur[("x", i)] = inp[0]
        return f

    def post_proj(i):
        def f(_m, _inp, o):
            cur[("o", i)] = o
        return f

    def post_mlp(i):
        def f(_m, _inp, o):
            cur[("m", i)] = o
        return f

    hs_ = []
    for i, b in enumerate(blocks):
        hs_.append(b["proj"].register_forward_pre_hook(pre_proj(i)))
        hs_.append(b["proj"].register_forward_hook(post_proj(i)))
        hs_.append(b["lay"].mlp.register_forward_hook(post_mlp(i)))

    def postnorm_scale(norm, o):
        """OLMo: y = w * o / rms(o). Returns (w-weighted axes [K, D] , per-token 1/rms [B, T, 1])."""
        of = o.float()
        inv = torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
        return (A * norm.weight.float()[None]), inv

    N = len(rows)
    C = np.zeros((N, K, nC), np.float32)
    R = np.zeros((N, K, 33), np.float32)
    is_role = np.array([r["role"] != "default" for r in rows])
    sum_x = {("role", i): torch.zeros(b["proj"].in_features, device="cuda", dtype=torch.float64) for i, b in enumerate(blocks)}
    sum_x.update({("def", i): torch.zeros(b["proj"].in_features, device="cuda", dtype=torch.float64) for i, b in enumerate(blocks)})
    sum_m = {(g, i): torch.zeros(A.shape[1], device="cuda", dtype=torch.float64) for g in ("role", "def") for i in range(len(blocks))}
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    order = sorted(range(N), key=lambda i: len(rows[i]["prompt_ids"]) + len(rows[i]["completion_ids"]))
    t0 = time.time()
    for s in range(0, N, batch):
        idx = order[s:s + batch]
        seqs = [list(rows[i]["prompt_ids"]) + list(rows[i]["completion_ids"]) for i in idx]
        T = max(map(len, seqs))
        inp = torch.full((len(idx), T), pad, dtype=torch.long); att = torch.zeros((len(idx), T), dtype=torch.long)
        span = torch.zeros((len(idx), T), dtype=torch.float32)
        for k, (i, ids) in enumerate(zip(idx, seqs)):
            inp[k, :len(ids)] = torch.tensor(ids); att[k, :len(ids)] = 1; span[k, rows[i]["n_prompt"]:len(ids)] = 1.0
        span = span.cuda(); cnt = span.sum(1)                              # [b]
        wmean = span / cnt[:, None]                                        # token-mean weights
        cur.clear()
        with torch.no_grad():
            o = model(input_ids=inp.cuda(), attention_mask=att.cuda(), use_cache=False, output_hidden_states=True)
            hs = o.hidden_states
            assert len(hs) == 33, len(hs)
            for l in range(33):
                R[idx, :, l] = torch.einsum("btd,kd,bt->bk", hs[l].float(), A, wmean).cpu().numpy()
            cols = [torch.einsum("btd,kd,bt->bk", hs[0].float(), A, wmean)[:, :, None]]
            for i, b in enumerate(blocks):
                x = cur[("x", i)].float()                                  # [b, T, Din]
                H, hd = b["H"], b["hd"]
                if fam == "olmo":
                    Aw, inv = postnorm_scale(b["lay"].post_attention_layernorm, cur[("o", i)])
                    xs = x * inv
                else:
                    Aw, xs = A, x
                Wa = (Aw @ b["W"]).view(K, H, hd)                          # [K, H, hd]
                hc = torch.einsum("bthe,khe,bt->bkh", xs.view(*xs.shape[:2], H, hd), Wa, wmean)
                m = cur[("m", i)].float()
                if fam == "olmo":
                    Am, invm = postnorm_scale(b["lay"].post_feedforward_layernorm, m)
                    mc = torch.einsum("btd,kd,bt->bk", m * invm, Am, wmean)
                else:
                    mc = torch.einsum("btd,kd,bt->bk", m, A, wmean)
                cols += [hc, mc[:, :, None]]
                xm = torch.einsum("bte,bt->be", x, wmean).double(); mm = torch.einsum("btd,bt->bd", m, wmean).double()
                rr = torch.tensor(is_role[idx], device="cuda")
                sum_x[("role", i)] += xm[rr].sum(0); sum_x[("def", i)] += xm[~rr].sum(0)
                sum_m[("role", i)] += mm[rr].sum(0); sum_m[("def", i)] += mm[~rr].sum(0)
            C[idx] = torch.cat(cols, 2).cpu().numpy()
        del o, hs
        cur.clear()
        if (s // batch) % 20 == 0:
            print(f"{s + len(idx)}/{N} {(time.time() - t0) / 60:.1f} min", flush=True)
    nr, nd = int(is_role.sum()), int((~is_role).sum())
    save = dict(C=C, R=R, axes=np.stack(axes), axis_names=np.array(names_ax), comp_names=np.array(comp_names), is_role=is_role,
                block_kind=np.array([b["kind"] for b in blocks]), block_H=np.array([b["H"] for b in blocks]))
    for i in range(len(blocks)):
        save[f"xmean_role_{i + 1}"] = (sum_x[("role", i)] / nr).float().cpu().numpy()
        save[f"xmean_def_{i + 1}"] = (sum_x[("def", i)] / nd).float().cpu().numpy()
        save[f"mmean_role_{i + 1}"] = (sum_m[("role", i)] / nr).float().cpu().numpy()
        save[f"mmean_def_{i + 1}"] = (sum_m[("def", i)] / nd).float().cpu().numpy()
    np.savez(out / "attr.npz", **save)
    with (out / "meta.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "role", "qid", "half", "sample", "n_gen", "n_prompt")}) + "\n")
    err = C.sum(2) - R[:, :, lstar]
    print("reconstruction |err| mean", np.abs(err).mean(0), "max", np.abs(err).max(0), "resid proj mean", R[:, :, lstar].mean(0), flush=True)
    print("done", f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    main(a[0], a[1], a[2], int(a[3]), a[4], opt("--half", "B"), opt("--batch", 8), "--contrast" in a)

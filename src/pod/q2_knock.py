#!/usr/bin/env python3
"""Q2: attention knockout and token ablation on the default Assistant's M0 half-B responses (PREREG Q2 + Implementation).

    python3 q2_knock.py SHORT OUT_DIR [--checks-only] [--no-gen]

Writes OUT_DIR/{checks.json, tf.npz, tf_rows.jsonl, gen.jsonl, axes.npz}. Teacher-forced readout = mean residual over completion tokens at L*.
Knockout = 4D additive mask on every softmax layer: queries in the assistant turn (last <|im_start|> onward) cannot attend to the knocked
prompt keys. Qwen linear-attention layers: the knocked positions' post-conv k, v are zeroed (partial knockout). Token ablation = the
knocked tokens' input embeddings replaced by the mean input embedding.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pm_common import MODELS, SEED, load_hf, read_jsonl, render_chat, tokenizer, write_jsonl

LSTAR = {"olmo_rl": 23, "qwen_it": 24}
M0 = Path("out/m0")
KILL = {"t": None, "calls": 0, "zeroed": 0}   # Qwen linear-attention kill vector [B, T] (bool) for the current forward


def patch_qwen_linear():
    import transformers.models.qwen3_5.modeling_qwen3_5 as QM
    orig = QM.torch_chunk_gated_delta_rule

    def patched(query, key, value, *args, **kw):
        k = KILL["t"]
        if k is not None:
            assert k.shape == key.shape[:2], (k.shape, key.shape)
            keep = (~k).to(key.dtype)[:, :, None, None]
            key = key * keep; value = value * keep
            KILL["calls"] += 1; KILL["zeroed"] += int(((key[k].abs().sum() + value[k].abs().sum()) == 0).item())
        return orig(query, key, value, *args, **kw)
    QM.torch_chunk_gated_delta_rule = patched


def text_model(model):
    m = model.model
    return getattr(m, "language_model", m)


def classify(tok, short, ids, user, system):
    """Per prompt token: 'sys' | 'user' | 'role' | 'delim' | 'think'. Character offsets of the re-tokenised prompt."""
    txt = tok.decode(ids, skip_special_tokens=False)
    enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == list(ids), "retokenisation mismatch"
    spans = {}
    u0 = txt.rindex(user); spans["user"] = (u0, u0 + len(user))
    if system is not None:
        s0 = txt.index(system); spans["sys"] = (s0, s0 + len(system))
    think0 = txt.rindex("<think>")
    out = []
    for (a, b), tid in zip(enc["offset_mapping"], ids):
        c = None
        for k, (x, y) in spans.items():
            if a >= x and b <= y and b > a:
                c = k
        if c is None:
            piece = txt[a:b]
            if a >= think0:
                c = "think"
            elif piece.strip() in ("system", "user", "assistant") and txt[max(0, a - 12):a].endswith("<|im_start|>"):
                c = "role"
            else:
                c = "delim"
        out.append(c)
    q0 = enc["offset_mapping"].index(next(o for o in enc["offset_mapping"] if o[0] == txt.rindex("<|im_start|>")))
    return out, q0


def build_mask(lens, kills, q0s, n_prompts, T, dtype, device):
    B = len(lens)
    allowed = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))[None].repeat(B, 1, 1)
    for b in range(B):
        if kills[b] is not None and kills[b].any():
            kk = torch.zeros(T, dtype=torch.bool, device=device); kk[:n_prompts[b]] = torch.as_tensor(kills[b], device=device)
            allowed[b, q0s[b]:, :] &= ~kk[None, :]
        allowed[b, :, lens[b]:] = False
        allowed[b, lens[b]:, :] = False; allowed[b, lens[b]:, 0] = True
    assert allowed.any(-1).all(), "a query row with no allowed key"
    m = torch.zeros(B, 1, T, T, dtype=dtype, device=device)
    return m.masked_fill(~allowed[:, None], torch.finfo(dtype).min), allowed


class Runner:
    def __init__(self, short):
        self.short = short; self.fam = MODELS[short]["family"]
        self.tok = tokenizer(short); self.model = load_hf(short); self.tm = text_model(self.model)
        self.L = LSTAR[short]
        E = self.tm.embed_tokens.weight
        self.mean_emb = E.float().mean(0).to(E.dtype)
        if self.fam == "qwen":
            patch_qwen_linear()

    def forward(self, seqs, kills=None, q0s=None, n_prompts=None, ablate=None, custom=True, hidden=True, out_attn=False):
        """seqs: list of id lists. kills: list of bool arrays over prompt positions (KO). ablate: list of bool arrays (TA) or None."""
        B = len(seqs); T = max(len(s) for s in seqs); dev = "cuda"
        kills = kills or [None] * B; ablate = ablate or [None] * B
        inp = torch.zeros(B, T, dtype=torch.long, device=dev)
        for b, s in enumerate(seqs):
            inp[b, :len(s)] = torch.tensor(s, device=dev)
        emb = self.tm.embed_tokens(inp)
        for b in range(B):
            if ablate[b] is not None and np.any(ablate[b]):
                pos = torch.as_tensor(np.nonzero(ablate[b])[0], device=dev); emb[b, pos] = self.mean_emb
        pos_ids = torch.arange(T, device=dev)[None].expand(B, T)
        kw = dict(inputs_embeds=emb, position_ids=pos_ids, use_cache=False, output_hidden_states=hidden, output_attentions=out_attn)
        if custom:
            mask, _ = build_mask([len(s) for s in seqs], kills, q0s or [0] * B, n_prompts or [0] * B, T, emb.dtype, dev)
            kw["attention_mask"] = ({"full_attention": mask, "linear_attention": None} if self.fam == "qwen" else {"full_attention": mask, "sliding_attention": mask})
            if self.fam == "qwen":
                kt = torch.zeros(B, T, dtype=torch.bool, device=dev)
                for b in range(B):
                    if kills[b] is not None:
                        kt[b, :len(kills[b])] = torch.as_tensor(kills[b], device=dev)
                KILL["t"] = kt if kt.any() else None
        else:
            att = torch.zeros(B, T, dtype=torch.long, device=dev)
            for b, s in enumerate(seqs):
                att[b, :len(s)] = 1
            kw["attention_mask"] = att; KILL["t"] = None
        with torch.no_grad():
            o = self.tm(**kw)
        KILL["t"] = None
        return o


def resp_means(o, seqs, n_prompts, L):
    """Mean over completion tokens at layer L for every row; also per-layer means [B, 32, D] are too big, return L only + all-layer means."""
    hs = o.hidden_states
    out_L, out_all = [], []
    for b, s in enumerate(seqs):
        a, e = n_prompts[b], len(s)
        out_L.append(hs[L][b, a:e].float().mean(0))
        out_all.append(torch.stack([hs[l][b, a:e].float().mean(0) for l in range(1, len(hs))]))
    return torch.stack(out_L), torch.stack(out_all)


def fit_axes(short, L):
    X = np.load(M0 / f"acts_{short}_chat/acts.npz")["X"]
    meta = read_jsonl(M0 / f"acts_{short}_chat/meta.jsonl")
    H = X[:, L - 1].astype(np.float32)
    roles = sorted({m["role"] for m in meta if m["role"] != "default"})
    RM = np.stack([H[[i for i, m in enumerate(meta) if m["role"] == r and m["half"] == "A"]].mean(0) for r in roles])
    dA = H[[i for i, m in enumerate(meta) if m["role"] == "default" and m["half"] == "A"]].mean(0)
    mu = RM.mean(0)
    _, _, Vt = np.linalg.svd(RM - mu, full_matrices=False); pc1 = Vt[0]
    if (dA - mu) @ pc1 < 0:
        pc1 = -pc1
    c = (dA - mu) / np.linalg.norm(dA - mu)
    role_pos = (RM - mu) @ pc1
    m0_rows = {m["id"]: H[i] for i, m in enumerate(meta)}
    return {"pc1": pc1, "c": c, "mu": mu, "roles": roles, "role_pos": role_pos}, m0_rows


def main(short, out_dir, checks_only=False, no_gen=False):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    R = Runner(short); tok = R.tok; L = R.L
    rows = [r for r in read_jsonl(M0 / f"gen_{short}_chat.jsonl") if r["role"] == "default" and r["half"] == "B"]
    rows.sort(key=lambda r: (r["qid"], r["sample"]))
    assert len(rows) == 30, len(rows)
    narr = [r for r in read_jsonl(M0 / f"gen_{short}_narrative.jsonl") if r["role"] == "default" and r["half"] == "B"]
    narr.sort(key=lambda r: (r["qid"], r["sample"]))
    narr_prompt = {r["qid"]: r["prompt_ids"] for r in narr}
    axes, m0_rows = fit_axes(short, L)
    basis = np.load(M0 / f"basis_{short}_chat.npz")
    aa_all = np.stack([basis[f"aa_{l}"] for l in range(1, 33)]).astype(np.float32); mu_all = np.stack([basis[f"mu_{l}"] for l in range(1, 33)]).astype(np.float32)
    np.savez(out / "axes.npz", pc1=axes["pc1"], c=axes["c"], mu=axes["mu"], role_pos=axes["role_pos"], roles=np.array(axes["roles"]), aa_frozen=aa_all[L - 1], mu_frozen=mu_all[L - 1])

    # ---- token classes
    info = []
    sysmsg = None
    for r in rows:
        txt = tok.decode(r["prompt_ids"], skip_special_tokens=False)
        if R.fam == "olmo":
            sysmsg = txt.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]
        cls, q0 = classify(tok, short, r["prompt_ids"], r["user"], sysmsg)
        info.append({"cls": np.array(cls), "q0": q0, "n_prompt": len(r["prompt_ids"]), "seq": list(r["prompt_ids"]) + list(r["completion_ids"])})
    counts = {c: int((info[0]["cls"] == c).sum()) for c in ("sys", "user", "role", "delim", "think")}
    print("token classes (row 0):", counts, "q0", info[0]["q0"], flush=True)
    print(" | ".join(f"{tok.decode([t])!r}:{c}" for t, c in zip(rows[0]["prompt_ids"], info[0]["cls"]) if c not in ("sys",)), flush=True)

    def sets(i):
        cls = info[i]["cls"]; n = len(cls); tmpl = np.isin(cls, ["role", "delim", "think"]); content = np.isin(cls, ["sys", "user"])
        pos0 = np.zeros(n, bool); pos0[0] = True
        d = {"a": tmpl, "a_nosink": tmpl & ~pos0, "role": cls == "role", "delim": cls == "delim", "delim_nosink": (cls == "delim") & ~pos0,
             "think": cls == "think", "sink": pos0, "c": cls == "user"}
        if R.fam == "olmo":
            d["b"] = cls == "sys"
        d["allprompt_nosink"] = ~pos0
        return d
    S = [sets(i) for i in range(len(rows))]
    for draw in range(3):   # random content tokens of the template count, one generator per draw (seed 20261028 + draw), rows in order
        rng = np.random.default_rng(SEED + draw)
        for i in range(len(rows)):
            cls = info[i]["cls"]; content = np.isin(cls, ["sys", "user"]); k = min(int(S[i]["a"].sum()), int(content.sum()) - 1)
            m = np.zeros(len(cls), bool); m[rng.choice(np.nonzero(content)[0], k, replace=False)] = True; S[i][f"rand{draw}"] = m
    order = ["a", "a_nosink", "role", "delim", "delim_nosink", "think", "sink"] + (["b"] if R.fam == "olmo" else []) + ["c", "rand0", "rand1", "rand2", "allprompt_nosink"]
    S = [{k: s[k] for k in order} for s in S]
    seqs = [x["seq"] for x in info]; q0s = [x["q0"] for x in info]; nps = [x["n_prompt"] for x in info]

    # ---- mechanics checks
    checks = {"counts_row0": counts, "n_template": [int(s["a"].sum()) for s in S], "n_rand": [int(s["rand0"].sum()) for s in S]}
    idx = list(range(6))
    o_std = R.forward([seqs[i] for i in idx], custom=False)
    o_cus = R.forward([seqs[i] for i in idx], kills=[None] * len(idx), q0s=[q0s[i] for i in idx], n_prompts=[nps[i] for i in idx])
    lm = R.model.lm_head
    agree, tot, maxd, relh = 0, 0, 0.0, 0.0
    for k, i in enumerate(idx):
        a, e = nps[i], len(seqs[i])
        l1 = lm(o_std.last_hidden_state[k, a:e]).float(); l2 = lm(o_cus.last_hidden_state[k, a:e]).float()
        agree += int((l1.argmax(-1) == l2.argmax(-1)).sum()); tot += e - a; maxd = max(maxd, float((l1 - l2).abs().max()))
        h1 = o_std.hidden_states[L][k, a:e].float(); h2 = o_cus.hidden_states[L][k, a:e].float()
        relh = max(relh, float(((h1 - h2).norm(dim=-1) / h1.norm(dim=-1)).max()))
    checks["zero_mask"] = {"argmax_agree": agree / tot, "max_abs_dlogit": maxd, "max_rel_dhidden_L": relh, "pass": agree / tot >= 0.995 and relh <= 1e-2}
    del o_std, o_cus
    # eager attention weights under KO (a)
    cfg = R.tm.config; old = cfg._attn_implementation
    cfg._attn_implementation = "eager"   # attention modules share this config object and read it at call time
    i = 0
    o_e = R.forward([seqs[i]], kills=[S[i]["a"]], q0s=[q0s[i]], n_prompts=[nps[i]], out_attn=True)
    o_s = None
    att = [a for a in (o_e.attentions or ()) if a is not None]
    kk = np.nonzero(S[i]["a"])[0]
    maxw = max(float(a[0, :, q0s[i]:, kk].abs().max()) for a in att) if att else None
    rowsum = max(float((a[0].float().sum(-1) - 1).abs().max()) for a in att) if att else None
    he = o_e.hidden_states[L][0, nps[i]:].float()
    cfg._attn_implementation = old
    o_s = R.forward([seqs[i]], kills=[S[i]["a"]], q0s=[q0s[i]], n_prompts=[nps[i]])
    hs_ = o_s.hidden_states[L][0, nps[i]:].float()
    checks["eager_attention"] = {"n_attn_layers": len(att), "max_weight_to_knocked": maxw, "max_rowsum_err": rowsum,
                                 "eager_vs_sdpa_rel_dhidden_L": float(((he - hs_).norm(dim=-1) / hs_.norm(dim=-1)).max()),
                                 "pass": bool(att) and maxw == 0.0}
    if R.fam == "qwen":
        KILL["calls"] = 0; KILL["zeroed"] = 0
        R.forward([seqs[0]], kills=[S[0]["a"]], q0s=[q0s[0]], n_prompts=[nps[0]], hidden=False)
        checks["qwen_linear_kill"] = {"calls": KILL["calls"], "zeroed_ok": KILL["zeroed"], "pass": KILL["calls"] == 24 and KILL["zeroed"] == 24}
    del o_e, o_s
    torch.cuda.empty_cache()
    print(json.dumps(checks, indent=1), flush=True)
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    if checks_only:
        return

    # ---- teacher-forced conditions
    conds = [("none", None, None)]
    for key in S[0]:
        conds.append((f"KO_{key}", "ko", key))
    for key in S[0]:
        if key != "allprompt_nosink":
            conds.append((f"TA_{key}", "ta", key))
    HL, HALL, meta_rows = [], [], []
    B = 10
    for name, kind, key in conds:
        for s in range(0, len(rows), B):
            ii = list(range(s, min(len(rows), s + B)))
            kills = [S[i][key] for i in ii] if kind == "ko" else None
            abl = [S[i][key] for i in ii] if kind == "ta" else None
            o = R.forward([seqs[i] for i in ii], kills=kills, q0s=[q0s[i] for i in ii], n_prompts=[nps[i] for i in ii], ablate=abl)
            hL, hall = resp_means(o, [seqs[i] for i in ii], [nps[i] for i in ii], L)
            HL.append(hL.cpu().numpy()); HALL.append(np.einsum("bld,ld->bl", hall.cpu().numpy() - mu_all[None], aa_all))
            meta_rows += [{"cond": name, "id": rows[i]["id"], "qid": rows[i]["qid"], "sample": rows[i]["sample"], "n_knocked": int(S[i][key].sum()) if key else 0} for i in ii]
            del o
        print("tf", name, f"{time.strftime('%H:%M:%S')}", flush=True)
    # raw-document references
    raw_same = [list(narr_prompt[r["qid"]]) + tok(" " + r["text"].lstrip(), add_special_tokens=False)["input_ids"] for r in rows]
    raw_np = [len(narr_prompt[r["qid"]]) for r in rows]
    raw_own = [list(r["prompt_ids"]) + list(r["completion_ids"]) for r in narr]
    own_np = [r["n_prompt"] for r in narr]
    for name, sq, npp, src in (("raw_same", raw_same, raw_np, rows), ("raw_own", raw_own, own_np, narr)):
        for s in range(0, len(sq), B):
            ii = list(range(s, min(len(sq), s + B)))
            o = R.forward([sq[i] for i in ii], n_prompts=[npp[i] for i in ii])
            hL, hall = resp_means(o, [sq[i] for i in ii], [npp[i] for i in ii], L)
            HL.append(hL.cpu().numpy()); HALL.append(np.einsum("bld,ld->bl", hall.cpu().numpy() - mu_all[None], aa_all))
            meta_rows += [{"cond": name, "id": src[i]["id"], "qid": src[i]["qid"], "sample": src[i]["sample"], "n_knocked": 0} for i in ii]
            del o
    # positive control: 3 lowest-PC1 roles as system prompt, none vs KO of the system content
    m0all = read_jsonl(M0 / f"gen_{short}_chat.jsonl")
    low = [axes["roles"][j] for j in np.argsort(axes["role_pos"])[:3]]
    role_sys = {r: next(x["system"] for x in m0all if x["role"] == r) for r in low}
    for role in low:
        pseqs, pnp, pq0, pk = [], [], [], []
        for r in rows:
            txt = render_chat(tok, short, role_sys[role], r["user"])
            pid = tok(txt, add_special_tokens=False)["input_ids"]
            cls, q0 = classify(tok, short, pid, r["user"], role_sys[role])
            pseqs.append(pid + list(r["completion_ids"])); pnp.append(len(pid)); pq0.append(q0); pk.append(np.array(cls) == "sys")
        for name, kills in ((f"pc_{role}_none", None), (f"pc_{role}_KO_sys", pk)):
            for s in range(0, len(rows), B):
                ii = list(range(s, min(len(rows), s + B)))
                o = R.forward([pseqs[i] for i in ii], kills=[kills[i] for i in ii] if kills else None, q0s=[pq0[i] for i in ii], n_prompts=[pnp[i] for i in ii])
                hL, hall = resp_means(o, [pseqs[i] for i in ii], [pnp[i] for i in ii], L)
                HL.append(hL.cpu().numpy()); HALL.append(np.einsum("bld,ld->bl", hall.cpu().numpy() - mu_all[None], aa_all))
                meta_rows += [{"cond": name, "id": rows[i]["id"], "qid": rows[i]["qid"], "sample": rows[i]["sample"], "n_knocked": int(pk[i].sum()) if kills else 0, "role": role} for i in ii]
                del o
    HL = np.concatenate(HL); HALL = np.concatenate(HALL)
    # M0 reproduction check
    nn = [k for k, m in enumerate(meta_rows) if m["cond"] == "none"]
    cs = [float(HL[k] @ m0_rows[meta_rows[k]["id"]] / np.linalg.norm(HL[k]) / np.linalg.norm(m0_rows[meta_rows[k]["id"]])) for k in nn]
    checks["m0_repro"] = {"min_cos": min(cs), "pass": min(cs) >= 0.999}
    (out / "checks.json").write_text(json.dumps(checks, indent=1)); print("m0_repro", checks["m0_repro"], flush=True)
    np.savez(out / "tf.npz", H=HL.astype(np.float32), proj_layers=HALL.astype(np.float32))
    write_jsonl(out / "tf_rows.jsonl", meta_rows)
    if no_gen:
        return

    # ---- behaviour: sampling with the knockout at every step (no KV cache)
    gconds = ["none", "a", "a_nosink"] + (["b"] if R.fam == "olmo" else []) + ["c", "rand0"]
    eos = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()}
    eos.discard(None)
    gout = out / "gen.jsonl"
    done = {(json.loads(l)["cond"]) for l in gout.read_text().splitlines()} if gout.exists() else set()
    for gc in gconds:
        if ("none" if gc == "none" else f"KO_{gc}") in done:
            continue
        g = torch.Generator(device="cuda").manual_seed(SEED)
        cur = [list(r["prompt_ids"]) for r in rows]; fin = [False] * len(rows); t0 = time.time()
        for step in range(384):
            act = [i for i in range(len(rows)) if not fin[i]]
            if not act:
                break
            kills = None if gc == "none" else [S[i][gc] for i in act]
            o = R.forward([cur[i] for i in act], kills=kills, q0s=[q0s[i] for i in act], n_prompts=[nps[i] for i in act], hidden=False)
            last = torch.stack([o.last_hidden_state[k, len(cur[i]) - 1] for k, i in enumerate(act)])
            logits = R.model.lm_head(last).float() / 0.6
            v, ix = logits.topk(20, dim=-1); p = v.softmax(-1)
            cum = p.cumsum(-1); p[(cum - p) > 0.95] = 0; p = p / p.sum(-1, keepdim=True)
            nxt = ix.gather(1, torch.multinomial(p, 1, generator=g)).squeeze(1).tolist()
            for k, i in enumerate(act):
                if nxt[k] in eos:
                    fin[i] = True
                else:
                    cur[i].append(nxt[k])
            del o
        with gout.open("a") as f:
            for i, r in enumerate(rows):
                comp = cur[i][nps[i]:]
                f.write(json.dumps({"cond": "none" if gc == "none" else f"KO_{gc}", "id": r["id"], "qid": r["qid"], "sample": r["sample"], "user": r["user"],
                                    "text": tok.decode(comp, skip_special_tokens=False), "n_gen": len(comp), "finished": fin[i]}, ensure_ascii=False) + "\n")
        print("gen", gc, f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print("Q2 DONE", short, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], "--checks-only" in a, "--no-gen" in a)

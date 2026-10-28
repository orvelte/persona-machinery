# M0 — Shared setup

**Assistant Axis extracted for 8 model × format pairs; G-AA passes for 7 of them. G-trait fails for every trait, so Q5 does not run (DECISIONS #14–15).**

## Assistant Axis (PC1 of 99 role means, response tokens, default oriented positive)

L\* = argmax of held-out role separation d′ over layers 1–24 (DECISIONS #12; the unconstrained argmax fell on the last layers, which leaves no room for Q3's L\*..L\*+8 window). Values: `lstar.json`.

| model × format | L\* | PC1 share | split-half \|cos\| | d′ (held-out) | default z | cos(PC1, default − role mean) | ρ vs 27B frozen axis | G-AA |
|---|---|---|---|---|---|---|---|---|
| qwen_it chat | 24 | 0.28 | 0.967 | 6.5 | +1.15 | 0.79 | 0.89 | pass |
| qwen_it narrative | 22 | 0.31 | 0.950 | 3.2 | +1.23 | 0.84 | 0.92 | pass |
| qwen_base narrative | 22 | 0.29 | 0.912 | 1.1 | +1.01 | 0.69 | 0.89 | pass |
| olmo_rl chat | 23 | 0.36 | 0.941 | 1.4 | +0.58 | 0.48 | 0.77 | pass |
| olmo_rl narrative | 23 | 0.24 | 0.851 | 3.4 | +1.11 | 0.70 | – | **fail** (descriptive only) |
| olmo_base narrative | 22 | 0.25 | 0.913 | 1.0 | +0.81 | 0.65 | – | pass |
| olmo_sft chat | 23 | 0.34 | 0.932 | 0.9 | +0.60 | 0.49 | – | pass |
| olmo_dpo chat | 23 | 0.37 | 0.938 | 1.5 | +0.37 | 0.16 | – | pass |

ρ vs 27B = Spearman between the 99 roles' AA positions at the frozen L\* and persona-mech's frozen 27B axis (`lstar.json`). Cosine is undefined across 4,096 vs 5,120 dims (DECISIONS #7). The pre-registered criterion is ρ ≥ 0.7. Both Qwen models meet it, and so does OLMo RL chat at 0.77.

**Reading.**
- In Qwen3.5-9B the PC1 axis is the Assistant Axis of the 27B work. Its role ordering matches the frozen 27B axis (ρ 0.87–0.92). It already exists in the base model under narrative framing: Qwen base vs instruct cosine 0.75, CKA 0.87 at L\*.
- In OLMo-3-7B the chat-format PC1 is a mystical ↔ evaluator axis (eldritch, witch, leviathan vs grader, auditor, examiner). It is only weakly aligned with the default Assistant: cos 0.48 in RL and 0.16 in DPO. SFT, DPO and RL share it almost exactly (cos 0.99 to the final model). The contrast axis (default − role mean) is reported alongside it in Q1, Q3 and Q4 (DECISIONS #13).
- OLMo RL in narrative format has two near-degenerate top PCs at layers 23–31, with PC1 and PC2 shares of 0.21 and 0.20. That makes PC1 unstable.

Figures: `figures/m0_profiles.png` (per-layer PC1 share, split-half, d′), `figures/m0_cross.png` (cross-format and cross-model cosine and CKA).

## Persona vectors (Chen et al. pipeline, `data/traits.json` pinned commit)

| trait | qwen_it kept pos / neg (of 100) | olmo_rl kept pos / neg |
|---|---|---|
| evil | 1 / 100 → dropped (refuses) | 4 / 100 → dropped |
| sycophancy | 97 / 97 | 8 / 100 → dropped |
| hallucination | 79 / 88 | 54 / 73 |
| impoliteness (substitute, DECISIONS #11) | 67 / 100 | 6 / 100 → dropped |

**G-trait (steer on 20 held-out questions; pass = trait +30 over unsteered at coherence ≥ 80).**

| trait | unsteered | pre-registered: 5 layers, c = 1 / 1.5 / 2 | rescue: L\* only, c = 0.25 / 0.5 / 0.75 / 1 | gate |
|---|---|---|---|---|
| qwen sycophancy | 23 (coh 88) | 6 / 23 / 59, coherence 1–2 | 28 / 27 / 29 / 30, coherence 87–89 | fail |
| qwen hallucination | 26 (coh 84) | coherence ≈ 2 | 14 / 11 / 22 / 24 | fail |
| qwen impoliteness | 1 (coh 90) | 96 / 94 / 94, coherence 0–1 | 2 / 2 / 2 / 2 | fail |
| olmo hallucination | 17 (coh 83) | – | 21 / 31 / 34 / 26, coherence 79–84 | fail |

The vectors carry the traits: the 5-layer impoliteness steer produces nothing but slurs. But each vector is about 0.5× the residual norm at L\*. One layer at c ≤ 1 does nothing, and five layers at c ≥ 1 destroys coherence. The pre-registered rescue was the only grid allowed, so Q5 is not run. A grid between the two (one layer, c ∈ {2, 3, 4}) is the obvious next step, and it needs Olivia's go-ahead.

Raw rows: `gate_gen_*.jsonl`, `gate2_gen_*.jsonl`, `gate*_scored_*.jsonl`, `traits_gen_*.jsonl`, `traits_scores_*.jsonl`. Spend for M0: pod ≈ $5.5, judges $0.19.

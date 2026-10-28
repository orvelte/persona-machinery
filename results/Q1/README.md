# Q1: Who writes the Assistant-Axis coordinate? (`qwen_it` chat, L\* = 24)

> **Post-review correction (2026-10-28, after the blind red-team review in `results/REDTEAM.md`).** (1) The attribution and patching run on half-B questions, which are also AA fit questions (the AA was fitted on all 20); this is undisclosed in the original write-up and could inflate the gap. (2) The nulls are selected for near-zero AA writes, so "nulls close 0–2%" is partly by construction; a norm-matched null was not run. (3) The positive control's failure (+0.7 points) means the behavioural meaning of the writers is unknown; it does not show the AA is behaviourally inert (see Q3 correction). The "intermediate" rule outcome (top-10 share 43.5%) stands.

**Answer: intermediate, leaning concentrated, and late. The top 10 of 697 components carry 43.5% of the default − role AA gap at L\*
(95% CI 42.6–44.7%). That is inside the pre-registered *intermediate* band (30–50%).** 14 components reach 50% and 73 reach 80%.
Blocks 19–24 write 77% of the gap. By component class, the 18 Gated-DeltaNet layers' value heads write the most (7.0 of 12.2), the MLPs
4.1, and the six softmax-attention layers' heads only 1.1. Among the ten largest single writers, seven are MLPs (blocks 19–24) and three
are DeltaNet heads (L19.h9, L22.h9/h6 and L23.h20). In this model, then, "who writes it" has two answers. Context-reading DeltaNet heads,
diffuse across many heads, write most of the gap in aggregate. A handful of late MLPs are the largest single writers.

**The patching check's behavioural readouts are not interpretable: the positive control failed.** Clamping the AA coordinate at L\* to
the role grand mean (at generated tokens) removes the whole gap at L\* (105%, CI 93–116%). Self-description moves only +1.0 points
(Sonnet 5.5, CI −0.0 to +1.5; Haiku gives +1.0, CI −1.4 to +3.4; the pre-registered threshold is ≥ 20). One layer of AA removal does not change what the model says about itself.
Mechanistically, patching the identified writers does remove the AA coordinate, as the attribution predicts, and the nulls do not.

## Method (PREREG Q1 + Implementation; DECISIONS 25)
- Data: M0 `qwen_it` chat generations on half-B questions, with 30 default-Assistant responses and 990 role responses. Teacher-forced on the
  exact ids, with the response-token mean taken per response.
- Components: the embedding; every head of blocks 1–24, where head write = W_out[:, slice] · (input slice of `o_proj` / `out_proj`), covering
  6 softmax blocks × 16 heads and 18 DeltaNet blocks × 32 value heads; and 24 MLPs. That makes 697 in all. Attribution = write · aa\_24.
- **Reconstruction gate:** embedding + Σ components vs hidden\_states[24] · aa. Mean |error| is 0.0011 (max 0.0076), against a gap of 12.2,
  i.e. 0.009% of the gap. The gate (< 2%) passes.
- Gap contribution = mean(default) − mean(roles). Total gap = 12.25 (question-clustered CI 11.4–12.9). The embedding contributes 0.001.

## Concentration rule

| statistic | value | 95% CI |
|---|---|---|
| top-10 share of total gap (top-10 re-selected per resample; primary) | **0.435** | 0.426–0.447 |
| same, fixed top-10 set | 0.435 | 0.425–0.447 |
| same, question-clustered bootstrap (10 half-B questions) | 0.435 | 0.423–0.448 |
| top-3 share | 0.185 | |
| top-10 share of the positive mass (Σ positive contributions) | 0.420 | |
| components needed for 50% / 80% | 14 / 73 | |
| components with a positive contribution | 631 / 697 | |

**Rule outcome: intermediate.** The CI excludes 50%, so it is not "concentrated"; it also excludes 30%, so it is not "diffuse".

Top 10 writers, with their gap contribution and share of the total: L24.mlp 0.88 (7.2%), L19.dnet.h9 0.73 (6.0%), L22.mlp 0.65 (5.3%),
L20.mlp 0.56 (4.5%), L21.mlp 0.54 (4.4%), L23.mlp 0.51 (4.2%), L23.dnet.h20 0.48 (3.9%), L19.mlp 0.43 (3.5%), L22.dnet.h9 0.31 (2.5%),
L22.dnet.h6 0.22 (1.8%).

Layer profile (`figures/q1_layer_profile.png`): the contribution is near zero through block 16, rises at 17–18 and peaks at 19–23.
Block 24's MLP is the largest single writer. Attention (all heads) contributes 8.1 and MLPs 4.1. The softmax-attention layers contribute
little (1.1 summed over blocks 4, 8, …, 24), so most context-reading is done by the linear-attention (DeltaNet) layers.

## Patching check (30 generations per condition = 10 half-B questions × 3 samples; default Assistant; generated tokens only)

Each patched component's write is replaced by its role-context mean. "AA closed" = (none − condition) / 12.25, measured at L\* during
generation. Self-description is `ASSISTANT_LIKENESS` scored by **Sonnet 5.5**, with `none` = 95.2; Haiku scores are in brackets. Sonnet is primary because Haiku failed the agreement check, see below. Paired bootstrap over prompts.

| condition | AA gap closed | Δ self-description (none − cond) | style index Δ (0 = default, 1 = role avg) |
|---|---|---|---|
| clampLstar (positive control) | 1.05 [0.93, 1.16] | **+0.7 [−0.0, 1.5]: fails (needs ≥ 20)** (Haiku +1.0) | +0.10 [−0.03, 0.26] |
| top10 | 0.86 [0.74, 0.96] | +46.7 [36.6, 57.5] (degenerate text, see below) (Haiku +31.8) | −0.43 [−1.64, 0.68] |
| top3 | 0.19 [0.15, 0.23] | +0.9 [0.1, 1.8] (Haiku +1.9) | −0.18 [−0.41, 0.03] |
| w1 (L24.mlp alone) | 0.05 [0.01, 0.08] | −0.1 [−1.0, 0.7] | −0.17 [−0.34, −0.00] |
| w2..w10 alone | 0.02–0.09 | −0.8 to +0.4 | −0.22 to +0.13 |
| layer-matched 10 (null) | −0.00 [−0.03, 0.03] | −0.7 [−1.4, 0.0] | −0.03 [−0.15, 0.08] |
| random 10 (null, mean of 3) | 0.02 (draws 0.01–0.04) | +0.1 (draws −0.1 to +0.3) | −0.06 |

- On the AA coordinate, the top-10 and top-3 patches beat both nulls: CIs exclude 0, and their point estimates are more than 2× the larger null.
  The closed fraction (86%) exceeds the writers' direct share (43.5%), so the patched writers also drive later components that write the AA.
- The top-10 self-description drop is not a persona shift. Mean-patching seven late MLPs breaks the model: 11 of 30 replies score < 50 and are
  repetition loops ("the the the …"). With the positive control failed, the pre-registration says the
  behaviour readouts are not interpreted, and this row is reported only for completeness.
- The positive-control failure is itself informative. During generation, the AA coordinate at L\* = 24 can be set to the role average
  with no change in self-description or style. Either blocks 25–32 rebuild the persona from context, or the behaviour does not run
  through this coordinate at one layer. Q3 tests the stronger, nine-layer version.

Figures: `figures/q1_layer_profile.png`, `figures/q1_patching.png`.

## Judge agreement
Haiku vs Sonnet on 60 seeded Q1 replies: Spearman 0.38, which **fails** the pre-registered 0.8. So Sonnet re-scored all 540 replies
(Spearman over all 536 scored pairs: 0.34). Both judges put nearly every reply in 85–98, and that ceiling depresses rank agreement. Condition
means agree (none 88.8 / 95.2, clamp 87.8 / 94.5, top10 57.1 / 48.5, Haiku / Sonnet), and the positive control fails under both.
`judge_agreement.json`; Haiku scores are kept as `likeness_haiku` in `patch_scored.jsonl`.

## OLMo (`olmo_rl` chat, L\* = 23; secondary)
Attribution only; no patching, as pre-registered. 30 default and 990 role responses on half-B. The post-attention and
post-feedforward RMSNorms are linearised by the per-token scale of each block's actual pre-norm output. Reconstruction |error| is
0.0006 for PC1 and 0.0010 for the contrast axis (max 0.004 / 0.005), against gaps of 1.25 / 2.55, so the gate passes. Reported for
both axes (DECISIONS #13):

| axis | total gap | top-10 share [95% CI] | rule | attention / MLP | blocks 17–23 share | top-10: heads / MLPs |
|---|---|---|---|---|---|---|
| PC1 (`aa_23`) | 1.25 | 0.402 [0.372, 0.449] | intermediate | 0.98 / 0.26 | 72% | 6 / 4 |
| contrast (default − mu) | 2.55 | 0.315 [0.306, 0.332] | intermediate (near the diffuse line) | 2.25 / 0.30 | 72% | 7 / 3 |

The top writers are mostly the same components on both axes: heads L23.h9, L19.h23 and L22.h19, plus the MLPs of blocks 19, 22 and 23.
So the PC1-vs-contrast flag (#13) does not change the conclusion. In OLMo, unlike Qwen, attention writes most of the gap even among the
top single writers, and the MLPs are minor. The shared pattern across both models: the coordinate is written in the ~6 blocks before
L\*, by tens of components, not by a few. `gap_olmo_rl_aa.json`, `gap_olmo_rl_contrast.json`, `attr_olmo_rl/`.

## Files
- `attr_qwen_it/attr.npz` (gitignored, 4.6 MB): per-response attributions, per-layer projections, and role / default mean component
  outputs. `attr_qwen_it/meta.jsonl` gives row metadata.
- `gap_aa.json`: all statistics above. `patch_qwen_it.jsonl`: raw generations with per-row AA. `patch_scored.jsonl`: with judge scores
  and style. `patch_summary.json`: the table.
- Code: `src/pod/q1_attr.py`, `src/q1_analyse.py`, `src/q1_patch_spec.py` (→ `data/Q1/patch_spec.json`), `src/pod/q1_patch.py`,
  `src/q1_patch_analyse.py`, `src/q13_agree.py`, `src/q1_fig.py`. The combined GPU chain is `src/pod/q13_run.sh`.

## Deviations
DECISIONS 25: the patch run was lost in the pod migration and re-run identically, reading the same aa/mu values from a compact basis,
with a paired bootstrap. No other deviations.

## Spend
Judges, Q1: $0.87. That is Haiku patch scoring $0.04, the agreement check $0.09, and the Sonnet re-score of all 540 replies $0.74.
GPU: attribution about 0.5 min on the H100 (old pod), patch 4.2 min and OLMo attribution 0.4 min on the H200 (new pod), about $0.5 in all.
A first patch run was lost when the old pod's volume hung.

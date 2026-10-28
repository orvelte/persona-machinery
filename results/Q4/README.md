# Q4: Pretraining origin. Is the Assistant Axis about narratives?

**Answer.** The axis exists before post-training, but it is not a "helpful character in a narrative" direction.
- **Qwen: the base model's narrative AA is the instruct AA** (cos +0.81 at L24 with the instruct chat AA; +0.88 with the instruct
  narrative AA; ceiling ≈ 0.93; Q4.1 rule *same axis* fires).
- **OLMo: it is not** by the 0.7 rule (cos +0.59 base narrative vs RL chat). Most of the OLMo gap is format: in matched narrative
  format, base vs RL is +0.91, though that comparison is descriptive only.
- **Narrative battery (Q4.2): the pre-registered prediction fails, in the reversed direction.** Projected on the instruct AA, the
  passages are ordered non-narrative expository text (encyclopaedia, code) ≫ first-person / narrator / helpful ≫ villains, in every
  model. Helpful-character passages sit *below* the mean of the other four classes: Qwen base g = −0.73 [−1.05, −0.44], instruct −0.79
  [−1.12, −0.50]; OLMo base −0.93, RL −1.19 (all CIs exclude 0). The instruct models are no better than the bases.
- **Exploratory, not pre-registered**: among the narrative / human classes only, Qwen's helpful characters do sit toward the Assistant
  end (base +0.52 [+0.23, +0.84], instruct +0.45 [+0.15, +0.78]), equally in base and instruct. OLMo shows nothing (≈ 0).
- **Stage tracking (Q4.3)**: OLMo's SFT creates the chat-format axis direction (later stages cos ≥ 0.99 to RL). The battery ordering is
  identical at every stage (Spearman ρ = 1.0 with RL). SFT makes the helpful class slightly *more* negative
  (Δg −0.27 [−0.49, −0.06]). DPO gives the largest d′ increase (+0.56).

- **Convention-vs-persona tasks (Q4.4)**: base and post-trained models give the same mix of narrative-convention answers in
  narrative format (pooled Δ Qwen +0.05 [−0.06, +0.15], OLMo −0.07 [−0.18, +0.04]). The frame matters more than training:
  qwen_it narrative vs chat is −0.15 [−0.25, −0.05]. The AA-ablation arm is uninterpretable because its positive control failed.

On raw passages, the axis reads expository, informational register (what Assistants write) far more than helpful
characters in stories.

## Q4.1 Base-model AA vs instruct AA (M0 bases; `q41.json`, `q41_rows.jsonl`, `figures/q4_axis_cos.png`)

Pre-registered rule: *same axis* iff cos ≥ 0.7 at L\*. The primary layer is the instruct model's L\* (Implementation).

| a vs b | layer (b's L\*) | cos PC1 [95% CI over roles] | cos contrast axis [CI] | CKA (role means) | split-half ceiling | cov-null \|cos\| (mean of 3) | at a's L\* (cos / CKA) | argmax-cos layer |
|---|---|---|---|---|---|---|---|---|
| qwen_base narrative vs qwen_it chat | 24 | **+0.81** [+0.59, +0.85] | +0.68 [+0.62, +0.73] | 0.80 | 0.94 | 0.49 | +0.82 / 0.79 | 22 (+0.82) |
| qwen_base narrative vs qwen_it narrative | 22 | **+0.88** [+0.74, +0.91] | +0.77 [+0.73, +0.81] | 0.86 | 0.93 | 0.57 | – | 1 (+0.94) |
| olmo_base narrative vs olmo_rl chat | 23 | +0.59 [+0.23, +0.85] | +0.40 [+0.30, +0.50] | 0.50 | 0.93 | 0.40 | +0.64 / 0.49 | 19 (+0.68) |
| olmo_base narrative vs olmo_sft chat | 23 | +0.58 [+0.19, +0.86] | +0.44 [+0.34, +0.53] | 0.51 | 0.92 | 0.39 | +0.63 / 0.50 | 19 (+0.68) |
| olmo_base narrative vs olmo_dpo chat | 23 | +0.58 [+0.06, +0.84] | +0.10 [+0.05, +0.18] | 0.51 | 0.93 | 0.39 | +0.63 / 0.50 | 19 (+0.67) |
| olmo_base narrative vs olmo_rl narrative (descriptive: G-AA fail) | 23 | +0.91 [+0.25, +0.93] | +0.81 [+0.77, +0.84] | 0.86 | 0.88 | 0.28 | +0.90 / 0.86 | 21 (+0.92) |
| olmo_sft chat vs olmo_rl chat | 23 | +0.99 [+0.97, +0.99] | +0.71 [+0.67, +0.76] | 0.97 | 0.94 | 0.67 | – | 23 |
| olmo_dpo chat vs olmo_rl chat | 23 | +0.99 [+0.97, +1.00] | +0.22 [+0.17, +0.29] | 0.98 | 0.94 | 0.68 | – | 23 |
| olmo_sft chat vs olmo_dpo chat | 23 | +0.98 [+0.92, +0.98] | +0.24 [+0.19, +0.31] | 0.96 | 0.93 | 0.68 | – | 19 |

**Reading.**
- **Qwen: same axis (rule fires).** The base model's narrative AA has cosine +0.81 with the instruct chat AA at L24 and +0.88 with the
  instruct narrative AA. This is close to the split-half ceiling of ≈ 0.93, and role-mean CKA is 0.80–0.86. The Assistant Axis of
  Qwen3.5-9B-Instruct is already present in the base model under narrative role play.
- **OLMo: not the same axis by the rule.** Base narrative vs any chat stage gives cos ≈ 0.58–0.59 at L23 (CKA ≈ 0.50). The 99-role
  bootstrap CI is wide ([+0.06, +0.86]): OLMo's PC1 shares little with the default Assistant (DECISIONS #13), so resampling roles
  swaps it with PC2. In matched (narrative) format, base vs RL is +0.91 with CKA 0.86, but RL narrative failed G-AA, so this is
  descriptive only. Most of the OLMo base-vs-chat gap is therefore a *format* difference (narrative vs chat role space), not a
  post-training one. For comparison, RL chat vs RL narrative is only +0.52 in M0.
- **Stages.** SFT, DPO and RL chat share PC1 almost exactly (≥ 0.98). The contrast axis (default − role mean) is where the stages
  differ: DPO's default Assistant sits off the SFT/RL direction (cos 0.22–0.24, vs 0.71 for SFT vs RL). The DPO "default Assistant"
  is the outlier stage.
- **Null caveat.** A covariance-matched random direction (drawn from the b model's role covariance) already has \|cos\| 0.3–0.7
  with a's AA, because both live in the same low-dimensional role subspace. The Qwen cosines clear the 0.7 rule, but not the
  shared-design "2 × null" intervention rule (0.81 vs 2 × 0.49). That rule is written for intervention effects, not for direction
  comparisons, so it is reported, not applied. The bootstrap CIs are percentile intervals and are biased low: duplicated roles in a
  resample perturb PC1.

## Q4.2 Narrative battery (`battery.json`, `battery_rows.jsonl`, `figures/q4_battery.png`)

200 passages, 40 per class. Written by Sonnet 5.5 and blind-classified by Haiku 5.5: 198/200 correct on the first pass, 2 rewritten
once, final purity 100% (`data/Q4/battery*.jsonl`, sha256 in PREREG). Each passage is fed as raw text with no template; the score is
the mean residual over tokens 1..n (position 0 excluded; the all-token mean gives identical g to 2 decimals), projected on the
family's instruct chat axis at its L\* (Qwen L24, OLMo L23). g = (helpful mean − mean of the other four class means) / pooled
within-class SD. 95% CI from a 2,000-resample bootstrap of passages within class.

| model | g on instruct PC1 [95% CI] | g on contrast axis (OLMo; Qwen for reference) | g on own AA | human helpers / AI helpers (PC1) | null \|g\| rand / cov | readout positive control (default − low-role texts, SD units) |
|---|---|---|---|---|---|---|
| qwen_base | −0.73 [−1.05, −0.44] | −1.26 [−1.63, −1.00] | −1.14 | −0.84 / −0.59 | 0.43 / 0.48 | +8.9 (raw +31.2 [+29.0, +33.4]) |
| qwen_it | −0.79 [−1.12, −0.50] | −1.33 [−1.70, −1.05] | −0.79 | −0.93 / −0.62 | 0.45 / 0.49 | +8.7 (raw +27.7 [+25.7, +29.8]) |
| olmo_base | −0.93 [−1.32, −0.61] | −1.19 [−1.59, −0.87] | −1.07 | −0.84 / −1.01 | 0.45 / 0.82 | +5.3 (raw +5.3 [+4.0, +6.5]) |
| olmo_sft | −1.20 [−1.59, −0.86] | −1.53 [−1.90, −1.21] | −1.07 | −1.18 / −1.20 | 0.39 / 1.00 | +5.8 |
| olmo_dpo | −1.22 [−1.62, −0.88] | −1.56 [−1.94, −1.23] | −1.21 | −1.19 / −1.23 | 0.37 / 1.00 | +5.8 |
| olmo_rl | −1.19 [−1.59, −0.85] | −1.53 [−1.91, −1.21] | −1.19 | −1.16 / −1.20 | 0.38 / 0.99 | +5.8 (raw +4.7 [+3.6, +5.8]) |

Class means in within-class SD units, centred, on the instruct PC1:

| model | non-narrative | first-person | narrator | helpful | villain |
|---|---|---|---|---|---|
| qwen_base | +4.05 | −0.72 | −0.83 | −0.58 | −1.92 |
| qwen_it | +4.01 | −0.65 | −0.81 | −0.63 | −1.91 |
| olmo_base | +3.20 | +0.50 | −0.58 | −0.75 | −2.37 |
| olmo_rl | +3.67 | +0.24 | −0.46 | −0.95 | −2.50 |

**Pre-registered outcome: neither "present in base" nor "created".** The helpful-class gap is negative with its CI excluding 0 in
both base and instruct, in both families. It is "reversed in both", an outcome the prediction did not list. Instruct − base Δg:
Qwen −0.06 [−0.12, −0.01], OLMo −0.26 [−0.47, −0.05], so post-training moves helpful passages slightly *further* from the Assistant
end. No gap "beats the nulls" (that requires a positive gap). The null \|g\| is 0.4–1.0, below the observed \|g\| of 0.7–1.6.

**Positive control passes.** The same pipeline separates M0 default-Assistant replies from replies of the 10 lowest-AA roles
(fit-question texts, presented as raw passages) by +5 to +9 SD in every model. So the passage readout works; the battery result is
not a dead readout.

**Exploratory (not pre-registered; chosen after seeing that the non-narrative class dominates the mean).** Helpful vs the mean of
villain, narrator and first-person only (pooled SD over those 4 classes):
- Qwen base +0.52 [+0.23, +0.84], instruct +0.45 [+0.15, +0.78] on PC1; contrast axis +0.24 / +0.21 (CIs include 0).
- OLMo PC1 +0.07 / −0.04 / −0.06 / −0.04 (base / SFT / DPO / RL; all CIs include 0); OLMo contrast −0.59 to −0.93 (negative).

Within narrative text, then, Qwen's axis places helpful characters on the Assistant side of villains, narrators and diarists. That
ordering is already fully present in the base model and is not sharpened by post-training. OLMo has no such ordering.

## Q4.3 Stage tracking, OLMo-3-7B (`battery.json` → `olmo.stages`, `figures/q4_stages.png`)

| stage | cos(stage AA, RL chat AA) L23 | cos contrast axes | held-out d′ (M0) | battery g, final PC1 [CI] | battery g, final contrast | g on own AA | ρ class order vs RL |
|---|---|---|---|---|---|---|---|
| base (narrative) | +0.59 | +0.40 | 1.04 | −0.93 [−1.32, −0.61] | −1.19 | −1.07 | 1.0 |
| SFT (chat) | +0.99 | +0.71 | 0.89 | −1.20 [−1.59, −0.86] | −1.53 | −1.07 | 1.0 |
| DPO (chat) | +0.99 | +0.22 | 1.45 | −1.22 [−1.62, −0.88] | −1.56 | −1.21 | 1.0 |
| RL (chat) | 1 | 1 | 1.42 | −1.19 [−1.59, −0.85] | −1.53 | −1.19 | 1.0 |

Transitions (paired bootstrap on g):
- base→SFT: Δd′ −0.15, Δg −0.27 [−0.49, −0.06].
- SFT→DPO: Δd′ **+0.56**, Δg −0.02 [−0.03, −0.00].
- DPO→RL: Δd′ −0.03, Δg +0.03 [+0.02, +0.03].

**Reading (pre-registered: the stage with the largest increase "creates or sharpens").**
- **d′ is sharpened by DPO** (+0.56). But base d′ is narrative-format and the rest are chat, so the base→SFT step mixes format with
  training.
- **Battery gap**: the only increase is DPO→RL (+0.03), which is negligible; the substantive change is SFT's decrease.
- **Direction**: SFT sets the chat-format axis direction, and DPO and RL leave it unchanged (cos 0.99).
- **Contrast axis**: DPO moves the default Assistant off the SFT/RL direction (cos 0.22), and RL brings it most of the way back (SFT
  vs RL 0.71).
- **Battery ordering**: identical at every stage, including base.

## Q4.4 Convention-vs-persona tasks (`q44.json`, `q44_labels.jsonl`, `q44_narr_*.jsonl`, `q44_chat_qwen_it.jsonl`, `figures/q4_tasks.png`)

**Items.** 59 items: kind white lie (19), genie grants without caveats (20), human character stays in character when asked "are you
an AI?" (20). Written by Sonnet 5.5 and blind-checked by Haiku (59/59 pass; DECISIONS #19).

**Cells.**
- Narrative format: the story + `Then {speaker} answered: "`, run on qwen_base, qwen_it, olmo_base and olmo_rl.
- qwen_it chat: "Continue this story with {speaker}'s reply…". Conditions: none; AA projection-out at L24–32 on generated positions;
  random and covariance-matched projection-out nulls (3 draws each).
- 3 samples per item (1 per null draw).
- Readout: Haiku sees the scene, the reply and both reply-type descriptions (order counterbalanced), never the model or condition,
  and labels each reply convention / norm / neither.

Share of narrative-convention answers among labelled replies [Wilson 95%]:

| cell | kind lie | genie | stays human | all |
|---|---|---|---|---|
| narr qwen_base | 23/52 (0.44) | 36/57 (0.63) | 36/53 (0.68) | 95/162 (0.59) |
| narr qwen_it | 22/53 (0.42) | 33/56 (0.59) | 32/52 (0.62) | 87/161 (0.54) |
| narr olmo_base | 17/44 (0.39) | 22/50 (0.44) | 40/54 (0.74) | 79/148 (0.53) |
| narr olmo_rl | 23/54 (0.43) | 35/57 (0.61) | 40/52 (0.77) | 98/163 (0.60) |
| chat qwen_it none | 30/48 (0.63) | 28/59 (0.47) | 55/56 (0.98) | 113/163 (0.69) |
| chat qwen_it AA ablation | 29/49 (0.59) | 26/56 (0.46) | 52/53 (0.98) | 107/158 (0.68) |
| chat random null (pooled) | 31/49 | 28/57 | 56/56 | 115/162 |
| chat cov-matched null (pooled) | 33/53 | 22/56 | 57/58 | 112/167 |

**Contrasts (Newcombe 95%), all tasks pooled.**
- Base − instruct, narrative format: Qwen +0.05 [−0.06, +0.15]; OLMo −0.07 [−0.18, +0.04].
- AA ablation − none: −0.02 [−0.12, +0.09]. Nulls: rand +0.02 [−0.08, +0.12], cov −0.02 [−0.12, +0.08].
- No per-task contrast between base and instruct, or between ablation and none, has a CI excluding 0.
- Format matters more than training. qwen_it narrative − qwen_it chat is −0.15 [−0.25, −0.05], driven by "stays human":
  0.62 in raw narrative vs 0.98 in chat. In raw story continuation the character often turns out to be an AI (a plot twist); asked
  to continue the story in chat, the instruct model keeps the human in character almost always.

**Ablation positive control: FAILED, so the ablation results are not interpreted (PREREG).** Clamping the AA coordinate to the
5th-percentile role coordinate at L24–32, on generated positions, did not lower Haiku Assistant-likeness on 20 Chen held-out
questions: 85.5 → 87.4, Δ +1.9 [−1.8, +7.0]. The intervention at this strength does not move a known AA behaviour. This matches
M0's G-trait experience: single-site, response-only edits of this size do little in qwen_it. The null ablation effect above is
therefore uninformative about the AA, not evidence against it.

**Reading (descriptive, no threshold).**
- None of the three narrative conventions differs between base and post-trained models in narrative format.
- The convention-vs-Assistant-norm split is set by the frame (raw story vs chat request), not by post-training.
- The "AA ablation" readout could not be validated.

## Spend
- **API.** Q4 item writing (Sonnet 5.5) $0.49, blind checks (Haiku) $0.15, Q4.4 labels (Haiku, 1,496 calls) $0.08. Q4 total ≈ $0.71.
  With Q4b ($0.41), Agent D's API total is ≈ $1.12 of the $4.00 share.
- **GPU.**
  - Battery forward passes ran in the same model loads as Q4b Arm A (≈ 4 min total for six models).
  - Q4.4 generations ≈ 6 min (4 vLLM runs at 0.3 min each plus start-up; HF intervention run 2.5 min).
  - Total ≈ 10 GPU-min ≈ $0.9.

## Deviations (DECISIONS #17–20)
- **#17.** Q4.1 ran from the M0 bases before the Implementation sub-section was written. Its analysis choices were fixed in the
  script beforehand.
- **#19.** Q4.4 item check: the genie brief was reworded after a content-filter refusal; kind_lie has 19 items; checker answers with
  no answer line were re-asked.
- **#20.** Passage means exclude token 0. The all-token sensitivity gives the same g.
- **Exploratory, labelled as such.** The narrative-only helpful contrast in Q4.2; the human − AI interaction in Q4b.
- **Q4.2 outcome.** "Reversed in both" was not one of the pre-registered outcome labels. It is reported as what happened, and the
  prediction counts as failed.

Code: `src/q4_q41.py`, `src/q4_items.py`, `src/q4_build.py`, `src/q4_analyse.py`, `src/q4_tasks.py`, `src/pod/q4_fwd.py`,
`src/pod/q4_run_*.sh`. Raw rows: `q41_rows.jsonl`, `battery_rows.jsonl`, `q44_*.jsonl`. Per-passage activations
`fwd_<model>/passages.npz` are gitignored (96 MB each).

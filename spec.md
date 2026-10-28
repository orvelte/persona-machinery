# Spec: how the persona machinery is represented (new repo `persona-machinery`)

## Goal

Neel's high-priority basic-science questions:

- How is the machinery that persona vectors and the Assistant Axis plug into actually implemented?
- What does the Assistant Axis correspond to in pretraining? Is it about narratives?

This repo asks five concrete questions, each with a pre-registered test.

| Question | Content |
|---|---|
| Q1 Writers | Which components write the Assistant-Axis (AA) coordinate, and where? |
| Q2 Triggers | Which input tokens trigger it? In particular, is it the chat template? (Our raw-document result says gaming disappears without the template.) |
| Q3 Readers | What reads the AA coordinate downstream, and what behaviours depend on it? |
| Q4 Pretraining origin | Does the AA exist in base models, and is it a "helpful character in a narrative" direction? |
| Q5 Persona vectors vs the AA | Are trait persona vectors (sycophancy, evil, hallucination) separate features, or do their effects run through the AA? |

**Stretch, only if time remains:** Q6 attractor dynamics. After a persona perturbation, does the residual stream return to the Assistant region over tokens? How fast, and does that differ between base and instruct?

## Models

Use small models so sweeps are cheap. All fit on one H100 in bf16.

- **Qwen3.5-9B base and Qwen3.5-9B instruct.** Same lineage as our 27B work; this release ships a base checkpoint.
- **OLMo-3-7B, base / SFT / DPO / RLVR (Think line).** The recipe is known, so stage-by-stage questions have answers. Confirm the exact HF names first.
- **Optional:** one model with public SAEs or transcoders (e.g. Gemma 3 with Gemma Scope 2). Use it for feature-level decomposition in Q1/Q5. The agent must first confirm the release exists and covers the needed layers; if not, skip it.

## Inputs

- **`vendor/persona-mech` (read-only):**
  - role library (99 roles × 20 questions);
  - P-subspace extraction code;
  - ablation and steering hooks;
  - capture and replay code.
- **Persona-vector trait definitions:** sycophancy, evil, hallucination. Use the Chen et al. pipeline as reimplemented in persona-mech (or the `charactertraits` repo if that's where it lives).
- **Never write to `vendor/`.**

## Rules

- **Pre-register first.** Write each module in `docs/PREREG.md` before reading its data, and log deviations in `docs/DECISIONS.md`.
- **Positive control before any null.**
  - Every ablation or steering result needs a positive control showing the intervention can move a known behaviour.
  - Every result needs random and covariance-matched subspace nulls.
- **No self-report as evidence of mechanism.** What the model says about itself is a behavioural readout only.
- **Report effect sizes with CIs.**
  - Bootstrap over prompts.
  - Hold out a prompt split for every direction that is fitted, so no direction is evaluated on the prompts it was fitted on.
- **Git:** commit locally; don't push.
- **Budget:**
  - Report the GPU burn rate after the first hour.
  - **Hard cap $90; stop and ask at $70.**
  - Pod about $3–4/h; API judges under $10.

## Shared setup (M0, about 30 min)

1. **Extract the AA** in each model following the Assistant Axis method:
   - role-response mean activations;
   - PC1 of the role means;
   - orient it so that the default Assistant sits at the positive end.
2. **Report:**
   - the variance explained by PC1;
   - its split-half stability;
   - its cosine with the persona-mech frozen axis (Qwen only, via the 9B ↔ 27B comparison where possible; otherwise skip).
3. **Extract persona vectors** for three traits (Chen-style contrastive prompts) and **P_k**, the top k role PCs, at k = 5 and 15.
4. **Gates:**
   - split-half stability of AA ≥ 0.9;
   - each trait vector steers its trait (judge-scored) at coherence ≥ 0.8.

## Q1. Who writes the AA coordinate?

**Method: direct attribution, plus activation patching to check it.**

1. On assistant-turn tokens, compute the residual projection onto AA at layer L*. L* is the layer of maximal role separation, chosen on held-out prompts.
2. Decompose that projection into contributions from each attention head and MLP output at layers ≤ L*.
3. Take the top 10 writers, ranked by mean contribution difference between Assistant and role contexts.
4. Patch each writer's output from role runs into Assistant runs, alone and in sets of 3 and 10. Measure the AA shift and the behaviour shift: self-description and style, scored by a judge.

**Pre-registered question:** is the AA written by a few components (top 10 explain ≥ 50% of the Assistant − role difference) or diffusely?

**Report:**

- the answer to that question;
- the layer profile;
- whether the writers are attention (reading from context) or MLP (computing).

## Q2. What triggers it? Chat template vs content

**Method: attention knockout and token ablation.**

1. **Template knockout.** In instruct models, block attention from assistant-turn positions to:
   - (a) the chat-template tokens (role markers, `<|im_start|>assistant` etc.);
   - (b) the system prompt;
   - (c) the user content.
2. Measure the AA projection and the behaviour for each.
3. **Raw-document control.** Repeat with the same content formatted as a raw document, no template. Compare the AA projection.

**Pre-registered prediction (from our M5 result):** knocking out the template tokens moves the AA projection toward the raw-document value more than knocking out the user content does.

**Why it matters.** This tests whether "being the Assistant" is keyed to the template. If so, it explains why the raw-document frame removed gaming.

## Q3. What reads it, and what depends on it?

**Method:**

1. **Behaviour panel under AA manipulation.** Ablate the AA (projection removal at layers L*–L*+8, generated tokens only), and separately clamp it to role-like values. Score a panel of behaviours, each with a judge or a deterministic metric:
   - refusal of harmful requests;
   - sycophancy;
   - self-description;
   - style;
   - honesty under a false-premise question;
   - instruction following on format constraints, as a near-neutral control.
2. **Readers.** For the behaviours that move, find which downstream heads and MLPs mediate the effect. Use path patching restricted to the AA direction: patch only the AA component of the residual into each downstream component's input.

**Pre-registered question:** does the AA act as a broad "who am I" switch (moves ≥ 3 behaviours) or a narrow one? Check that the format-following control stays at 90% or more of baseline.

## Q4. Pretraining origin: is the AA about narratives? (Neel's high priority)

**Method:**

1. **Base-model AA.** Extract the AA in the base model using narrative-formatted role play. Example prompts:
   - "The following is a transcript of a conversation between a user and a helpful AI assistant."
   - "...between a traveller and a pirate captain."

   Do this for the same 99 roles. Compare it with the instruct AA (cosine at each layer, via the residual basis, which is shared within a lineage; check with CKA).
2. **Narrative battery.** Project base and instruct activations onto the instruct AA for about 200 short narrative passages, sorted into five classes:
   - helpful servant or assistant characters (butlers, guides, AI assistants in fiction);
   - villains and tricksters;
   - neutral narrators;
   - non-narrative text (encyclopedia, code);
   - first-person human speakers.

   The passages are written by one agent and reviewed blind by another for class purity. **Pre-registered prediction:** helpful-character passages sit on the Assistant end in the base model, and post-training shifts and sharpens this rather than creating it.
3. **Stage tracking (OLMo-3).** Compute the following at base, SFT, DPO and RLVR:
   - AA cosine with the final model's AA;
   - role separation along AA;
   - the narrative-battery ordering.

   Which stage creates or sharpens the axis?
4. **Coin-toss-style edge case.** Neel's PSM example is narrative convention vs persona. Build 3 small tasks where the narrative-convention answer and the Assistant-persona answer differ. Example: a story where the helpful character is expected to lie kindly, vs the Assistant's honesty norm. Ask whether base vs instruct, and AA ablation vs no ablation, move the answer.

## Q5. Persona vectors vs the AA

**Method:**

1. Decompose each trait vector into its component along the AA, its component in P15 ⊥ AA, and the remainder.
2. Steer with:
   - (a) the full vector;
   - (b) the vector orthogonalised to AA;
   - (c) the vector orthogonalised to P15;
   - (d) the AA component alone.
3. Measure trait expression and coherence for each.
4. Also ablate AA while steering with the full trait vector.

**Pre-registered question:** is a trait's steering effect mediated by the AA? The rule is that orthogonalising to AA removes ≥ 50% of the effect, or ablating AA blocks it. Otherwise the trait vectors are independent features riding next to it.

## Stretch Q6. Attractor dynamics

1. **Perturb.** Inject a role vector for 5 tokens mid-response.
2. **Track.** Follow the AA projection over the next 200 tokens in base vs instruct and across OLMo stages. Report the recovery time constant.
3. **Prediction:** the instruct model returns to the Assistant region faster. If so, that is a mechanistic signature of "persona selection" as an attractor that post-training strengthens.

## Order and parallelism

Split the work across three agents after M0.

| Agent | Modules |
|---|---|
| Agent A | Q1 + Q3 (writers and readers) |
| Agent B | Q2 + Q4 (template and pretraining), on the 9B base/instruct pair and the OLMo stages |
| Agent C | Q5, then Q6 if time remains |

Each agent:

- pre-registers before running;
- shares the M0 artifacts read-only;
- logs to its own `results/Qx/`.

**Blind red-team agent at the end.** It receives only code and results. It checks for:

- hook leakage into prompt tokens;
- directions fitted on the evaluation prompts;
- nulls that aren't variance-matched;
- judge prompts that leak condition labels.

## Deliverables

- `results/Qx/README.md` for each module, stating which pre-registered rule fired.
- **One summary table:** question → answer → effect size → confidence.
- **Figures (matplotlib):**
  - serif font (Liberation Serif), inward ticks on all sides, faint horizontal grid;
  - colour = category, hatch or line style = model or stage;
  - palette `#3B73C8 #E07B39 #4CAF7D #E5A93B #D9769C #8E6BBF #3BA7A7`;
  - bootstrap or Wilson 95% intervals, italic subtitle with n and setup.
- **Research-log artifact entry** with actual spend.

## Not in scope

- 27B models.
- RL training.
- SAE training from scratch. Only use existing public SAEs or transcoders, if they exist.
# Red-team review: M0, Q1, Q2, Q2b, Q3, Q4 (+ Q4.2b), Q4b, Q6

Blind review from code and raw rows. Q5 not reviewed. All recomputations used `.venv/bin/python` on the committed raw files and
`results/M0/basis_*.npz`. Severity: H = a headline claim is wrong or unsupported as stated; M = a conclusion or an interpretation needs to
change or be qualified; L = a hygiene or reporting issue.

## Issues, ranked by severity

### 1. [H] Q4.2 / Q4.2b: the register and character effects are not specific to the AA. Random directions produce the same effect sizes.
- **Evidence.** I re-projected `results/Q4/fwd_<model>/register.npz` (X_excl0 at L24 Qwen / L23 OLMo, about `mu_L`) onto the M0 null
  directions and onto other PCs, using the same pooled-SD R/C/K code as `src/q4_register.py`. On the AA, my values reproduce the README
  exactly (qwen_base R +1.92, C +1.20, K +2.66). The same statistics on other directions:
  - isotropic `rand1_24_{0,1,2}`, qwen_base: R −0.08 / −1.31 / −0.17, K +1.15 / −2.63 / −1.31;
  - `cov1_24_{0,1,2}`, qwen_base: R +0.25 / −2.22 / +1.94, C +1.97 / −1.57 / +1.61;
  - olmo_rl, isotropic `rand1_23_0`: K +2.85, larger than the AA's +2.46; `cov1_23_0`: R +2.09, K +3.15;
  - PC5 of the qwen_it role means: K +2.79 (qwen_base).
  The six classes are far apart in activation space, so almost any direction separates them by 1–3 within-class SDs. Under the shared-design
  "beats the nulls" rule (point estimate > 2 × the larger null family's mean |effect|), no R, C or K beats the nulls in any model. For
  qwen_base: R 1.92 vs 2 × 1.47 (cov); C 1.20 vs 2 × 1.72; K 2.66 vs 2 × 1.70 (rand). Q4.2b reports no nulls at all
  (`src/q4_register.py` has no rand/cov projection), although the shared design applies the nulls to every module.
- The battery has the same problem. The "reversed" helpful gap is |g| 0.73–1.19 against null |g| 0.43–1.00 (`results/Q4/battery.json`),
  which fails the 2× rule in every model (qwen_base 0.73 < 0.96; olmo_rl 1.19 < 1.98). The README's line "no gap beats the nulls (that
  requires a positive gap)" hides that the negative gaps do not beat the nulls in magnitude either.
- **Changes a headline?** Yes. "On raw passages the axis is first a third-person reference register … character is a second, additive
  dimension" describes the passage geometry, not a property of the AA. "The prediction fails" survives. "Reversed in both" and the
  register reading do not, as AA-specific claims.
- **Fix.** Report R, C, K and g on the 3 + 3 null directions and on PCs 2–10 for every model. State AA claims only where they beat
  those nulls, or express effects as a percentile of a null distribution (e.g. 1,000 isotropic directions plus role-span directions
  orthogonal to the AA).

### 2. [H] Q3 (with Q1): "the AA coordinate is a readout, not a lever" is not supported. The intervention was never shown to work, and Q4.4 reads the identical null the opposite way.
- **Evidence.**
  - Q3 projects out `aa_L` at L24–32 on generated positions (`data/Q3/conditions.json`). There is no efficacy positive control: the format
    control only checks that the ablation is not destructive. The repo's own rule ("positive control before any null", `README.md`) is
    therefore not met.
  - The same intervention family failed every positive control it was given. Q1's clamp at L\*: +0.7 points, needing ≥ 20 (recomputed
    from `results/Q1/patch_scored.jsonl`: Sonnet +0.73, Haiku +1.0). Q4.4's clamp to the 5th-percentile role coordinate at L24–32: +1.9
    [−1.8, +7.0].
  - Q4.4 uses the identical op (`data/Q4/q44_conditions.json` "ablate") and concludes that the null is "uninformative about the AA, not
    evidence against it" (`results/Q4/README.md:227`). Q3 draws the opposite conclusion (`results/Q3/README.md:11–12`).
  - The prompt-position KV is untouched by construction (`src/pod/pm_gen_hf.py:65`), so later layers can still read the default persona
    from the prompt.
  - Four of the five readouts sit at a floor or ceiling: refusal 97.5%, honesty 100%, sycophancy 0.9/100, self-description 94.9/100
    (recomputed).
- **Changes a headline?** Yes, the reading (the rule label "narrow" is computed correctly). The defensible statement is: "a response-only
  AA edit at L24–32 has no detectable effect; whether the AA is causal cannot be decided with this intervention."
- **Fix.** Make the Q3 and Q1 READMEs consistent with Q4.4. Add an efficacy control: for example, the same op with prompt positions
  included, or adding a role's AA displacement (Q6-style, at 1× and 3×) and checking that assistant-likeness moves. Choose panel items
  with baseline headroom.

### 3. [M] Layer 32 is fitted on post-final-norm states but intervened on pre-norm states (Q3 projout/clamp, Q4.4 ablate/clamp_low, nulls at L32).
- **Evidence.**
  - `src/pod/pm_acts.py:54` stores `o.hidden_states[1:]`. In HF, the last element is the output after the final RMSNorm.
  - `pm_gen_hf.py:80` hooks `layers[31]`'s output, which is the pre-norm residual.
  - The basis shows the break: ‖mu_L‖ in qwen_it chat is 62, 64, 81, 97, 107, 100 for L26–31, then 43 at L32. cos(aa_L, aa_{L−1}) is
    ≈ 0.97 up to L31, then 0.88 at L32. In OLMo, ‖default − mu‖ jumps from 4.06 to 11.74 at L32.
  - So at L32 the op sets the pre-norm coordinate along a post-norm direction to a post-norm target (`medrole_tgt_32`, `mu_32`, clamp_low
    −26.7). That is an arbitrary shift, not an AA ablation.
- The same mismatch affects M0's layer-32 descriptives, and with them the DECISIONS #12 rationale ("argmax d′ fell on the final layer").
- **Changes a headline?** Maybe. It affects one of nine layers in Q3 and Q4.4. Both results were null and the format control passed, but
  the layer-32 edit is mis-specified.
- **Fix.** Refit layer 32 from the decoder-layer output hook, or drop L32 from the windows. Flag it in M0.

### 4. [M] Q2: the "≤ 0.15 role SDs" size claim holds only on PC1. On the contrast axis the effects are 2–6× larger.
- **Evidence.** The README claim is at `results/Q2/README.md:7` and `:98`. Across-role SD on c, computed from the M0 role means on Q2's
  half-A axes: Qwen 8.05, OLMo 1.02. In those units:
  - Qwen KO(a) on c: −2.78 → 0.35 SD;
  - OLMo KO(a) on c: +0.35 → 0.34 SD;
  - OLMo KO(b), the system message: −0.88 → **0.86 SD**. That is the same size as a role system prompt (0.5–1.1 SD).
- DECISIONS #13 requires flagging any PC1-vs-c disagreement, and this is one. "Teacher-forcing … keeps the coordinate Assistant-like
  whatever the prompt shows" is false on c for OLMo.
- Recomputed rule statistics (from `tf_rows.jsonl` + `tf.npz`, cluster bootstrap) match the README:
  - Qwen PC1 Δt − Δu: 0.131 [0.010, 0.267];
  - OLMo PC1: 2.86 [−9.4, 12.1], with a near-zero denominator (gap 0.031). Report the un-normalised shift difference instead: +0.089
    [−0.021, +0.199].
- **Changes a headline?** Maybe. It affects the "size" bullet and the M5 bearing, not the rule outcome.
- **Fix.** Report the shifts in each axis's own role-SD units, and qualify the "mostly carried by the response tokens" claim for OLMo c.

### 5. [M] Q4b: decision-table row 4 fires only because of a 4-story percentile bootstrap.
- **Evidence.** The human-protagonist per-story effects are 0.041, 0.016, −0.001 and 0.012 (`results/Q4b/decision.json`). The
  percentile bootstrap (`src/q4b_analyse.py:71`) gives [+0.003, +0.034]. A t-interval gives [−0.011, +0.045]. An exact sign-flip test
  gives p = 0.25, and with n = 4 its minimum possible p is 0.125. No 4-story cell can ever be significant at 5%, so row 4 can never
  legitimately fire. The authors call it "not robust" but still report it as fired.
- **Changes a headline?** Yes, for the decision label (qwen_base becomes "no row fires"). The substantive reading (≈ 0 in Arm A) is
  unchanged. The 12-story overall t-interval [−0.005, +0.035] agrees with the reported result.
- **Fix.** Use exact permutation or t intervals for n ≤ 12. Declare protagonist-level rows untestable at n = 4.

### 6. [M] Null construction (`src/pod/pm_basis.py:69–74`; `src/q6_build.py:59–60`): neither null is a variance-matched control.
- **Recomputed at L\*,** from the basis npz, as the variance of the 99 role means captured relative to the AA:
  - isotropic rand1: 0.07–0.11%;
  - cov1: 10–80% (qwen_it chat 0.48 / 0.24 / 0.30).
- The cov nulls are partly the AA itself: |cos(cov1, aa)| at L\* is 0.45–0.63 (qwen_it chat) and up to 0.85–0.88 (Qwen narrative pairs).
  They also displace the default Assistant by 58–68% of the AA's displacement (qwen_it chat: 6.7–7.8 vs 11.5).
- So the random null is a straw man that moves almost nothing, and the cov null contains part of the signal.
  - Q3: the cov null raises the bar (style cov_0 +23.9); conservative.
  - Q4.1: the null |cos| is 0.3–0.7 (disclosed).
  - Q4.2 / Q4.2b: see issue 1.
  - Q6: issue 7.
- P5 / P15 cov nulls capture 53–64% and 77–82% of the real subspaces' role variance. Those are closer to matched.
- **Changes a headline?** Maybe (it matters through issue 1).
- **Fix.** Add a null drawn from the role span, orthogonalised to the AA, and rescaled to the AA's role variance or default displacement.
  Report all three.

### 7. [M] Q6: the positive control is tautological, the nulls are weak by construction, and the README's mechanism is wrong.
- **Evidence.**
  - W is read at the layer where the vector is added (`src/pod/q6_attractor.py:51–58`), so W_role ≈ the vector's own AA component
    (`data/Q6/roles.json` aa_comp −29 to −32 in qwen_it chat; W −30.5).
  - The isotropic null has AA component |0.0–0.7| (`rand_aa_comp`), so W_rand ≈ 0 by construction.
  - The cov null's signed AA components cancel within the family (qwen_it chat: −16.9, −1.4, +3.4, +29.0, +21.5). W_cov/W_role is
    therefore 2.7× for olmo_dpo, 3.4× for olmo_base and 4.4× for qwen_it chat, not "an order of magnitude or more"
    (`results/Q6/README.md:35`).
  - The perturbation is added after block L\*, so the layer-≤ L\* K/V of tokens 20–24 are never perturbed. The L\* readout at t ≥ 25 can
    change **only** through changed sampled tokens, not through "attention to the perturbed tokens' cached states" (`README.md:9`).
  - Verified from `results/Q6/raw/traj_*.npz`: on post-window stretches where the tokens are identical to `none`, |Δ| has median 0.0
    (Qwen) and max ≤ 0.19 (bf16 batch noise). 73% of role trajectories in qwen_it chat diverge in tokens by t ≤ 25.
  - So Q6 already measures "a 5-token kick, then read the model's own altered text", which is the README's proposed fix, only weakly.
  - R50 of the cov family equals or exceeds the role's in qwen_base (−0.37 vs −0.34). The residual displacement is generic divergence,
    not role-specific.
- **Recomputed** W, K and f50 for qwen_it chat, olmo_base and qwen_base from the raw npz: these match `summary.json` (e.g. qwen_it chat W
  −30.48, K −0.861 [−1.549, −0.257], f50 0.004 [−0.011, 0.017]).
- **Changes a headline?** No. "No detectable change" stands. The explanation and the proposed follow-up need rewriting.
- **Fix.** Read out at layers > L\* as well. Use nulls matched on the AA component (aa_comp·aa plus a random orthogonal remainder). Report
  f50 net of the null family's R50.

### 8. [M/L] Undisclosed evaluation on fit data (Q1, Q6). The disclosed cases are adequate.
- **Q1.** The attribution, top-10 selection and patching readouts all use half-B questions, while `aa_24` is fitted on all 20 questions
  (`pm_basis.py:48`; `data/Q1/patch_jobs.jsonl`: 10/10 questions are half B). The Q1 README does not mention it. The share statistic is
  relative and the patching readouts are null, so the impact is small, but the selection and evaluation use the same responses.
- **Q6.** The prompts are default × half-B (`data/Q6/jobs_*.jsonl`). The role vectors, `default_L` and `aa_L` are fitted on all 20
  questions, and that includes the evaluated default responses. Not disclosed. Impact on dynamics: small.
- **Disclosed and adequate:**
  - Q3 style on half-B (DECISIONS #23; the other five Q3 sets have 0 overlap with the role library or Chen questions, verified);
  - the Q4.2 readout positive control on fit texts (PREREG Q4 Implementation).
- **Clean:** Q2 (half-A axes, half-B evaluation); Chen extraction (80 questions) vs held-out (60 used), 0 overlap; Q4.4's positive-control
  questions are Chen held-out.
- **Changes a headline?** No.
- **Fix.** Refit on half A (as Q2 did) for Q1 and Q6, or disclose in both READMEs.

### 9. [L/M] M0: the post-hoc L\* window flipped a gate.
- DECISIONS #12 caps L\* at 24 after seeing the profiles. That turns qwen_base narrative from G-AA *fail* (split-half 0.896 at the
  unconstrained L\* = 30; 0.87–0.90 at every layer from 24 to 32, `basis_qwen_base_narrative.json`) into *pass* (0.912 at L22).
- The flip is logged, but its consequence is not. qwen_base narrative is an intervention arm of Q6's primary contrast and of the Qwen
  stage steps, which would otherwise be "not testable".
- **Changes a headline?** Maybe (Q6 primary contrast), though the outcome was null.
- **Fix.** State this in the M0 and Q6 READMEs. Report Q6 qwen_base as conditional on the re-freeze.

### 10. [L] Q1: the nulls are selected to have near-zero AA writes, and the positive control's AA readout is tautological.
- `rand10` draws from the 686 non-top components, most of them early heads with near-zero gap. `lm10` takes the smallest |gap| per block
  (`src/q1_patch_spec.py:29–42`). Beating them on "AA closed" is guaranteed by the selection, and neither is matched on write norm or
  disruption.
- The top-10 patch breaks 11 of 30 replies. Among the non-degenerate replies it still closes 0.80 of the gap (recomputed), so the
  mechanistic claim survives.
- `clampLstar`'s AA is read after the clamp at the same hook (`src/pod/q1_patch.py:83–92`), so the 105% figure is mechanical.
- **Changes a headline?** No.
- **Fix.** Add a null of 10 components matched on output norm (same blocks, top by ‖write‖, low AA gap). Read the clamp's AA effect at a
  later layer.

### 11. [L] Judges: no label leakage, but the readout is insensitive and the agreement protocol differs between the two judges.
- **No leakage.** Every judge prompt receives only question/request + reply, or for Q4.4 story + reply + the two type descriptions in
  counterbalanced order:
  - `src/judge.py:20–33`, `src/q3_score.py:25–31,96–102`, `src/q1_patch_analyse.py:30–31`, `src/q2_analyse.py:172–173`,
    `src/q4_tasks.py:30,65–71`, `src/m0_traits.py:48,97`;
  - no condition, model, coefficient or role label is sent (Q4.4's hashed key only sets the order);
  - the cache is keyed on the payload (`src/api.py:67`).
- **Insensitive readout.** ASSISTANT_LIKENESS sits at 85–98 at baseline and has never been shown to drop on fluent non-assistant text: it
  drops only on degenerate output (Q1 top10, Q2 KO). That weakens every failed positive control (Q1, Q4.4) as evidence about the AA.
- **Different protocols.** Haiku was run "number only, 12 tokens, no thinking"; the Sonnet agreement and re-scores used "reasoning, number
  on the last line" (`src/q13_agree.py:27–33`). The agreement failures (ρ 0.30–0.57) therefore confound the model with the protocol.
- **Changes a headline?** Maybe, through issue 2.
- **Fix.** Validate the judge on M0 role-prompted replies (it should span 0–100). Use one protocol for both judges.

### 12. [L] Bootstrap units are slightly too fine in places.
- Q3 self-description resamples (prompt, sample) pairs, with 2 samples per prompt.
- Q4.2 / Q4.2b resample passages within class, ignoring the 4 sub-theme calls × 10 passages.
- Q1's primary CI resamples responses (the question-clustered CI is also reported and agrees).
- The Q4.4 Newcombe intervals treat 3 samples per item as independent. Re-done with an item-cluster bootstrap, the conclusions are
  unchanged: narr qwen_it − chat none −0.153 [−0.257, −0.046]; base − instruct +0.046 [−0.016, +0.108]; ablate − none −0.016
  [−0.082, +0.054].
- **Changes a headline?** No.
- **Fix.** Cluster on prompt or sub-theme.

### 13. [L] Factual slips in the READMEs.
- Q2b: "gaming is confined to two tasks" (`results/Q2b/README.md:16`), but `floor_div` was gamed 1/8 in W2 and in W3
  (`summary.json`).
- Q6: "order of magnitude" (issue 7).
- Q2: the Qwen random control knocks out 11–12 of 12–14 user tokens (`checks.json` n_rand; `src/pod/q2_knock.py:86–90`), so
  "template − random" is close to "template − user". Say so.

## Checked and clean
- **pm_gen_hf hooks touch decode steps only** (`pm_gen_hf.py:65`). Batches never mix conditions (`:100–107`); left padding stays in the
  prefill. Clamp targets are relative to `mu`, as the op assumes: `medrole_tgt_L` equals (rolemean_bartender − mu_L)·aa_L at L24/28/32
  exactly, and `clamp_low` lies within the relative role range.
- **Q6 prompt and early tokens are untouched.** Max |Δ| for t < 20 is exactly 0 in the three pairs I recomputed, and sampling uses common
  random numbers across conditions.
- **Q1 patch hooks are decode-only** (`q1_patch.py:59,75,83`). The torch seed is reset per condition. All Q1 patch table numbers were
  recomputed from `patch_scored.jsonl` and match: clamp AA 1.050, top10 0.857 / +46.7, top3 0.189 / +0.9, lm10 −0.002, rand10 0.009–0.036.
- **Q1 attribution.** Response-span only, reconstruction gate as reported, and the Qwen gap and top-10 come from `attr.npz` via
  `q1_analyse.py`.
- **Q2 masks.** The mechanics checks pass (`checks.json`): zero weight to knocked keys, 24/24 linear-layer kills, and replay cos 1.0000.
  The rule statistics were recomputed for both models and both axes, and they match.
- **Q2b.** Gate counts recomputed from `scored_gate_W*.jsonl`: W1 5 / 41, W2 7 / 36, W3 9 / 40. No truncations.
- **Q3.** All projout, clamp and null deltas recomputed from `scored_qwen_it.jsonl` (refusal −2.5, sycophancy +0.48, self-description −0.55,
  honesty 0, format −2.5, style +39.5 [6.3, 77.6]). They match. The rule logic in `q3_score.py:133–164` follows the pre-registration.
  Note that one prompt (+1.74) carries 44% of the style effect.
- **Q4.1.** Cosines recomputed (qwen_base narr vs qwen_it chat L24 0.810; OLMo 0.594).
- **Q4.2b.** R, C, K and their CIs recomputed for four models; they match `register.json`. The rule labels are applied correctly given
  the numbers.
- **Q4.4.** Judge inputs carry no labels, and the conclusions are robust to item clustering.
- **Q4b Arm A.** The overall judge effects match `decision.json`. The design-validity control was applied as pre-registered.
- **M0.** d′ and split-half are computed with the fixed A/B halves (`pm_basis.py:56–64`). The Chen extraction and held-out question sets
  are disjoint. The G-trait judge sees question + answer only.

# Q6b: Context-perturbation basin test

**RL deepens the Assistant basin relative to DPO; no other OLMo stage changes it detectably, and "each stage deepens" is not supported.** The test puts 48 tokens of the model's own off-persona text into the start of the default Assistant's reply, then lets the model continue. It measures how much of the resulting Assistant-Axis displacement is still there 100–149 tokens later (f; lower = faster return = deeper basin).

- **Positive control passes in all 7 pairs at both layers.** The off-persona prefix moves the axis by 5–43 units at the start of the continuation. A default-voice prefix from another question (the null) moves it by 2–6% of that.
- **OLMo, retained fraction f at L\*+4 (primary).** The ordering is not monotone. DPO makes the basin shallower than SFT (not significantly), and RL brings it back to the SFT level.

| stage | f [95% CI] |
|---|---|
| base | 0.56 [0.38, 0.69] |
| SFT | 0.64 [0.48, 0.83] |
| DPO | 0.77 [0.68, 0.88] |
| RLVR | 0.63 [0.55, 0.72] |

- **Pre-registered stage verdicts (Δf at L\*+4; L\* agrees in every case):**

| step | Δf [95% CI] | verdict |
|---|---|---|
| OLMo base → SFT (format-confounded) | +0.07 [−0.20, +0.38] | no detectable change |
| OLMo SFT → DPO | +0.13 [−0.07, +0.30] | no detectable change |
| **OLMo DPO → RL** | **−0.14 [−0.25, −0.04]** | **deepens** (L\*: −0.13 [−0.27, −0.00]) |
| Qwen base → instruct, chat (format-confounded) | −0.15 [−0.26, −0.04] | deepens |
| Qwen base → instruct, narrative (format-matched) | −0.15 [−0.30, +0.01] | no detectable change (same sign) |

- **Qwen.** Base f 0.84 [0.75, 0.94] vs instruct 0.70 (chat) and 0.70 (narrative). By the Q6 amendment's confound rule, the chat "deepens" counts as a training effect because the format-matched contrast agrees in sign. That contrast is the same size (−0.15), but its CI just includes 0.

## Reading
- **Bearing on the "deeper basin" clause** (pre-registered wording, Q6 amendment): the DPO→RL step deepens. That supports the clause at 7B for the RL stage specifically, with a known recipe, which is the controlled delta the Qwen releases cannot isolate.
- **The clause does not hold stage by stage.** SFT and DPO do not deepen; DPO's point estimate goes the other way. The net base→RL difference is +0.07 (not a pre-registered contrast, so untested). The base model, in narrative format, returns at least as fast as RL.
- **Bearing on the SPAR terraforming hypothesis:** this is partial support. Post-training changes how hard the Assistant pulls a trajectory back, but not monotonically. The clearest strengthening is from RL, after DPO had weakened it.
- **Early return.** Over the first 20–40 tokens, the OLMo base model recovers about 40% of the displacement (f_early 0.62). Every post-trained stage keeps about 90%, so the base model's fast early return is the larger difference. That comparison is format-confounded: the base model runs in narrative format, the stages in chat. It is not a pre-registered contrast.

## Design
- Pre-registration: `docs/PREREG.md` "Q6b", written before any Q6b data.
- **Prompt:** the default prompt for each of 10 half-B questions (M0 token ids), plus a 48-token prefix of the model's own text, plus a 160-token free continuation (4 seeds shared across conditions; EOS suppressed until 150 tokens).
- **Prefixes:**
  - role: its own response under each of its 5 lowest-AA roles;
  - default: its own default response;
  - null: its own default response to another question.
  - Role prefixes under 48 tokens are dropped: 28–29 (role, question) pairs in the two base models, none in the post-trained ones.
- **Readout:** AA refitted on **half-A** M0 responses only, which fixes the fit-on-eval issue from the red-team review of Q6. The readout runs at L\* and L\*+4, above the layers the perturbation enters.
- **Statistics:** d(t) = projection with the role prefix − projection with the default prefix, paired by question and seed. D0 = mean over t 0–9; f = mean over t 100–149 ÷ D0. 2,000 question-cluster bootstrap draws.
- **Code:** `src/pod/q6b.py`, `src/pod/run_q6b.sh`, `src/q6b_analyse.py`.
- **Results:** `summary.json`; raw per-token projections in `raw/` (local). Figure: `figures/q6b_return.png`.

## Limits
- Only 10 questions per pair, so the CIs are wide. DPO→RL is the only OLMo step whose CI excludes 0, at both layers.
- The off-persona context is 48 tokens of the model's own role-conditioned text, not a full role.
- f is a ratio. D0 is similar across the OLMo stages (−5.5 to −6.5 at L\*), so the ratios are comparable within OLMo.

## Spend
- One H200 pod (um2u4eu13edqi0), about 25 minutes from creation to deletion (≈ $2.2). The two stopped pods could not restart, so the run used a fresh pod. Both that pod and the H200 pod 0334j1xh2w72cz were deleted afterwards.
- Judges: $0.

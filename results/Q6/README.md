# Q6 — Does each post-training stage deepen the Assistant basin?

**No detectable deepening at any stage, but the design cannot resolve basin depth.** A role vector injected for 5 response tokens moves the Assistant-Axis coordinate strongly in every model, so the positive control passes. Within 10 tokens of the window ending, though, only 0–4% of that displacement remains. Nothing beyond the nulls remains 50 tokens later (f50 ≈ 0 everywhere).

- **τ cannot be fitted.** The exponential fits sit at their bounds, with CIs spanning [0.5, 1000] tokens.
- **Every pre-registered stage step returns "no detectable change"** on the retained fraction f50: base→SFT, SFT→DPO, DPO→RL, Qwen base→instruct chat, and the format-matched Qwen base→instruct narrative.
- **The prediction (instruct returns faster) is not supported.** No model has a slow return to compare against.

**Why the design cannot answer the question.** The injected vector and the readout sit at the same layer, L\*. While the vector is added, the readout includes it directly. Once it stops, the only remaining route is attention from later tokens to the perturbed tokens' cached states, and that route turns out to be weak in all models. "Snaps back within a token" is therefore mostly a statement about that route, not about an attractor. A basin test would need a perturbation that changes what the model *generates*: steer long enough to put role-like text into the context, then stop, and track the coordinate as the model reads its own off-persona text. Q6 does not do that.

Pre-registration: `docs/PREREG.md` Q6, "Q6 Implementation" and "Q6 amendment (stage framing)", all written before any Q6 data. Deviation: DECISIONS #28 (runs lost in the pod volume hang, re-run unchanged). Code: `src/q6_build.py`, `src/pod/q6_attractor.py`, `src/q6_analyse.py`. Results: `summary.json` and `trajectories_<pair>.jsonl.gz`; the raw npz files are local only.

## Setup
- **Prompts:** 80 per pair (default Assistant × 10 half-B questions × 8 samples). Sampling uses shared random numbers across conditions, so the conditions differ only by the perturbation.
- **Perturbation:** the role vector (role mean − default mean at L\*) for the 5 roles with the lowest AA z, at its own norm. It is added at L\* on decode steps for response tokens 20–24 only.
- **Nulls:** a random direction and a covariance-matched direction of equal norm, plus 3× versions of the role and random vectors.
- **Readout:** the AA projection at L\* per token for t = 0–224, as the paired difference from the no-perturbation run.
- **Leak check:** max |Δ| before the window is 0.0 in every pair, so the hook does not touch the prompt or the early tokens.

## Results (AA units; role vector at 1×, 3× in the last two columns)

W = mean displacement during the window. K = mean displacement over the 10 tokens right after the window. f50 = displacement at t = 70–79 as a fraction of W. 95% CIs come from a question-cluster bootstrap.

| pair | W (role) | K (role) [CI] | K / W | f50 [CI] | W (3×) | K / W (3×) |
|---|---|---|---|---|---|---|
| olmo_base_narrative | -9.2 | -0.28 [-0.42, -0.16] | 3.0% | 0.018 [-0.013, 0.053] | -28.5 | 8.3% |
| olmo_sft_chat | -6.6 | -0.05 [-0.14, 0.05] | 0.7% | 0.001 [-0.016, 0.018] | -20.2 | 3.0% |
| olmo_dpo_chat | -7.4 | -0.09 [-0.21, 0.01] | 1.2% | -0.005 [-0.043, 0.033] | -22.8 | 3.8% |
| olmo_rl_chat | -7.4 | 0.01 [-0.05, 0.06] | 0.1% | 0.009 [-0.017, 0.033] | -22.8 | 4.1% |
| olmo_rl_narrative | -10.4 | -0.06 [-0.23, 0.10] | 0.5% | 0.013 [-0.006, 0.034] | -33.3 | 10.7% |
| qwen_base_narrative | -25.3 | -0.66 [-1.08, -0.27] | 2.6% | 0.013 [-0.001, 0.032] | -79.7 | 11.3% |
| qwen_it_narrative | -27.5 | -1.01 [-1.43, -0.64] | 3.7% | 0.032 [0.018, 0.045] | -84.5 | 7.3% |
| qwen_it_chat | -30.5 | -0.86 [-1.55, -0.26] | 2.8% | 0.004 [-0.011, 0.017] | -92.7 | 4.1% |

- **Positive control passes in all 8 pairs.** W for the role vector beats the random and covariance-matched nulls by an order of magnitude or more. The random null's W is about 0.
- **Carry-over (K beats the nulls) passes only for olmo_base, qwen_base and qwen_it narrative.** Even there K is under 4% of W. At 3× it passes everywhere, at 3–11% of W.
- **Stage contrasts.** Δf50 per step, 95% CI:

| step | Δf50 | verdict |
|---|---|---|
| OLMo base → SFT (format-confounded) | −0.016 [−0.059, +0.025] | no detectable change |
| OLMo SFT → DPO | −0.006 [−0.052, +0.035] | no detectable change |
| OLMo DPO → RL | +0.013 [−0.046, +0.072] | no detectable change |
| Qwen base → instruct, chat (format-confounded) | −0.010 [−0.033, +0.011] | no detectable change |
| Qwen base → instruct, narrative (format-matched) | +0.018 [−0.006, +0.041] | no detectable change |

- **Descriptive only, not pre-registered:** at 3× the retained share right after the window is lower in the post-trained chat models (OLMo SFT, DPO, RL 3.0–4.1%; Qwen instruct chat 4.1%) than in the base models (8.3% and 11.3%). That would be the direction of a deeper basin. But the format-matched comparisons disagree. Qwen instruct in narrative format is 7.3% against base 11.3%, while OLMo RL in narrative format is 10.7% against base 8.3%. So the gap tracks the chat format at least as much as training.

## Bearing on the hypotheses
- **SPAR terraforming and the thesis's "deeper basin" clause:** neither is supported or refuted by Q6. With this perturbation every model returns within about one token, so a deeper basin has nothing to show up in. The generation-level perturbation described above is the test that could tell them apart.

## Figures
`figures/q6_trajectories.png` (1×), `figures/q6_trajectories_3x.png`, `figures/q6_control.png`. `figures/q6_tau.png` shows the τ fits; most of them sit at their bounds, so read it as "not identifiable".

## Spend
About 100 GPU-min on the H200 (≈ $9) plus about 5 min of smoke tests. Judges: $0.

# Q5 — Persona vectors vs the Assistant Axis (`qwen_it` chat)

**Outcome: not run. The final pre-registered steering gate (DECISIONS #29) fails for all three traits, so no trait vector is a validated steering vector, and the Q5 mediation rule cannot be applied (DECISIONS #30).**

## Question and pre-registered rule
Are Chen-style trait vectors separate features, or do their steering effects run through the AA? (PREREG Q5.) Rule: a trait is *mediated* if steering with v ⊥ AA removes ≥ 50% of the full-v effect beyond the nulls, or if ablating the AA at L\*..L\*+8 blocks ≥ 50% of it. It is *independent* if both are ≤ 25%, and *partly mediated* otherwise. Q5 runs only on traits that pass G-trait (PREREG M0 step 7).

## Gate (step 1): add c·v at L\* = 24 only, c ∈ {2, 3, 4}
Pass = trait score ≥ unsteered + 30 at mean coherence ≥ 80 (Haiku 5.5, Chen trait and coherence prompts). n = 20 Chen held-out questions × 1 sample. 95% bootstrap CIs over questions, 2,000 resamples. The unsteered baseline is the 'none' rows of `results/M0/gate_gen_qwen_it.jsonl`.

| trait | unsteered trait (coh) | c = 2: trait [CI] / coh [CI] | c = 3 | c = 4 | gate |
|---|---|---|---|---|---|
| sycophancy | 23 [17, 30] (88) | **76** [68, 84] / 52 [41, 64] | 79 [67, 89] / 3 [3, 4] | 25 [9, 45] / 1 | fail |
| hallucination | 26 [11, 44] (84) | 55 [36, 73] / 60 [51, 67] | **89** [81, 94] / 11 [6, 16] | 92 / 3 | fail |
| impoliteness | 1 [1, 2] (90) | **57** [38, 76] / 39 [23, 56] | 91 [81, 97] / 3 [2, 4] | 95 / 2 | fail |

Bold marks the cells that clear the +30 trait bar. In all of them, coherence is below 80.

Together with M0 (`results/M0/gate_qwen_it.json`, `gate2_qwen_it.json`), the single-layer dose-response is now mapped from c = 0.25 to 4 (`figures/q5_gate.png`):
- **c ≤ 1:** coherence stays ≥ 80, but no trait gains more than +7.
- **c = 2:** sycophancy and impoliteness gain ≥ +30, but coherence is already 39–60.
- **c ≥ 3:** the text is degenerate (coherence ≤ 11).

No pre-registered c sits in a window with both effects. There may be a narrow window between c = 1 and c = 2. Testing it would need a new grid, which #29 rules out.

Observation (descriptive, not used for any decision): at c = 2, many sycophancy and hallucination replies read as fluent (see `gate3_gen_qwen_it.jsonl`). Their Haiku coherence scores cluster at 62 and 72, not near 0. The Chen coherence prompt may partly penalise the trait expression itself, so "coherence < 80" at c = 2 is not the same as degenerate text. That would be a reason to revisit the gate's coherence criterion in a future pre-registration, not to reinterpret this one.

## Step 2 (decomposition, conditions (a)–(e), nulls, AA readout)
Not run, because no trait passed. The "Q5 Implementation" sub-section of PREREG was not written: it would pre-register a run that cannot happen (DECISIONS #30). The following were not computed:
- shares of ‖v‖² along the AA and in P15;
- mediation fractions;
- positive control;
- random and covariance-matched nulls.

The gate itself is the positive-control stage for Q5, and it failed.

## Files
- `gate3_qwen_it.json`: gate summary (per c: trait, CI, delta, coherence, CI, n, non-numeric judge replies).
- `gate3_gen_qwen_it.jsonl`: 180 generations, with prompt, condition and text.
- `gate3_scored_qwen_it.jsonl`: per-row trait and coherence scores, including the 60 unsteered baseline rows.
- Code:
  - `src/q5_gate3.py` (`jobs`, `score`): data in `data/Q5/gate3_jobs_qwen_it.jsonl` and `gate3_conditions_qwen_it.json`.
  - Generation: `src/pod/pm_gen_hf.py`, unchanged (sampling T = 0.6, top-p 0.95, top-k 20, max 256 new tokens, seed 20261028; vectors `traits_qwen_it.npz`, md5 matches the pod copy).
  - Figure: `src/q5_gate_fig.py` produces `figures/q5_gate.png`.

## Spend
- **Judges:** $0.028 (360 Haiku calls; 120 of them cached).
- **Pod:** about 4 GPU-minutes on the H200 (≈ $0.35). This covers a 1.5-min generation plus the model load, and a first launch killed during load (DECISIONS #30).
- **Project total after Q5:** $21.6 (`src/spend.py`).

## Deviations
DECISIONS #30 (gate outcome; Implementation sub-section not written; the relaunched job).

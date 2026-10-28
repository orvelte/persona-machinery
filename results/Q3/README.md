# Q3: What reads the Assistant-Axis coordinate, and what depends on it? (`qwen_it` chat, L\* = 24)

**Answer: narrow. Removing the AA coordinate at layers 24–32 on every generated token moves 1 of 5 behaviours (style), so the
pre-registered rule gives *narrow* (≤ 1 mover).** Refusal, sycophancy, self-description and false-premise honesty are unchanged, and
none of them moves beyond the random or covariance-matched nulls. The format-following control stays at 97% of baseline, so the ablation
is not destructive. The one mover, style, is fragile. It rests on 10 prompts, and clamping to the median role's coordinate does not
reproduce it (+7 points, CI −22 to +39), even though that clamp is nearly the same intervention (it sets the coordinate to +0.98
instead of 0, from a default of about +12.9). **Readers were not run:** under the pre-registered condition they are traced only for
movers with a continuation metric (refusal, sycophancy, honesty), and none of those moved.

Taken together with Q1's failed positive control, in Qwen3.5-9B the AA coordinate at the response positions is a *readout* of the
persona, not a lever on behaviour. Setting it to the role-average value, alone (Q1, layer 24) or across all nine late layers (Q3),
leaves what the model says, and how it describes itself, intact.

## Panel and conditions (PREREG Q3 + Implementation; DECISIONS 22–24)
- Sets: refusal 40 (JailbreakBench JBB-Behaviors harmful, 4 per category, seeded; DECISIONS 22); sycophancy 40; self-description 20 × 2
  samples; honesty (false premise) 40; format 40 (deterministic checker); style = 10 half-B role-library questions, default Assistant
  (AA fit questions; DECISIONS 23). Prompts: `data/Q3/panel.jsonl`, `data/Q3/refusal.jsonl`, generator `data/Q3/make_panel.py`.
- Blind purity (Haiku, 6-way, before any generation): refusal 0.95, sycophancy 1.00, self-description 1.00, honesty 0.97, format 1.00.
  All pass the ≥ 0.90 bar (`purity.json`).
- 9 conditions × 210 generations = 1,890. Conditions: `none`; `projout`, which sets the AA coordinate to `mu_L`'s at layers 24..32; `clampmed`,
  which clamps to the median role (*bartender*) at 24..32; and three random plus three covariance-matched 1-d projection-outs at the same
  layers. Generated tokens only, temperature 0.6, no think leaks; 72–74% of replies finish within the cap in every condition.
- Effects are condition − none with a paired bootstrap over prompts (2,000). "Moves" = CI excludes 0 and |Δ| > 2 × the larger null family's
  mean |Δ|.

## Results (points: % for binary sets, 0–100 for judged scores, style index × 100 = % of the default → role-average distance)

| behaviour | none | AA projection-out Δ [95% CI] | clamp → median role Δ | null: random / cov (mean \|Δ\|) | moves? |
|---|---|---|---|---|---|
| refusal (% refused) | 97.5 | −2.5 [−7.5, 0.0] | +0.0 [−7.5, 7.5] | 0.0 / 0.0 | no |
| sycophancy (agree, 0–100; Sonnet) | 0.9 | +0.5 [−0.6, 1.8] | +0.9 [−0.6, 3.4] | 0.6 / 0.6 | no |
| self-description (assistant-likeness; Sonnet) | 94.9 | −0.6 [−1.2, 0.2] | −0.6 [−1.3, 0.3] | 0.3 / 0.3 | no |
| honesty (% corrects false premise) | 100.0 | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] | 0.0 / 0.0 | no |
| style index (n = 10) | 0 | **+39.5 [6.3, 77.6]** | +7.0 [−21.7, 39.4] | 9.5 / 13.8 (cov draws 23.9, 7.1, 10.3) | yes |
| *format control (% pass)* | 90.0 | −2.5 [−12.5, 5.0]; ratio 0.97 ≥ 0.90 | −2.5 [−12.5, 5.0] | 1.7 / 1.7 | control OK |

**Rule outcome: narrow** (1 mover of 5). The format control passes (97% of baseline), so the L\*..L\*+4 re-run is not triggered.

Figure: `figures/q3_panel.png`.

**Caveats.**
- Ceiling and floor. At baseline the model refuses 97.5% of JBB requests, corrects 100% of false premises, and barely agrees with stated
  misconceptions (0.9/100). Shifts toward compliance or agreement were possible but did not happen. Shifts in the other direction could
  not be detected.
- The style mover: n = 10, one covariance draw also excludes 0 (+23.9), and the near-identical clamp does not reproduce it. Treat it as
  weak evidence that response formatting (fewer headers, bullets and bold) leans role-ward when AA is removed.
- The nulls are 1-d projection-outs. A random direction carries a far smaller coordinate than AA does in the default Assistant (≈ 12.9
  above the role mean), so the nulls also displace the stream less. That is how the pre-registration defines them, and it is the
  usual reading. Here it does not matter, since AA itself does nothing.

## Readers (path patching)
Not run, by the pre-registered condition (DECISIONS 24): readers are traced only for behaviours that move *and* have a continuation
metric (refusal, sycophancy, honesty). None of the three moved, and style has no such metric. The script is ready
(`src/pod/q3_readers.py`: mixer / MLP / unembed units of blocks 25–32, AA-restricted input patching, early-continuation log-odds),
but there is no effect for it to decompose.

## Judge agreement (Sonnet 5.5 vs Haiku 5.5, 60 seeded items per prompt; `judge_agreement.json`)
- refusal: κ = 1.0 (100% agreement, n = 50 parsed). honesty: κ = 1.0 (100%, n = 60). Both pass. Both sets are almost one class, so κ
  is near-degenerate. Raw agreement is the informative number.
- sycophancy: Spearman 0.30, and self-description: 0.57, both **fail** (< 0.8). So Sonnet re-scored every item of both sets, 720 calls,
  and the table uses the Sonnet scores. Haiku's scores are kept as `agree_haiku` / `likeness_haiku`. Under Haiku the conclusions are
  the same: sycophancy projout +2.1 [−0.6, 6.7] and self-description −0.1 [−1.6, 1.4], neither moving. The low rank agreement reflects
  ceiling and floor: nearly every reply scores 0–5 (sycophancy) or 85–98 (self-description).

## Files
`gen_qwen_it.jsonl` (raw generations), `scored_qwen_it.jsonl` (judge labels, format pass, style features), `summary_qwen_it.json`
(every number above), `purity.json`, `judge_agreement.json`. Code: `src/q3_purity.py`, `src/pod/q3_refusal_set.py`, `src/q3_score.py`,
`src/q3_style.py`, `src/q3_sonnet_rescore.py`, `src/q13_agree.py`, `src/q3_fig.py`, `src/pod/q3_readers.py`; conditions
`data/Q3/conditions.json` and compact basis `data/Q3/q3_basis_qwen_it.npz` (M0 keys for layers 24–32 plus clamp targets).

## OLMo (`olmo_rl`, secondary)
Not run. Primary first; given the null result in Qwen and the shared pod's priority queue, the OLMo panel (PC1 and contrast axes) was
not worth the GPU time. It is the natural next step if someone wants a cross-family check.

## Deviations
DECISIONS 22 (refusal items from JBB), 23 (style on fit questions, as pre-registered, flagged), 24 (continuation metric instead of
first-token; mixer/MLP units; κ for binary agreement).

## Spend
Judges, Q3: $1.24. That is purity $0.005, Haiku scoring $0.10, agreement $0.35, and the Sonnet re-score $0.79. GPU: 10.9 min of
generation on the H200, about $1.0.

# Q2 — What triggers the AA coordinate: chat template vs content

> **Post-review correction (2026-10-28, after the blind red-team review in `results/REDTEAM.md`).** The size claim "every template knockout moves ≤ 0.15 role SDs" holds only on PC1. On OLMo's contrast axis the template knockout moves 0.34–0.35 role SDs and the system-message knockout 0.86 SD, about the size of a role system prompt. So for OLMo, the default system message (not the template delimiters) carries much of the default-Assistant signal on c. The pre-registered rule outcomes are unchanged.

**The pre-registered prediction is not supported in the primary model (`olmo_rl`).** For the same response text, the raw-document (narrative) projection on OLMo's PC1 is the same as the chat projection (gap 0.03 [−0.03, 0.09]). The ratio Δ is therefore undefined, and Δ_template − Δ_user = +2.9 [−7.4, +13.6] includes 0. On OLMo's contrast axis c the result runs the other way: knocking out the template moves the coordinate *away* from the raw-document value, while knocking out OLMo's default system message moves it 3.8× past it.

In the secondary model (`qwen_it`) the prediction holds on PC1, narrowly for attention knockout (+0.13 [+0.003, +0.28]) and robustly for token ablation (+0.42 [+0.24, +0.61]). The knockout result disappears when position 0, the attention sink, is spared. In Qwen most of the template effect comes from the closed think block that sits right before the response.

Every template knockout moves the coordinate by ≤ 0.15 across-role SDs. A role system prompt moves it by 0.7–1.1 SDs. The AA coordinate of a teacher-forced response is mostly carried by the response tokens themselves.

Pre-registration: `docs/PREREG.md` Q2 + "Q2 Implementation". Deviation: DECISIONS #21 (axes refitted from a fresh replay after the pod migration). Code: `src/pod/q2_knock.py`, `src/q2_analyse.py`.

## Setup
- **Data:** the 30 default-Assistant half-B responses from M0 (10 questions × 3 seeds), teacher-forced on their exact token ids. The readout is the mean residual over response tokens at L\* (OLMo 23, Qwen 24).
- **Axes:** half-A PC1 of the 99 role means, and the contrast axis c = half-A default − role mean. Both are fitted on half A only and evaluated on half B. The frozen M0 AA per layer is used only for the layer profile.
- **Knockout (KO):** a 4D additive mask on every softmax layer. Assistant-turn queries (the last `<|im_start|>` onward) cannot attend to the knocked keys.
  - In Qwen's 24 Gated-DeltaNet layers, the knocked positions' k and v are zeroed instead. This is a partial knockout: it applies to every later position, and the width-4 conv still leaks.
- **Token ablation (TA):** the knocked tokens' input embeddings are replaced by the mean embedding.
- **Token sets:**
  - template: OLMo 18 tokens, Qwen 12;
  - (a′) template sparing position 0;
  - each template class separately (role names, delimiters, think tags);
  - position 0 alone;
  - (b) system content (OLMo only, 59 tokens);
  - (c) user content (~14 tokens);
  - 3 random draws of content tokens of the template count.
- **Raw-document reference** (`raw_same`): the M0 narrative prompt for the same question, followed by the same chat response text. It is projected on the chat axes at the chat L\*.
- **Statistics:** Δ_X = mean(none − X) / mean(none − raw_same). CIs are a cluster bootstrap over the 10 questions (2,000 resamples).

## Mechanics checks (all pass; `*/checks.json`)
| check | olmo_rl | qwen_it |
|---|---|---|
| all-allowed custom mask vs standard forward: argmax agreement, max \|Δlogit\| | 100%, 0.0 | 100%, 0.0 |
| eager attention under KO (a): max weight from assistant queries to knocked keys | 0.0 in 32/32 layers | 0.0 in 8/8 softmax layers |
| linear-attention kill fired, k and v = 0 at knocked positions | – | 24/24 layers |
| replayed M0 library vs stored basis (default mean; min over 99 role means), cos | 1.0000; 1.0000 | 1.0000; 1.0000 |
| eager vs sdpa hidden at L\* under KO (max relative diff, bf16 drift) | 0.054 | 0.043 |

## Rule (PC1, the pre-registered axis) and the contrast axis
Shifts below are in raw projection units. The across-role SD of half-A PC1 positions is 2.04 (OLMo) and 9.80 (Qwen).

| model, axis | none − raw_same | KO(a) shift | KO(c) shift | KO(b) shift | Δ_template − Δ_user, KO | same, KO sink spared | same, TA |
|---|---|---|---|---|---|---|---|
| **olmo_rl PC1** | +0.03 [−0.03, +0.09] | −0.16 [−0.21, −0.12] | −0.07 [−0.16, 0.00] | +0.14 [+0.09, +0.19] | **+2.9 [−7.4, +13.6]** (Δ undefined) | +3.2 [−11.1, +18.6] | +2.7 [−10.1, +11.9] |
| olmo_rl c | +0.23 [+0.08, +0.36] | **+0.35 [+0.17, +0.55]** (away from raw) | −0.18 [−0.30, −0.09] | **−0.88 [−0.98, −0.79]** | −2.3 [−9.2, −0.9] | −2.3 [−8.3, −0.9] | −0.03 [−1.1, +0.4] |
| **qwen_it PC1** | +1.30 [+1.00, +1.63] | −0.86 [−1.01, −0.71] (Δ 0.66 [0.53, 0.83]) | −0.69 [−0.81, −0.58] (Δ 0.53 [0.40, 0.70]) | – | **+0.13 [+0.003, +0.28]** | −0.00 [−0.10, +0.09] | +0.42 [+0.24, +0.61] |
| qwen_it c | +2.78 [+2.10, +3.48] | −2.78 (Δ 1.00 [0.83, 1.24]) | −1.07 (Δ 0.38 [0.32, 0.47]) | – | +0.62 [+0.48, +0.82] | +0.14 [+0.05, +0.26] | +0.48 [+0.35, +0.66] |

The OLMo PC1 and c results disagree, which DECISIONS #13 requires us to flag:
- **PC1:** both the template and the user knockouts lower the coordinate slightly, and the raw value is no different from chat.
- **c:** the default system message carries the Assistant signal. Removing it from view (KO) or from the input (TA) drops c by 0.88. OLMo's system message reads "You are Olmo, a helpful … AI assistant".
- **Template on c:** removing the template *raises* c. The think tags alone raise it by +0.66.

**Which template tokens matter** (Δ on PC1; KO / TA):

| tokens | olmo_rl | qwen_it |
|---|---|---|
| role names | undefined (OLMo Δ has no meaningful denominator; shift +0.04 / −0.09) | 0.00 / 0.21 |
| delimiters, all | shift −0.10 / −0.11 | 0.13 / 0.54 |
| delimiters, sink spared | shift −0.11 / −0.12 | 0.00 / 0.57 |
| think tags | shift −0.16 / −0.17 | 0.62 / 1.04 |
| position 0 alone | shift +0.01 / −0.00 | 0.04 / 0.16 (on c: 0.31) |

In Qwen the closed think block (`<think>\n\n</think>\n\n`) does nearly all the work. Its knockout alone matches the full template knockout. Role-name tokens do nothing. Two caveats apply: these tokens sit directly before the response, so the effect is confounded with locality, and Qwen's knockout is partial.

**Random control.** In Qwen, KO of random content tokens (12 of ~14, i.e. most of the user text) gives Δ 0.33–0.40. That is below the template's 0.66: template − random = +0.29 [+0.15, +0.46]. In OLMo the draws come from the 73 system + user tokens. Their PC1 shifts are ≈ 0 (+0.02 to +0.03), and on c they match KO(c) (Δ ≈ 0.8–1.0).

**Positive control (passes).** For the 3 lowest-PC1 roles (OLMo: eldritch, pirate, witch; Qwen: eldritch, fool, infant), we gave the role text as the system prompt and teacher-forced the same default responses. The role prompt lowers PC1:
- OLMo: by 1.1–2.2 (0.5–1.1 SD);
- Qwen: by 1.5–2.7.

KO of the role system prompt then moves the coordinate back:
- OLMo PC1: 98–105% of the way;
- OLMo c: 29–49%;
- Qwen: 88–93%.

All of these CIs exclude 0. The knockout can move the coordinate, and the role prompt's effect on it runs mostly through direct attention.

## Behaviour (30 generations per condition, knockout applied at every step; `*/gen_judged.jsonl`)
| condition | olmo assistant-likeness | olmo coherence | qwen assistant-likeness | qwen coherence |
|---|---|---|---|---|
| none | 90.7 [89.2, 92.2] | 90.9 | 89.5 [87.4, 91.0] | 92.3 |
| KO (a) template | 66.8 (−23.9 [−41.5, −6.6]) | 43.2 | 73.9 (−15.6 [−29.8, −2.3]) | 37.2 |
| KO (a′) template, sink spared | 72.0 (−18.7 [−34.2, −5.3]) | 56.0 | 82.1 (−7.4 [−14.0, −2.4]) | 84.4 |
| KO (b) system | 91.5 (+0.9 [−0.3, +2.0]) | 92.1 | – | – |
| KO (c) user | 89.9 (−0.6 [−2.2, +0.9]) | 43.8 | 82.9 (−6.2 [−10.0, −3.1]) | 14.2 |
| KO random | 88.3 (−2.4 [−3.8, −1.0]) | 91.8 | 83.4 (−6.1 [−12.4, −0.7]) | 49.6 |

Judge: Haiku 5.5, question + reply only, no condition label.

- **Template knockout breaks generation.** OLMo with the sink knocked out produces "ThatThatThat…". Qwen echoes the template (the question, then `<think>` tags). The drop in assistant-likeness comes with a collapse in coherence, so it is not a clean persona switch.
- **Without the user content, both models stay assistants.** They answer some other question, or ask "it seems your message is incomplete". Assistant-likeness barely moves, while coherence (which includes relevance) collapses.
- **Removing OLMo's system message changes nothing behaviourally,** even though c drops by 0.88.

Style z-scores (SD across the 99 role means; `summary_behaviour.json`) follow the same pattern. Template KO in OLMo removes headers, bullets and bold (−3 to −5 SD). User KO in Qwen adds hedges and questions (+17–19 SD).

## Reading
- The coordinate does not key on the template in OLMo-3-7B-Think. With the same content, the raw document sits where the chat sits on PC1. The default system message, which names the model a helpful AI assistant, is the prompt component that carries OLMo's default-vs-role contrast.
- In Qwen3.5-9B, removing the template moves the coordinate the full chat → raw distance on c and two-thirds of it on PC1. This mostly comes from the empty think block, plus the attention sink.
- **Size:** all of these shifts are ≤ 0.15 role SDs, small next to what a role prompt does. Teacher-forcing the same Assistant response keeps the coordinate Assistant-like whatever the prompt shows.
- This bears on the M5 claim that "the template is what made gaming possible". What separates chat from raw for the *same* content is small on the AA. The chat-vs-raw gaming difference in M5 is more likely carried by what the model generates in each frame than by a template-gated AA value.

## Figures
- `figures/q2_delta.png`: shift per knocked token set, KO vs TA, PC1 and c, with the raw-document value as a band.
- `figures/q2_layers.png`: per-layer projection on the frozen AA.
- `figures/q2_behaviour.png`: assistant-likeness and coherence per condition.

## Files
`<model>/tf_rows.jsonl` + `tf.npz` (per-row mean residual at L\* for every condition; projections on every layer's frozen AA), `<model>/axes.npz`, `<model>/checks.json`, `<model>/gen.jsonl`, `<model>/gen_judged.jsonl`, `summary_tf.json`, `summary_behaviour.json`, `style_role_sd.json`.

## Spend
- Judges: $0.05 (600 Haiku calls).
- Pod: two HF jobs of ≈ 7 and ≈ 8 min on the H200, ≈ $1.3.

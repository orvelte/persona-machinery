#!/usr/bin/env python3
"""Build the research-log page (Artifact) from repo state: plan, module status, results, figures, spend.

    .venv/bin/python src/build_page.py OUT.html

Content that changes as results land lives in MODULES below (status, verdict, numbers) and in results/*/README.md;
figures are embedded as data URIs from figures/*.png. Spend comes from src/spend.py.
"""
from __future__ import annotations

import base64
import html
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# status: done | running | queued | blocked | deferred
MODULES = [
    {"id": "M0", "name": "Shared setup", "status": "done",
     "question": "Extract the Assistant Axis (PC1 of 99 role means, default oriented positive), its stability, P5/P15, and Chen-style trait vectors; gate on stability ≥ 0.9 and on trait steering at coherence ≥ 80.",
     "verdict": "Axis stable in 7 of 8 model × format pairs. Trait steering gate failed for every trait.",
     "points": [
         "Qwen3.5-9B's axis matches the 27B Assistant Axis from earlier work: Spearman ρ of the 99-role ordering is 0.89 (chat) and 0.92 (narrative).",
         "The axis already exists in the Qwen base model under narrative framing (ρ 0.89 with the 27B axis; cosine 0.75 with the instruct axis).",
         "In OLMo-3-7B chat, PC1 runs from mystical roles to evaluator roles and only weakly tracks the default Assistant (cos 0.16–0.49), so a default-minus-roles contrast axis is reported alongside it.",
         "Evil was dropped because both models refuse the evil prompts; impoliteness was substituted. No trait passed the steering gate: 5 layers at c = 1 destroys coherence, and 1 layer at c ≤ 1 barely moves the trait.",
     ],
     "figures": ["m0_profiles.png", "m0_cross.png"]},
    {"id": "Q2", "name": "Triggers: template vs content", "status": "done", "priority": True,
     "question": "Knock out attention from the assistant turn to (a) template tokens, (b) the system prompt, (c) the user content; compare the axis projection with the raw-document value.",
     "verdict": "Prediction not supported in OLMo (primary); holds narrowly in Qwen. The template barely moves the axis coordinate.",
     "points": [
         "OLMo RL: for the same response text, the raw-document projection equals the chat projection (gap +0.03 [−0.03, +0.09]), so there is no template-driven shift to explain. Δ_template − Δ_user = +2.9 [−7.4, +13.6].",
         "Qwen instruct: template knockout moves the coordinate toward the raw-document value more than user-content knockout (+0.13 [+0.003, +0.28]; token ablation +0.42 [+0.24, +0.61]). The empty think block right before the response carries most of it.",
         "Every template knockout moves the coordinate by at most 0.15 across-role SDs, against 0.7–1.1 SDs for a role system prompt. The coordinate of a response is mostly carried by the response tokens themselves.",
         "Positive control passed: knocking out a role system prompt restores 88–105% of the coordinate. Mechanics checks (null mask = identical logits; zero attention to knocked tokens) all passed.",
         "In generation, template knockout breaks the output rather than switching persona.",
     ],
     "figures": ["q2_delta.png", "q2_layers.png", "q2_behaviour.png"]},
    {"id": "Q2b", "name": "Lift the raw-document floor, then vary the judge", "status": "done", "priority": True,
     "question": "Build a raw document that carries the incentive as text; gate on gaming ≥ 30% (15 of 48) in 9B instruct; then compare the automated, human-reviewer and checker judge wordings across 9B base, 9B instruct and 27B 3.6/3.8.",
     "verdict": "Gate failed for all three pre-registered wordings (gaming 5, 7, 9 of 48); stopped as pre-registered. The judge comparison was not run.",
     "points": [
         "Qwen3.5-9B instruct, raw repository document with a manager's note, a 'made CI green' git-log line and grader.py: it writes spec-correct code in about 80% of samples and fits the contradicting test in 10–19%.",
         "All gaming falls on two of six tasks (leap year, round-half). Spec-only accuracy is 36–41 of 48, so the task is solvable.",
         "Because the floor could not be lifted, Q2b cannot say whether the automated-vs-human judge distinction exists before post-training. Q4b answers that with log-odds instead.",
     ],
     "figures": ["q2b_gate.png"]},
    {"id": "Q4b", "name": "Narrative convention vs the Assistant's judge distinction", "status": "done", "priority": True,
     "question": "Arm A: forced-choice log-odds (fit the test vs follow the spec) on 48 story templates crossed by judge × protagonist × stakes. Arm B: the base model simulating an AI-assistant transcript, scored by execution.",
     "verdict": "No evidence of an automated-vs-human convention in base-model story endings, but the test was underpowered and not validated against Qwen3.8, so this null cannot carry weight. The narrative-convention explanation stays open.",
     "points": [
         "Arm A: base models show no automated-vs-human effect in story endings. Qwen base +0.015 nats/token [−0.001, +0.033], OLMo base +0.012 [−0.024, +0.047]; the scheduler-vs-none validity control is ≈ 0 in both.",
         "Post-trained models lean the other way: Qwen instruct −0.038 [−0.060, −0.015] and Qwen3.6-27B −0.071 [−0.120, −0.023], meaning more test-fitting under a human reviewer in stories.",
         "Qwen3.8-27B is near 0 (−0.006 [−0.035, +0.023]) but its validity control fails. The pre-registered ordering 3.8 > 3.6 holds (+0.065 [+0.003, +0.122]), yet the story measure does not reproduce 3.8's +48-point behavioural effect.",
         "Arm B: neither base model games when simulating an Assistant (Qwen 0–1, OLMo 4–5 of 48), even after the one allowed escalation, so the behavioural column is empty.",
         "Decision table: no row fires. The earlier row-4 call rested on a percentile bootstrap over 4 stories; an exact sign-flip test gives p = 0.25, and with 4 stories the smallest attainable p is 0.125.",
     ],
     "figures": ["q4b_armA.png", "q4b_armB.png"]},
    {"id": "Q4", "name": "Pretraining origin of the axis", "status": "done",
     "question": "Base vs instruct axis cosine and CKA; a 200-passage narrative battery in five classes; OLMo stage tracking; three convention-vs-persona tasks.",
     "verdict": "About voice, not narrative character. In pretraining the axis separates the impersonal reference voice (the encyclopaedia voice, nobody in particular) from characters speaking in their own voices. Helper-vs-villain separation is not specific to the axis. This holds for projection on raw passages.",
     "points": [
         "Qwen: the base model's narrative axis is the instruct axis (cos +0.81 with the instruct chat axis at L* 24; +0.88 in matched narrative format; split-half ceiling ≈ 0.93).",
         "OLMo: not the same axis by the 0.7 rule (+0.59 base narrative vs RL chat), mostly a format difference: base vs RL in matched narrative format is +0.91 (descriptive only).",
         "Narrative battery: the prediction fails in reverse. Encyclopaedia and code passages sit at the Assistant end; helpful-character passages sit below the other classes in every model (Qwen base g = −0.73 [−1.05, −0.44], instruct −0.79; OLMo base −0.93, RL −1.19).",
         "Register control (Olivia's 2×2, 40 passages per class): the pre-registered rule fires 'both' in all six models (Qwen base: register +1.9 SD, character +1.2 SD, interaction ≈ 0). After review, these were re-tested against 2,000 random directions.",
         "Only the entry-versus-manual contrast is specific to the axis: +3.83 SD in Qwen base and +3.60 in instruct, above at least 99.7% of both null families. The character effect (79–96th percentile), the expository-vs-story gap and the villain entry − helper manual contrast (45–95th) are not; most directions separate these passage classes by 1–3 SD.",
         "The battery's reversed helpful-class gap does not beat its nulls either. So the earlier 'reversed' reading is withdrawn; 'helpful characters do not sit at the Assistant end' stands.",
         "Exploratory: among narrative and human passages only, Qwen's helpful characters do lean toward the Assistant end, equally in base (+0.52) and instruct (+0.45).",
         "OLMo stages: SFT creates the chat-format axis (later stages cos ≥ 0.99 to RL); the battery ordering is identical at every stage; DPO gives the largest gain in role separation.",
         "Convention-vs-persona tasks: base and post-trained models answer alike in narrative format; the frame matters more than training. The axis-ablation arm is uninterpretable because its positive control failed.",
     ],
     "figures": ["q4_register.png", "q4_axis_cos.png", "q4_battery.png", "q4_stages.png", "q4_tasks.png"]},
    {"id": "Q1", "name": "Writers of the axis coordinate", "status": "done",
     "question": "Attribute the axis projection at L* to every attention head and MLP; is it written by a few components (top 10 explain ≥ 50% of the Assistant − role gap) or diffusely? Check by patching.",
     "verdict": "Intermediate: the top 10 components write 43.5% [42.6, 44.7] of the gap, mostly late MLPs and linear-attention heads. Behavioural positive control failed, so what the writers do for behaviour is unknown.",
     "points": [
         "Qwen3.5-9B instruct, L* = 24. The embedding plus all 697 components reconstructs the residual with 0.009% error.",
         "14 components reach 50% of the default − role gap, 73 reach 80%. Blocks 19–24 write 77%. In aggregate, linear-attention (Gated DeltaNet) heads write 7.0, MLPs 4.1 and softmax-attention heads 1.1 of the 12.25 gap.",
         "Patching confirms the attribution on the coordinate: the top 10 writers patched to their role-context mean close 86% [74, 96] of the gap; the top 3 close 19%; the nulls close 0–2%.",
         "Positive control failed: clamping the coordinate at L* to the role mean removes 105% of the gap but moves self-description by only +0.7 points (20 needed). Patching behaviour readouts are therefore not interpreted.",
         "OLMo RL (attribution only): also intermediate (top-10 share 0.40 on PC1, 0.32 on the contrast axis), with attention dominating.",
     ],
     "figures": ["q1_layer_profile.png", "q1_patching.png"]},
    {"id": "Q3", "name": "Readers and dependent behaviours", "status": "done",
     "question": "Ablate or clamp the axis at L*..L*+8 on generated tokens; score refusal, sycophancy, self-description, style and honesty, with format-following as the control. Then find readers by axis-restricted path patching.",
     "verdict": "Undecided. The pre-registered label is narrow (1 of 5 moved), but the edit was never shown able to move behaviour, so this null does not show the axis is causally inert.",
     "points": [
         "Removing the coordinate at layers 24–32 leaves refusal (−2.5 points [−7.5, 0]), sycophancy, self-description and false-premise honesty unchanged. Format following stays at 97% of baseline, so the intervention is not destructive.",
         "Style moved +39.5% of the default-to-role distance [6, 78] against a null of 14, but on only 10 prompts, and clamping to the median role does not reproduce it (+7 [−22, +39]).",
         "Several baselines sit at ceiling (refusal 97.5%, honesty 100%), so only movement in one direction was detectable.",
         "Readers were not traced: the pre-registration runs path patching only for behaviours with a continuation metric that moved, and none did.",
         "Harmful-request prompts are 40 items drawn from the public JailbreakBench set. Haiku-vs-Sonnet judge agreement failed for sycophancy and self-description, so Sonnet re-scored those as pre-registered.",
     ],
     "figures": ["q3_panel.png"]},
    {"id": "Q6", "name": "Does each stage deepen the Assistant basin?", "status": "done",
     "question": "After a 5-token role perturbation, how fast does the axis projection return, and how much displacement remains? Compared across OLMo base → SFT → DPO → RLVR and Qwen3.5-9B base vs instruct. Tests the terraforming hypothesis and the thesis's 'deeper basin' clause.",
     "verdict": "Design could not test basin depth: a same-layer kick that decays within a few tokens measures neither recovery after off-persona context nor the displacement needed to leave the Assistant region. Every model snaps back within about one token.",
     "points": [
         "Positive control passes in all 8 pairs: during the window the role vector moves the coordinate by 7–30 units, and the random null by about 0. The prompt and pre-window tokens are untouched (max change 0.0).",
         "Within 10 tokens after the window only 0–4% of the displacement remains, and 50 tokens later nothing beyond the nulls (retained fraction ≈ 0 everywhere). Exponential recovery times cannot be fitted.",
         "Every pre-registered stage step (base→SFT, SFT→DPO, DPO→RL, Qwen base→instruct in chat and in matched narrative format) returns the registered label 'no detectable change'; under the basin-depth definition each is 'design could not test'.",
         "Why (corrected after review): injection and readout sit at the same layer, so the cached states the readout depends on are never perturbed. Any carry-over can only come from the 5 tokens the perturbation changed. A real basin test steers until off-persona text fills the context, then reads out above L* as the model returns.",
         "Descriptive only: at 3× strength the post-trained chat models retain less right after the window (3–4%) than the bases (8–11%). But the format-matched comparisons disagree, so this tracks chat format at least as much as training.",
     ],
     "figures": ["q6_trajectories.png", "q6_trajectories_3x.png", "q6_control.png"]},
    {"id": "Q6b", "name": "Basin test with off-persona context", "status": "done",
     "question": "Put 48 tokens of the model's own off-persona text at the start of the default Assistant's reply, let it continue, and measure how much axis displacement remains 100–149 tokens later. Compared across OLMo base → SFT → DPO → RLVR and Qwen base vs instruct.",
     "verdict": "RL deepens the Assistant basin relative to DPO; no other OLMo stage changes it detectably, so 'each stage deepens' is not supported.",
     "points": [
         "Positive control passes in all 7 pairs at both readout layers: the off-persona prefix moves the axis by 5–43 units, and a default-voice prefix from another question moves it by only 2–6% of that.",
         "OLMo retained fraction after about 100 tokens (lower = faster return): base 0.56, SFT 0.64, DPO 0.77, RL 0.63. DPO→RL deepens, Δ −0.14 [−0.25, −0.04], at both L* and L*+4. Base→SFT and SFT→DPO show no detectable change; DPO's trend runs shallower.",
         "Qwen: base 0.84 vs instruct 0.70 in both chat and matched narrative format. The chat step is significant (−0.15 [−0.26, −0.04]); the format-matched step is the same size but its CI just includes 0.",
         "Reading: post-training changes how hard the Assistant pulls a trajectory back, but not monotonically. The clearest strengthening is from RL, after DPO had weakened it. Only 10 questions per pair, so the CIs are wide.",
         "The readout axis was refit on half-A questions only, and read at L* and L*+4, which addresses the red team's Q6 points. About 25 pod-minutes in total.",
     ],
     "figures": ["q6b_return.png"]},
    {"id": "Q5", "name": "Persona vectors vs the axis", "status": "blocked",
     "question": "Does orthogonalising a trait vector to the axis remove its steering effect?",
     "verdict": "Not run: no trait vector steers its trait at usable coherence. The final pre-registered grid (1 layer, c = 2–4) failed for all three traits.",
     "points": [
         "Qwen3.5-9B instruct, layer 24 only, 20 held-out questions. At c ≤ 1 coherence holds (≥ 80) but no trait gains more than 7 points; at c = 2 sycophancy rises 23 → 76 and impoliteness 1 → 57, but coherence falls to 39–60; at c ≥ 3 the text is degenerate.",
         "A usable window may lie between c = 1 and c = 2; the pre-registration allowed no further grid. At c = 2 many replies read as fluent while scoring 62–72 on coherence, so Chen's coherence prompt may partly penalise the trait itself — a question for a future pre-registration.",
     ],
     "figures": ["q5_gate.png"]},
]

STATUS_LABEL = {"done": "Done", "running": "Running", "queued": "Queued", "blocked": "Blocked", "deferred": "Deferred"}


def spend():
    out = subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "src/spend.py")], capture_output=True, text=True).stdout.strip()
    parts = dict(p.strip().split(" $", 1) for p in out.split("|")[:3])
    return {k: float(v) for k, v in parts.items()}


def img(name):
    p = REPO / "figures" / name
    if not p.exists():
        return ""
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{html.escape(name)}" loading="lazy"><figcaption>{html.escape(name)}</figcaption></figure>'


def esc(s):
    return html.escape(s, quote=False)


def summary_table():
    md = (REPO / "results/SUMMARY.md").read_text().splitlines()
    rows = [l for l in md if l.startswith("|") and not l.startswith("|---")]
    if not rows:
        return ""
    import re
    def cell(t):
        t = html.escape(t.strip(), quote=False)
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t).replace("\\*", "*")
    head = [c for c in rows[0].strip("|").split("|")]
    body = "".join("<tr>" + "".join(f"<td>{cell(c)}</td>" for c in r.strip("|").split("|")) + "</tr>" for r in rows[1:])
    return '<div class="tablebox"><table class="summary"><thead><tr>' + "".join(f"<th>{cell(h)}</th>" for h in head) + f"</tr></thead><tbody>{body}</tbody></table></div>"


def main(out_path):
    summary_html = summary_table()
    sp = spend(); now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = sp["total"]; pct = min(100, total / 90 * 100); stop_pct = 70 / 90 * 100
    rows = "".join(f'<tr><td class="mid">{m["id"]}</td><td>{esc(m["name"])}{" <span class=\"prio\">priority</span>" if m.get("priority") else ""}</td>'
                   f'<td><span class="pill s-{m["status"]}">{STATUS_LABEL[m["status"]]}</span></td><td>{esc(m["verdict"]) or "–"}</td></tr>' for m in MODULES)
    sections = []
    for m in MODULES:
        pts = "".join(f"<li>{esc(p)}</li>" for p in m["points"])
        figs = "".join(img(f) for f in m["figures"])
        sections.append(f'''<section class="mod" id="{m["id"].lower()}">
  <header><span class="mid">{m["id"]}</span><h3>{esc(m["name"])}</h3><span class="pill s-{m["status"]}">{STATUS_LABEL[m["status"]]}</span></header>
  <p class="q">{esc(m["question"])}</p>
  {f'<p class="verdict">{esc(m["verdict"])}</p>' if m["verdict"] else '<p class="pending">No results yet.</p>'}
  {f"<ul>{pts}</ul>" if pts else ""}
  {f'<div class="figs">{figs}</div>' if figs else ""}
</section>''')
    page = f'''<title>Persona Machinery</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column of lab-notebook sections over a pale sky ground; white cards only for figures and the status table. Single light look by request. */
:root {{
  --sky: #DCEBF7; --sky-deep: #C6DDF0; --paper: #FFFFFF; --ink: #0B0F14; --muted: #3A4652; --rule: #A9C4DB;
  --accent: #1F5FA8; --ok: #2E7D4F; --run: #1F5FA8; --wait: #8A6A12; --bad: #A33A2B; --off: #5B6773;
  --display: "Source Serif 4", "Liberation Serif", Georgia, serif; --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
  color-scheme: light;
}}
body {{ background: var(--sky); color: var(--ink); font-family: var(--body); font-size: 15.5px; line-height: 1.55; }}
.wrap {{ max-width: 980px; margin: 0 auto; padding-block: 36px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 36px; }}
h1, h2, h3 {{ font-family: var(--display); text-wrap: balance; margin: 0; color: var(--ink); }}
h1 {{ font-size: clamp(2rem, 4.5vw, 2.8rem); font-weight: 700; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.45rem; font-weight: 600; padding-bottom: 6px; border-bottom: 1px solid var(--rule); }}
h3 {{ font-size: 1.2rem; font-weight: 600; }}
p, li {{ max-width: 68ch; }}
.lede {{ display: grid; gap: 12px; }}
.lede p {{ margin: 0; font-size: 1.05rem; }}
.meta {{ font-family: var(--mono); font-size: 0.8rem; color: var(--muted); letter-spacing: 0.02em; }}
.block {{ display: grid; gap: 14px; }}
.block ul, .mod ul {{ margin: 0; padding-left: 1.2em; display: grid; gap: 6px; }}
.tablebox {{ overflow-x: auto; background: var(--paper); border: 1px solid var(--rule); border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
th, td {{ text-align: left; padding: 9px 12px; border-bottom: 1px solid #E3ECF4; vertical-align: top; }}
th {{ font-family: var(--mono); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 500; }}
tr:last-child td {{ border-bottom: 0; }}
.mid {{ font-family: var(--mono); font-weight: 500; color: var(--accent); white-space: nowrap; }}
.pill {{ display: inline-block; font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.04em; padding: 2px 8px; border-radius: 999px; border: 1px solid currentColor; white-space: nowrap; background: var(--paper); }}
.s-done {{ color: var(--ok); }} .s-running {{ color: var(--run); }} .s-queued {{ color: var(--wait); }} .s-blocked {{ color: var(--bad); }} .s-deferred {{ color: var(--off); }}
.prio {{ font-family: var(--mono); font-size: 0.68rem; color: var(--accent); margin-left: 6px; letter-spacing: 0.06em; text-transform: uppercase; }}
.budget {{ display: grid; gap: 10px; }}
.bar {{ position: relative; height: 14px; background: var(--paper); border: 1px solid var(--rule); border-radius: 7px; overflow: hidden; }}
.bar .fill {{ position: absolute; inset: 0 auto 0 0; background: var(--accent); }}
.bar .stop {{ position: absolute; top: -2px; bottom: -2px; width: 2px; background: var(--bad); }}
.nums {{ display: flex; flex-wrap: wrap; gap: 8px 28px; font-variant-numeric: tabular-nums; }}
.nums b {{ font-family: var(--display); font-size: 1.3rem; }}
.mod {{ display: grid; gap: 10px; padding-top: 4px; }}
.mod header {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; }}
.mod .q {{ margin: 0; color: var(--muted); }}
.mod .verdict {{ margin: 0; font-weight: 600; }}
.mod .pending {{ margin: 0; color: var(--off); font-style: italic; }}
.figs {{ display: grid; gap: 16px; }}
figure {{ margin: 0; background: var(--paper); border: 1px solid var(--rule); border-radius: 6px; padding: 10px; display: grid; gap: 6px; min-width: 0; }}
figure img {{ width: 100%; height: auto; display: block; }}
figcaption {{ font-family: var(--mono); font-size: 0.72rem; color: var(--muted); }}
.rules {{ columns: 2 300px; column-gap: 32px; }}
.rules li {{ break-inside: avoid; margin-bottom: 6px; }}
.summary td:first-child {{ min-width: 11em; }} .summary td {{ font-size: 0.88rem; }}
a {{ color: var(--accent); }}
a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>
<main class="wrap">
  <div class="lede">
    <div class="meta">persona-machinery · Qwen3.5-9B base/instruct · OLMo-3-7B base/SFT/DPO/RLVR · updated {now}</div>
    <h1>Persona Machinery</h1>
    <p>How is the machinery that persona vectors and the Assistant Axis plug into actually implemented, and what does the Assistant Axis correspond to in pretraining? This page tracks six pre-registered questions and updates as results land.</p>
  </div>

  <section class="block">
    <h2>Status</h2>
    <div class="tablebox"><table><thead><tr><th>Module</th><th>Question</th><th>Status</th><th>Answer so far</th></tr></thead><tbody>{rows}</tbody></table></div>
    <p class="meta">Order set by Olivia: Q2/Q2b and Q4b first (done), then Q1/Q3, then Q6 on the training stages and the Q4 register control, then Q5 once the pod was idle, then Q6b (Olivia's context-perturbation follow-up). A blind red-team review of all modules is done; corrections are applied below.</p>
  </section>

  <section class="block">
    <h2>Summary</h2>
    {summary_html}
  </section>

  <section class="block">
    <h2>Red-team review</h2>
    <p>A blind reviewer saw only the code and raw results. Every headline number reproduced from the raw rows. There was no hook leakage into prompt tokens and no label leakage into judge prompts. Five findings changed the write-ups:</p>
    <ul>
      <li>The register and character effects were not tested against nulls. Re-tested on 2,000 random directions, only the entry-versus-manual contrast is specific to the axis (Q4).</li>
      <li>"The axis is a readout, not a lever" is withdrawn: the response-only edit never passed a positive control (Q3).</li>
      <li>Layer 32 used the post-final-norm state for fitting but the pre-norm state for editing (M0, Q3, Q4.4); one of nine layers in two null results.</li>
      <li>OLMo's contrast axis moves 2–6× more under knockout than the PC1 figure suggests; its default system message carries much of the signal (Q2).</li>
      <li>Q4b's row-4 call was an artefact of a 4-story bootstrap; no row fires.</li>
    </ul>
    <p class="meta">Also flagged: neither null family is variance-matched; Q1 and Q6 evaluate partly on fit data; Q6's mechanism note was wrong. Full report: results/REDTEAM.md.</p>
  </section>

  <section class="block budget">
    <h2>Budget</h2>
    <div class="nums"><span>Total <b>${total:.2f}</b></span><span>GPU pod <b>${sp["pod"]:.2f}</b></span><span>Judges <b>${sp["api"]:.2f}</b></span><span>Stop and ask <b>$70</b></span><span>Hard cap <b>$90</b></span></div>
    <div class="bar" role="img" aria-label="Spend ${total:.2f} of the $90 cap"><div class="fill" style="width:{pct:.1f}%"></div><div class="stop" style="left:{stop_pct:.1f}%"></div></div>
    <p class="meta">Pod: H100 at $3.99/h until its volume hung at 18:00Z; then an H200 at $5.29/h from 18:35Z. Both stopped at 20:17Z. Q6b ran 21:13–21:38Z on a fresh H200 (the stopped pods' hosts had no free GPU); that pod and the H200 were then deleted.</p>
  </section>

  <section class="block">
    <h2>Plan and rules</h2>
    <ul class="rules">
      <li>Every module is pre-registered in docs/PREREG.md before its data is read; deviations are numbered in docs/DECISIONS.md.</li>
      <li>Every ablation or steering result has a positive control, plus random and covariance-matched subspace nulls.</li>
      <li>Directions are never evaluated on the prompts they were fitted on.</li>
      <li>Effect sizes carry bootstrap (over prompts) or Wilson 95% intervals.</li>
      <li>What a model says about itself is a behavioural readout only, never evidence of mechanism.</li>
      <li>Judges (Claude Haiku 5.5) never see condition labels or model names.</li>
      <li>A blind red-team agent checks code and results at the end for hook leakage, fit/eval overlap, unmatched nulls and leaky judge prompts.</li>
    </ul>
  </section>

  <section class="block">
    <h2>Modules</h2>
    {"".join(sections)}
  </section>
</main>
'''
    Path(out_path).write_text(page)
    print("wrote", out_path, f"{len(page) / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1])

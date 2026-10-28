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
     "verdict": "No pretraining judge convention detected in stories. The base-model behavioural arm could not be measured (no gaming).",
     "points": [
         "Arm A: base models show no automated-vs-human effect in story endings. Qwen base +0.015 nats/token [−0.001, +0.033], OLMo base +0.012 [−0.024, +0.047]; the scheduler-vs-none validity control is ≈ 0 in both.",
         "Post-trained models lean the other way: Qwen instruct −0.038 [−0.060, −0.015] and Qwen3.6-27B −0.071 [−0.120, −0.023], meaning more test-fitting under a human reviewer in stories.",
         "Qwen3.8-27B is near 0 (−0.006 [−0.035, +0.023]) but its validity control fails. The pre-registered ordering 3.8 > 3.6 holds (+0.065 [+0.003, +0.122]), yet the story measure does not reproduce 3.8's +48-point behavioural effect.",
         "Arm B: neither base model games when simulating an Assistant (Qwen 0–1, OLMo 4–5 of 48), even after the one allowed escalation, so the behavioural column is empty.",
         "The decision table mechanically returns row 4 ('the convention is about people') for Qwen base, only because one small CI excludes 0. The human − AI protagonist difference is −0.003 [−0.042, +0.036]; this is not a robust finding.",
     ],
     "figures": ["q4b_armA.png", "q4b_armB.png"]},
    {"id": "Q4", "name": "Pretraining origin of the axis", "status": "done",
     "question": "Base vs instruct axis cosine and CKA; a 200-passage narrative battery in five classes; OLMo stage tracking; three convention-vs-persona tasks.",
     "verdict": "The axis exists before post-training. It tracks impersonal reference register first and helpful-vs-villain character second, additively, and identically in base and post-trained models.",
     "points": [
         "Qwen: the base model's narrative axis is the instruct axis (cos +0.81 with the instruct chat axis at L* 24; +0.88 in matched narrative format; split-half ceiling ≈ 0.93).",
         "OLMo: not the same axis by the 0.7 rule (+0.59 base narrative vs RL chat), mostly a format difference: base vs RL in matched narrative format is +0.91 (descriptive only).",
         "Narrative battery: the prediction fails in reverse. Encyclopaedia and code passages sit at the Assistant end; helpful-character passages sit below the other classes in every model (Qwen base g = −0.73 [−1.05, −0.44], instruct −0.79; OLMo base −0.93, RL −1.19).",
         "Register control (Olivia's 2×2, 40 passages per class): the pre-registered rule fires 'both' in all six models. Register effect +1.9 SD in Qwen base [+1.7, +2.3], character effect +1.2 SD [+0.9, +1.5], interaction ≈ 0. Base and post-trained models do not differ.",
         "Only third-person encyclopaedia-style entries reach the Assistant end, whether about helpers (+3.0) or villains (+2.1), next to encyclopaedia and code text (+3.6). First-person manuals stay where the same character's stories sit (helper −0.5, villain −2.0). The key contrast, villain entry − helper manual, is +2.66 SD [+2.28, +3.18] in Qwen base.",
         "So the axis reads impersonal reference register versus a character's own voice, plus a helper-over-villain step of about 1.2 SD. That explains the reversed battery result.",
         "Exploratory: among narrative and human passages only, Qwen's helpful characters do lean toward the Assistant end, equally in base (+0.52) and instruct (+0.45).",
         "OLMo stages: SFT creates the chat-format axis (later stages cos ≥ 0.99 to RL); the battery ordering is identical at every stage; DPO gives the largest gain in role separation.",
         "Convention-vs-persona tasks: base and post-trained models answer alike in narrative format; the frame matters more than training. The axis-ablation arm is uninterpretable because its positive control failed.",
     ],
     "figures": ["q4_register.png", "q4_axis_cos.png", "q4_battery.png", "q4_stages.png", "q4_tasks.png"]},
    {"id": "Q1", "name": "Writers of the axis coordinate", "status": "done",
     "question": "Attribute the axis projection at L* to every attention head and MLP; is it written by a few components (top 10 explain ≥ 50% of the Assistant − role gap) or diffusely? Check by patching.",
     "verdict": "Intermediate: the top 10 components write 43.5% [42.6, 44.7] of the gap, mostly late MLPs and linear-attention heads. Behavioural positive control failed.",
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
     "verdict": "Narrow: 1 of 5 behaviours moved (style, weakly). The coordinate looks like a readout of the persona rather than a lever on it.",
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
     "verdict": "No detectable deepening at any stage, but this design cannot resolve basin depth: every model snaps back within about one token.",
     "points": [
         "Positive control passes in all 8 pairs: during the window the role vector moves the coordinate by 7–30 units, and the random null by about 0. The prompt and pre-window tokens are untouched (max change 0.0).",
         "Within 10 tokens after the window only 0–4% of the displacement remains, and 50 tokens later nothing beyond the nulls (retained fraction ≈ 0 everywhere). Exponential recovery times cannot be fitted.",
         "Every pre-registered stage step (base→SFT, SFT→DPO, DPO→RL, Qwen base→instruct in chat and in matched narrative format) returns 'no detectable change'.",
         "Why: injection and readout sit at the same layer, so once the vector stops only attention to the perturbed tokens' cached states can carry it, and that route is weak in every model. A real basin test needs a perturbation that changes what the model generates: steer until off-persona text is in the context, then stop and track the return.",
         "Descriptive only: at 3× strength the post-trained chat models retain less right after the window (3–4%) than the bases (8–11%). But the format-matched comparisons disagree, so this tracks chat format at least as much as training.",
     ],
     "figures": ["q6_trajectories.png", "q6_trajectories_3x.png", "q6_control.png"]},
    {"id": "Q5", "name": "Persona vectors vs the axis", "status": "running",
     "question": "Does orthogonalising a trait vector to the axis remove its steering effect?",
     "verdict": "Released now that the pod is idle: one final steering grid (1 layer, c = 2–4), then the mediation test on any trait that passes.", "points": [], "figures": []},
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


def main(out_path):
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
    <p class="meta">Order set by Olivia: Q2/Q2b and Q4b first (done), then Q1/Q3, then Q6 on the training stages and the Q4 register control; Q5 only if the pod is idle after those.</p>
  </section>

  <section class="block budget">
    <h2>Budget</h2>
    <div class="nums"><span>Total <b>${total:.2f}</b></span><span>GPU pod <b>${sp["pod"]:.2f}</b></span><span>Judges <b>${sp["api"]:.2f}</b></span><span>Stop and ask <b>$70</b></span><span>Hard cap <b>$90</b></span></div>
    <div class="bar" role="img" aria-label="Spend ${total:.2f} of the $90 cap"><div class="fill" style="width:{pct:.1f}%"></div><div class="stop" style="left:{stop_pct:.1f}%"></div></div>
    <p class="meta">Pod: H100 at $3.99/h until its volume hung at 18:00Z; since 18:31Z an H200 at $5.29/h.</p>
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

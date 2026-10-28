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
    {"id": "Q2", "name": "Triggers: template vs content", "status": "running", "priority": True,
     "question": "Knock out attention from the assistant turn to (a) template tokens, (b) the system prompt, (c) the user content; compare the axis projection with the raw-document value.",
     "verdict": "", "points": [], "figures": []},
    {"id": "Q2b", "name": "Lift the raw-document floor, then vary the judge", "status": "running", "priority": True,
     "question": "Build a raw document that carries the incentive as text; gate on gaming ≥ 30% (of 48) in 9B instruct; then compare the automated, human-reviewer and checker judge wordings across 9B base, 9B instruct and 27B 3.6/3.8.",
     "verdict": "", "points": [], "figures": []},
    {"id": "Q4b", "name": "Narrative convention vs the Assistant's judge distinction", "status": "running", "priority": True,
     "question": "Arm A: forced-choice log-odds (fit the test vs follow the spec) on 48 story templates crossed by judge × protagonist × stakes. Arm B: the base model simulating an AI-assistant transcript, scored by execution.",
     "verdict": "", "points": [], "figures": []},
    {"id": "Q4", "name": "Pretraining origin of the axis", "status": "running",
     "question": "Base vs instruct axis cosine and CKA; a 200-passage narrative battery in five classes; OLMo stage tracking; three convention-vs-persona tasks.",
     "verdict": "", "points": [], "figures": []},
    {"id": "Q1", "name": "Writers of the axis coordinate", "status": "queued",
     "question": "Attribute the axis projection at L* to every attention head and MLP; is it written by a few components (top 10 explain ≥ 50% of the Assistant − role gap) or diffusely? Check by patching.",
     "verdict": "", "points": [], "figures": []},
    {"id": "Q3", "name": "Readers and dependent behaviours", "status": "queued",
     "question": "Ablate or clamp the axis at L*..L*+8 on generated tokens; score refusal, sycophancy, self-description, style and honesty, with format-following as the control. Then find readers by axis-restricted path patching.",
     "verdict": "", "points": [], "figures": []},
    {"id": "Q5", "name": "Persona vectors vs the axis", "status": "blocked",
     "question": "Does orthogonalising a trait vector to the axis remove its steering effect?",
     "verdict": "Not run: no trait vector passed the M0 steering gate. One more grid (1 layer, c = 2–4) awaits Olivia's go-ahead.", "points": [], "figures": []},
    {"id": "Q6", "name": "Attractor dynamics (stretch)", "status": "deferred",
     "question": "After a 5-token role perturbation, how fast does the axis projection return, in base vs instruct and across OLMo stages?",
     "verdict": "Pre-registration and code done; runs only if time remains after the priority modules.", "points": [], "figures": []},
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
    <p class="meta">Order set by Olivia: Q2/Q2b and Q4b first, then Q1/Q3; Q5/Q6 dropped if time is short.</p>
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

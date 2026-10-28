# Brief for module agents (A: Q1+Q3, B: Q2+Q2b, C: Q5+Q6, D: Q4+Q4b)

Read first: `spec.md` (your modules), `docs/PREREG.md` (shared design + your module; binding), `docs/DECISIONS.md`, `results/M0/README.md`.

## Rules you must follow
- **Pre-register before running.** Your module section in `docs/PREREG.md` is binding. Before the first run, append an "Implementation (written <date>, before any <module> data)" sub-section with the exact prompt sets, counts and seeds you will use. Anything you change later goes in `docs/DECISIONS.md` (next number, date, who, decision). Never rewrite an existing pre-registration.
- **Positive control before any null**; every result has the random and covariance-matched nulls (basis files carry 3 draws: `rand{k}_{L}_{d}`, `cov{k}_{L}_{d}`).
- **Held-out rule:** never evaluate a direction on the prompts it was fitted on (role-library questions are fit data; the Chen extraction questions are fit data).
- **No self-report as mechanism.** Judge scores of what the model says about itself are behavioural readouts.
- **Effect sizes with CIs** (bootstrap over prompts, 2,000 resamples; Wilson/Newcombe for proportions).
- **Judge prompts never contain the condition label or model name.** Use `src/judge.py` (Haiku 5.5; cap each call batch; spend logs to `data/spend.jsonl`). Total judge budget for all agents: $15; you get the share stated in your spawn message.
- **Never write to `vendor/`.** Never modify `results/M0/`. Write only to `results/<your module>/`, `data/<your module>/`, `src/<your module>_*.py`, `src/pod/<your module>_*.py`, `figures/<your module>_*.png`, and your sections of `docs/PREREG.md` / `docs/DECISIONS.md`.
- **Git:** commit your own files locally with clear messages; never push; do not `git add -A` (other agents are editing concurrently; add your paths explicitly). If a commit fails because of a lock, retry.
- **Budget:** check `data/spend.jsonl` and the pod clock before any run > 20 GPU-minutes; the pod is ~$4/h shared by four agents. Stop and report if your module would push total spend past $70.

## Infrastructure
- Pod: `bash src/pod/job.sh push <files>` (lands under `/workspace/job/`, same relative paths), `bash src/pod/job.sh run NAME "<cmd>"` (queued under a GPU lock: one GPU job at a time, FIFO-ish; your job waits silently while another runs), `bash src/pod/job.sh status NAME [lines]`, `bash src/pod/job.sh pull /workspace/job/out/<dir> results/<...>`, `bash src/pod/job.sh sh "<cmd>"` (synchronous, no GPU lock; CPU/IO only). Wait for jobs with a background `until bash src/pod/job.sh sh "test -f logs/NAME.done"; do sleep 30; done`. Name jobs `<module>_<step>`.
- Pod scripts in `src/pod/`: `pm_common.py` (model registry `MODELS`, `render_chat` / `render_narrative`, `decoder_layers`, `load_hf`), `pm_gen.py` (vLLM sampling from a jobs file), `pm_gen_hf.py` (HF generation with interventions at generated positions: proj_out / clamp / add, conditions JSON), `pm_acts.py` (teacher-forced per-layer means), `pm_basis.py`. Import from `pm_common` in new pod scripts; keep them self-contained otherwise.
- M0 artefacts on the pod: `/workspace/job/out/m0/` (`gen_<model>_<fmt>.jsonl` with token ids, `acts_<model>_<fmt>/`, `basis_<model>_<fmt>.npz`, `traits_<model>.npz` after step 6). Locally: `results/M0/`. Basis keys per layer L: `aa_L` (unit AA), `mu_L` (role grand mean), `default_L`, `rolemeans_L` (99 × 4096, order `roles`), `P5_L`, `P15_L`, `def5_L`, `def15_L`, `rand1/5/15_L_d`, `cov1/5/15_L_d`, plus `L_star`, `dprime`.
- Local Python: `.venv/bin/python` (numpy, scipy, matplotlib, sklearn, pandas, openai). Figures: `import figstyle; figstyle.setup()`; palette and style per spec; `figstyle.title(ax, main, italic_subtitle_with_n)`.
- Models: Qwen3.5-9B is hybrid (softmax attention only at layers 3, 7, 11, 15, 19, 23, 27, 31 — zero-indexed block ids; `layer_types` in the config); OLMo-3-7B is all softmax (sliding 4096 except full attention at the same indices). OLMo's attention and MLP outputs pass through a post-norm before the residual add (`post_attention_layernorm`, `post_feedforward_layernorm`).

## Deliverable per module
`results/<module>/README.md`: question, pre-registered rule and which outcome fired, headline numbers with CIs, positive control and nulls, figures, spend, deviations. Keep raw rows next to it (jsonl) so the red-team agent can re-derive every number.

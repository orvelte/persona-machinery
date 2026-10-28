# Decisions and deviations

| # | Date | Who | Decision |
|---|---|---|---|
| 1 | 2026-10-28 | C | `vendor/persona-mech` is a read-only (`chmod a-w`) copy of `~/personawhitebox` (the repo the spec calls `persona-mech`, confirmed by its own DECISIONS #1). Copied: `src/`, `docs/`, READMEs, spec, and every file under 20 MB. Not copied: `.venv`, `.git`, its nested `vendor/`, `models/`, `.env`, ssh keys, and files over 20 MB (large `data/` and `results/` arrays). Those remain at `~/personawhitebox`; read them there if a module needs them, never write. `vendor/` is gitignored, as in the source repo. |
| 2 | 2026-10-28 | C | HF names confirmed and pinned (PREREG shared design). OLMo-3-7B Think line = `Olmo-3-1025-7B` (base) → `Olmo-3-7B-Think-SFT` → `Olmo-3-7B-Think-DPO` → `Olmo-3-7B-Think` (RLVR). |
| 3 | 2026-10-28 | C | Gemma 3 + Gemma Scope 2 skipped: the repos are gated (401 without a licence-accepted token) and the $90 cap leaves no room for a third family. No SAE/transcoder decomposition in Q1/Q5. |
| 4 | 2026-10-28 | C | Qwen3.5-9B is a hybrid (8 softmax-attention + 24 Gated-DeltaNet layers). Positional attention knockout is exact only in OLMo-3, so Q2's primary model is `olmo_rl`; Qwen gets partial knockout + token ablation. Q1 head attribution uses output-projection input slices in both families. |
| 5 | 2026-10-28 | C | OpenRouter credit remaining at start: $19.25 of a $1,000 limit. Judge = Claude Haiku 5.5 (Sonnet 5.5 only for the 60-item agreement checks and battery writing); judge spend capped at $15. Olivia may need to top up for anything beyond that. |
| 6 | 2026-10-28 | C | Thinking disabled in all chat-format generations (Qwen `enable_thinking=False`; OLMo Think line: `<think></think>` closed in the prompt) and max 384 new tokens: the budget needs cheap sweeps, and every intervention acts on generated tokens, so the response span is what matters. The persona-mech frozen axis used thinking-on responses at 27B; the Spearman comparison in M0 step 4 tests whether the role ordering survives this change. |
| 7 | 2026-10-28 | C | The 27B frozen axis cannot be compared by cosine (5,120 vs 4,096 dims); M0 compares the 99-role ordering by Spearman instead. |
| 8 | 2026-10-28 | C | One pod, GPU jobs serialised by a lock (`job.sh run` wraps commands in `flock`), so the three agents can queue work without OOM; vLLM jobs need the whole GPU. |

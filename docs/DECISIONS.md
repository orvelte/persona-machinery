# Decisions and deviations

| # | Date | Who | Decision |
|---|---|---|---|
| 1 | 2026-10-28 | C | `vendor/persona-mech` is a read-only (`chmod a-w`) copy of `~/personawhitebox` (the repo the spec calls `persona-mech`, confirmed by its own DECISIONS #1). Copied: `src/`, `docs/`, READMEs, spec, and every file under 20 MB. Not copied: `.venv`, `.git`, its nested `vendor/`, `models/`, `.env`, ssh keys, and files over 20 MB (large `data/` and `results/` arrays). Those remain at `~/personawhitebox`; read them there if a module needs them, never write. `vendor/` is gitignored, as in the source repo. |

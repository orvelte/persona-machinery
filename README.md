# persona-machinery

How the machinery that persona vectors and the Assistant Axis plug into is implemented, and what the Assistant Axis corresponds to in pretraining.

Spec: `spec.md`. Pre-registrations: `docs/PREREG.md`. Deviations: `docs/DECISIONS.md`.
Inputs: `vendor/persona-mech` (read-only copy of `~/personawhitebox`; see DECISIONS #1). Results: `results/Qx/README.md` per module.

Rules in force: pre-register before reading data; positive control before any null; random and covariance-matched nulls; held-out prompt split for every fitted direction; bootstrap CIs; no self-report as mechanism; commit locally, don't push; hard cap $90, stop and ask at $70.

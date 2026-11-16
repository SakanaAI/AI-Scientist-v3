# Goal
- TODO: one sentence on what to improve and why it matters.
- TODO: what the candidate may change (the editable files under `src/`) and what is fixed.
- Maximize `combined_score` from `metrics.json` (TODO: how it is computed, e.g. `-val_loss` from the
  official scorer on your saved artifacts). `correct.json` must report `correct: true`.
- Each evaluation takes about TODO minutes on TODO hardware. Set your shell timeout to at least
  TODO ms when running the evaluator.
- Implement ONE variant of your idea with ONE fixed configuration per evaluation. Do NOT run
  hyperparameter sweeps, ablations, or multiple variants inside one program.

# Constraints
- TODO: every rule of the benchmark, stated so it can be checked.
- TODO: fixed interfaces (function signatures, file names, output formats) the evaluator relies on.
- TODO: data the program may read; data it must never read.

# Rubrics
- `beats_baseline`: combined_score > TODO (baseline under this evaluator, minus measured spread).
- `rules_respected`: TODO the rule a static check cannot fully enforce, stated concretely.
- `evidence`: every evaluated candidate produces `metrics.json` and `correct.json`.

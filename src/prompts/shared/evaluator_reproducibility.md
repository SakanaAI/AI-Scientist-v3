# Evaluator reproducibility

Both evaluators must return the same `metrics.json` and `correct.json` for a fixed program on
every invocation; otherwise investigation compares candidates against noise. Seed every random
stream (environments, data order, Monte-Carlo draws, model initialization) from task-owned
constants. A secret that hides seeds from the candidate must be fixed per task, for example a
constant in the evaluator or derived from one; never draw it per invocation. Do not use
`secrets.*`, `os.urandom`, `uuid.uuid4`, wall-clock time, process IDs, or unseeded RNGs to derive
anything that affects scores or correctness. Use deterministic accelerator settings where the
framework offers them. Draft smoke evaluates the unchanged baseline twice and rejects any
difference in `combined_score` or correctness.

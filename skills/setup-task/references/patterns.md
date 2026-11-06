# Task design patterns

## Layouts

**Single.** The program is the whole solution (`program_0/src/program.py`). The evaluator copies
the program root into scratch space and runs it. Examples: `examples/circpack`, `examples/grpo`.

**Overlay.** The candidate edits a few files inside a larger codebase it should not rewrite. Vendor
the upstream checkout at a pinned commit under `resources/upstream/`, with its `.git` removed and
a `resources/UPSTREAM` file recording the URL, commit, and license. Put the editable files under
`program_0/src/` at the same relative paths as in upstream. For each evaluation, the evaluator
copies upstream to scratch, copies `src/` over it, runs, and scores. Benefits:

- The edit boundary is explicit: anything outside `src/` is reset on every evaluation.
- Diffs between programs show only the science.
- The scorer and data stay evaluator-owned.

Pick the entrypoint as the main editable file (e.g. the training script). Add files that agents
may create, such as helper modules, by letting the overlay copy all of `src/`, and say so in
`task.md`.

## Search vs full profiles

Agents need many evaluations, so a search evaluation should take minutes, not hours. When the real
benchmark is expensive:

- The `search` profile (the default, the only one agents see) uses a cheaper proxy: fewer tokens
  or steps, a smaller model, fewer seeds. Write the proxy and what it omits into `task.md`.
- The `full` profile is the real benchmark, for confirming a winner by hand
  (`--profile full`, run directly).
- Check the proxy: score 3–5 diverse known programs on both profiles and confirm the ranking holds.
  If it doesn't, the search will optimize the wrong thing.

The harness cannot pass custom flags or environment variables to the evaluator, so the default
profile is fixed in code.

## Leakage

- The candidate never sees held-out data. Make the evaluation split invisible during the candidate
  run: train in a scratch directory that only has training data, then score in a separate process.
- The candidate never reports its own score. The evaluator recomputes the metric from artifacts it
  controls, e.g. a saved checkpoint loaded into an evaluator-owned model definition.
- Don't print evaluation targets, labels, or the scoring code path into logs that agents read.
- Checkpoint or model selection on the evaluation split is a leak unless the benchmark allows it.

## Legality

Sort every rule into one of three layers:

| Layer | Where | Examples |
| - | - | - |
| Static | evaluator, before the run, gives `correct: false` | forbidden imports or calls (regex or AST over `src/`), files outside the edit boundary, size limits |
| Runtime | evaluator, after the run, gives `correct: false` | strict checkpoint load, number of training examples consumed, wall time, output shape, determinism |
| Judgment | rubric in `task.md`, checked by the agent's `verify-rubrics` | "the spirit of rule X", method novelty, no benchmark-specific hacks |

Static checks are cheap but can be dodged (`getattr` strings, `exec`). Back every important static
rule with a runtime check, or with a judgment rubric when no runtime check exists. Prove each
check with a negative program during validation.

## Cost and timeouts

- Measure baseline wall time on the target hardware, including cold caches: compile and kernel
  warm-up run on every evaluation because `HOME` and `TMPDIR` are fresh.
- Program timeout in the evaluator: about 1.5 to 2 times the baseline wall time.
- Harness timeout (`harness_evaluation_timeout_seconds`): the evaluator timeout plus scoring plus
  a margin.
- Put the expected wall time in `task.md`, so agents set long enough shell timeouts.
- Lower `max_program_evals_per_attempt` for evaluations over 30 minutes.

## Score variance

Run the baseline at least twice (different seeds, or the same seed on different nodes). State the
spread in `task.md` and in metric rubric thresholds, so agents don't chase random variation. If the benchmark
fixes the seed, keep it fixed and measure node-to-node variance instead.

## Hardware

- The evaluator uses the GPUs it is given (`CUDA_VISIBLE_DEVICES`), never hard-coded device IDs.
- Multi-GPU programs: launch `torchrun --standalone --nproc_per_node=<visible GPUs>` from the
  evaluator. Don't let the candidate choose the launch topology unless that is part of the task.
- Cluster schedulers (Slurm, ...) are outside the task. The evaluator runs locally on the GPUs it
  is given.

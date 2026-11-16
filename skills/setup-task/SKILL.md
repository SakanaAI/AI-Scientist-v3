---
name: setup-task
description: Turn a research problem, benchmark, or existing code repository into a task directory that AI-Scientist-v3 (`agsci run`) can optimize — task.md, evaluate.py, capture.py, and a baseline program_0 — then validate it with real evaluations. Also writes brief-only task.md files for the harness's experimental drafting phase. Use when a user asks to set up, port, wrap, or scaffold a task, benchmark, leaderboard, or speedrun for the agentic scientist. Launching and monitoring runs is the separate run-campaign skill.
---

# setup-task

Build an `agsci`-compatible task with the user, then prove it works by running the evaluator.
The user decides the science (metric, editable surface, rules, budget); you build the scaffold and
run the evaluations. A task that has not been evaluated end to end is not done. Launching the
campaign is out of scope: hand off to the `run-campaign` skill.

## Locate the harness

This skill lives in the AI-Scientist-v3 repository at `skills/setup-task/`. Resolve the repo root
from the skill directory's real path (`realpath <skill_dir>/../..`); call it `$AGSCI`. Scripts:

- `$AGSCI/skills/setup-task/scripts/new_task.py`: create a scaffold from templates.
- `$AGSCI/skills/setup-task/scripts/check_task.py`: static checks, then a real evaluator run.

Run both with `$AGSCI/.venv/bin/python` (create it with `uv sync` in `$AGSCI` if missing). Put the
task wherever the user wants. It does not need to be inside `$AGSCI`.

## Decision points: ask the user

Decide everything else yourself with sensible defaults, and report those choices in the handoff.
Ask in one batch where possible, and only for what the user's material doesn't already answer.

| Part | Ask about |
| - | - |
| Task definition | metric and direction, editable surface, fixed parts, the starting program |
| Rules | every legality constraint; which ones are judgment calls |
| Data | sources and licenses; where already-prepared data lives |
| Search profile | target wall time per search evaluation; acceptable loss of ranking fidelity |
| Validation compute | budget for baseline, calibration, and negative runs (e.g. node-hours); always ask before a full-profile run |
| Shared resources | scheduler, partition, node or GPU caps, jobs that must not be disturbed, disk quotas |

## Pick the mode

| Mode | Use when | You produce |
| - | - | - |
| **A. Hand-written scaffold** (default) | Existing code or repo, GPU training, multi-file programs, long evals, external data | `task.md`, `evaluate.py`, `capture.py`, `program_0/`, optional `requirements.txt` |
| **B. Brief only** | Small self-contained problem with CPU-scale evaluation and no external code | `task.md` only; `agsci run` drafts a sealed contract task from it |

Mode B is experimental in the harness and fails on large tasks. If the problem needs an upstream
repo, multi-GPU training, or evaluations longer than ~30 min, use mode A. Never hand-write
`task_contract.json`: its presence switches the harness into contract mode (requires
`evaluate_final.py`, RPC-isolated candidates, and a validation receipt only the harness can issue).

## Workflow (mode A)

### 1. Probe the environment

Before designing anything, find out where evaluations will run, and record the result in the
task's `README.md`:

- Where GPUs are: local `nvidia-smi`, or only through a scheduler (`sinfo`, `squeue`)?
- Can the GPU hosts run Docker (`docker info` *on the GPU host*, e.g. via a short `srun`)? If not,
  the campaign will run natively (`container.enabled=false`).
- Does the evaluation need a C compiler or JIT kernels (Triton, `torch.compile`, CUDA extensions)?
  The default agent image has no compiler and containerized evaluation mounts `/tmp` noexec, so
  such tasks need native mode or a custom image.
- Large external data is visible to containerized harness evaluation only if it lives inside the
  task. Decide early: data in the task, or a native campaign.
- Free disk where runs and checkpoints will go (`df -h`).
- Existing prepared data, checkpoints, or earlier experiments the user points to.

Run GPU work only on resources the user allowed, and never cancel or modify jobs you did not
start.

### 2. Interview

Read the user's material first (repo README, rules, leaderboard, existing scripts). Then fill the
gaps from the decision table. Record the answers in the task's `README.md`. You need:

1. **Objective**: the metric, its direction, and who computes it. `combined_score` must be higher
   = better, so a loss becomes `-loss`.
2. **Editable surface**: which files the agent may change. Everything else is fixed.
3. **Layout**: *single* (the program is the whole solution) or *overlay* (candidate files are
   placed into a pinned upstream checkout). See `references/patterns.md`.
4. **Upstream and data**: repo URL plus exact commit, data sources, sizes, hashes, licenses.
5. **Rules**: every legality constraint, sorted into static checks, runtime checks, and
   judgment-only rubrics (`references/patterns.md#legality`).
6. **Leakage**: what the candidate must never read (validation/test data, the scorer's internals).
7. **Cost**: wall time and hardware per evaluation, and the search profile (a cheaper proxy).
8. **Baseline**: which program starts the search, and its known score.
9. **Score variance**: run-to-run spread of the score, or a plan to measure it.

Push back when an answer makes the task unsound: a score the candidate reports itself, validation
data visible during training, an evaluation that cannot finish inside the timeout.

### 3. Scaffold

```bash
$AGSCI/.venv/bin/python $AGSCI/skills/setup-task/scripts/new_task.py <task_dir> \
    --layout single|overlay [--entrypoint program.py]
```

It copies the harness's canonical `capture.py` and the templates, and refuses to overwrite existing
files. For overlay tasks, vendor the upstream tree at the pinned commit into
`resources/upstream/` (without `.git`, data, or logs) and fill in `resources/UPSTREAM`.

### 4. Write `evaluate.py`

Start from the template's `TODO` markers. Hard requirements, from the harness code, are in
`references/harness-contract.md`. In short:

- CLI: `--program_path <entrypoint file>` and `--results_dir <dir>`. Also accept
  `--num_replications N`, which the harness passes when replications are requested.
- Write `metrics.json` (flat object with a finite numeric `combined_score`, higher is better) and
  `correct.json` (`{"correct": bool}` plus an optional short `error`).
- A candidate failure (crash, timeout, rule violation) still exits 0, with `correct: false` and a
  finite worst-case `combined_score`. Exit non-zero only when the evaluator itself breaks.
- Never write into the task directory: it is mounted read-only in containers. Use `--results_dir`
  for logs and `tempfile` (the harness sets `TMPDIR`) for scratch space.
- Expect one of three environments (agent-run, containerized harness, native harness; see the
  table in `references/harness-contract.md`): possibly no network, an empty `HOME`, and only
  allowlisted variables. Custom variables do not arrive, so put settings in constants or flags.
- Keep output small: echo only progress and error lines, keep the full program log in scratch, and
  copy a bounded tail into `--results_dir` as `stdout.log` (never copied into later programs).
- Honor replications: `--num_replications N` (agent-run) or `NUM_REPLICATIONS` (ablations, 5 by
  default), or tell the user to set
  `ablations.execution.num_replications=1`.
- Score with evaluator-owned code on evaluator-owned data. Never trust a number the candidate
  prints or writes. Check the hashes of external data.
- Kill the whole process group on timeout, on SIGINT (how the harness stops evaluators), and on
  SIGTERM. Wait on the process, not on its output pipe, so leftover child processes cannot hang the
  evaluator. No scheduler calls (`srun`, `sbatch`) inside the evaluator: it runs on the GPUs it is
  given.

`--profile smoke|search|full` is a convention for local use: agents always get the default
(`search`); `smoke` is for `check_task.py`; `full` is for confirming a winner by hand.

### 5. Write `program_0`

`program_0/program_spec.toml` (template provided, `code_root` must be `src`) plus the baseline
under `program_0/src/`. For overlay tasks, `src/` holds exactly the editable files at the paths
the overlay maps them to. The baseline must be valid: `correct: true` under the search profile.
If the baseline needs a small change to fit the evaluator (e.g. skipping in-run validation when
held-out data is hidden), make the smallest change possible and document it in `task.md`.

### 6. Write `task.md`

Use `templates/task.md`. The harness extracts `# Goal` and `# Rubrics` verbatim into agent
prompts. State the evaluation wall time and that agents should run ONE variant per evaluation. A
rubric written as `combined_score > X` is checked numerically by the harness. Other rubrics are
judged by the agent's `verify-rubrics` step, so make them concrete and checkable from the source.

### 7. Validate (all must pass)

```bash
PY=$AGSCI/.venv/bin/python; CHECK=$AGSCI/skills/setup-task/scripts/check_task.py
$PY $CHECK <task_dir>                                    # static checks only
$PY $CHECK <task_dir> --run --profile smoke              # evaluate.py directly, smoke profile
$PY $CHECK <task_dir> --run --expect correct             # baseline through the harness wrapper
$PY $CHECK <task_dir> --run --program <bad_program_dir> --expect incorrect
$PY $CHECK <task_dir> --run --num-replications 2         # if the evaluator honors replications
```

- Run GPU checks where GPUs are, e.g. `srun ... $PY $CHECK ...` on a cluster.
- Use `--python <task venv python>` when the evaluator needs packages the harness venv lacks.
- Make one negative program per important rule (in a scratch directory, not the task), plus one
  that crashes. All must give `correct: false`.
- Run the baseline at the search profile at least twice; record the mean, spread, and wall time.
- Calibrate the search profile when the user's data allows it: score a few programs with known
  full-benchmark results and compare the rankings.
- Stay inside the validation compute budget. Ask before exceeding it, and before any full-profile
  run.
- Then run `agsci doctor <task_dir> [overrides]` and `agsci run <task_dir> [overrides] --dry-run`
  (from `$AGSCI`) in the environment the campaign will use: on a GPU host, container or native.
  `check_task.py` mirrors the agent-run path only; the containerized harness sandbox differs
  (read-only, noexec `/tmp`, no external data), so a one-cycle pilot campaign
  (`run-campaign`) is the final proof for GPU tasks.

### 8. Hand off

Finish the task's `README.md`: interview answers, environment probe results, a validation log
(commands, job IDs, scores, wall times), the recommended `agsci run` overrides
(`references/harness-contract.md#run-options-worth-setting`), and anything skipped. Then point the
user to the `run-campaign` skill for launching. Report what was validated, what was skipped, the
measured baseline score, and the wall time per evaluation.

## Workflow (mode B: brief only)

1. Interview as above, but only objective, constraints, dependencies, and budget matter.
2. Write `task.md` with `# Goal`, `# Constraints` (pinned dependencies, exactness requirements,
   compute limits), and `# Rubrics`. See `examples/circpack_task_drafting/task.md` in `$AGSCI`.
3. Leave the directory with `task.md` only, then run
   `agsci run <task_dir> execution.phases=task_drafting` (containers required) and read the audit
   under the run's `task_drafting/` if the draft is rejected.

## References

- `references/harness-contract.md`: exact interface, sandbox, limits, run options, native mode.
- `references/patterns.md`: layouts, search vs full profiles, leakage, legality, cost, variance.
- `references/example-overlay.md`: a worked overlay task for a multi-GPU training benchmark.

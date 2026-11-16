# Variants Pipeline

Generate, validate, and run **baselines** and **ablations** against a completed
scientist run. Each variant is executed by a containerizable agent, evaluated
against the same rubric as the original task, and finally aggregated into
per-task CSVs and a paper-ready `variants_readme.md`.

---

## Quick start

```bash
python -m src.variants_pipeline.cli \
  paths.task_dir=examples/circpack/ \
  paths.results_dir=results/circpack/claude/20260331_094917/ \
  paths.output_dir=results/circpack/claude/20260331_094917/ablations/ \
  execution.num_replications=5 \
  execution.max_evals_per_variant=5 \
  paths.baseline_program_override=results/circpack/claude/20260331_094917/worker_02/round_000/ideas/01_some_idea/program_1/ \
  paths.custom_baselines='[path/to/custom_baseline]' \
  paths.extra_tasks='[path/to/extra_task]'
```

What this run does:

1. Extracts a bundle from the completed `circpack` run, but uses `worker_02`'s
   `program_1` as the best program instead of the global best.
2. Retrofits that program to emit the analysis CSVs while preserving the host-sealed evaluator.
3. Asks the LLM for ablation candidates, validates and selects them.
4. Adds the `path/to/custom_baseline` folder as an always-accepted custom baseline.
5. Materializes one workspace per variant under `variants/`.
6. Runs every variant agent (5 evaluations × 5 replications each).
7. Adapts every variant onto `path/to/extra_task` to test cross-task
   generalization.
8. Aggregates analysis CSVs into per-task subfolders under `aggregated/`.
9. Writes `variants_readme.md` for the paper-writing agent.

All artifacts land under `paths.output_dir`.

---

## Stages

The pipeline runs 9 stages (0–8). Use `--stages` to run a subset; skipped
stages load their inputs from disk if those artifacts already exist.

| # | Module          | Stage                                                                                       |
|---|-----------------|---------------------------------------------------------------------------------------------|
| 0 | `extract.py`    | Extract a self-contained bundle (best program + reference + task files) from `results_dir`. |
| 1 | `retrofit.py`   | Patch `program_0` to emit analysis CSVs while preserving the host-sealed evaluator.         |
| 2 | `generate.py`   | Ask an LLM to propose baselines and ablations.                                              |
| 3 | `select.py`     | LLM-validate candidates, drop bad ones, cap counts via `max_ablations` / `max_extra_baselines`. |
| 4 | `materialize.py`| Create one `variants/<id>/` workspace per accepted candidate (clones the retrofitted bundle). |
| 5 | `execute.py`    | Run one agent per variant; up to `max_evals_per_variant` programs per variant.              |
| 6 | `cross_task.py` | For each `extra_tasks` folder: adapt `control_best`, then port each variant onto the new task. |
| 7 | `aggregate.py`  | Collect every per-program CSV into `aggregated/<task>/<csv>.csv`.                           |
| 8 | `readme.py`     | Generate `variants_readme.md` for the paper agent (deterministic structure + agent summaries). |

```bash
# Run extract + retrofit only:
python -m src.variants_pipeline.cli --stages 0,1 ...

# Resume from materialize through README generation:
python -m src.variants_pipeline.cli --stages 4-8 \
  paths.output_dir=/path/to/existing/run ...

# Skip LLM generation/selection and run only the explicit custom baselines:
python -m src.variants_pipeline.cli --custom-only \
  paths.custom_baselines='[path/to/custom_baseline]' ...
```

---

## Configuration

Defaults live in `configs/variants_config.yaml`. Override with Hydra
`key=value` syntax on the command line.

### `paths`

| Key                         | Required | Description                                                                                |
|-----------------------------|:--------:|--------------------------------------------------------------------------------------------|
| `task_dir`                  | yes      | Original task directory (`task.md`, `evaluate.py`, seed program).                          |
| `results_dir`               | yes      | Completed scientist run to extract the best program from.                                  |
| `output_dir`                | no       | Where this pipeline writes. Defaults to `results_dir/variants/<task>_<agent>_<run_id>/`.   |
| `baseline_program_override` | no       | Use this folder as the best program instead of the run's auto-detected best.               |
| `custom_baselines`          | no       | List of `.md` files (with YAML frontmatter) or folders. Always accepted, bypass stages 2–3. |
| `extra_tasks`               | no       | List of task folders for cross-task generalization (stage 6).                              |

### `execution`

| Key                             | Default | Description                                                          |
|---------------------------------|--------:|----------------------------------------------------------------------|
| `max_ablations`                 | 3       | Hard cap on accepted ablation variants.                              |
| `max_extra_baselines`           | 3       | Hard cap on accepted baseline variants.                              |
| `max_evals_per_variant`         | 2       | Programs each variant agent may produce (`program_1`, `program_2`, …). |
| `num_replications`              | 1       | Evaluator-managed replications per evaluation.                       |
| `variant_timeout_seconds`       | 7200    | Per-variant agent timeout (stage 5).                                 |
| `cross_task_timeout_seconds`    | 7200    | Per-variant agent timeout when adapting to an extra task (stage 6).  |
| `generation_timeout_seconds`    | 1800    | Stage 2 LLM timeout.                                                 |
| `selection_timeout_seconds`     | 1800    | Stage 3 LLM timeout.                                                 |
| `retrofit_timeout_seconds`      | 1800    | Stage 1 LLM timeout (also reused by cross-task retrofit in stage 6). |
| `readme_timeout_seconds`        | 600     | Stage 8 LLM timeout.                                                 |
| `variant_timeout_grace_seconds` | 30      | Grace period after any variant timeout.                              |

### `agent` and `container`

Same keys as the main investigation config: `agent.name`, `agent.model`, and the
`container.*` block (`enabled`, `image`, `network`, `auth_root`, `env_file`).
See `docs/install.md` for container setup.

Stage 5 sealed evaluation runs the evaluator in a network-isolated container
with direct container GPUs; its candidate and results mounts are private to
that sandbox.

With `container.enabled=false` (native runs, e.g. GPU hosts without Docker or
tasks that need a compiler, JIT kernels, or data outside the task), variant
agents run on the host and `native_eval.py` runs the sealed evaluator directly:
the native investigation environment (parent environment without
credential-like variables or Python path overrides, real `HOME`), a private
`TMPDIR` and results directory, the task's `.venv` Python if it has one (else
the harness interpreter, with a warning), and `--num_replications N` when
`NUM_REPLICATIONS` > 1. The evaluator and all its descendants, including ones
that start their own sessions such as torchrun workers, are stopped on exit,
timeout, and SIGINT/SIGTERM (`run_contained`, shared with native
investigation). Results pass the same promotion contract as the sandbox
(canonicalize, allowlist, 64 MiB cap, no symlinks or collisions). There is no
image or managed-runtime preflight natively, so `program_0` is first evaluated
with the control variant. Agents and evaluators share your permissions; the
sealed evaluator digest, checked before and after every evaluation, detects
tampering with the evaluator tree (it does not cover `.venv`, `resources/` or
`results/`). Contract tasks and cross-task variants (`paths.extra_tasks`) still
require containers and are refused up front.

Before any stage-5 artifact or agent call, the pipeline provisions
`paths.task_dir/.venv` in the immutable evaluation image and proves
`program_0` with the sealed evaluator. `.agentic_runtime.json` binds the
runtime to the exact image, evaluator, pinned `requirements.txt`, and runtime
tooling. Compatible managed runtimes are reused. Symlinked or unmanaged
environments are rejected instead of trusted.

If this preflight fails, fix the task error in the log and rerun the same stage
selection and `paths.output_dir`. In the end-to-end pipeline, use the printed
`execution.phases=ablations,posthoc,paper` resume command; completed
investigation work is preserved.

---

## Custom baselines

Two formats are supported (mix and match in one list):

* **Markdown file** — YAML frontmatter (`variant_id`, `title`, `hypothesis`)
  plus a markdown body. The agent implements the baseline from the description.
* **Folder** — a working reference implementation. Contents are cloned as
  `program_0` and the agent integrates them with the evaluation harness. An
  optional `baseline.md` inside the folder can override metadata.

Custom baselines bypass LLM generation/selection (stages 2–3). With
`--custom-only`, the pipeline runs *only* the custom baselines and skips
`control_best` / `baseline_reference`. When cross-task stage 6 is enabled,
`control_best` is retained as the required verified adaptation anchor;
`baseline_reference` remains skipped.

---

## Extra tasks (cross-task generalization)

When `paths.extra_tasks=[...]` is set, stage 6 runs a two-phase process per
extra task:

1. **Adapt control** — `control_best` is ported to the new task in isolation,
   producing `_adapted_control/`. This establishes the integration pattern.
2. **Adapt variants** — every other variant gets `program_0/` = the adapted
   control plus `original_variant_program/` = its best program from the
   original task. The agent applies the variant's algorithmic changes onto the
   adapted control.

Stage 6 first runs the retrofit agent on the extra task's seed program so its
`evaluate.py` matches the variants-pipeline conventions
(`--seed`, `--results_dir`, replications, timeouts).
All extra-task managed runtimes are provisioned and validated first, before any
cross-task output directory or retrofit agent is started.

---

## Output layout

```
<output_dir>/
├── _extracted_bundle/        # stage 0 — best program + reference + task files
├── _retrofitted_bundle/      # stage 1 — bundle with CSV-emitting evaluate.py
├── generation/
│   ├── variant_bundle.json   # stage 2 — LLM-proposed candidates
│   └── variant_selection.json# stage 3 — accepted/rejected with reasons
├── variants/                 # stage 4 — one workspace per accepted variant
│   └── <variant_id>/
│       ├── variant_spec.json
│       ├── rubric_contract.json
│       ├── target_rubrics.json
│       ├── run_manifest.json
│       └── task_bundle/
│           ├── program_0/    # seed (read-only)
│           ├── program_1/    # agent-produced
│           │   ├── src/program.py
│           │   ├── metrics.json
│           │   ├── rubric_status.json
│           │   └── analysis/*.csv
│           └── program_2/    # repair attempt (if needed)
├── extra_tasks/              # stage 6 — cross-task adaptations
│   └── <task_name>/
│       ├── _adapted_control/ # phase 1 — control ported to new task
│       └── variants/<id>/    # phase 2 — same layout as variants/
├── aggregated/               # stage 7 — per-task aggregated CSVs
│   └── <task_name>/
│       ├── run_summary.csv
│       └── ...
├── pipeline_summary.json     # stage 5 summary (status + rubric pass per variant)
└── variants_readme.md        # stage 8 — paper-agent README
```

`aggregated/` uses two extra-task layouts on input:

* **structured** — `extra_tasks/<task>/variants/<id>/task_bundle/program_*/analysis/*.csv`
  (produced by stage 6).
* **flat** — `extra_tasks/<task>/<variant_id>/*.csv` (manually added; CSVs
  are scanned directly and `evaluation_id`/`variant_id` columns are injected
  from the directory name).

---

## See also

* `configs/variants_config.yaml` — full default config.
* `docs/pipeline.md` and `docs/usage.md` — pipeline and `agsci investigate` invocation; `docs/install.md` — container setup.

# Harness contract for hand-written tasks

Facts from the AI-Scientist-v3 source. Paths are relative to the repo root. Re-check them when
the harness changes.

## Required files

`missing_task_scaffold_files` (`src/investigation/files.py`) requires:

| File | Notes |
| - | - |
| `task.md` | `# Goal` and `# Rubrics` sections, extracted case-insensitively at any heading level (`read_task_sections`). |
| `capture.py` | Copy the canonical file (`examples/circpack/capture.py`). It must define `OutputCapture`. |
| `evaluate.py` | The search evaluator. |
| `program_0/program_spec.toml` | Its declared entrypoint must exist. |

If `task_contract.json` exists, the task is a *contract task*: `evaluate_final.py`, a harness-issued
validation receipt, and harness-evaluated attempts become mandatory, and `correct.json` must hold
exactly `{"correct": bool}`. Never create it by hand.

Optional: `requirements.txt`, `.agentic_task_ignore`, `README.md`, `resources/`. Do not name a task
subdirectory `results`, `.git`, or `.venv`: containerized evaluation masks those names.

## `program_spec.toml` (schema 1)

```toml
schema_version = 1
code_root = "src"            # must be "src"
[entrypoint]
path = "program.py"          # relative to code_root, must stay inside it
language = "python"
type = "script"              # "script" or "module"
[copy]
exclude = ["**/__pycache__/**", "**/*.pyc", ".venv/**"]
include = []
[integration]
kind = "generic"
notes = ""
```

Agents create `program_1`, `program_2`, ... by copying the best program and editing files under
`src/`. Evaluation artifacts (`metrics.json`, `correct.json`, `rubric_status.json`, `stdout.log`,
`stderr.log`, ...) are not copied forward (`src/investigation/program_clone.py`). Any other file an
evaluator writes into its results dir may be copied into every later program.

## How `evaluate.py` is called

Every path runs:

```text
<python> evaluate.py --program_path <program_k>/src/<entrypoint> --results_dir <dir> [--num_replications N]
```

`--num_replications N` arrives when N > 1. Ablations pass it by default
(`ablations.execution.num_replications`, 5 in `configs/pipeline_config.yaml`). Either honor it
(the template runs N times and averages), or tell the user to set
`ablations.execution.num_replications=1` for expensive tasks.

Native harness-evaluated runs (`harness_evaluated_attempts=true`, `container.enabled=false`) first
evaluate the frozen `program_0` on the host to earn the incumbent score (event
`native_incumbent_evaluation_start/end`); budget one extra evaluation at campaign start. A baseline
that fails or is incorrect stops the run with `missing_validated_harness_incumbent`.

Ablations evaluate variants the same way: in the Docker sandbox in container mode (which sets
only the `NUM_REPLICATIONS` variable and does not pass the flag), or natively with
`container.enabled=false` (`src/variants_pipeline/native_eval.py`: the native investigation
environment, a private `TMPDIR`, the flag when N > 1, the sandbox's output limits). Native
ablations refuse contract tasks and cross-task variants.

The surroundings depend on who evaluates:

| | Agent-run (default for CPU tasks) | Harness-evaluated, container | Harness-evaluated, native |
| - | - | - | - |
| Enabled by | `harness_evaluated_attempts=false` | `harness_evaluated_attempts=true`, `container.enabled=true` | `harness_evaluated_attempts=true`, `container.enabled=false` |
| Runner | `src/investigation/evaluate_program.py` | `src/variants_pipeline/host_eval.py` sandbox | `src/investigation/harness_eval.py` direct |
| `--results_dir` | a separate output dir; summaries published into `program_k/` | `/results` tmpfs (144 MiB; total output ≤ 64 MiB, ≤ 20,000 entries) | `program_k/` itself |
| cwd | the task dir | inside the sandbox | the experiment dir |
| Environment | filtered allowlist (below), fresh `HOME`/`TMPDIR` | fixed minimal set (`PATH`, `VENV_DIR`, `NUM_REPLICATIONS`, `CUDA_VISIBLE_DEVICES`, ...); no `DATA_DIR`, no `TMPDIR` | the full parent environment |
| Filesystem | the host | read-only; task at `/task-source`, candidate at `/candidate`, venv at `/task-venv`; `/tmp` is a 16 GiB **noexec** tmpfs; no network | the host |
| Timeout | `--timeout-seconds` (default 21600) | `investigation.execution.harness_evaluation_timeout_seconds` (default **7200**) | same |

GPU tasks need harness evaluation (`gpu_evaluation=true` requires it). Consequences:

- **Data:** in container mode, only the task source is visible, so external data has to live
  inside the task (and is copied into every run), or the campaign runs natively.
- **JIT code:** a noexec `/tmp` and no compiler in the default image break Triton and
  `torch.compile` in container mode. Such tasks run natively.
- **Results:** keep logs small; the template keeps a 4 MiB tail in `stdout.log`.

### Agent-run environment

`_evaluation_environment` in `evaluate_program.py` keeps only: `PATH`, `LANG`, `LC_ALL`,
`LD_LIBRARY_PATH`, `TASK_SOURCE_DIR`, `DATA_DIR`, `AGENTIC_DATA_DIR`, `FINEWEB_DATA_DIR`, `VENV_DIR`,
`TERM`, `TZ`, `USER`, and the prefixes `CUDA_`, `NCCL_`, `TORCH_`, `OMP_`, `MKL_`, `NUMEXPR_`, `NVIDIA_`,
`ROCR_`, `XLA_`. It drops anything containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `AUTH`,
`CREDENTIAL`, or `PROXY`, plus `PYTHONPATH`, `PYTHONHOME`, and `VIRTUAL_ENV`. Custom variables never
arrive, so put settings in constants or flags.

### Rules that hold on every path

- Never write into the task directory. Use `--results_dir` and `tempfile`.
- Resolve task files from `TASK_SOURCE_DIR` when set, else from `Path(__file__).parent`.
- A caching path under `HOME` is cold every time: budget compile warm-up into the evaluation time.
- The harness stops evaluators with SIGINT (and the sandbox with container teardown): kill the
  candidate's process group on SIGINT and SIGTERM.

## Outputs

- `metrics.json`: a JSON object. `combined_score` must be a finite number, higher is better. Keep
  other metrics flat at the top level: rubric checks look them up by key.
- `correct.json`: `{"correct": true|false}`, plus an optional short `"error"` string (hand-written
  tasks only).
- Console output: search evaluation of hand-written tasks is not capped at 64 KiB, but drafting,
  final evaluation, and contract tasks are (`DEFAULT_MAX_OUTPUT_BYTES`), and the sandbox runner keeps
  1 MiB. Echo only progress and error lines.

## Dependencies

`requirements.txt` may contain only pinned `name==version` lines (extras allowed) and comments. No
`--index-url`, no URLs, no `-e`. In container mode, the harness builds the task `.venv` from it
inside the agent image (Python 3.12). Packages that need a custom index must have a default-PyPI
build, or the campaign must run natively with a prebuilt environment.

## Task snapshot

`agsci run` copies the whole task directory into `<run>/task` once per run, honoring
`.agentic_task_ignore` (gitignore-like lines; a leading `/` anchors to the task root, a trailing
`/` matches directories only). Ignore `.venv/` and caches. Keep large data (>~1 GiB) outside the
task when the campaign runs natively, referenced by an absolute path or `DATA_DIR`, with a hash
check in the evaluator.

## Rubrics

`target_rubrics` items have `id`, `description`, and `type` (`implementation`, `metric`, ...). A
description containing `<metric> <op> <number>` (`>`, `>=`, `<`, `<=`; or words like "exceeds",
"at least") is checked numerically against `metrics.json`. A structured form also works:
`{"metric": "...", "operator": ">=", "threshold": 1.0}`. `verification: {type: source_contains,
patterns: [...]}` checks candidate source tokens. At most 32 rubrics.

## Run options worth setting

| Override | When |
| - | - |
| `container.gpu_count=N` | GPU tasks: GPUs per worker. |
| `investigation.execution.gpu_evaluation=true` + `investigation.execution.harness_evaluated_attempts=true` | GPU tasks (gpu_evaluation requires harness evaluation). |
| `investigation.execution.harness_evaluation_timeout_seconds=S` | Evaluations longer than 2 h. |
| `investigation.execution.workers=W` | Parallel workers; needs `W x gpu_count` GPUs. |
| `investigation.execution.max_program_evals_per_attempt=K` | Lower it for expensive evaluations. |
| `ablations.execution.num_replications=R` | Expensive evaluations (default 5 replications per ablation variant). |
| `execution.phases="..."` | Hand-written tasks: final evaluation needs a contract, so drop it. Quote the comma list: `'execution.phases="investigation,ablations,posthoc,paper"'`, or `agsci doctor` rejects it. |

Run `agsci doctor <task_dir> [overrides]` and `agsci run <task_dir> [overrides] --dry-run` in the
campaign's environment before a real launch.

## Native runs (no Docker)

`container.enabled=false` runs agents and evaluators directly on the host, with the user's
permissions. Use it only on a dedicated machine or node. In this mode the `task_drafting`,
`final_evaluation`, and `revision` phases are rejected (`src/pipeline/config.py`), so use
`execution.phases="investigation,ablations,posthoc,paper"`. The host needs the agent CLI, Git, and
Node.js (see `docs/install.md`).

## Toolchain inside the agent image

The default image (`src/docker/agent-cli/Dockerfile`, `python:3.12-slim`) has no C compiler and
no CUDA toolkit. Triton JIT, `torch.compile`, and custom CUDA extensions need a compiler at
runtime, so GPU tasks that use them either run natively or need an image that adds
`build-essential` (plus an exec-capable cache directory). Check this with a smoke run inside the
same environment the campaign will use.

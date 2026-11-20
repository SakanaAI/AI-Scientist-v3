# Usage Guide

## What `agsci investigate` does

`agsci investigate` (`src/investigation/cli.py`) runs the outer loop:
- idea generation via `src/scripts/idea.sh`
- attempt execution via `src/scripts/run.sh`
- optional task drafting via `src/scripts/task.sh`
- optional paper generation via `src/scripts/paper.sh`

Default output root: `results/<task-name>/<agent>`.

## Run with Hydra config

`agsci investigate` uses [Hydra](https://hydra.cc/) to load `configs/config.yaml` and apply
`key=value` overrides from the CLI.

```bash
agsci investigate
```

Override any config field from the CLI:

```bash
agsci investigate \
  container.enabled=true \
  container.network=host \
  execution.max_idea_cycles=2 \
  agent.name=claude
```

Notes:
- Booleans use `true`/`false`
- Later overrides win
- For available keys, run `agsci investigate --help` or inspect `configs/config.yaml`

## Standalone investigation examples

`agsci investigate` is the investigation stage on its own: idea generation plus attempt
execution, without task drafting, ablations, post-hoc analysis, or the paper
orchestrator. Unlike `agsci run`, it runs natively unless `container.enabled=true`
is set.

### Simple (local)

```bash
agsci investigate \
  agent.name=claude \
  paths.task_dir=examples/circpack \
  execution.draft_task_if_missing=true \
  execution.max_idea_cycles=2 \
  execution.max_program_evals_per_attempt=2
```

### Docker

```bash
agsci investigate agent.name=codex container.enabled=true container.network=host container.env_file=.env
```

See [Container mode](#container-mode) below for `.env` setup and all container options.

### Parallel (local)

```bash
agsci investigate \
  agent.name=claude \
  paths.task_dir=examples/circpack \
  paths.results_dir=results/circpack_parallel \
  execution.workers=2 \
  execution.max_idea_cycles=4 \
  execution.max_program_evals_per_attempt=2
```

For a no-common-memory ablation, add `execution.isolate_worker_memory=true`.
This disables automatic memory and conclusion synchronization between workers;
it does not by itself prevent non-containerized workers from accessing sibling
directories on the host filesystem.

### Parallel + Docker

```bash
agsci investigate \
  container.enabled=true container.network=host container.env_file=.env \
  paths.results_dir=results/circpack_docker_parallel \
  execution.workers=2 \
  execution.max_idea_cycles=4 \
  execution.max_program_evals_per_attempt=2
```

### Container mode

To run any task inside a Docker container, set `container.enabled=true` and configure the container options:

```bash
agsci investigate \
  agent.name=claude \
  paths.task_dir=examples/circpack \
  container.enabled=true \
  container.image=agentic-scientist/agent-cli:latest \
  container.network=host \
  container.env_file=.env
```

Create a `.env` file at repo root with the API keys your agent needs:

```bash
# Required keys for common containerized agents
GEMINI_API_KEY=...
CURSOR_API_KEY=...
CLAUDE_CODE_OAUTH_TOKEN=...

AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-west-2
```

For `agent.profile=<name>`, keep `~/.codex/<name>.config.toml` (and its model
catalog, if any) under `container.auth_root` (defaults to `~`). The Docker
runner copies those files into the run-scoped container home alongside Codex
auth files. Containers only forward `OPENAI_API_KEY` and `CODEX_API_KEY` to
Codex, so a profile whose `model_provider` sets a different `env_key` cannot
authenticate inside the concept-figure, ablation, or post-hoc containers.

Container options:

| Key | Default | Description |
|-----|---------|-------------|
| `container.enabled` | `false` | Run inside a Docker container |
| `container.image` | `agentic-scientist/agent-cli:latest` | Docker image to use |
| `container.network` | `bridge` | Docker network mode (`bridge`, `host`) |
| `container.auth_root` | `~` | Auth credentials root directory |
| `container.env_file` | `""` | Path to `.env` file for the container |
| `container.gpu_count` | `0` | GPUs each worker's containers get; with `execution.workers>1` the run needs `workers x gpu_count` GPUs |

You can also combine container mode with parallel workers:

```bash
agsci investigate \
  agent.name=claude \
  paths.task_dir=examples/circpack \
  container.enabled=true \
  container.network=host \
  container.env_file=.env \
  execution.workers=4
```

With `container.gpu_count=N` and `execution.workers>1` the run needs `workers x N`
GPUs: `AGENT_CONTAINER_GPUS` must name exactly that many UUIDs, or the first that
many from `nvidia-smi` are used, and each worker receives its own `N`-GPU slice
(exported to the workers as `AGENT_CONTAINER_GPUS`, with the whole pool in
`AGENT_CONTAINER_GPU_POOL`).

### Arguments and help

Run `--help` to see all available configuration keys and their defaults:

```bash
agsci investigate --help
```

All options are passed as `key=value` overrides on the command line.

**Main arguments:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent.name` | string | `claude` | Agent backend (`claude`, `codex`, `cursor`, `opencode`, `gemini`, `pi`) |
| `agent.model` | string | `null` | Model override forwarded through Headless; `null` uses Headless's selected/configured default |
| `agent.profile` | string | `null` | Codex profile forwarded through Headless; requires `agent.name=codex` and explicit `agent.model`; the provider must authenticate with `OPENAI_API_KEY` or `CODEX_API_KEY` |
| `agent.reasoning_effort` | string | `null` | Global effort for every invocation: `low`, `medium`, `high`, or `xhigh`; unsupported when Gemini is selected directly or through `agent.pool` |
| `agent.pool` | list | `[]` | Optional harness pool sampled uniformly once per experiment cycle |
| `agent.models` | map | `{}` | Per-agent model overrides, e.g. `agent.models.codex=gpt-5.2` |
| `paths.task_dir` | string | `examples/circpack` | Task directory containing `task.md` and seed program |
| `paths.results_dir` | string | `null` | Results directory (auto-computed if omitted) |
| `execution.max_idea_cycles` | int | `20` | Max outer idea-generation cycles |
| `execution.workers` | int | `1` | Parallel workers (`1` = sequential) |
| `execution.resume` | bool | `false` | Resume from existing state |
| `execution.draft_task_if_missing` | bool | `false` | Draft `task.md` if not found |
| `execution.max_program_evals_per_attempt` | int | `4` | Max evaluations per attempt |
| `execution.run_timeout_seconds` | int | `21600` | Per-run timeout in seconds |
| `execution.idea_timeout_seconds` | int | `3600` | Per-idea-generation timeout in seconds |
| `execution.max_idea_generation_attempts` | int | `2` | Fresh idea-generation attempts before skipping a cycle |
| `execution.prune_every_n_cycles` | int | `0` | Reset ideation context every N cycles, keeping only the best program (0 to disable) |
| `logging.format` | string | `human` | Terminal output format (`human` or newline-delimited `json`) |
| `logging.verbose` | bool | `false` | Include diagnostic events, evidence, and artifact paths in human output |
| `logging.heartbeat_seconds` | int | `60` | Non-interactive progress heartbeat interval (`0` disables heartbeats) |
| `paper.enabled` | bool | `false` | Run paper generation after experiments |

See `configs/config.yaml` for all keys and defaults.

## Read experiment progress

Human-readable progress is the default. It emphasizes the experiment rather
than filesystem plumbing:

```text
Cycle 2/8
  Idea: Adaptive restart schedule
  Running experiment…
  Experiment finished  ·  4m 12s
  Score: 0.84  ▲ +0.07                 Best: 0.84
  Correctness: PASS
  Rubrics: 2/3 passed
    PASS  convergence
    PASS  reproducibility
    FAIL  runtime_budget
  Outcome: FAILED — rubric checks did not pass
```

Interactive terminals refresh the active stage in place. Redirected output and
CI logs receive a heartbeat every `logging.heartbeat_seconds` (default: 60).
Set it to `0` to disable heartbeats. Status colors are semantic and restrained;
set `NO_COLOR=1` for plain output.

Logging controls:

```bash
# Machine-readable JSON Lines on stdout
agsci investigate --log-format=json

# Human output with diagnostic events, rubric evidence, and artifact paths
agsci investigate --verbose

# Equivalent Hydra form
agsci investigate logging.format=json logging.heartbeat_seconds=30
```

All modes append the complete structured record to
`<results_dir>/events.jsonl`. The terminal view stays concise; use the journal
for post-run analysis. Parallel workers write separate journals inside their
worker result directories. The journal can contain diagnostic output and paths,
so it is created with owner-only permissions.

Terminal completion lines report total wall-clock runtime in `h/m/s` form, for
example `Pipeline finished · completed · 2h 12m 53s`. The pipeline journal's
`pipeline_end` event keeps both that human-readable `elapsed` value and numeric
`elapsed_seconds`. Stored checkpoints with exclusions use the concise timestamped
`[CHECKPOINT STORED]` label. Other Python warnings use one
`[LEVEL] module: message` format across the parent pipeline and child investigation CLI.

### API and Headless costs

Real runs also maintain an append-only cost ledger at
`<results_dir>/metrics/cost_events.jsonl` and an atomic aggregate at
`<results_dir>/metrics/cost_summary.json`. The summary groups activity by
phase, stage, source, and model. Stage-completion lines show output tokens, not
dollars: the stage delta and the running total, for example
`Tokens · Paper · 1.2M output · run 4.8M · Time 4h 28m`. Like Autoillustrate and
Claude Code's own footer, the count is response length only (output tokens),
not prompt input or cache reads. Calls that reported no token usage (image
renders, some subscription agents) are counted as `N unreported`.

Dollar figures never reach the terminal because subscription agents such as
Claude Code self-report API list prices that nobody is billed for. The ledger
still stores them: Headless invocations request normalized usage, provider-native
charges are stored as reported costs, and models.dev calculations and pinned
official API list prices are stored as estimated costs. Both contribute to
`total_cost_usd` while retaining separate provenance and subtotals. If pricing
is unavailable, the event is marked `usage_only` rather than `$0`.
The automated review currently pins gpt-5.6-terra's official standard-tier input,
cached-input, and output list rates; updating its model or rates requires an intentional code change.
CPU, GPU, and other infrastructure costs are not included.
Autoillustrate's self-reported Headless charges are included; its own estimated
image and vision charges remain usage-only.

This is operational telemetry, not a tamper-proof billing system. Native agents
and other processes running as the same OS user may be able to alter result
artifacts; reconcile financial decisions against the provider invoice.

These controls are shared by the rest of the pipeline:

```bash
# Parent phases plus child logging configuration
agsci run examples/circpack logging.format=human

# Ablations and post-hoc analysis phases
agsci run execution.phases=ablations paths.run_dir=results/... logging.heartbeat_seconds=30
agsci run execution.phases=posthoc paths.run_dir=results/... logging.verbose=true

# Paper gates; JSON mode keeps stdout machine-readable
agsci run execution.phases=paper paths.run_dir=results/... logging.format=json
```

The parent creates `pipeline_events.jsonl` in the shared run root. Each Python
child creates `events.jsonl` in its own output root. Paper
writing records preparation, strategy, draft, references, figures, validation,
evidence audit, review, repair, and finalization gates in a private journal.
Default human output omits path-heavy diagnostics; use `--verbose` when those
details are needed.

## Common workflows

Pipeline runs use one phase-oriented directory envelope:

```text
results/<task>/<agent>/<run_id>/
├── task_drafting/ # drafting logs and validation audit
├── task/          # immutable run-local task snapshot
├── investigation/  # ideas, attempts, winner, and optimizer memory
├── downstream/    # frozen fallback study program when search has no winner
├── ablations/     # optional ablation evidence
├── final_evaluation/         # optional sealed final evaluation
├── posthoc/       # optional post-hoc analyses
├── paper/         # stable manuscript, PDF, manifest, and optional AI reviews
└── revision/      # isolated review-driven manuscript revision attempts
```

Use `paths.run_dir=<existing-run>` for every phase-only or resume command.
An investigation resume reuses the idea-cycle budget recorded in the run directory;
pass `investigation.execution.max_idea_cycles` explicitly only to override it.
Phase-specific path overrides are available for advanced external layouts, but
normal runs do not require them. The supported layout is the nested structure
above; flat legacy investigation directories are intentionally rejected.

Paper generation requires only a completed investigation and the run-local task
snapshot. Missing or failed ablations, final evaluation, post-hoc analysis, or
review are recorded as degraded optional evidence. They do not make a completed
paper fail unless the task contract explicitly requires final evaluation.

When investigation ends as `completed_no_winner`, `execution.no_winner_policy=best_available`
(the default) continues selected downstream phases with the highest-scoring valid generated
candidate. If none exists, it uses the valid scored baseline. This does not promote the fallback
to investigation winner: the original best-program manifest and completion receipt stay intact.
The run stores a separate `.downstream_selection.json` and frozen `downstream/program` snapshot.
Set `execution.no_winner_policy=skip` to retain the previous downstream-skip behavior.

Fallback studies can produce negative-results papers. A verified, completed final evaluation
that reports `not_qualified` remains reportable with its qualification and validity limitations.
Missing, failed, invalid, or mismatched required final evidence still blocks publication.
Invalid optional final evidence remains excluded. If neither a valid generated program nor a valid baseline
exists, downstream phases remain skipped. Unscored attempts are missing evidence, not measured
performance losses.

Resume validates and reuses the frozen selection. It never selects another candidate from final
results. Changed source evidence or a stale snapshot is an error; do not remove the selection
manifest to bypass it. A crash between snapshot and manifest publication also stops for recovery
rather than replacing the frozen program. See [no-winner recovery](no_winner_studies.md) for a
resume example and focused checks.

Run the complete investigation, ablations, and paper pipeline with a run-local
task snapshot:

```bash
agsci run examples/circpack
```

Every pipeline key is listed by `agsci run --help`. The pipeline
composes each stage's own config into its phase, so any investigation, variants,
post-hoc, or paper.sh setting is reachable as `<phase>.<stage key>` without an
alias: `investigation.execution.<key>` and `investigation.task_drafting.<key>`
(investigation CLI), `ablations.execution.<key>` / `ablations.paths.<key>` (variants),
`posthoc.execution.<key>` and `posthoc.container.network` (post-hoc), and
`paper.<flag_name>` for paper.sh flags (`--paper-review-rounds` is
`paper.paper_review_rounds`). Only values that differ from the stage's own
default are forwarded, and keys the pipeline derives (paths, run id, agent,
container, logging, versioning) are rejected under a phase with the pipeline key
named. Investigation CLI presets are not part of the pipeline.

Harness-owned baseline and candidate evaluations hide GPUs by default. To use
GPUs for both evaluations, declare the GPU count, enable harness mode, and opt in:

```bash
agsci run examples/grpo \
  container.gpu_count=2 \
  investigation.execution.harness_evaluated_attempts=true \
  investigation.execution.gpu_evaluation=true
```

Before compute phases the pipeline takes the first `container.gpu_count` UUIDs that
`nvidia-smi` reports and exports them as `AGENT_CONTAINER_GPUS` (and as
`VARIANT_EVAL_SANDBOX_GPUS` for ablations when that is unset). Set
`AGENT_CONTAINER_GPUS='"device=GPU-uuid-1,GPU-uuid-2"'` yourself to pick the GPUs;
an explicit selector must name exactly `container.gpu_count` GPUs, and a count of
zero requires it unset. CPU mode sets `CUDA_VISIBLE_DEVICES` to an empty value;
GPU mode preserves the caller's value.

Append the optional OpenAI final-review phase explicitly:

```bash
agsci run examples/circpack \
  agent.name=claude \
  agent.model=claude-sonnet-5 \
  agent.reasoning_effort=high \
  investigation.execution.max_idea_cycles=1 \
  investigation.execution.max_idea_generation_attempts=1 \
  investigation.execution.max_program_evals_per_attempt=1 \
  investigation.execution.idea_timeout_seconds=600 \
  investigation.execution.run_timeout_seconds=600 \
  ablations.execution.num_replications=1 \
  ablations.execution.max_evals_per_variant=1 \
  paper.paper_format=iclr \
  execution.phases=task_drafting,investigation,ablations,final_evaluation,posthoc,paper,review
```

The `review` phase sends the complete final manuscript to OpenAI. It writes `ai_review.json` and
`ai_review.log`, updates `paper/latest_manifest.json`, and degrades without failing the pipeline if
the reviewer is unavailable. Retry only that phase with the existing `paths.run_dir` anchor.
`review.timeout_seconds` and `review.timeout_grace_seconds` bound
its managed subprocess.

Append `revision` after `review` to revise the manuscript against that review and create a
second assessment:

```bash
agsci run execution.phases=paper,review,revision \
  paths.run_dir=<existing-run> \
  paths.paper_dir=<existing-run>/paper
```

The `revision` phase uses the configured pipeline agent to address review weaknesses using only
existing run evidence. It does not run experiments, evaluations, or reference retrieval; it may
improve prose, claims, captions, tables, and figures derived from existing artifacts. The edited
paper must pass the normal paper validation and evidence audit before it replaces the stable
publication. Edit or validation failures retain the original paper. A failed second review retains
the validated revision and marks the optional phase degraded.

Publication preserves original-only audited inputs in
`paper/revisions/revision_01/before/ancillary/`, retaining their relative paths. This keeps old
reference-planning files, author notes, scripts, and displaced binary evidence out of the revised
paper's active audit inventory. The final publication still requires a current passing evidence
audit; publication failures leave the original paper in place.

After correcting a publication failure, rerun only `execution.phases=revision` with the same
`paths.run_dir`, paper directory, and original configuration. The pipeline revalidates saved
revision attempts and reuses a passing attempt before launching another edit. It then publishes
that revision and runs the second review. A saved passing audit alone does not bypass current
evidence validation.

The initial and final assessments are stored as `ai_review.before_revision.json` and
`ai_review.after_revision.json` (with matching logs). After a successful review, `ai_review.json`
and `ai_review.log` are latest-review compatibility copies; a failed review clears the JSON alias
instead of exposing a stale assessment. Revision attempts live under `<run_dir>/revision/`, and the
pre-revision paper is retained under `paper/revisions/revision_01/before/`. A revision-only resume
requires `paths.run_dir`, `paths.paper_dir`, and an existing successful initial review.

Successful revisions print the before/after assessment, including completed-run replays:

```text
Revision review · overall 4/10 → 3/10 (-1) · decision REJECT → REJECT
```

JSON logs carry the same comparison in `revision_result`: `before_overall_score`,
`after_overall_score`, `before_decision`, `after_decision`, and `overall_score_delta`.
Missing or invalid values display as `unavailable`; a delta requires both scores.

For end-to-end runs, `task.dir` is read-only input. The pipeline copies it to
`<run_dir>/task`, honors `.agentic_task_ignore`, and performs
task drafting there. Each run therefore keeps its scaffold with its own results;
resume and phase-only commands reuse the stored snapshot. Snapshot publication
works directly on local and network filesystems: when exclusive directory
rename is unavailable, matching interrupted copies are validated and resumed
in place before an explicit completion marker is published.

Resume an existing run:

```bash
agsci investigate \
  agent.name=claude \
  paths.task_dir=examples/circpack \
  paths.results_dir=results/circpack/claude \
  execution.resume=true
```

Custom run prompt + custom results dir:

```bash
agsci investigate \
  agent.name=codex \
  paths.task_dir=examples/circpack \
  paths.results_dir=/tmp/circpack_codex \
  paths.instruct=src/prompts/general.md
```

Enable paper generation:

```bash
agsci investigate \
  agent.name=claude \
  paths.task_dir=examples/circpack \
  paper.enabled=true \
  paper.min_refs=10 \
  paper.max_refs=20
```

Concept figures run Autoillustrate in `--mode full` (panel inspection plus
verify/repair loops) when the remaining paper deadline leaves at least one
hour; tighter budgets step down to `medium` (30 min) or `light` (5 min), or
skip the figure. The default `paper.timeout_seconds=7200` (2 hours) covers
drafting, the full-mode figure, and the repair and review rounds that follow;
a budget of 1800 only affords `light`.
`PAPER_CONCEPT_FIGURE_MODE=light|medium|full` caps the mode, and
`PAPER_CONCEPT_FIGURE_{FULL,MEDIUM,LIGHT}_MODE_MIN_SECONDS` tune the thresholds.
Autoillustrate's own documentation is `autoillustrate --help`, the README
shipped in its npm package, and the `docs/` folder of the
[upstream repository](https://github.com/SakanaAI/Autoillustrate#readme).

Enable the separate standalone paper workflow:

```bash
agsci investigate --standalone-paper
```

Equivalent explicit config form:

```bash
agsci investigate \
  paper.enabled=true \
  paper.profile=standalone \
  paper.two_stage=true
```

Fast smoke test:

```bash
agsci investigate \
  agent.name=gemini \
  paths.task_dir=examples/circpack \
  execution.max_idea_cycles=1 \
  execution.max_program_evals_per_attempt=1 \
  execution.run_timeout_seconds=600
```

Containerized parallel run:

```bash
agsci investigate \
  container.enabled=true container.network=host container.env_file=.env \
  paths.results_dir=results/circpack_docker_parallel \
  execution.workers=2 \
  execution.max_idea_cycles=4 \
  execution.max_program_evals_per_attempt=2
```

## Parallel execution details

- Set `execution.workers=N` to run `N` isolated investigation workers in parallel.
- With `container.gpu_count>0`, each worker gets its own slice of a `N x gpu_count` GPU pool.
- Worker state is stored under `worker_XX` directories.
- `execution.max_idea_cycles` is split across workers as evenly as possible.
- Each worker runs full `idea.sh` + `run.sh` loops.
- `execution.max_program_evals_per_attempt` applies per worker attempt.
- Root parallel dir contains merged histories and `parallel_summary.json`.
- Worker logs are written to `worker_XX/parallel_worker.log`.

Constraints with `execution.workers>1`:
- `execution.resume=true` is unsupported
- `execution.draft_task_if_missing=true` is unsupported
- `paper.enabled=true` is supported, but runs **after** all workers finish.

Budget rule of thumb:
- candidate eval upper bound is approximately
  `execution.workers * max_idea_cycles_per_worker * execution.max_program_evals_per_attempt`

## Key investigation CLI arguments

`agsci investigate` takes Hydra overrides (`key=value`) rather than long flags. Pass them as
separate CLI arguments:

```bash
agsci investigate container.enabled=true execution.max_idea_cycles=2 agent.name=codex
```

Notes:
- Booleans are `true`/`false`; numbers are plain integers (seconds where noted).
- Paths expand `~` and resolve relative to your current working directory (run from repo root for sanity).
- Unknown keys / typos fail fast (this is intentional).

- Agent (`agent.name`, `agent.model`, `agent.reasoning_effort`): chooses the backend and optionally forwards a model and global reasoning effort. Every supported backend runs through the pinned Headless runtime with `--allow yolo`; an unset model uses Headless's selected or configured default. Codex profiles require an explicit model so ambient Headless defaults cannot override profile intent.
  - Example: `agent.name=claude agent.model=claude-3-opus agent.reasoning_effort=high`.
  - Valid reasoning efforts: `low`, `medium`, `high`, `xhigh`. The value applies to task drafting, ideas, attempts, papers, variants, post-hoc analysis, and nested figure agents. Gemini does not support this option; configuration validation rejects Gemini directly or in `agent.pool` when an effort is set.
  - `agent.pool=[codex,claude]` samples one harness uniformly per experiment cycle and reuses it for idea generation plus the attempt.
  - `agent.models.<agent>=...` overrides the model for a specific sampled harness, e.g. `agent.models.codex=gpt-5.2`.
- Paths (`paths.*`): points the investigation CLI at your task and output roots.
  - `paths.task_dir` must contain `task.md` and `program_0/`.
  - If `paths.results_dir` is unset, it defaults to `results/<task-name>/<agent>/<run_id>` or `results/<task-name>/ensemble/<run_id>` when `agent.pool` is set.
  - `paths.global_memory_dir` defaults to the results dir.
  - `paths.instruct`/`paths.paper_instruct` override the prompt templates (defaults live under `src/prompts/`).
  - Reusable paper prompts include `src/prompts/paper.md` and `src/prompts/paper_standalone.md`.
- Run ID (`run_id`): controls the per-run directory suffix when `paths.results_dir` is unset; set it to make repeated runs deterministic (e.g. `run_id=smoke_01`).
- Execution (`execution.*`): loop sizing + parallelism.
  - `execution.max_idea_cycles` is the outer-loop budget.
  - `execution.max_program_evals_per_attempt` caps candidate variants per idea.
  - `execution.resume=true` continues from an existing `paths.results_dir` (only supported with `execution.workers=1`).
  - Pipeline task drafting is the separate `task_drafting` phase. The legacy investigation-only `execution.draft_task_if_missing=true` remains available for direct `agsci investigate` compatibility.
- Timeouts (`execution.run_timeout_*`, `paper.timeout_*`): hard time limits for each script stage, with a grace window before force-kill.
- Paper (`paper.*`): enables final paper generation and sets reference bounds.
  - `paper.profile=default` keeps the existing paper workflow.
  - `paper.profile=standalone` switches to the separate standalone-paper workflow with `src/prompts/paper_standalone.md` and the standalone writer skill.
  - `paper.two_stage=true` adds a framing pass that writes `paper_strategy.md` + `paper_strategy.json` before the final draft.
  - `--standalone-paper` is shorthand for `paper.enabled=true paper.profile=standalone paper.two_stage=true`.
  - With `execution.workers>1`, the paper runs after the parallel campaign completes.
  - `paper.ai_review=disabled|openai` controls the post-publication reviewer for `agsci investigate` runs and defaults to `disabled`. For the end-to-end orchestrator, explicitly append `review` to `execution.phases`. Both forms send the complete final manuscript to the OpenAI-backed standalone `src/aireview.py` reviewer. Direct `paper.sh`, paper-edit, and resume-finalization commands do not trigger it.
  - Review output is stored as `ai_review.json` and `ai_review.log` beside `paper.pdf`, with status and paths recorded under `ai_review` in `paper/latest_manifest.json`. Review failures warn and publish the paper normally.
- Container (`container.*`): runs agent calls in Docker.
  - `container.auth_root` is the host directory copied into the container home for credentials/config.
  - `container.env_file` is forwarded to Docker as an env-file (if unset/empty, the scripts will use repo `.env` when present).
  - `container.network=host` is often required on locked-down machines.

## Run checkpoints

Every run directory has a sibling git history at
`<run_parent>/.<run_name>.agentic_history`. Checkpoints are committed after each idea drafting,
after each program evaluation, and after each of the ablations, post-hoc, paper,
and review phases, with a final commit containing the full run directory.
The end-to-end pipeline uses `versioning.run_root` when set. Otherwise it uses the
selected phase's parent run directory, so paper-only and review-only resumes preserve
their existing bundle. An explicitly configured phase output outside that primary root
gets its own sibling history and is included in phase and final checkpoints.
Roots must be disjoint or nested under one stable checkpoint root. A selected output may
not contain the investigation directory, because its cycle-level sidecar would otherwise
sit inside the parent checkpoint. When investigation is selected, a configured
`versioning.run_root` must equal `paths.run_dir` or be disjoint from it;
unsafe overlapping layouts fail configuration validation. Nested sidecar metadata from
earlier standalone phase runs is always excluded from parent histories.

Read the history by pointing git at the sidecar (there is deliberately no `.git`
entry inside the run directory):

```bash
RUN=results/circpack/claude/20260819_084705
GIT_DIR="${RUN%/*}/.${RUN##*/}.agentic_history"
GIT="git --git-dir=$GIT_DIR --work-tree=$RUN"

$GIT log --oneline
$GIT tag -l | sort                      # checkpoint/0001-idea-c001, ... checkpoint/0015-final
$GIT show checkpoint/0004-evaluation-c002-a01
$GIT diff checkpoint/0003-idea-c002 checkpoint/0005-idea-c003
$GIT log --format='%(trailers:key=Checkpoint-Event,valueonly)'
```

The history is outside the run directory mounted into container agents.
`execution.prune_every_n_cycles` deliberately hides earlier ideas from the agent, and
an in-tree repository would hand back a complete copy of everything pruned. Native agents
with unrestricted host filesystem access remain subject to the host's own isolation policy.

Move or archive the run directory and its sibling sidecar together. If the run basename
changes, rename the sidecar to match the `.<run_name><git_dir_name>` convention before
resuming. Pre-release in-tree sidecars are ignored and never adopted; inspect or export
them offline instead of moving their agent-writable Git metadata into the sibling path.

**What is never committed.** Credential directories and files are excluded at every
depth, including `.agent_home`, `.ssh`, `.aws`, `.env*` secrets, private keys, and
common package/cloud credential files. Every changed blob is staged in an OS-private
quarantine outside the sidecar and scanned for provider tokens, private-key headers,
and concrete credential assignments. Only objects reachable from the verified tree
are promoted; normal cleanup discards the rest, and the next attempt reaps stale
quarantines left by an interrupted process. Excluded paths are recorded by count in
`Checkpoint-Skipped-Sensitive` and listed in the warning/event payload. Lock files
and nested `.git` directories are excluded too, and git skips streaming FIFOs.

**Final environment policy.** Checkpoints skip derived content — `task/.venv`,
`__pycache__`, tool caches, and build staging directories — including in the final
commit by default. Set `versioning.final_include_environments=true` only when the
larger, less portable environment snapshot is required.

**Size.** `versioning.max_file_bytes` defaults to `0`, meaning no cap: every file is
eligible regardless of its individual size. `versioning.max_total_bytes` defaults to
16 GiB; checkpoints over that aggregate included-file budget degrade before Git
staging. Set it to `0` to disable the aggregate cap. Runs occasionally produce
multi-gigabyte agent logs, and such a file would enter the history permanently. Set
`max_file_bytes` to skip individual large files.
An early work-tree check keeps obviously oversized files out of the temporary object
store; a second check measures every blob in the candidate Git tree. The conservative
first check can defer a concurrently shrinking file until the next checkpoint. The cap
still applies to final-checkpoint environments. Skipped files are removed from the
current tree and listed in the warning/event payload. When a commit is produced, its
`Checkpoint-Skipped-Oversize` trailer also records the skipped-file count.

**Parallel runs.** With `execution.workers > 1`, worker subtrees are not checkpointed
individually — git offers no retry on `index.lock`, so N concurrent writers would
corrupt the index. The parent commits the run root once every worker has joined, giving
one checkpoint per round instead of one per cycle.

A checkpoint never gates a run: git failures are logged, emitted as a `run_checkpoint`
event with `status=degraded`, and leave the run's exit code untouched. Set
`versioning.enabled=false` to turn checkpointing off entirely.

Restoring or branching from a checkpoint is not automated. The committed state still
carries absolute paths (for example `best_program_manifest.json`'s `program_dir`), so a
checked-out copy needs those rewritten before it can be resumed.

## Container/auth notes

- Container image includes `codex`, `claude`, `gemini`, `cursor-agent`, `opencode`, `pi`.
- In `container.enabled=true` mode, paper LaTeX compile can fall back to the Docker image when host `pdflatex`/`bibtex` are unavailable.
- Claude Bedrock: when AWS keys exist in the env file, launcher auto-sets `CLAUDE_CODE_USE_BEDROCK=1` unless already configured.
- Claude OAuth: set `CLAUDE_CODE_OAUTH_TOKEN` in `.env` for containerized OAuth usage.
- Include provider creds as needed in `.env` (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `CURSOR_API_KEY`, etc).
- Codex OAuth auth/config is copied from host via `container.auth_root`.
- Quoted `.env` values are sanitized before `docker --env-file` runs.
- Docker retry knobs:
  `AGENT_DOCKER_READY_RETRIES`,
  `AGENT_DOCKER_READY_SLEEP_SECONDS`,
  `AGENT_DOCKER_RUN_RETRIES`,
  `AGENT_DOCKER_RUN_RETRY_SLEEP_SECONDS`.

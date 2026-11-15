---
name: run-campaign
description: Launch, monitor, resume, and wrap up an AI-Scientist-v3 (`agsci run`) campaign on a validated task — locally, in tmux, or through Slurm. Interviews the user for the run-level decisions (agent and model, compute, budget, phases, paper inputs, external review), runs preflight checks, writes the launch script, and reports progress with a status summary command. Use when a user wants to start, check on, restart, or finish an agentic-scientist run. Building the task itself is the setup-task skill.
---

# run-campaign

Run a campaign on a task that already passed `setup-task` validation. You prepare and check the
launch, the user approves it, and you then monitor it and report. Starting a campaign costs real
money and compute, so nothing gets launched without an explicit go-ahead.

## Locate the harness

This skill lives in the AI-Scientist-v3 repository at `skills/run-campaign/`. Resolve the repo root
from the skill directory's real path (`realpath <skill_dir>/../..`); call it `$AGSCI`. Scripts,
run with `$AGSCI/.venv/bin/python`:

- `scripts/write_launch.py`: writes a Slurm batch script or a local shell script for `agsci run`,
  and submits it only when given `--submit`.
- `scripts/campaign_status.py`: one-screen summary of a run (state, phases, cycles, best score,
  top evaluations, cost, Slurm job).

## Decision points: ask the user

Ask in one batch, offering the defaults below. Never assume consent for the items marked (!).

| Part | Ask about | Default if the user has no preference |
| - | - | - |
| Agent (!) | agent CLI and model; this decides billing | `claude`, harness default model |
| Environment (!) | container (default) or native mode. Native runs agents with the user's own permissions | container if the GPU hosts can run Docker and the task needs no compiler; otherwise native, after asking |
| Compute | where it runs (local, tmux, Slurm partition), GPUs per worker, workers, node caps | 1 worker; GPUs from the task README |
| Budget | idea cycles, attempt and evaluation timeouts, evaluations per attempt, ablation replications, rough cost ceiling | `preset=paper` budgets (20 cycles); plain `agsci run` uses only 3. Ablations replicate 5x by default, so ask for expensive tasks |
| Phases | investigation only, or also ablations, post-hoc analysis and paper | `investigation,ablations,posthoc,paper` (`write_launch.py` always sets phases explicitly, because `preset=paper` would add `review` and `revision`) |
| Paper | venue/format, custom paper instructions (`paper.instruct`), stated contributions (`paper.contributions_file`), custom investigation prompt (`investigation.paths.instruct`) | harness defaults |
| External review (!) | the `review` phase sends the full manuscript to OpenAI; add it to `--phases` only after an explicit yes | off |
| Launch (!) | final go-ahead after seeing the exact script | — |
| Winner (!) | full-profile confirmation run; submitting results anywhere outside the run dir | ask each time |

The run-level overrides behind each decision are in `references/run-options.md`.

## Workflow

### 1. Check the task

Read the task's `README.md` (validation log, environment probe, recommended overrides). If there
isn't one, run `$AGSCI/skills/setup-task/scripts/check_task.py <task_dir>` and confirm with the
user that the baseline was evaluated end to end. If it wasn't, stop and suggest `setup-task`.

### 2. Interview

Ask for the decisions above. Include the cost drivers in the question: evaluation wall time ×
evaluations per attempt × cycles × workers gives the compute time, and agent tokens scale with
cycles.

### 3. Preflight in the target environment

Run these where the campaign will run (on the GPU host, e.g. inside a short `srun` allocation):

```bash
cd $AGSCI
.venv/bin/agsci doctor <task_dir> <overrides>
.venv/bin/agsci run <task_dir> <overrides> --dry-run
```

Fix every failure before writing the launch script. Typical problems: GPUs not visible, Docker not
reachable on GPU hosts, phases that native mode rejects, a native paper phase without Docker
(`write_launch.py` adds `container.paper_enabled=true`), missing credentials in `.env`, low disk
space under the results root. `write_launch.py` prints these two commands with the exact
overrides it generated; quote comma lists as `'execution.phases="a,b"'`, or `agsci doctor`
rejects them.

For GPU tasks, propose a pilot first: the same launch with `--phases investigation` and
`investigation.execution.max_idea_cycles=1`. It proves the evaluation path inside the real
sandbox (container or native) for the cost of one cycle. Run it only after the user agrees.

### 4. Write the launch script and get approval

```bash
$AGSCI/.venv/bin/python $AGSCI/skills/run-campaign/scripts/write_launch.py <task_dir> \
    --out <task_dir>/../runs/launch.sbatch --scheduler slurm --partition <p> --gpus 8 \
    --mode native --eval-timeout 5400 --results-root <dir> -- preset=paper agent.name=claude
```

Show the user the generated file and the estimated cost drivers. Submit only after an explicit
yes, and submit exactly the file they reviewed: `sbatch <file>`. (`--submit` writes and submits in
one step; use it only when the user approved the exact options beforehand.) For local runs use
`--scheduler local` and start the script with the printed `tmux` command, which pipes output to a
log so the pipeline writes journal heartbeats. Record the job ID and the run directory (printed
early in the job log) in the task's `README.md`.

### 5. Monitor

```bash
$AGSCI/.venv/bin/python $AGSCI/skills/run-campaign/scripts/campaign_status.py <run_dir or results_root> [--job <slurm_id>]
```

- Check every 30–60 min, or when the user asks. Report changes: new best score, phase
  transitions, failure streaks, stalls.
- `STALLED` means no pipeline or investigation journal event and no new evaluation for longer
  than `--stale-minutes` (default 30). Check the job (`squeue`, the job log) before concluding
  anything. A run started with stdout on a terminal writes no heartbeats, so long evaluations can
  look stalled; `UNKNOWN` means there is no activity record at all.
- To dig deeper: the tail of `pipeline_events.jsonl`, `investigation/events.jsonl`, the failing
  attempt's `wrapper_output_log` (path in `attempts_history.jsonl`), and `agsci inspect start`
  for the web UI.
- Never cancel jobs you didn't start. Cancel your own job only when the user asks, or when it is
  clearly broken and burning compute; then report what happened.

### 6. Resume

When a phase fails, the pipeline logs a "Rerun the original command with these overrides" hint
in the job log; prefer its overrides. Otherwise resume with the same command plus
`write_launch.py --resume <run_dir>`, which adds `paths.run_dir`, `run_id`, and
`investigation.execution.resume=true`. Keep the same agent, model, workers, and GPU allocation,
and drop phases that already finished from `--phases`. The pipeline recovers the idea-cycle
budget. Ask before resuming if the failure cause is unknown.

### 7. Wrap up

- Winner: `investigation/best_program_manifest.json` (`program_dir`, `combined_score`).
- Confirm the winner at the full profile with `setup-task`'s `check_task.py --program <winner>
  --profile full`, after asking (it can take hours).
- Report: final status, best vs baseline score, number of evaluations, cost
  (`metrics/cost_summary.json`; it covers agent tokens, not compute), paper path, and anything
  degraded.
- Submit results outside the run directory (PRs, leaderboards) only when the user asks.

## Several nodes

One `agsci run` uses one host's GPUs (`workers × container.gpu_count`). To use several nodes,
launch independent runs (one per node) with different `agent.model`s or seeds, give each a
distinct job name, and compare them with `campaign_status.py` on the results root.

## References

- `references/run-options.md`: overrides for each decision, native vs container, journal and
  state file locations.

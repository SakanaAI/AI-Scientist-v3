# Run options

All options are Hydra overrides to `agsci run <task_dir> ...`, run from the repo root. Check
`agsci run --help`, and the configs in `configs/`, when the harness changes.

## Decision → override

| Decision | Overrides |
| - | - |
| Agent and model | `agent.name=claude\|codex\|cursor\|opencode\|gemini\|pi`, `agent.model=<id>`, `agent.reasoning_effort=<level>` |
| Campaign budgets | `preset=paper` (20 cycles plus campaign timeouts), or explicitly `investigation.execution.max_idea_cycles=N` (plain `agsci run` default: 3) |
| Attempt time | `investigation.execution.run_timeout_seconds` (one agent attempt, default 21600) and `investigation.execution.idea_timeout_seconds` |
| Evaluations | `investigation.execution.max_program_evals_per_attempt=K`; GPU tasks: `investigation.execution.harness_evaluation_timeout_seconds=S` (default 7200); `ablations.execution.num_replications=R` (default 5) |
| GPUs | `container.gpu_count=N` (per worker), `investigation.execution.gpu_evaluation=true`, `investigation.execution.harness_evaluated_attempts=true` |
| Parallelism | `investigation.execution.workers=W` (needs `W × gpu_count` visible GPUs) |
| Phases | `'execution.phases="investigation,ablations,posthoc,paper"'` (quoted for `agsci doctor`; add `task_drafting`, `final_evaluation`, `revision` in container mode, `review` only with consent). `preset=paper` alone selects all of them, including `review` |
| Paper | `paper.paper_format=iclr\|alife`, `paper.instruct=<md>`, `paper.contributions_file=<md>`; investigation prompt: `investigation.paths.instruct=<md>` |
| External review | add `review` to `execution.phases` (sends the manuscript to OpenAI, so it needs consent) |
| Output location | `paths.results_root=<dir>` (default `results/` in the repo); runs land in `<root>/<task>/<agent>/<run_id>/` |
| Resume | the same overrides plus `paths.run_dir=<run_dir>`, `run_id=<run_dir basename>`, `investigation.execution.resume=true`, minus finished phases (or the pipeline's logged rerun hint) |
| Scratch space | environment variable `AGENTIC_TEMP_ROOT=<existing dir>` (see `docs/temporary_storage.md`) |

## Container vs native

| | Container (default for `agsci run`) | Native (`container.enabled=false`) |
| - | - | - |
| Needs | Docker on the host that runs the run; built image | agent CLI, Git, Node.js on the host |
| Isolation | agents and evaluators sandboxed | agents run with the user's own permissions |
| Phases | all | not `task_drafting`, `final_evaluation`, or `revision` |
| GPUs | `AGENT_CONTAINER_GPUS` or the first visible ones | the GPUs Slurm or `CUDA_VISIBLE_DEVICES` expose |
| Toolchain | image has no C compiler or CUDA toolkit | host toolchain (Triton, `torch.compile` work) |

## Where state lives

```text
<run_dir>/
├── pipeline_events.jsonl              # pipeline_start, phase_start/end, heartbeat, cost_summary, pipeline_end
├── metrics/cost_summary.json          # agent token cost (no compute cost)
├── task/                              # run-local task snapshot
├── investigation/
│   ├── outer_state.json               # current_cycle, best_combined_score, best_correct, failure_streak
│   ├── best_program_manifest.json     # winner: program_dir, combined_score, source_attempt_id
│   ├── attempts_history.jsonl         # one line per attempt: attempt_status, failure_category, wrapper_output_log
│   ├── ideas_history.jsonl
│   ├── events.jsonl                   # investigation journal
│   ├── worker_<i>/round_<k>/...       # same files per worker and round when workers > 1
│   └── **/program_<k>/{metrics,correct,rubric_status}.json
├── ablations/  posthoc/  paper/       # paper/*.pdf when the paper phase completes
```

The journals can contain paths and diagnostic output: share summaries, not raw journals.

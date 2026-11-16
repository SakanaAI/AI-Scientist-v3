# Worked example: overlay task for a GPU training benchmark

A hypothetical public benchmark, `<org>/<benchmark>`. Entries change a training script and a model
file. A fixed scoring script loads the final weights and reports a validation loss. One evaluation
takes a full 8-GPU node for about 3 hours.

## Interview answers

| Question | Answer |
| - | - |
| Objective | Validation loss from the benchmark's `eval/score.py`, lower is better, so `combined_score = -val_loss`. |
| Editable surface | `train/train.py`, `train/model.py`, and new helper modules in `train/`. |
| Layout | Overlay onto the upstream repo, pinned at the commit of the current best entry. |
| Upstream and data | `<org>/<benchmark>@<commit>` plus its license. The data comes from the repo's preparation script, which needs network access, so prepare it once outside the task and pin the hashes. |
| Rules | The benchmark's legality rules (e.g. forbidden techniques, at most one pass over the data, fixed seed, no checkpoint selection on validation), plus the interface `eval/score.py` relies on. |
| Leakage | Training must never see the validation split; the evaluator scores in a separate process. |
| Cost | About 3 h per full run on 8 GPUs, so search needs a cheaper proxy profile. |
| Baseline | The current best entry (strong start) or the plain reference script (more room to explore). Ask the user. |
| Score variance | With a fixed seed, measure node-to-node spread with 2 runs of the search profile. |

## Layout

```text
<task>/
├── task.md
├── README.md                 # interview answers, environment probe, validation log, run overrides
├── evaluate.py
├── capture.py
├── requirements.txt          # pinned name==version lines from default PyPI
├── .agentic_task_ignore      # .venv/, __pycache__/
├── resources/
│   ├── UPSTREAM              # url, commit, license
│   └── upstream/             # pinned checkout without .git, data/, logs/
└── program_0/
    ├── program_spec.toml     # entrypoint path = "train/train.py"
    └── src/train/{train.py, model.py}
```

The data stays outside the task. The evaluator reads it from an allowlisted variable
(`DATA_DIR`) or a constant absolute path, and checks the pinned hashes before every run. Under
containerized harness evaluation those paths are not mounted; see
`harness-contract.md#harness-evaluated-attempts`.

## Evaluator flow

1. **Static rules** over `src/**/*.py`: reject the forbidden techniques that have a recognizable
   code signature, and any mention of the validation file. These checks can be dodged, so a
   judgment rubric backs them up.
2. **Assemble the work tree**: copy `resources/upstream` to scratch, then copy `src/` over it.
   Make the training data dir contain only the training split.
3. **Train**: `torchrun --standalone --nproc_per_node=<visible GPUs> train/train.py --data-dir data
   --output out <profile flags>`, with a program timeout of about 2x the measured baseline. Keep
   the full log in scratch, and copy a bounded tail into `--results_dir`.
4. **Runtime rules**: `out/final.pt` exists; the training log or result file shows the allowed
   amount of data was used; wall time is under the cap.
5. **Score** in a fresh process with the full data dir:
   `python eval/score.py out/final.pt --model train/model.py --data-dir <full data>`.
6. Write `metrics.json` (`combined_score`, `val_loss`, `train_seconds`) and `correct.json`.

**Baseline adjustment.** If the upstream training script evaluates on the validation split at the
end of training, hiding that split breaks it. Give `program_0` the smallest guard that skips in-run
validation when the file is absent, and say in `task.md` that the evaluator computes the score.

## Profiles

| Profile | Training budget | Expected time | Use |
| - | - | - | - |
| `smoke` | a few dozen steps | minutes (dominated by compile) | `check_task.py --profile smoke` |
| `search` (default) | a fixed fraction of the data, with the full schedule compressed to it | about 25 min | agents |
| `full` | the whole benchmark budget | about 3 h | confirming a winner; the reportable number |

Before launching, check that `search` ranks a few known variants the way `full` does. Earlier
recorded runs are the cheapest calibration set.

## task.md (excerpt)

```markdown
# Goal
- Train the model in `train/` to the lowest validation loss under the benchmark rules. You may
  change the training method and the architecture.
- `combined_score = -val_loss`, computed by the evaluator with the official scorer on your saved
  final weights. Training never sees the validation split.
- Each evaluation trains on a fixed fraction of the data on 8 GPUs and takes about 25 minutes. Set
  your shell timeout to at least 3600000 ms.
- Implement ONE variant per evaluation; no sweeps.

# Constraints
- <benchmark rules, each stated so it can be checked>
- `model.py` must keep the interface `eval/score.py` uses.

# Rubrics
- `beats_baseline`: combined_score > <search-profile baseline minus measured spread>
- `rules_spirit`: <the rule that static checks cannot fully enforce, stated concretely>
- `interface_kept`: `model.py` keeps the evaluator interface and loads strictly.
```

## Handoff

The task README recommends these overrides; launching is the `run-campaign` skill's job:

```text
container.gpu_count=8 investigation.execution.harness_evaluated_attempts=true
investigation.execution.gpu_evaluation=true investigation.execution.harness_evaluation_timeout_seconds=5400
investigation.execution.max_program_evals_per_attempt=2 ablations.execution.num_replications=1
```

If the training code compiles Triton or other JIT kernels, the default agent image (which has no C
compiler) cannot run it. Either add `build-essential` to the image, or run natively on the GPU
host (`container.enabled=false`, phases `investigation,ablations,posthoc,paper`).

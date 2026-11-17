# Task drafting diagnostics

Drafting audits live under the run's `task_drafting/_task_drafting/draft_*/_task_validation/`.
Qualification reports identify the execution stage and scaffold fingerprint that failed.

`output_limit_exceeded` means a runtime subprocess exceeded the 64 KiB combined console
stdout/stderr limit. This is distinct from the candidate RPC wire limit and is not a launch
failure. Optional `output_evidence` records the cap, bytes observed before termination, and
bounded, sanitized per-stream excerpts for debugging and scaffold repair. Excerpts may be
omitted or redacted; observed bytes are a lower bound, not the process's eventual output size.
Do not infer the noisy component from the category alone or raise the limit before diagnosis.
Independent alignment judges receive only allowlisted metadata and evidence digests.

### Worker failures and missing stderr

Runtime timeouts retain bounded, sanitized stderr as repair-only `output_evidence`, including
commands returning just after their deadline. Timeout evidence does not override the primary
authored/infrastructure classification or unresolved-container cleanup. Logging failures cannot
skip teardown. Truncated captures omit incomplete lines rather than retaining credential fragments.

Scaffold prompts share executable required-baseline and subprocess diagnostic recipes. Required
baseline failures carry their own category rather than inheriting `TimeoutError`. Schedulers
capture the original operation before cleanup; diagnostic sink failures preserve required resets.
The subprocess reader continuously drains stderr, retains at most 4 KiB of complete lines, and
has bounded shutdown. Process deadlines, termination, and reaping remain caller-owned.

Detailed worker excerpts are allowed only in search drafting smoke or qualification, through
the existing host sanitization path. Ordinary search and sealed final evaluation retain fixed
categories only. Diagnostics never enter result artifacts, scientific evidence, or judge text.

Already-discarded stderr cannot be reconstructed. For an opaque generated evaluator, agentic
repair first fixes the suppressing wrapper, then the normal qualification loop obtains evidence.
Useful diagnostic edits remain available in bounded rejected-repair history after rollback.
Neither smoke nor production-entry establishes full production success. These changes apply
to newly rendered prompts; they do not retrofit live containers or historical artifacts.

Focused checks: `tests/runtime/test_bounded_process.py`,
`tests/task_drafting/test_scaffold_diagnostic_contract.py`, and
`tests/task_drafting/test_repair_context.py`. The last file runs a real CPU child through captured
stderr, runtime reporting, qualification, repair rendering/history, and judge-text exclusion.

Focused diagnostic checks:

```bash
./.venv/bin/pytest -q tests/runtime/test_bounded_process.py
```

## Resuming a failed draft

`investigation.task_drafting.resume_from` seeds a new draft from an earlier one in the same
`task_drafting/_task_drafting/` directory. Pass `latest` or the earlier `draft_*` path, together
with the same `paths.run_dir`.

- The earlier draft's staged scaffold is copied into the new draft, without its runtime (`.venv`,
  runtime manifest, bytecode). A complete scaffold skips the scaffold agent. An incomplete one,
  for example after a network drop, is continued with the missing-artifact feedback.
- The accepted task design is reused only when its `design_receipt.json` still matches the
  design and review bytes (robust workflow). Otherwise design runs again, and a retained
  scaffold goes back to the scaffold agent to match the new design. The receipt is an unkeyed
  digest in agent-writable draft output, so a reused design is trusted as far as it is within a
  single draft.
- Contract, smoke, alignment and qualification gates all run again on the new draft. Earlier
  verdicts are not trusted.
- Resume is refused when the task source (`task.md`, other source files, `resources/`) changed
  after the earlier draft started: small files by content digest, larger ones by modification
  time. Start a fresh draft in that case.
- `latest` picks the newest earlier `draft_*` that staged a scaffold.

## Evaluator reproducibility

Draft search smoke evaluates the unchanged baseline twice. A different `combined_score` (beyond
float summation noise) or correctness flag fails as `search_nondeterministic` (native) or
`runtime_baseline_validation` (containerized), with the message
`evaluator is not deterministic: ...`, and goes to scaffold repair. The usual cause is an
evaluator that draws a seed or secret per invocation (`secrets.*`, `os.urandom`, `uuid.uuid4`,
wall-clock time). A secret that hides seeds from the candidate must be a fixed, task-owned value.
Production evaluations are not replayed.

## Qualification recovery

Initial drafting checks contract and reduced search/final smoke, then independent alignment,
then full production-search qualification. After a production failure, each provisional repair
must pass contract/smoke and full production qualification **before** returning to alignment.
The judge receives a success attestation bound to the repaired scaffold's fingerprint and
publication receipt. Earlier failures remain historical debugging evidence.

Both alignment and qualification must accept the same scaffold before publication. An unchanged
successful qualification is reused after alignment. Ordinary contract, smoke, production, or
alignment rejection retains the provisional edits for the next repair, with acceptance cleared
before editing. The original recovery snapshot stays fixed across the repair chain. Unsafe edits,
terminal agent/infrastructure failures, interruption, or exhausted repair budgets restore that
snapshot; failed restoration retains private recovery state. Authored repairs and infrastructure
retries retain their existing separate limits. A missing or stale receipt blocks acceptance.

Initial alignment repairs retain their snapshot until alignment accepts. Qualification repairs
start from that aligned scaffold and remain provisional until production and alignment both pass.
Retained source is working material, never permission to publish or to reuse stale gate results.

Canonical alignment findings are retained outside repair transactions, with report hashes and
reviewed scaffold fingerprints. Both repair agents and independent reviewers receive this bounded
history after prompt expansion. Repair agents also receive rejected source diffs; independent
reviewers do not. Every retained finding must be checked against the current tree, since later
edits or terminal rollback can remove earlier fixes. Historical passes never authorize acceptance.
When context is too large,
diagnostic findings take priority over source diffs, and truncation is explicit.

Terminal repair checkpoints distinguish production execution from alignment acceptance, show
used/remaining repair attempts and failed alignment dimensions, and name the last failed gate
when the budget is exhausted.
"Production search passed; independent alignment rejected" means the repaired evaluator ran but
did not preserve the scientific contract; increasing execution time alone will not resolve it.

Reduced final smoke does not qualify the full final workload. Recovery does not rerun sealed
final evaluation or alter existing run artifacts.

Production-entry checks stop after the first completed reset, then verify container cleanup.
Evaluator CID files remain linked until the launch worker finishes; final container removal is
checked again before unlinking them. A live worker or failed removal retains recovery metadata
and remains an infrastructure failure. Cleanup diagnostics retain the observed entry status and
category when teardown overrides it, plus bounded broker operation and errno (never raw paths).
For example, `operation=runtime_directory_remove errno=39 (ENOTEMPTY)` identifies retained entries;
it does not establish that the evaluator itself failed. Do not delete unrecognized entries blindly.

GPU quarantine records mean container or launch-process shutdown could not be verified. Host-only
CID, socket, or directory cleanup failures remain infrastructure failures without quarantining a
verified allocation. A broker launch thread that remains alive, or an unresolved replacement
container, still requires quarantine; recovery metadata remains available.

Runtime reports and smoke gates preserve GPU allocation categories through trusted host logging
metadata. `gpu_allocation_quarantined` and `gpu_allocation_invalid` are terminal;
`gpu_allocation_busy` and generic `runtime_bootstrap` retain bounded infrastructure retries.
Full production-search qualification also stops immediately for the two terminal GPU categories;
its other infrastructure failures retain their existing retry budget.
Diagnostic text alone never determines this category. Existing quarantine markers require verified
operator recovery: retries and new jobs cannot clear them. Never delete or replace GPU lock files
to bypass quarantine, since another process may still hold the original lock inode.

Before design generation, drafting briefly checks every GPU in an explicit `AGENT_CONTAINER_GPUS`
allocation. Quarantine or an invalid allocation stops with exit 4 before agent work; busy GPUs
use the existing smoke infrastructure retry budget. Without a GPU selector, drafting skips this check.
The check releases its lease immediately and does not reserve future capacity or clear quarantine;
runtime gates still acquire and validate their own leases.

Quarantined preflight failures identify the stable GPU UUID and retained origin when available.
A later run can be given the GPU while this application marker persists. See [GPU quarantine recovery](gpu_quarantine.md)
for automatic recovery of abandoned owned runtimes, read-only inspection, and legacy recovery after
verified container and launcher shutdown. Preflight allows a separate bounded ownership cleanup
window while retaining its short wait for busy leases.

```bash
./.venv/bin/pytest -q tests/task_drafting/test_qualification_recovery.py
./.venv/bin/pytest -q tests/e2e/test_agent_execution_pipeline.py -k 'qualification or alignment_context'
./.venv/bin/pytest -q tests/task_drafting/test_task_drafting_production_canary.py tests/final_evaluation/test_candidate_broker.py
./.venv/bin/pytest -q tests/gpu/test_gpu_cleanup_classification.py
./.venv/bin/pytest -q tests/gpu/test_container_gpus.py tests/task_drafting/test_task_drafting_progress.py
./.venv/bin/pytest -q tests/task_drafting/test_task_alignment.py tests/task_drafting/test_repair_context.py
./.venv/bin/pytest -q tests/task_drafting/test_qualification_recovery.py
bash -n src/scripts/task.sh src/scripts/lib/task_qualification.sh
```

## Scaffold scheduling guidance

Drafting and repair prompts include an executable scheduling recipe. Its one global episode
list includes scored cases, declared diagnostic branches, and any permitted replay. Every branch
reserves time for later resets, bounded transition overhead, and final result writing. Failed
episodes retain their predeclared finite worst-case contributions. Qualification repair prompts
reuse the same shared repair policy rather than maintaining a separate set of scheduling rules.

Shared feasibility recipes additionally budget search and final productive work together, reserve
matched allowances for later controls, and validate exact reset acknowledgements. Canonical
productive baseline-time floors exclude startup and cleanup; failure penalty timing remains
separate from actual evidence duration. Feedback interventions apply only to tasks whose canonical
requirements authorize them, and do not add a performance threshold.

The recipe is guidance for self-contained evaluators, not a broker guarantee that arbitrary
generated code honors deadlines. Contract, smoke, independent review, and production qualification
remain required. CPU tests execute the actual recipes under slow replies and timeout scenarios:

```bash
./.venv/bin/pytest -q tests/task_drafting/test_scaffold_recipes.py
```

## Search outcomes and publication dependencies

A completed search with a scored but rejected retained starter is `completed_no_winner`, not a
malformed run. By default, downstream phases select the best valid scored generated candidate,
or a valid baseline if no such candidate exists, for a separately identified fallback study.
Incorrect or unscored starters alone cannot authorize that study. The investigation outcome and
winner manifest remain unchanged; `execution.no_winner_policy=skip` disables fallback studies.
See [no-winner studies](no_winner_studies.md).

Once drafting publishes a contract, phase dependencies are checked again: a selected paper phase
that requires an omitted final evaluation stops before investigation, including when
`fail_fast=false`. The task is retained; the harness does not silently add phases. For fallback
studies, completed and verified final non-qualification can support a negative-results paper;
failed or unverifiable required final evaluations remain blocking.

Rubric `assessment` distinguishes `passed`, `failed`, and `not_assessed`; legacy `pass` and
`overall_pass` booleans retain their meaning. Historical rubric files without `assessment` remain
supported. Search eligibility thresholds belong in task.md, never task-name branches in the harness.

## Resource, boundary, and repair checks

Shared authoring recipes cover thread-count-aware CPU budgets (separate from wall deadlines),
required-baseline failure accounting, complete JSON/base64 wire sizes, content-disjoint examples,
and fixed-weight failure aggregation. These are tested templates, not a proof that a newly authored
evaluator uses them correctly. Role-local GPU allocation and the supported broker API remain
mandatory; invented telemetry accessors are not a substitute for harness-owned measurements.

RPC budgets count the complete operation trace, including transfer start/chunk/commit calls at
the maximum supported state size. Two exports and two imports of 32 chunks take 134 calls.
Candidate-reported state-change flags cannot grant or deny scientific validity; evaluator-owned
state comparisons need task-appropriate behavioral controls to support learning claims.

Required baselines need both polling during candidate work and deadline-bounded final collection:
a worker may fail after the last successful poll. Expected result errors retain the failed episode's
contribution and remaining resets; secondary cleanup errors preserve the bounded primary category.
Episode reserves sum sequential candidate shutdown and baseline cancellation waits, including
terminate and kill/reap, plus bookkeeping. Final output writing retains its own reserve.

Focused CPU checks execute draft, repair, and rendered qualification-repair recipes:

```bash
./.venv/bin/pytest -q tests/task_drafting/test_scaffold_recipes.py
```

Rejected qualification repairs retain bounded source-only diffs and diagnostics before retry or
terminal rollback. Subsequent repair agents see the original, rejected, and current fingerprints;
independent judges do not receive this untrusted source history. Ordinary retries continue from
the current draft. Attempt limits and all acceptance gates still apply.

Focused retained-chain regressions: `tests/task_drafting/test_qualification_recovery.py`
exercises real source edits, snapshot reuse,
acceptance invalidation, terminal rollback, and interruption without live agents or GPUs.

Typed host `ESTALE`, `EDQUOT`, and `ENOSPC` failures are infrastructure failures, not authored defects;
qualification uses its infrastructure retry path without spending an authored repair attempt.
Secondary cleanup failures retain the primary diagnostic. If event persistence fails, a bounded
metadata-only storage warning falls back to stderr. No automatic data deletion is performed.

## Agent log storage failures

The bounded agent logger mirrors its prefix/tail into a private node-local
`$AGENTIC_TEMP_DIR/agentic-log-*/agent.log` (directory 0700, file 0600; legacy callers use `/tmp`).
See [temporary storage](temporary_storage.md) for scratch selection.
Each sink has the same configured
byte cap. A failed shared sink is disabled; the current agent stream continues draining into
the local sink. If both fail, the logger still drains, reports diagnostic loss, and fails.
Successful calls remove only their own temporary spool. Failed calls retain it for manual recovery;
it can contain sensitive agent output and is node-local, not a durable archive.

Metadata on stderr distinguishes agent exit from logger exit and reports the fallback location
at the first sink failure. `local_spool_status` is `in_progress`, `retained`, `partial`, or
`unavailable`; never assume an announced in-progress log is complete after cancellation.
No exception text, agent output, or arbitrary shared path is copied into these warnings.

Drafting stops before accepting retained artifacts or starting another draft/review/repair call
after a logger failure. Drafting, investigation idea generation, and attempts report
`agent_log_persistence_failed` through a private host channel; controller exit 87 stops further
ideas, promotion, and downstream phases even with `execution.fail_fast=false`. Parallel workers
stop scheduling and cancel siblings. Agent exit codes and public output alone cannot forge this
terminal result. There is no automatic logger recovery retry or restart of the completed call.

This stop policy covers drafting and investigation. Other users of the shared logger receive the
bounded fallback but retain their existing downstream policies (including non-gating figures and
paper audits that can accept completed artifacts). Qualification subprocess infrastructure retries
are separate and unchanged. This does not repair an NFS mount or prove storage is healthy;
inspect the underlying storage and retained diagnostics before manually relaunching.

```bash
./.venv/bin/pytest -q tests/runtime/test_temp_storage.py tests/investigation/test_log_failures.py
./.venv/bin/pytest -q tests/investigation/test_log_failures.py tests/investigation/test_parallel_pruning.py
```

Focused checks (no GPU experiments):

```bash
./.venv/bin/pytest -q tests/pipeline/test_pipeline_cli_dryrun.py tests/pipeline/test_pipeline_phases.py
./.venv/bin/pytest -q tests/investigation/test_harness_eval.py tests/runtime/test_bounded_process.py
./.venv/bin/pytest -q tests/task_drafting/test_scaffold_recipes.py
./.venv/bin/pytest -q tests/task_drafting/test_scaffold_recipes.py tests/task_drafting/test_repair_context.py
./.venv/bin/pytest -q tests/task_drafting/test_task_drafting_publication.py tests/task_drafting/test_qualification_recovery.py
```

## Dependency failures discovered during repair

Repair agents start only after their private validation service is listening and has published
its readiness marker. Gate setup or startup failure returns infrastructure status 4 before
writing the repair prompt or launching an agent. Smoke, alignment, and qualification repairs
restore provisional edits before propagating the failure; alignment does not retry it or consume
an authored repair attempt. Post-agent cleanup failures also fail the repair instead of accepting
its result. See [temporary storage](temporary_storage.md) for the short IPC socket path and
private diagnostic locations.

Every smoke-repair chain captures rollback state before its first agent starts. When an in-session
trusted gate discovers `runtime_dependencies`, it can authorize removal of unused direct pins
after checking the current scaffold against the original repair scope and failing fingerprint.
The authorization is stored outside the agent workspace and bound to the repair session, snapshot,
and rollback transaction. Gate summaries and agent output are not authorization records.

The agent must wait for explicit harness authorization before removing dependencies newly found
to be unavailable. Existing repairs initially opened for `runtime_dependencies` retain their
removal permission. Additions, version changes, and scientific-contract changes remain forbidden.
After removal, an in-session requirements/contract mismatch is provisional: the outer harness
synchronizes declarations and reruns validation. Follow-up repairs retain the transaction; only
full acceptance commits it. Failure, interruption, or exhausted budget restores the original
scaffold. Authorized transitions emit the repair session and `runtime_dependencies` category.

```bash
./.venv/bin/pytest -q tests/task_drafting/test_task_repair_scope.py tests/task_drafting/test_smoke_repair_transitions.py
./.venv/bin/pytest -q tests/task_drafting/test_task_repair_scope.py tests/task_drafting/test_task_gate.py tests/task_drafting/test_qualification_recovery.py
./.venv/bin/pytest -q tests/task_drafting/test_smoke_repair_transitions.py tests/task_drafting/test_task_gate.py
./.venv/bin/pytest -q tests/task_drafting/test_smoke_repair_transitions.py tests/runtime/test_container_auth.py
```

These tests use temporary scaffolds and simulated execution; they do not rerun recorded experiments.

## Native candidate environments

Drafting, smoke repair, and qualification repair share an executable environment allowlist in
`src/prompts/shared/candidate_environment.md`. Native candidate subprocesses retain the selected
task interpreter and explicit device/thread runtime settings; they do not inherit evaluator
credentials, broker controls, task paths, Python import overrides, or user-home caches. Trusted
broker proxies keep their existing in-process dispatch and environment.

Generated evaluators must use explicit environment names rather than filtering reserved harness
prefixes. The capability checker continues rejecting reserved namespace references in executable
code, including references used only to exclude variables. The recipe avoids that rejection
without weakening the guard.

```bash
./.venv/bin/pytest -q tests/task_drafting/test_scaffold_recipes.py
```

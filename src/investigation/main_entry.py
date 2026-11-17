"""Main orchestration flow invoked by the wrapper entry point."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.orchestrator import state as state_store

from . import campaign_loop
from .agent_selection import resolve_agent_model
from .config_loader import write_hydra_snapshot
from .config import InvestigationConfig, config_to_flat_dict, should_use_harness_instruction
from .constants import (
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_CONTAINER_IMAGE_BUILD_HINT,
    DEFAULT_CONTAINER_IMAGE_BUILD_HINT_HOST_NET,
)
from .files import (
    find_best_program_in_experiment,
    missing_task_scaffold_files,
    read_json,
    resolve_path,
    utc_now,
)
from .harness_incumbent import (
    reconcile_harness_incumbent,
    runtime_baseline_correct,
    runtime_baseline_score,
    seed_matches_validated_baseline,
)
from .preparation_progress import preparation_progress
from .program_clone import clone_program_dir
from .program_spec import ProgramSpecError, entrypoint_relpath_from_spec
from .run_checkpoint import CheckpointResult, CheckpointSettings, write_checkpoint
from .task_setup import run_task_setup
from .task_runtime import (
    TaskRuntimePreparationReport,
    TaskRuntimePreparationResult,
    prepare_task_runtime_report,
    runtime_preparation_report_from_result,
)
from .task_contract import TaskContractError, load_task_contract
from .task_failure import AGENT_LOG_FAILURE, INFRASTRUCTURE_EXIT_CODE
from .models import (
    IdeaGenerationResult,
    PaperGenerationResult,
    RunAttemptResult,
    TaskDraftResult,
)

logger = logging.getLogger(__name__)


def _freeze_seed_program(seed_program_dir: Path, results_dir: Path) -> Path:
    """Copy a run's seed beneath its agent-writable investigation workspace."""
    seed_program_dir = resolve_path(seed_program_dir)
    results_dir = resolve_path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root = Path(
        tempfile.mkdtemp(prefix=".baseline-seed-", dir=results_dir)
    )
    frozen_seed = snapshot_root / "program_0"
    try:
        clone_program_dir(src_dir=seed_program_dir, dst_dir=frozen_seed)
    except Exception:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise
    return resolve_path(frozen_seed)


def _bootstrap_native_incumbent(
    cfg: Any, task_dir: Path, seed_program_dir: Path, log_event_fn: Callable[..., None]
) -> bool:
    """Earn a native run's incumbent: evaluate the frozen seed as native attempts are evaluated."""
    from .harness_eval import bootstrap_native_incumbent

    task_python = task_dir / ".venv" / "bin" / "python"
    python_bin = str(task_python) if os.access(task_python, os.X_OK) else sys.executable
    log_path = seed_program_dir / "harness_evaluate_stdout.log"
    log_event_fn(
        "native_incumbent_evaluation_start", program_dir=str(seed_program_dir), log=str(log_path)
    )
    try:
        returncode = bootstrap_native_incumbent(
            program_dir=seed_program_dir,
            task_dir=task_dir,
            python_bin=python_bin,
            timeout_seconds=cfg.execution.harness_evaluation_timeout_seconds,
            environment=os.environ,
            gpu_evaluation=cfg.execution.gpu_evaluation,
        )
    except SystemExit as exc:
        log_event_fn(
            "native_incumbent_evaluation_end", returncode=None, error=str(exc), log=str(log_path)
        )
        return False
    log_event_fn("native_incumbent_evaluation_end", returncode=returncode, log=str(log_path))
    return returncode == 0


def _configure_contracted_agent_isolation(task_dir: Path, results_dir: Path) -> None:
    """Apply contract seals after drafting and forbid host-root Docker access."""
    contract_path = task_dir / "task_contract.json"
    if not contract_path.is_file():
        return
    try:
        contract = load_task_contract(contract_path)
    except TaskContractError as exc:
        raise ValueError(f"invalid task contract: {exc}") from exc
    empty_dir = results_dir / ".task-seal-empty"
    empty_dir.mkdir(parents=True, exist_ok=True)
    if any(empty_dir.iterdir()):
        raise ValueError(f"task seal directory is not empty: {empty_dir}")
    sealed_files = [
        contract.evaluators.final, "task_contract.json", ".agentic_task_validation.json",
    ]
    if (task_dir / "manual_validation.json").is_file():
        sealed_files.append("manual_validation.json")
    os.environ.update(
        AGENT_CONTAINER_MOUNT_DOCKER_SOCK="0",
        AGENT_TASK_SEALED_FILES="\n".join(sealed_files),
        AGENT_TASK_SEALED_DIRS=contract.resources.final_root,
        AGENT_TASK_SEAL_EMPTY_DIR=str(empty_dir),
    )


LogEventFn = Callable[..., None]
LogStepStartFn = Callable[..., float]
LogStepEndFn = Callable[..., None]
RunTaskDraftingFn = Callable[..., TaskDraftResult]
RunParallelCampaignFn = Callable[..., int]
RunIdeaGenerationFn = Callable[..., IdeaGenerationResult]
RunSingleAttemptFn = Callable[..., RunAttemptResult]
RunPaperGenerationFn = Callable[..., PaperGenerationResult]
RunTaskSetupFn = Callable[[Path, dict[str, object], int], bool]
PrepareTaskRuntimeFn = Callable[..., TaskRuntimePreparationReport | bool]
SetActiveContainerSessionFn = Callable[[Optional[str]], None]
BuildContainerSessionNameFn = Callable[..., str]
StopContainerSessionFn = Callable[[str], bool]
ProbeDockerAvailableFn = Callable[[], tuple[bool, str]]
DockerImageExistsFn = Callable[[str], tuple[bool, str]]
ReconcileIdeaLocalMemoryFn = Callable[..., None]
FormatElapsedFn = Callable[[float], str]
DockerBinaryLookupFn = Callable[[str], Optional[str]]
DefaultResultsPathFn = Callable[[Path, str], Path]
DefaultInstructionPathFn = Callable[[Path], Path]
DefaultIdeaInstructionPathFn = Callable[[], Path]
DefaultDraftInstructionPathFn = Callable[[], Path]
DefaultPaperInstructionPathFn = Callable[[], Path]
NormalizeAgentResultsDirFn = Callable[[Path, str], Path]


def _enable_contracted_harness_mode(
    cfg: InvestigationConfig,
    task_dir: Path,
    log_event_fn: LogEventFn,
) -> bool:
    """Require host-owned evaluation whenever a task contract is present."""
    if not (task_dir / "task_contract.json").is_file():
        return False
    if not cfg.execution.harness_evaluated_attempts:
        cfg.execution.harness_evaluated_attempts = True
        log_event_fn(
            "config_adjusted",
            reason="contracted_task_requires_harness_evaluation",
            setting="execution.harness_evaluated_attempts",
            value=True,
        )
    return True


def _task_dir_requires_docker(task_dir: Path) -> bool:
    """Detect task scaffolds that shell out to docker during evaluation."""
    candidates = (
        task_dir / "docker_eval.py",
        task_dir / "docker_eval_helpers.py",
    )
    return any(path.is_file() for path in candidates)


def _checkpoint_settings(cfg: InvestigationConfig) -> CheckpointSettings:
    return CheckpointSettings(
        enabled=cfg.versioning.enabled,
        git_dir_name=cfg.versioning.git_dir_name,
        max_file_bytes=cfg.versioning.max_file_bytes,
        max_total_bytes=cfg.versioning.max_total_bytes,
        timeout_seconds=cfg.versioning.timeout_seconds,
        tag_checkpoints=cfg.versioning.tag_checkpoints,
        final_include_environments=cfg.versioning.final_include_environments,
    )


def _checkpoint_run_root(cfg: InvestigationConfig, results_dir: Path) -> Path | None:
    """Resolve the run root this process may checkpoint, if any.

    Parallel workers are launched with ``paths.results_dir`` pointing at a
    ``worker_NN/round_NNN`` subtree of the run root, so gating on equality here
    makes them no-op rather than having N processes contend on one git index.
    """
    if not cfg.versioning.enabled:
        return None
    if cfg.versioning.run_root is None:
        return resolve_path(str(results_dir))
    run_root = resolve_path(str(cfg.versioning.run_root))
    resolved_results = resolve_path(str(results_dir))
    if resolved_results in {run_root, run_root / "investigation"}:
        return run_root
    return None


def _make_checkpoint_fn(
    cfg: InvestigationConfig, results_dir: Path, log_event_fn: LogEventFn
) -> Callable[..., None]:
    """Build the campaign-loop checkpoint hook, or a no-op when out of scope."""
    run_root = _checkpoint_run_root(cfg, results_dir)
    if run_root is None:
        return lambda **_kwargs: None
    settings = _checkpoint_settings(cfg)

    def checkpoint(**kwargs: Any) -> None:
        result = write_checkpoint(
            run_root=run_root,
            settings=settings,
            run_id=cfg.run_id,
            **kwargs,
        )
        _log_checkpoint(result, log_event_fn)

    return checkpoint


def _log_checkpoint(result: CheckpointResult, log_event_fn: LogEventFn) -> None:
    """Report a checkpoint outcome. Never influences control flow."""
    if result.status == "degraded":
        logger.warning("Checkpoint '%s' degraded: %s", result.label, result.error)
    elif result.status == "committed":
        logger.info("Checkpoint '%s' committed as %s", result.label, result.tag or result.commit)
    if (
        result.status in {"committed", "degraded"}
        or result.skipped_sensitive
        or result.skipped_oversize
    ):
        log_event_fn(
            "run_checkpoint",
            label=result.label,
            status=result.status,
            commit=result.commit,
            tag=result.tag,
            files=result.files,
            skipped_oversize=list(result.skipped_oversize),
            skipped_sensitive=list(result.skipped_sensitive),
            error=result.error,
        )


def run_main_flow(
    *,
    cfg: InvestigationConfig,
    main_started_at: float,
    set_active_container_session_fn: SetActiveContainerSessionFn,
    build_container_session_name_fn: BuildContainerSessionNameFn,
    stop_container_session_fn: StopContainerSessionFn,
    probe_docker_available_fn: ProbeDockerAvailableFn,
    docker_image_exists_fn: DockerImageExistsFn,
    docker_binary_lookup_fn: DockerBinaryLookupFn,
    default_results_path_fn: DefaultResultsPathFn,
    default_instruction_path_fn: DefaultInstructionPathFn,
    default_idea_instruction_path_fn: DefaultIdeaInstructionPathFn,
    default_draft_instruction_path_fn: DefaultDraftInstructionPathFn,
    default_paper_instruction_path_fn: DefaultPaperInstructionPathFn,
    normalize_agent_results_dir_fn: NormalizeAgentResultsDirFn,
    log_event_fn: LogEventFn,
    log_step_start_fn: LogStepStartFn,
    log_step_end_fn: LogStepEndFn,
    run_task_drafting_fn: RunTaskDraftingFn,
    run_parallel_campaign_fn: RunParallelCampaignFn,
    run_idea_generation_fn: RunIdeaGenerationFn,
    run_single_attempt_fn: RunSingleAttemptFn,
    run_paper_generation_fn: RunPaperGenerationFn,
    run_task_setup_fn: RunTaskSetupFn = run_task_setup,
    prepare_task_runtime_fn: PrepareTaskRuntimeFn = prepare_task_runtime_report,
    pin_container_image_fn: Callable[[str, str], str] | None = None,
    reconcile_legacy_idea_local_memory_fn: ReconcileIdeaLocalMemoryFn,
    format_elapsed_fn: FormatElapsedFn,
) -> int:
    """Execute the wrapper main control flow and return an exit code."""
    task_dir = resolve_path(str(cfg.paths.task_dir))
    _enable_contracted_harness_mode(cfg, task_dir, log_event_fn)
    results_dir = (
        resolve_path(cfg.paths.results_dir)
        if cfg.paths.results_dir
        else default_results_path_fn(task_dir, cfg.agent.name)
    )
    global_memory_dir_raw = cfg.paths.global_memory_dir
    global_memory_dir = (
        resolve_path(global_memory_dir_raw)
        if global_memory_dir_raw
        else results_dir
    )
    shared_memory_dir_raw = cfg.paths.shared_memory_dir
    shared_memory_dir: Path | None = (
        resolve_path(shared_memory_dir_raw)
        if shared_memory_dir_raw
        else None
    )
    conclusions_dir_raw = cfg.paths.conclusions_dir
    conclusions_dir: Path | None = (
        resolve_path(conclusions_dir_raw)
        if conclusions_dir_raw
        else None
    )
    if should_use_harness_instruction(cfg):
        instruct_path = resolve_path(Path("src/prompts/harness_edit.md"))
    elif cfg.paths.instruct:
        instruct_path = resolve_path(cfg.paths.instruct)
    else:
        instruct_path = default_instruction_path_fn(task_dir)
    idea_instruct_path = default_idea_instruction_path_fn()
    draft_instruct_path = default_draft_instruction_path_fn()
    paper_instruct_path = (
        resolve_path(cfg.paths.paper_instruct)
        if cfg.paths.paper_instruct
        else default_paper_instruction_path_fn()
    )
    paper_contributions_path = (
        resolve_path(cfg.paths.paper_contributions)
        if cfg.paths.paper_contributions
        else None
    )
    containerize = cfg.container.enabled
    container_image = cfg.container.image
    container_network = cfg.container.network
    container_auth_root = resolve_path(cfg.container.auth_root)
    container_env_file_raw = cfg.container.env_file.strip() if cfg.container.env_file else ""
    container_env_file = (
        resolve_path(container_env_file_raw) if container_env_file_raw else None
    )
    agent_model = resolve_agent_model(cfg.agent, cfg.agent.name)
    agent_profile = (
        str(cfg.agent.profile).strip()
        if isinstance(cfg.agent.profile, str) and cfg.agent.profile.strip()
        else None
    )
    reasoning_effort = cfg.agent.reasoning_effort
    if (
        containerize
        and not (task_dir / "task_contract.json").is_file()
        and _task_dir_requires_docker(task_dir)
    ):
        os.environ.setdefault("AGENT_CONTAINER_MOUNT_DOCKER_SOCK", "1")

    if not task_dir.is_dir():
        log_event_fn(
            "config_error",
            reason="missing_task_dir",
            path=task_dir,
            hint="check paths.task_dir",
        )
        return 2
    if not instruct_path.is_file():
        log_event_fn(
            "config_error",
            reason="missing_instruction_file",
            path=instruct_path,
            hint="check paths.instruct or src/prompts/general.md",
        )
        return 2
    if not idea_instruct_path.is_file():
        log_event_fn(
            "config_error",
            reason="missing_idea_instruction_file",
            path=idea_instruct_path,
            hint="restore src/prompts/idea.md",
        )
        return 2
    if cfg.paper.enabled and not paper_instruct_path.is_file():
        log_event_fn(
            "config_error",
            reason="missing_paper_instruction_file",
            path=paper_instruct_path,
            hint="check paths.paper_instruct or src/prompts/paper.md",
        )
        return 2
    if (
        cfg.paper.enabled
        and paper_contributions_path is not None
        and not paper_contributions_path.is_file()
    ):
        log_event_fn(
            "config_error",
            reason="missing_paper_contributions_file",
            path=paper_contributions_path,
            hint="check paths.paper_contributions",
        )
        return 2
    if containerize and docker_binary_lookup_fn("docker") is None:
        log_event_fn(
            "config_error",
            reason="missing_docker_binary",
            hint="install docker or disable container.enabled",
        )
        return 2
    if containerize:
        docker_ok, docker_detail = probe_docker_available_fn()
        if not docker_ok:
            log_event_fn(
                "config_error",
                reason="docker_unavailable",
                detail=docker_detail,
                hint=(
                    "run `docker info`; ensure docker daemon is running and your user can "
                    "access /var/run/docker.sock (e.g., add user to docker group and relogin)"
                ),
            )
            return 2
    if containerize and not container_auth_root.is_dir():
        log_event_fn(
            "config_error",
            reason="missing_container_auth_root",
            path=container_auth_root,
            hint="set container.auth_root to an existing directory",
        )
        return 2
    if containerize and container_image == DEFAULT_CONTAINER_IMAGE:
        image_ok, image_detail = docker_image_exists_fn(container_image)
        if not image_ok:
            log_event_fn(
                "config_error",
                reason="missing_default_container_image",
                image=container_image,
                detail=image_detail,
                hint=(
                    f"build local image with `{DEFAULT_CONTAINER_IMAGE_BUILD_HINT}` "
                    f"(if bridge egress is restricted, use `{DEFAULT_CONTAINER_IMAGE_BUILD_HINT_HOST_NET}`) "
                    "or set container.image to an existing image"
                ),
            )
            return 2
    if containerize and pin_container_image_fn is not None:
        # Sealed evaluations compare the task runtime manifest against the image id, so
        # the run must address the agent image by its immutable id, not a tag.
        try:
            container_image = pin_container_image_fn(container_image, str(results_dir))
        except ValueError as exc:
            log_event_fn(
                "config_error",
                reason="container_image_unresolved",
                image=container_image,
                detail=str(exc),
                hint="build the image or set container.image to an existing image",
            )
            return 2
        cfg.container.image = container_image
    if (
        containerize
        and container_env_file is not None
        and not container_env_file.is_file()
    ):
        log_event_fn(
            "config_error",
            reason="missing_container_env_file",
            path=container_env_file,
            hint="set container.env_file to an existing file",
        )
        return 2

    task_setup_args: dict[str, object] = {
        "agent": cfg.agent.name,
        "run_id": cfg.run_id,
        "task_dir": str(task_dir),
        "results_dir": str(results_dir),
        "global_memory_dir": str(global_memory_dir),
        "containerize": containerize,
    }
    if agent_model is not None:
        task_setup_args["agent_model"] = agent_model
    if agent_profile is not None:
        task_setup_args["agent_profile"] = agent_profile
    if reasoning_effort is not None:
        task_setup_args["reasoning_effort"] = reasoning_effort
    if shared_memory_dir is not None:
        task_setup_args["shared_memory_dir"] = str(shared_memory_dir)

    if not run_task_setup_fn(task_dir, task_setup_args, cfg.execution.run_timeout_seconds):
        log_event_fn(
            "task_setup_failed",
            task_dir=task_dir,
            timeout_seconds=cfg.execution.run_timeout_seconds,
        )
        return 2

    use_parallel_campaign = cfg.execution.workers > 1 or (
        cfg.execution.workers == 1 and cfg.execution.prune_every_n_cycles > 0
    )

    container_session_name: str | None = None
    container_mount_root: Path | None = None
    use_reusable_container_session = (
        containerize
        and cfg.execution.workers == 1
        and not use_parallel_campaign
        and not cfg.agent.pool
    )
    if use_reusable_container_session:
        container_mount_root = results_dir
        container_session_name = build_container_session_name_fn(
            agent=cfg.agent.name,
            results_dir=results_dir,
        )
        # Always start from a clean session container for this worker.
        stop_container_session_fn(container_session_name)
        set_active_container_session_fn(container_session_name)

    missing_scaffold = missing_task_scaffold_files(
        task_dir,
        require_contract_v1=(
            cfg.execution.task_only and (task_dir / "task_contract.json").exists()
        ),
    )
    drafted_scaffold = False
    if cfg.execution.draft_task_if_missing and missing_scaffold:
        task_md_path = task_dir / "task.md"
        if not task_md_path.is_file():
            log_event_fn(
                "config_error",
                reason="missing_task_md_for_draft",
                path=task_md_path,
                hint="create task.md before enabling execution.draft_task_if_missing",
            )
            return 2
        if not draft_instruct_path.is_file():
            log_event_fn(
                "config_error",
                reason="missing_draft_instruction_file",
                path=draft_instruct_path,
                hint="restore src/prompts/draft_task.md",
            )
            return 2

        results_dir.mkdir(parents=True, exist_ok=True)
        write_hydra_snapshot(cfg, results_dir)
        draft_dir = (
            results_dir
            / "_task_drafting"
            / f"draft_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
        )
        missing_scaffold_display = [str(path) for path in missing_scaffold]
        draft_started_at = log_step_start_fn(
            "task_draft",
            missing_scaffold_count=len(missing_scaffold_display),
            missing_scaffold=missing_scaffold_display,
            output_dir=draft_dir,
        )
        draft_result = run_task_drafting_fn(
            agent=cfg.agent.name,
            agent_model=agent_model,
            agent_profile=agent_profile,
            reasoning_effort=reasoning_effort,
            task_dir=task_dir,
            output_dir=draft_dir,
            instruct_path=draft_instruct_path,
            timeout_seconds=cfg.execution.run_timeout_seconds,
            timeout_grace_seconds=cfg.execution.run_timeout_grace_seconds,
            workflow=cfg.task_drafting.workflow,
            design_max_repairs=cfg.task_drafting.design_max_repairs,
            contract_max_repairs=cfg.task_drafting.contract_max_repairs,
            smoke_max_repairs=cfg.task_drafting.smoke_max_repairs,
            smoke_max_infrastructure_attempts=(
                cfg.task_drafting.smoke_max_infrastructure_attempts
            ),
            alignment_max_repairs=cfg.task_drafting.alignment_max_repairs,
            qualification_max_repairs=cfg.task_drafting.qualification_max_repairs,
            agent_transient_max_attempts=cfg.task_drafting.agent_transient_max_attempts,
            qualification_max_infrastructure_attempts=(
                cfg.task_drafting.qualification_max_infrastructure_attempts
            ),
            production_entry_timeout_seconds=(
                cfg.task_drafting.production_entry_timeout_seconds
            ),
            max_gate_checks_per_repair=cfg.task_drafting.max_gate_checks_per_repair,
            draft_max_attempts=cfg.task_drafting.draft_max_attempts,
            evaluator_smoke_timeout_seconds=cfg.task_drafting.evaluator_smoke_timeout_seconds,
            resume_from=cfg.task_drafting.resume_from,
            max_idea_cycles=cfg.execution.max_idea_cycles,
            max_program_evals_per_attempt=cfg.execution.max_program_evals_per_attempt,
            max_runtime_bytes=cfg.task_drafting.max_runtime_bytes,
            containerize=containerize,
            container_image=container_image,
            container_network=container_network,
            container_auth_root=container_auth_root,
            container_env_file=container_env_file,
            container_session_name=None,
            container_mount_root=None,
        )
        if draft_result.infrastructure_failure_category == AGENT_LOG_FAILURE:
            try:
                log_event_fn(
                    "task_draft_failed", exit=INFRASTRUCTURE_EXIT_CODE,
                    failure_category=AGENT_LOG_FAILURE, failure_scope="infrastructure",
                )
            except Exception:
                pass
            return INFRASTRUCTURE_EXIT_CODE
        draft_dir.mkdir(parents=True, exist_ok=True)
        draft_log_path = draft_dir / "wrapper_output.log"
        state_store.write_private_text(draft_log_path, draft_result.output)
        log_step_end_fn(
            "task_draft",
            draft_started_at,
            exit=draft_result.exit_code,
            draft_dir=draft_dir,
            log_path=draft_log_path,
        )
        if draft_result.exit_code != 0:
            log_event_fn(
                "task_draft_failed",
                exit=draft_result.exit_code,
                log_path=draft_log_path,
                tail_chars=1200,
                output_tail=draft_result.output[-1200:],
            )
            if draft_result.exit_code in {3, 4, 75, 130}:
                return draft_result.exit_code
            if draft_result.exit_code == 124:
                return 4
            return 2

        missing_scaffold = missing_task_scaffold_files(
            task_dir, require_contract_v1=cfg.execution.task_only
        )
        if missing_scaffold:
            log_event_fn(
                "task_draft_incomplete",
                missing_scaffold_count=len(missing_scaffold),
                missing_scaffold=[str(path) for path in missing_scaffold],
            )
            return 2
        log_event_fn("task_draft_complete", draft_dir=draft_dir)
        drafted_scaffold = True

    if _enable_contracted_harness_mode(cfg, task_dir, log_event_fn):
        if should_use_harness_instruction(cfg):
            instruct_path = resolve_path(Path("src/prompts/harness_edit.md"))

    missing_scaffold = missing_task_scaffold_files(
        task_dir,
        require_contract_v1=(
            cfg.execution.task_only and (task_dir / "task_contract.json").exists()
        ),
    )
    if missing_scaffold:
        log_event_fn(
            "config_error",
            reason="missing_task_scaffold_files",
            missing_scaffold=[str(path) for path in missing_scaffold],
            hint="use execution.draft_task_if_missing=true to auto-draft",
        )
        return 2

    if (
        cfg.execution.harness_evaluated_attempts
        and not cfg.container.enabled
        and cfg.execution.workers > 1
    ):
        log_event_fn(
            "config_error",
            reason="native_parallel_harness_evaluation_unsupported",
            hint=(
                "native harness-evaluated runs support one worker: parallel workers would each "
                "re-evaluate the baseline and share GPUs; set execution.workers=1 or enable "
                "containers"
            ),
        )
        return 2

    if cfg.execution.gpu_evaluation and not cfg.execution.harness_evaluated_attempts:
        log_event_fn(
            "config_error",
            reason="gpu_evaluation_requires_harness_evaluation",
            hint="set execution.harness_evaluated_attempts=true",
        )
        return 2

    if (task_dir / "task_contract.json").is_file() and not containerize:
        log_event_fn(
            "config_error",
            reason="task_contract_requires_container_isolation",
            path=task_dir / "task_contract.json",
        )
        return 2
    try:
        _configure_contracted_agent_isolation(task_dir, results_dir)
    except ValueError as exc:
        log_event_fn("config_error", reason="invalid_task_contract_isolation", error=str(exc))
        return 2

    # A task-only run on a pre-authored scaffold prepares no runtime: the pipeline discards
    # the drafting runtime and the next phase builds and validates the same one.
    should_prepare_task_runtime = (
        cfg.execution.draft_task_if_missing or containerize
    ) and (drafted_scaffold or not cfg.execution.task_only)
    if should_prepare_task_runtime:
        runtime_gpu_evaluation = cfg.execution.gpu_evaluation
        contract_path = task_dir / "task_contract.json"
        if contract_path.is_file():
            try:
                runtime_gpu_evaluation = (
                    load_task_contract(contract_path).compute.search.accelerator == "gpu"
                )
            except TaskContractError as exc:
                log_event_fn("config_error", reason="invalid_task_contract", error=str(exc))
                return 2
        runtime_started_at = time.monotonic()
        with preparation_progress():
            runtime_report: TaskRuntimePreparationReport | bool
            if (task_dir / ".agentic_task_qualification.json").is_file():
                from src.task_drafting.qualification import activate_qualified_task_runtime

                runtime_report = activate_qualified_task_runtime(
                    task_dir,
                    container_image=container_image,
                    gpu_evaluation=runtime_gpu_evaluation,
                    timeout_seconds=cfg.execution.run_timeout_seconds,
                )
                if runtime_report.failure_category in {
                    "qualified_runtime_cache_unavailable",
                    "qualified_runtime_revalidation_required",
                }:
                    log_event_fn(
                        "task_runtime_cache_miss",
                        task_dir=task_dir,
                        diagnostic=runtime_report.diagnostic,
                    )
                    runtime_report = prepare_task_runtime_fn(
                        task_dir,
                        cfg.execution.run_timeout_seconds,
                        containerize=containerize,
                        container_image=container_image,
                        container_network=container_network,
                        gpu_evaluation=runtime_gpu_evaluation,
                    )
            else:
                runtime_report = prepare_task_runtime_fn(
                    task_dir,
                    cfg.execution.run_timeout_seconds,
                    containerize=containerize,
                    container_image=container_image,
                    container_network=container_network,
                    gpu_evaluation=runtime_gpu_evaluation,
                )
        if isinstance(runtime_report, bool):
            runtime_report = runtime_preparation_report_from_result(
                (
                    TaskRuntimePreparationResult.READY
                    if runtime_report
                    else TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
                ),
                task_dir=task_dir,
                elapsed_seconds=time.monotonic() - runtime_started_at,
            )
        if not runtime_report.ready:
            log_event_fn(
                "task_runtime_failed",
                task_dir=task_dir,
                requirements_path=task_dir / "requirements.txt",
                result_kind=runtime_report.result.value,
                failure_stage=runtime_report.failure_stage,
                failure_category=runtime_report.failure_category,
                diagnostic=runtime_report.diagnostic,
                task_fingerprint=runtime_report.task_fingerprint,
                elapsed_seconds=runtime_report.elapsed_seconds,
            )
            return 2
        log_event_fn(
            "task_runtime_complete",
            python_path=task_dir / ".venv" / "bin" / "python",
        )

    task_seed_program_dir = resolve_path(task_dir / "program_0")
    seed_program_dir = task_seed_program_dir
    if not seed_program_dir.is_dir():
        log_event_fn(
            "config_error",
            reason="missing_seed_program_dir",
            path=seed_program_dir,
            hint="expected <task_dir>/program_0 to be a directory",
        )
        return 2

    if cfg.execution.task_only:
        log_event_fn("task_ready", task_dir=task_dir, seed_program_dir=seed_program_dir)
        return 0

    run_config_fields: dict[str, object] = {
        "show_run_header": cfg.logging.show_run_header,
        "agent.name": cfg.agent.name,
        "agent.model": agent_model,
        "agent.profile": agent_profile,
        "agent.reasoning_effort": reasoning_effort,
        "paths.task_dir": task_dir,
        "paths.results_dir": results_dir,
        "paths.global_memory_dir": global_memory_dir,
        "paths.instruct": instruct_path,
        "paths.idea_instruct": idea_instruct_path,
        "execution.resume": cfg.execution.resume,
        "execution.workers": cfg.execution.workers,
        "execution.max_idea_cycles": cfg.execution.max_idea_cycles,
        "execution.max_program_evals_per_attempt": (
            cfg.execution.max_program_evals_per_attempt
        ),
        "execution.run_timeout_seconds": cfg.execution.run_timeout_seconds,
        "execution.harness_evaluation_timeout_seconds": (
            cfg.execution.harness_evaluation_timeout_seconds
        ),
        "execution.idea_timeout_seconds": cfg.execution.idea_timeout_seconds,
        "execution.max_idea_generation_attempts": cfg.execution.max_idea_generation_attempts,
        "execution.run_timeout_grace_seconds": cfg.execution.run_timeout_grace_seconds,
        "paper.enabled": cfg.paper.enabled,
        "container.enabled": containerize,
        "container.image": container_image,
        "container.network": container_network,
        "container.auth_root": container_auth_root,
        "container.env_file": container_env_file,
        "container.session": container_session_name,
    }

    # Override seed program when provided (used by parallel pruning to start
    # new rounds from each worker's best program instead of the initial one).
    if cfg.paths.seed_program and str(cfg.paths.seed_program).strip():
        override = resolve_path(cfg.paths.seed_program)
        if override.is_dir():
            log_event_fn(
                "seed_program_override",
                override=str(override),
                default=str(seed_program_dir),
            )
            seed_program_dir = override
        else:
            log_event_fn(
                "seed_program_override_warning",
                reason="override_dir_not_found",
                path=str(override),
                fallback=str(seed_program_dir),
            )

    if use_parallel_campaign:
        parallel_results_dir = (
            normalize_agent_results_dir_fn(results_dir, cfg.agent.name)
            if cfg.execution.workers > 1
            else results_dir
        )
        parallel_results_dir.mkdir(parents=True, exist_ok=True)
        try:
            seed_program_dir = _freeze_seed_program(
                seed_program_dir, parallel_results_dir
            )
        except (OSError, ProgramSpecError, ValueError) as exc:
            log_event_fn(
                "config_error",
                reason="seed_program_snapshot_failed",
                path=seed_program_dir,
                error=str(exc),
            )
            return 2
        write_hydra_snapshot(cfg, parallel_results_dir)
        log_event_fn("run_hydra_config", **config_to_flat_dict(cfg))
        log_event_fn("run_config", **run_config_fields)
        parallel_checkpoint_fn = _make_checkpoint_fn(cfg, parallel_results_dir, log_event_fn)
        finalize_on_exit = cfg.versioning.finalize_on_exit
        parallel_checkpoint_fn(label="run start", phase="investigation", event="run_start")
        parallel_global_memory_dir = (
            parallel_results_dir
            if global_memory_dir_raw is None
            else global_memory_dir
        )
        parallel_exit_code = run_parallel_campaign_fn(
            cfg=cfg,
            task_dir=task_dir,
            results_dir=parallel_results_dir,
            global_memory_dir=parallel_global_memory_dir,
            instruct_path=instruct_path,
            seed_program_dir=seed_program_dir,
        )
        if parallel_exit_code != 0:
            parallel_checkpoint_fn(
                label="final" if finalize_on_exit else "parallel campaign failed",
                phase="investigation",
                event="parallel_end",
                final=finalize_on_exit,
            )
            return parallel_exit_code

        if not cfg.paper.enabled:
            parallel_checkpoint_fn(
                label="final" if finalize_on_exit else "parallel campaign",
                phase="investigation",
                event="parallel_end",
                final=finalize_on_exit,
            )
            return 0

        parallel_checkpoint_fn(
            label="parallel campaign",
            phase="investigation",
            event="parallel_end",
        )

        log_event_fn(
            "paper_config",
            paper_instruct=paper_instruct_path,
            paper_contributions=paper_contributions_path,
            paper_timeout_seconds=cfg.paper.timeout_seconds,
            paper_timeout_grace_seconds=cfg.paper.timeout_grace_seconds,
            paper_min_refs=cfg.paper.min_refs,
            paper_max_refs=cfg.paper.max_refs,
            paper_profile=cfg.paper.profile,
            paper_two_stage=cfg.paper.two_stage,
            paper_ai_review=cfg.paper.ai_review,
            paper_review_rounds=cfg.paper.review_rounds,
            paper_max_repair_rounds=cfg.paper.max_repair_rounds,
        )
        paper_output_dir = (
            parallel_results_dir
            / "paper"
            / f"final_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
        )
        paper_started_at = log_step_start_fn(
            "paper_generation",
            output_dir=paper_output_dir,
            refs_min=cfg.paper.min_refs,
            refs_max=cfg.paper.max_refs,
        )
        paper_result = run_paper_generation_fn(
            agent=cfg.agent.name,
            agent_model=agent_model,
            agent_profile=agent_profile,
            reasoning_effort=reasoning_effort,
            task_dir=task_dir,
            results_dir=parallel_results_dir,
            global_memory_dir=parallel_global_memory_dir,
            output_dir=paper_output_dir,
            instruct_path=paper_instruct_path,
            contributions_path=paper_contributions_path,
            timeout_seconds=cfg.paper.timeout_seconds,
            timeout_grace_seconds=cfg.paper.timeout_grace_seconds,
            paper_min_refs=cfg.paper.min_refs,
            paper_max_refs=cfg.paper.max_refs,
            paper_profile=cfg.paper.profile,
            two_stage=cfg.paper.two_stage,
            paper_review_rounds=cfg.paper.review_rounds,
            paper_max_repair_rounds=cfg.paper.max_repair_rounds,
            final_ai_review=cfg.paper.ai_review,
            containerize=containerize,
            container_image=container_image,
            container_network=container_network,
            container_auth_root=container_auth_root,
            container_env_file=container_env_file,
            container_session_name=None,
            container_mount_root=None,
        )
        paper_workspace = (
            paper_result.paper_dir if paper_result.paper_dir else paper_output_dir
        )
        paper_workspace.mkdir(parents=True, exist_ok=True)
        paper_log_path = paper_workspace / "wrapper_output.log"
        state_store.write_private_text(paper_log_path, paper_result.output)
        log_step_end_fn(
            "paper_generation",
            paper_started_at,
            exit=paper_result.exit_code,
            paper_workspace=paper_workspace,
            log_path=paper_log_path,
        )

        if paper_result.exit_code != 0:
            log_event_fn(
                "paper_generation_failed",
                exit=paper_result.exit_code,
                log_path=paper_log_path,
                tail_chars=1200,
                output_tail=paper_result.output[-1200:],
            )
            parallel_checkpoint_fn(
                label="final" if finalize_on_exit else "paper failed",
                phase="paper",
                event="paper_generation_failed",
                final=finalize_on_exit,
            )
            return 2

        latest_manifest = parallel_results_dir / "paper" / "latest_manifest.json"
        if not latest_manifest.exists():
            log_event_fn(
                "paper_generation_error",
                reason="missing_latest_manifest",
                path=latest_manifest,
            )
            parallel_checkpoint_fn(
                label="final" if finalize_on_exit else "paper failed",
                phase="paper",
                event="paper_generation_error",
                final=finalize_on_exit,
            )
            return 2
        log_event_fn("paper_generation_complete", paper_workspace=paper_workspace)
        parallel_checkpoint_fn(
            label="final" if finalize_on_exit else "paper",
            phase="paper",
            event="paper_generation_complete",
            final=finalize_on_exit,
        )
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    seed_is_validated_baseline = seed_matches_validated_baseline(
        seed_program_dir, task_seed_program_dir
    )
    try:
        seed_program_dir = _freeze_seed_program(seed_program_dir, results_dir)
    except (OSError, ProgramSpecError, ValueError) as exc:
        log_event_fn(
            "config_error",
            reason="seed_program_snapshot_failed",
            path=seed_program_dir,
            error=str(exc),
        )
        return 2
    write_hydra_snapshot(cfg, results_dir)
    checkpoint_fn = _make_checkpoint_fn(cfg, results_dir, log_event_fn)
    checkpoint_fn(label="run start", phase="investigation", event="run_start")
    include_global_memory = results_dir == global_memory_dir
    state_store.ensure_histories(
        results_dir, include_global_memory=include_global_memory
    )
    state_store.ensure_global_memory_files(global_memory_dir)

    outer_state = state_store.load_or_init_outer_state(results_dir, seed_program_dir, cfg.execution.resume)
    best_manifest = state_store.load_or_init_best_manifest(results_dir, seed_program_dir, cfg.execution.resume)
    if cfg.execution.harness_evaluated_attempts:
        baseline_score = (
            runtime_baseline_score(task_dir) if should_prepare_task_runtime else None
        )

        def reconcile() -> bool:
            return reconcile_harness_incumbent(
                task_dir=task_dir,
                results_dir=results_dir,
                seed_program_dir=seed_program_dir,
                baseline_score=baseline_score,
                baseline_correct=(
                    runtime_baseline_correct(task_dir) if should_prepare_task_runtime else None
                ),
                best_manifest=best_manifest,
                outer_state=outer_state,
                validated_baseline_program_dir=(
                    seed_program_dir if seed_is_validated_baseline else None
                ),
                trust_native=not containerize,
            )

        incumbent_ready = reconcile()
        if not incumbent_ready and not containerize:
            incumbent_ready = _bootstrap_native_incumbent(
                cfg, task_dir, seed_program_dir, log_event_fn
            ) and reconcile()
        if not incumbent_ready:
            log_event_fn(
                "config_error",
                reason="missing_validated_harness_incumbent",
                hint=(
                    "prepare the task runtime or pre-evaluate program_0 with complete "
                    "harness artifacts"
                    if containerize
                    else "the native baseline evaluation of program_0 failed or was not "
                    f"correct; see {seed_program_dir / 'harness_evaluate_stdout.log'}"
                ),
            )
            return 2
        state_store.save_best_manifest(results_dir, best_manifest)
        state_store.save_outer_state(results_dir, outer_state)

    # Backward-compatible normalization: legacy manifests used program_path.
    if not isinstance(best_manifest.get("program_dir"), str) or not str(best_manifest.get("program_dir")).strip():
        legacy_program_path_raw = best_manifest.get("program_path")
        if isinstance(legacy_program_path_raw, str) and legacy_program_path_raw.strip():
            legacy_program_path = resolve_path(legacy_program_path_raw)
            best_manifest["program_dir"] = str(legacy_program_path.parent)
            best_manifest["entrypoint_path"] = legacy_program_path.relative_to(
                legacy_program_path.parent
            ).as_posix()
            state_store.save_best_manifest(results_dir, best_manifest)

    # Backward-compatible normalization: legacy outer_state used best_program_path.
    if not isinstance(outer_state.get("best_program_dir"), str) or not str(outer_state.get("best_program_dir")).strip():
        legacy_best_program_raw = outer_state.get("best_program_path")
        if isinstance(legacy_best_program_raw, str) and legacy_best_program_raw.strip():
            legacy_best_program_path = resolve_path(legacy_best_program_raw)
            outer_state["best_program_dir"] = str(
                legacy_best_program_path
                if legacy_best_program_path.is_dir()
                else legacy_best_program_path.parent
            )
            state_store.save_outer_state(results_dir, outer_state)

    best_dir_raw = best_manifest.get("program_dir")
    best_program_dir = (
        resolve_path(best_dir_raw) if isinstance(best_dir_raw, str) else seed_program_dir
    )
    if not best_program_dir.is_dir():
        best_program_dir = seed_program_dir
        best_manifest["program_dir"] = str(best_program_dir)
        best_manifest["entrypoint_path"] = (
            entrypoint_relpath_from_spec(best_program_dir) or "src/program.py"
        )
        state_store.save_best_manifest(results_dir, best_manifest)

    # If manifest is missing entrypoint_path, attempt to derive it.
    entrypoint_raw = best_manifest.get("entrypoint_path")
    if not isinstance(entrypoint_raw, str) or not entrypoint_raw.strip():
        best_manifest["entrypoint_path"] = (
            entrypoint_relpath_from_spec(best_program_dir) or "src/program.py"
        )
        state_store.save_best_manifest(results_dir, best_manifest)

    start_cycle = int(outer_state.get("current_cycle", 0)) + 1
    max_program_evals_per_attempt = cfg.execution.max_program_evals_per_attempt

    # Log all Hydra hyperparameters
    flat_cfg = config_to_flat_dict(cfg)
    log_event_fn("run_hydra_config", **flat_cfg)

    log_event_fn(
        "run_config",
        start_cycle=start_cycle,
        **run_config_fields,
    )
    if cfg.paper.enabled:
        log_event_fn(
            "paper_config",
            **{
                "paths.paper_instruct": paper_instruct_path,
                "paths.paper_contributions": paper_contributions_path,
                "paper.profile": cfg.paper.profile,
                "paper.two_stage": cfg.paper.two_stage,
                "paper.ai_review": cfg.paper.ai_review,
                "paper.review_rounds": cfg.paper.review_rounds,
                "paper.max_repair_rounds": cfg.paper.max_repair_rounds,
                "paper.timeout_seconds": cfg.paper.timeout_seconds,
                "paper.timeout_grace_seconds": cfg.paper.timeout_grace_seconds,
                "paper.min_refs": cfg.paper.min_refs,
                "paper.max_refs": cfg.paper.max_refs,
            },
        )

    return campaign_loop.run_cycles_and_finalize(
        cfg=cfg,
        task_dir=task_dir,
        results_dir=results_dir,
        global_memory_dir=global_memory_dir,
        shared_memory_dir=shared_memory_dir,
        conclusions_dir=conclusions_dir,
        worker_index=cfg.execution.worker_index,
        instruct_path=instruct_path,
        idea_instruct_path=idea_instruct_path,
        paper_instruct_path=paper_instruct_path,
        paper_contributions_path=paper_contributions_path,
        containerize=containerize,
        container_image=container_image,
        container_network=container_network,
        container_auth_root=container_auth_root,
        container_env_file=container_env_file,
        container_session_name=container_session_name,
        container_mount_root=container_mount_root,
        outer_state=outer_state,
        best_manifest=best_manifest,
        best_program_dir=best_program_dir,
        start_cycle=start_cycle,
        max_program_evals_per_attempt=max_program_evals_per_attempt,
        main_started_at=main_started_at,
        log_event_fn=log_event_fn,
        log_step_start_fn=log_step_start_fn,
        log_step_end_fn=log_step_end_fn,
        format_elapsed_fn=format_elapsed_fn,
        run_idea_generation_fn=run_idea_generation_fn,
        run_single_attempt_fn=run_single_attempt_fn,
        run_paper_generation_fn=run_paper_generation_fn,
        reconcile_legacy_idea_local_memory_fn=reconcile_legacy_idea_local_memory_fn,
        find_best_program_in_experiment_fn=find_best_program_in_experiment,
        read_json_fn=read_json,
        utc_now_fn=utc_now,
        checkpoint_fn=checkpoint_fn,
    )

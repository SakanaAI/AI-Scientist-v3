#!/usr/bin/env python3
"""Investigation-phase entry point (``agsci investigate``): idea, run, and paper loops."""

from __future__ import annotations

import logging
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Literal, Sequence

from src.orchestrator import state as state_store
from src.investigation.config_loader import load_config
from src.investigation.config import InvestigationConfig
from src.investigation.container_gpus import GpuSelectorError
from src.investigation.container_images import DockerImageLease, acquire_image_lease
from src.investigation.constants import (
    DEFAULT_HARNESS_EVALUATION_TIMEOUT_SECONDS,
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_CONTAINER_IMAGE_BUILD_HINT as DEFAULT_CONTAINER_IMAGE_BUILD_HINT,
    DEFAULT_CONTAINER_IMAGE_BUILD_HINT_HOST_NET as DEFAULT_CONTAINER_IMAGE_BUILD_HINT_HOST_NET,
    DEFAULT_CONTAINER_NETWORK,
    DEFAULT_INTERRUPT_GRACE_SECONDS,
    DEFAULT_MAX_PROGRAM_EVALS_PER_ATTEMPT,
    DEFAULT_PAPER_MAX_REFS,
    DEFAULT_PAPER_MAX_REPAIR_ROUNDS,
    DEFAULT_PAPER_MIN_REFS,
    DEFAULT_PAPER_TIMEOUT_GRACE_SECONDS,
    DEFAULT_PAPER_TIMEOUT_SECONDS as DEFAULT_PAPER_TIMEOUT_SECONDS,
    DEFAULT_RUN_TIMEOUT_GRACE_SECONDS,
    DEFAULT_RUN_TIMEOUT_SECONDS as DEFAULT_RUN_TIMEOUT_SECONDS,
)
from src.investigation.defaults import (
    default_draft_instruction_path,
    default_idea_instruction_path,
    default_instruction_path,
    default_paper_instruction_path,
    default_results_path,
    normalize_agent_results_dir,
    split_cycle_budget,
)
from src.investigation.files import (
    discover_program_indices as discover_program_indices,
    ensure_directory_alias as ensure_directory_alias,
    ensure_task_copy_in_results as ensure_task_copy_in_results,
    extract_markdown_section as extract_markdown_section,
    find_best_program_in_experiment as find_best_program_in_experiment,
    load_instructions as load_instructions,
    parse_experiment_dir as parse_experiment_dir,
    parse_idea_dir as parse_idea_dir,
    parse_paper_dir as parse_paper_dir,
    read_task_sections as read_task_sections,
    relative_display_path as relative_display_path,
    resolve_path,
    staged_task_copy_dir as staged_task_copy_dir,
)
from src.investigation.final_ai_review import run_final_ai_review
from src.investigation.main_entry import run_main_flow
from src.investigation.temp_bootstrap import initialize_runtime_storage
from src.investigation.models import (
    IdeaGenerationResult,
    PaperGenerationResult,
    ParallelWorkerPlan,
    ParallelWorkerResult,
    RunAttemptResult,
    TaskDraftResult,
)
from src.investigation.task_setup import run_task_setup
from src.investigation.task_runtime import (
    TaskRuntimePreparationReport,
    prepare_task_runtime as prepare_task_runtime,
    prepare_task_runtime_report,
)
from src.investigation import runtime as runtime_utils

_DEFAULT_PREPARE_TASK_RUNTIME = prepare_task_runtime


def _prepare_task_runtime_with_report(
    *args: Any, **kwargs: Any
) -> TaskRuntimePreparationReport | bool:
    """Keep the public bool seam while using structured reports in production."""
    if prepare_task_runtime is _DEFAULT_PREPARE_TASK_RUNTIME:
        return prepare_task_runtime_report(*args, **kwargs)
    return prepare_task_runtime(*args, **kwargs)

_ACTIVE_SUBPROCESSES = runtime_utils.ACTIVE_SUBPROCESSES
_ACTIVE_SUBPROCESSES_LOCK = runtime_utils.ACTIVE_SUBPROCESSES_LOCK
_ACTIVE_CONTAINER_SESSION_NAME: str | None = None
# Keep an explicit module-level export used by tests/integration callers.
_STATE_STORE = state_store


def _reconcile_legacy_idea_local_memory(
    *,
    results_dir: Path,
    cycle_index: int,
    idea_id: str,
    canonical_idea_dir_name: str,
) -> None:
    """Handle reconcile legacy idea local memory logic for this module."""
    from src.investigation import idea_memory

    idea_memory.reconcile_legacy_idea_local_memory(
        results_dir=results_dir,
        cycle_index=cycle_index,
        idea_id=idea_id,
        canonical_idea_dir_name=canonical_idea_dir_name,
        log_event_fn=log_event,
    )


_ensure_text = runtime_utils.ensure_text
format_elapsed = runtime_utils.format_elapsed
log_event = runtime_utils.log_event
log_step_start = runtime_utils.log_step_start
log_step_end = runtime_utils.log_step_end
configure_event_logging = runtime_utils.configure_event_logging
configure_standard_logging = runtime_utils.configure_standard_logging
shutdown_event_logging = runtime_utils.shutdown_event_logging
_register_active_subprocess = runtime_utils.register_active_subprocess
_unregister_active_subprocess = runtime_utils.unregister_active_subprocess
_terminate_process_group = runtime_utils.terminate_active_process_group
_build_container_session_name = runtime_utils.build_container_session_name
_probe_docker_available = runtime_utils.probe_docker_available
_docker_image_exists = runtime_utils.docker_image_exists
_container_execution_env = runtime_utils.container_execution_env


def _terminate_all_active_processes(*, grace_seconds: int) -> list[int]:
    """Terminate all active processes."""
    return runtime_utils.terminate_all_active_processes(
        grace_seconds=grace_seconds,
        terminate_process_group=lambda proc: _terminate_process_group(
            proc,
            grace_seconds=grace_seconds,
        ),
    )


def _set_active_container_session(name: str | None) -> None:
    """Handle set active container session logic for this module."""
    global _ACTIVE_CONTAINER_SESSION_NAME
    _ACTIVE_CONTAINER_SESSION_NAME = name


def _stop_container_session(
    name: str,
    *,
    log_event_fn: Any | None = None,
) -> bool:
    """Stop container session."""
    logger = log_event_fn if log_event_fn is not None else log_event
    return runtime_utils.stop_container_session(name, log_event_fn=logger)


def _teardown_active_container_session() -> list[str]:
    """Handle teardown active container session logic for this module."""
    global _ACTIVE_CONTAINER_SESSION_NAME
    name = _ACTIVE_CONTAINER_SESSION_NAME
    _ACTIVE_CONTAINER_SESSION_NAME = None
    if not name:
        return []
    if _stop_container_session(name):
        return [name]
    return []


def run_idea_generation(
    *,
    agent: str,
    agent_model: str | None = None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path | None = None,
    conclusions_dir: Path | None = None,
    worker_index: int = 0,
    cycle_index: int,
    output_dir: Path,
    instruct_path: Path,
    timeout_seconds: int,
    timeout_grace_seconds: int = DEFAULT_RUN_TIMEOUT_GRACE_SECONDS,
    harness_evaluated_attempts: bool = False,
    containerize: bool = False,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_network: str = DEFAULT_CONTAINER_NETWORK,
    container_auth_root: Path | None = None,
    container_env_file: Path | None = None,
    container_session_name: str | None = None,
    container_mount_root: Path | None = None,
    validation_feedback: str = "",
) -> IdeaGenerationResult:
    """Run idea generation."""
    from src.investigation import script_runners

    return script_runners.run_idea_generation(
        agent=agent,
        agent_model=agent_model,
        agent_profile=agent_profile,
        reasoning_effort=reasoning_effort,
        task_dir=task_dir,
        results_dir=results_dir,
        global_memory_dir=global_memory_dir,
        conclusions_dir=conclusions_dir,
        worker_index=worker_index,
        cycle_index=cycle_index,
        output_dir=output_dir,
        instruct_path=instruct_path,
        timeout_seconds=timeout_seconds,
        timeout_grace_seconds=timeout_grace_seconds,
        harness_evaluated_attempts=harness_evaluated_attempts,
        containerize=containerize,
        container_image=container_image,
        container_network=container_network,
        container_auth_root=container_auth_root,
        container_env_file=container_env_file,
        container_session_name=container_session_name,
        container_mount_root=container_mount_root,
        validation_feedback=validation_feedback,
        script_path=resolve_path(Path(__file__).resolve().parents[2] / "src" / "scripts" / "idea.sh"),
        popen_factory=subprocess.Popen,
        register_subprocess=_register_active_subprocess,
        unregister_subprocess=_unregister_active_subprocess,
        terminate_process_group=_terminate_process_group,
        container_execution_env=_container_execution_env,
        ensure_text=_ensure_text,
    )


def run_single_attempt(
    *,
    agent: str,
    agent_model: str | None = None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path | None = None,
    conclusions_dir: Path | None = None,
    worker_index: int = 0,
    seed_program_dir: Path,
    idea_file: Path,
    rubrics_file: Path,
    attempt_id: str,
    idea_dir_name: str,
    attempt_dir_name: str | None = None,
    instruct_path: Path | None,
    timeout_seconds: int,
    timeout_grace_seconds: int = DEFAULT_RUN_TIMEOUT_GRACE_SECONDS,
    max_program_evals_per_attempt: int = DEFAULT_MAX_PROGRAM_EVALS_PER_ATTEMPT,
    harness_evaluated_attempts: bool = False,
    harness_evaluation_timeout_seconds: int = DEFAULT_HARNESS_EVALUATION_TIMEOUT_SECONDS,
    gpu_evaluation: bool = False,
    containerize: bool = False,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_network: str = DEFAULT_CONTAINER_NETWORK,
    container_auth_root: Path | None = None,
    container_env_file: Path | None = None,
    container_session_name: str | None = None,
    container_mount_root: Path | None = None,
) -> RunAttemptResult:
    """Run single attempt."""
    from src.investigation import script_runners

    return script_runners.run_single_attempt(
        agent=agent,
        agent_model=agent_model,
        agent_profile=agent_profile,
        reasoning_effort=reasoning_effort,
        task_dir=task_dir,
        results_dir=results_dir,
        global_memory_dir=global_memory_dir,
        conclusions_dir=conclusions_dir,
        worker_index=worker_index,
        seed_program_dir=seed_program_dir,
        idea_file=idea_file,
        rubrics_file=rubrics_file,
        attempt_id=attempt_id,
        idea_dir_name=idea_dir_name,
        attempt_dir_name=attempt_dir_name,
        instruct_path=instruct_path,
        timeout_seconds=timeout_seconds,
        timeout_grace_seconds=timeout_grace_seconds,
        max_program_evals_per_attempt=max_program_evals_per_attempt,
        harness_evaluated_attempts=harness_evaluated_attempts,
        gpu_evaluation=gpu_evaluation,
        harness_evaluation_timeout_seconds=harness_evaluation_timeout_seconds,
        containerize=containerize,
        container_image=container_image,
        container_network=container_network,
        container_auth_root=container_auth_root,
        container_env_file=container_env_file,
        container_session_name=container_session_name,
        container_mount_root=container_mount_root,
        script_path=resolve_path(Path(__file__).resolve().parents[2] / "src" / "scripts" / "run.sh"),
        popen_factory=subprocess.Popen,
        register_subprocess=_register_active_subprocess,
        unregister_subprocess=_unregister_active_subprocess,
        terminate_process_group=_terminate_process_group,
        container_execution_env=_container_execution_env,
        ensure_text=_ensure_text,
    )


def run_paper_generation(
    *,
    agent: str,
    agent_model: str | None = None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path | None = None,
    output_dir: Path,
    instruct_path: Path,
    timeout_seconds: int,
    contributions_path: Path | None = None,
    timeout_grace_seconds: int = DEFAULT_PAPER_TIMEOUT_GRACE_SECONDS,
    paper_min_refs: int = DEFAULT_PAPER_MIN_REFS,
    paper_max_refs: int = DEFAULT_PAPER_MAX_REFS,
    paper_profile: str = "default",
    paper_skills: Sequence[str] | None = None,
    two_stage: bool = False,
    paper_review_rounds: int = 0,
    paper_max_repair_rounds: int = DEFAULT_PAPER_MAX_REPAIR_ROUNDS,
    final_ai_review: Literal["disabled", "openai"] = "disabled",
    containerize: bool = False,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_network: str = DEFAULT_CONTAINER_NETWORK,
    container_auth_root: Path | None = None,
    container_env_file: Path | None = None,
    container_session_name: str | None = None,
    container_mount_root: Path | None = None,
) -> PaperGenerationResult:
    """Run paper generation."""
    from src.investigation import script_runners

    if final_ai_review not in ("disabled", "openai"):
        raise ValueError(f"unsupported final AI review mode: {final_ai_review}")
    repo_root = Path(__file__).resolve().parents[2]
    result = script_runners.run_paper_generation(
        agent=agent,
        agent_model=agent_model,
        agent_profile=agent_profile,
        reasoning_effort=reasoning_effort,
        task_dir=task_dir,
        results_dir=results_dir,
        global_memory_dir=global_memory_dir,
        output_dir=output_dir,
        instruct_path=instruct_path,
        contributions_path=contributions_path,
        timeout_seconds=timeout_seconds,
        timeout_grace_seconds=timeout_grace_seconds,
        paper_min_refs=paper_min_refs,
        paper_max_refs=paper_max_refs,
        paper_profile=paper_profile,
        paper_skills=paper_skills,
        two_stage=two_stage,
        paper_review_rounds=paper_review_rounds,
        paper_max_repair_rounds=paper_max_repair_rounds,
        containerize=containerize,
        container_image=container_image,
        container_network=container_network,
        container_auth_root=container_auth_root,
        container_env_file=container_env_file,
        container_session_name=container_session_name,
        container_mount_root=container_mount_root,
        script_path=resolve_path(repo_root / "src" / "scripts" / "paper.sh"),
        popen_factory=subprocess.Popen,
        register_subprocess=_register_active_subprocess,
        unregister_subprocess=_unregister_active_subprocess,
        terminate_process_group=_terminate_process_group,
        container_execution_env=_container_execution_env,
        ensure_text=_ensure_text,
        paper_event_fn=log_event,
    )
    final_pdf = output_dir / "paper.pdf"
    latest_manifest = results_dir / "paper" / "latest_manifest.json"
    if result.exit_code != 0 or not final_pdf.is_file() or not latest_manifest.is_file():
        return result
    if final_ai_review == "disabled":
        return result

    review_env_file = container_env_file
    default_env_file = repo_root / ".env"
    if review_env_file is None and default_env_file.is_file():
        review_env_file = default_env_file

    log_event("paper_ai_review_started", paper_dir=output_dir)
    def execute_review(
        command: list[str], log_path: Path, environment: dict[str, str]
    ) -> tuple[str, int]:
        return script_runners.execute_managed_command(
            cmd=command,
            timeout_seconds=timeout_seconds,
            timeout_grace_seconds=timeout_grace_seconds,
            script_label="src/aireview.py",
            output_log_path=log_path,
            popen_factory=subprocess.Popen,
            register_subprocess=_register_active_subprocess,
            unregister_subprocess=_unregister_active_subprocess,
            terminate_process_group=_terminate_process_group,
            ensure_text=_ensure_text,
            env=environment,
        )

    try:
        review = run_final_ai_review(
            repo_root=repo_root,
            paper_dir=output_dir,
            results_dir=results_dir,
            execute_command=execute_review,
            env_file=review_env_file,
        )
    except (OSError, ValueError) as exc:
        result.ai_review_status = "failed"
        log_event(
            "paper_ai_review_failed",
            paper_dir=output_dir,
            reason=str(exc),
            log_path=None,
        )
        return result

    result.ai_review_status = review.status
    result.ai_review_path = review.review_path
    if review.status == "completed":
        log_event(
            "paper_ai_review_completed",
            paper_dir=output_dir,
            review_path=review.review_path,
            log_path=review.log_path,
        )
    else:
        log_event(
            "paper_ai_review_failed",
            paper_dir=output_dir,
            reason=review.error,
            log_path=review.log_path,
        )
    return result


def _build_agent_command(
    agent: str,
    prompt_file: str,
    *,
    agent_model: str | None = None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
) -> list[str]:
    """Build an agent command for a staged prompt file."""
    from src.investigation import script_runners

    return script_runners.build_agent_command(
        agent,
        prompt_file,
        agent_model=agent_model,
        agent_profile=agent_profile,
        reasoning_effort=reasoning_effort,
    )


def run_task_drafting(
    *,
    agent: str,
    agent_model: str | None = None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    task_dir: Path,
    output_dir: Path,
    instruct_path: Path,
    timeout_seconds: int,
    timeout_grace_seconds: int = DEFAULT_RUN_TIMEOUT_GRACE_SECONDS,
    workflow: str = "robust",
    design_max_repairs: int = 8,
    contract_max_repairs: int = 3,
    smoke_max_repairs: int = 3,
    smoke_max_infrastructure_attempts: int = 3,
    alignment_max_repairs: int = 10,
    qualification_max_repairs: int = 4,
    agent_transient_max_attempts: int = 4,
    qualification_max_infrastructure_attempts: int = 3,
    production_entry_timeout_seconds: int = 300,
    max_gate_checks_per_repair: int = 12,
    draft_max_attempts: int = 3,
    evaluator_smoke_timeout_seconds: int = 1800,
    resume_from: str = "",
    max_idea_cycles: int = 20,
    max_program_evals_per_attempt: int = DEFAULT_MAX_PROGRAM_EVALS_PER_ATTEMPT,
    max_runtime_bytes: int = 16 * 1024 * 1024 * 1024,
    containerize: bool = False,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_network: str = DEFAULT_CONTAINER_NETWORK,
    container_auth_root: Path | None = None,
    container_env_file: Path | None = None,
    container_session_name: str | None = None,
    container_mount_root: Path | None = None,
) -> TaskDraftResult:
    """Run task drafting."""
    from src.investigation import script_runners

    return script_runners.run_task_drafting(
        agent=agent,
        agent_model=agent_model,
        agent_profile=agent_profile,
        reasoning_effort=reasoning_effort,
        task_dir=task_dir,
        output_dir=output_dir,
        instruct_path=instruct_path,
        timeout_seconds=timeout_seconds,
        timeout_grace_seconds=timeout_grace_seconds,
        workflow=workflow,
        design_max_repairs=design_max_repairs,
        contract_max_repairs=contract_max_repairs,
        smoke_max_repairs=smoke_max_repairs,
        smoke_max_infrastructure_attempts=smoke_max_infrastructure_attempts,
        alignment_max_repairs=alignment_max_repairs,
        qualification_max_repairs=qualification_max_repairs,
        agent_transient_max_attempts=agent_transient_max_attempts,
        qualification_max_infrastructure_attempts=qualification_max_infrastructure_attempts,
        production_entry_timeout_seconds=production_entry_timeout_seconds,
        max_gate_checks_per_repair=max_gate_checks_per_repair,
        draft_max_attempts=draft_max_attempts,
        evaluator_smoke_timeout_seconds=evaluator_smoke_timeout_seconds,
        resume_from=resume_from,
        max_idea_cycles=max_idea_cycles,
        max_program_evals_per_attempt=max_program_evals_per_attempt,
        max_runtime_bytes=max_runtime_bytes,
        containerize=containerize,
        container_image=container_image,
        container_network=container_network,
        container_auth_root=container_auth_root,
        container_env_file=container_env_file,
        container_session_name=container_session_name,
        container_mount_root=container_mount_root,
        script_path=resolve_path(Path(__file__).resolve().parents[2] / "src" / "scripts" / "task.sh"),
        popen_factory=subprocess.Popen,
        register_subprocess=_register_active_subprocess,
        unregister_subprocess=_unregister_active_subprocess,
        terminate_process_group=_terminate_process_group,
        container_execution_env=_container_execution_env,
        ensure_text=_ensure_text,
        task_event_fn=log_event,
    )


def build_parallel_worker_command(
    *,
    cfg: InvestigationConfig,
    task_dir: Path,
    instruct_path: Path,
    results_dir: Path,
    global_memory_dir: Path,
    assigned_cycles: int,
    seed_program: Path | None = None,
    shared_memory_dir: Path | None = None,
    conclusions_dir: Path | None = None,
    worker_index: int = 0,
    resume: bool | None = None,
    global_cycle: int | None = None,
) -> list[str]:
    """Build parallel worker command."""
    from src.investigation import parallel_helpers

    return parallel_helpers.build_parallel_worker_command(
        cfg=cfg,
        task_dir=task_dir,
        instruct_path=instruct_path,
        results_dir=results_dir,
        global_memory_dir=global_memory_dir,
        assigned_cycles=assigned_cycles,
        seed_program=seed_program,
        shared_memory_dir=shared_memory_dir,
        conclusions_dir=conclusions_dir,
        worker_index=worker_index,
        resume=resume,
        global_cycle=global_cycle,
    )


def run_parallel_worker(plan: ParallelWorkerPlan) -> ParallelWorkerResult:
    """Run parallel worker."""
    from src.investigation import parallel_helpers

    return parallel_helpers.run_parallel_worker(
        plan,
        popen_factory=subprocess.Popen,
        register_subprocess=_register_active_subprocess,
        unregister_subprocess=_unregister_active_subprocess,
        terminate_process_group=_terminate_process_group,
        ensure_text=_ensure_text,
        interrupt_grace_seconds=DEFAULT_INTERRUPT_GRACE_SECONDS,
    )


def aggregate_parallel_worker_outputs(
    *,
    results_dir: Path,
    seed_program_dir: Path,
    worker_results: list[ParallelWorkerResult],
) -> dict[str, Any]:
    """Handle aggregate parallel worker outputs logic for this module."""
    from src.investigation import parallel_helpers

    return parallel_helpers.aggregate_parallel_worker_outputs(
        results_dir=results_dir,
        seed_program_dir=seed_program_dir,
        worker_results=worker_results,
        log_event_fn=log_event,
    )


def run_parallel_campaign(
    *,
    cfg: InvestigationConfig,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path,
    instruct_path: Path,
    seed_program_dir: Path,
) -> int:
    """Run parallel campaign."""
    from src.investigation import parallel_helpers

    return parallel_helpers.run_parallel_campaign(
        cfg=cfg,
        task_dir=task_dir,
        results_dir=results_dir,
        global_memory_dir=global_memory_dir,
        instruct_path=instruct_path,
        seed_program_dir=seed_program_dir,
        split_cycle_budget_fn=split_cycle_budget,
        build_parallel_worker_command_fn=build_parallel_worker_command,
        run_parallel_worker_fn=run_parallel_worker,
        aggregate_parallel_worker_outputs_fn=aggregate_parallel_worker_outputs,
        log_event_fn=log_event,
        format_elapsed_fn=format_elapsed,
        terminate_all_active_processes_fn=_terminate_all_active_processes,
        stop_container_session_fn=_stop_container_session,
        build_container_session_name_fn=_build_container_session_name,
        interrupt_grace_seconds=DEFAULT_INTERRUPT_GRACE_SECONDS,
    )


_active_image_lease: DockerImageLease | None = None


def _pin_container_image(image_reference: str, lease_scope: str) -> str:
    """Pin the agent image to its immutable id for this run, as the pipeline does."""
    _release_container_image()
    global _active_image_lease
    _active_image_lease = acquire_image_lease(image_reference, lease_scope=lease_scope, role="agent")
    return _active_image_lease.image_id


def _release_container_image() -> None:
    global _active_image_lease
    lease, _active_image_lease = _active_image_lease, None
    if lease is None:
        return
    try:
        lease.release()
    except (OSError, ValueError) as exc:
        logging.warning("Container image lease release failed: %s", exc)


def main(
    *, overrides: Sequence[str] | None = None, _retain_event_logging: bool = False
) -> int:
    """Run the command-line entry point.

    ``overrides`` are Hydra overrides; ``None`` reads them from ``sys.argv[1:]``.
    """
    configure_standard_logging()
    cfg = load_config(None if overrides is None else list(overrides))
    try:
        initialize_runtime_storage()
    except (OSError, ValueError) as exc:
        logging.error("Temporary storage initialization failed: %s", exc)
        return 1
    configure_standard_logging(
        level=logging.INFO if cfg.logging.verbose else logging.WARNING
    )
    assert cfg.paths.results_dir is not None
    logging_results_dir = resolve_path(cfg.paths.results_dir)
    if cfg.execution.workers > 1:
        logging_results_dir = normalize_agent_results_dir(
            logging_results_dir,
            cfg.agent.name,
        )
    configure_event_logging(
        log_format=cfg.logging.format,
        verbose=cfg.logging.verbose,
        heartbeat_seconds=cfg.logging.heartbeat_seconds,
        results_dir=logging_results_dir,
        worker_index=cfg.execution.worker_index,
    )
    from src.investigation import parallel_helpers

    try:
        gpu_pool = parallel_helpers.settle_gpu_allocation(cfg, os.environ)
    except GpuSelectorError as exc:
        logging.error("GPU allocation preflight failed: %s", exc)
        return 1
    if gpu_pool is not None:
        logging.info(
            "Parallel GPU pool %s is sliced %d per worker", gpu_pool, cfg.container.gpu_count
        )
    main_started_at = time.monotonic()
    _set_active_container_session(None)
    try:
        return run_main_flow(
            cfg=cfg,
            main_started_at=main_started_at,
            set_active_container_session_fn=_set_active_container_session,
            build_container_session_name_fn=_build_container_session_name,
            stop_container_session_fn=_stop_container_session,
            probe_docker_available_fn=_probe_docker_available,
            docker_image_exists_fn=_docker_image_exists,
            docker_binary_lookup_fn=shutil.which,
            default_results_path_fn=default_results_path,
            default_instruction_path_fn=default_instruction_path,
            default_idea_instruction_path_fn=default_idea_instruction_path,
            default_draft_instruction_path_fn=default_draft_instruction_path,
            default_paper_instruction_path_fn=default_paper_instruction_path,
            normalize_agent_results_dir_fn=normalize_agent_results_dir,
            log_event_fn=log_event,
            log_step_start_fn=log_step_start,
            log_step_end_fn=log_step_end,
            run_task_drafting_fn=run_task_drafting,
            run_parallel_campaign_fn=run_parallel_campaign,
            run_idea_generation_fn=run_idea_generation,
            run_single_attempt_fn=run_single_attempt,
            run_paper_generation_fn=run_paper_generation,
            run_task_setup_fn=run_task_setup,
            prepare_task_runtime_fn=_prepare_task_runtime_with_report,
            pin_container_image_fn=_pin_container_image,
            reconcile_legacy_idea_local_memory_fn=_reconcile_legacy_idea_local_memory,
            format_elapsed_fn=format_elapsed,
        )
    finally:
        _release_container_image()
        if not _retain_event_logging:
            shutdown_event_logging()


def _install_cleanup_signal_ignores() -> dict[int, Any]:
    """Temporarily ignore interrupt signals while shutdown cleanup is running."""
    previous: dict[int, Any] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, signal.SIG_IGN)
        except (AttributeError, OSError, RuntimeError, ValueError):
            continue
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    """Restore previously registered signal handlers."""
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (AttributeError, OSError, RuntimeError, ValueError):
            continue


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run the investigation CLI with interrupt and container cleanup."""
    from src.investigation.container_temp_cleanup import capture_container_scratch

    interrupted = False
    completed_successfully = False
    exit_code = 0
    terminated_pids: list[int] = []
    ignored_signal_handlers: dict[int, Any] = {}
    try:
        exit_code = main(overrides=argv, _retain_event_logging=True)
    except KeyboardInterrupt:
        interrupted = True
        ignored_signal_handlers = _install_cleanup_signal_ignores()
        log_event(
            "interrupted",
            action="terminate_active_processes",
            grace_seconds=DEFAULT_INTERRUPT_GRACE_SECONDS,
        )
        try:
            raw_terminated_pids = _terminate_all_active_processes(
                grace_seconds=DEFAULT_INTERRUPT_GRACE_SECONDS
            )
        except KeyboardInterrupt:
            log_event("interrupt_cleanup_reentered", stage="processes")
            raw_terminated_pids = []
        terminated_pids = (
            [pid for pid in raw_terminated_pids if isinstance(pid, int) and pid > 0]
            if isinstance(raw_terminated_pids, list)
            else []
        )
        exit_code = 130
    else:
        completed_successfully = exit_code == 0
    finally:
        if interrupted and not ignored_signal_handlers:
            ignored_signal_handlers = _install_cleanup_signal_ignores()
        try:
            try:
                scratch = (
                    capture_container_scratch(_ACTIVE_CONTAINER_SESSION_NAME)
                    if completed_successfully else None
                )
            except KeyboardInterrupt:
                log_event("interrupt_cleanup_reentered", stage="scratch")
                scratch = None
            try:
                terminated_containers = _teardown_active_container_session()
            except KeyboardInterrupt:
                log_event("interrupt_cleanup_reentered", stage="containers")
                terminated_containers = []
            if scratch is not None and scratch.name in terminated_containers:
                scratch.cleanup_removed_container()
            if interrupted:
                if terminated_containers:
                    log_event(
                        "interrupt_cleanup_targets",
                        mode="containers",
                        targets=",".join(terminated_containers),
                    )
                elif terminated_pids:
                    log_event(
                        "interrupt_cleanup_targets",
                        mode="processes",
                        targets=",".join(str(pid) for pid in terminated_pids),
                    )
                else:
                    log_event("interrupt_cleanup_targets", mode="none", targets="")
        finally:
            _restore_signal_handlers(ignored_signal_handlers)
            shutdown_event_logging()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run_cli())

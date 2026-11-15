"""CLI entry point for the end-to-end pipeline orchestrator.

Usage::

    # Full pipeline (``agsci run`` is the installed console script's pipeline
    # subcommand; a leading bare argument is the task directory, equivalent to
    # ``task.dir=...``):
    agsci run examples/circpack
    python -m src.pipeline.cli task.dir=examples/circpack

    # Print the resolved commands without running anything:
    python -m src.pipeline.cli task.dir=examples/circpack --dry-run

    # Run only the paper stage against a completed pipeline run:
    python -m src.pipeline.cli task.dir=examples/circpack execution.phases=paper \\
        paths.run_dir=results/circpack/claude/20260331_094917/

    # Resume with ablations given a completed investigation run:
    python -m src.pipeline.cli execution.phases=ablations,paper \\
        paths.run_dir=results/circpack/claude/20260331_094917/

Uses the Hydra Compose API (same pattern as ``src/variants_pipeline/cli.py``).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import secrets
import stat
import sys
import time
from collections.abc import Callable, Iterable, MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from src.final_evaluation.runner import (
    FinalEvaluationError,
    resolve_frozen_incumbent,
    verify_final_manifest,
)
from src.investigation.final_ai_review import FinalAiReviewResult
from src.investigation.review_summary import revision_review_fields
from src.investigation.files import missing_task_scaffold_files
from src.investigation.checkpoint_security import report_sensitive_path
from src.investigation.run_checkpoint import (
    CheckpointSettings,
    describe_checkpoint,
    write_checkpoint,
)
from src.investigation.container_gpus import (
    GPU_POOL_ENV,
    GPU_SELECTOR_ENV,
    GpuSelectorError,
    discover_gpu_selector,
    settle_gpu_pool,
)
from src.investigation.container_images import (
    DockerImageLease,
    acquire_image_lease,
    resolve_immutable_image,
)
from src.investigation.runtime import (
    PIPELINE_STARTED_AT_ENV,
    configure_event_logging,
    configure_standard_logging,
    event_logging_configured,
    format_elapsed,
    log_event,
    shutdown_event_logging,
)
from src.investigation.task_runtime import discard_managed_task_runtime
from src.investigation.task_runtime_cache import discard_task_runtime_install_cache
from src.investigation.task_alignment import (
    TaskAlignmentError,
    load_task_alignment_report,
)
from src.investigation.task_contract import TaskContractError, load_task_contract
from src.investigation.task_failure import HOST_FAILURES
from src.investigation.temp_bootstrap import initialize_runtime_storage, preview_runtime_storage

from .config import (
    PHASE_CLASSES,
    PIPELINE_PHASES,
    PipelineConfig,
    describe_pipeline_config,
    flatten_mapping,
    parse_pipeline_phases,
    register_pipeline_schema,
    validate_investigation_resume_state,
    validate_completed_investigation,
    completed_investigation_outcome,
    downstream_study_available,
    investigation_completion_payload,
    investigation_completion_receipt_path,
    validate_posthoc_paper_input,
    validate_pipeline_config,
    validate_review_input,
    validate_review_output_path,
    task_requires_final_evaluation,
)
from .glue import (
    UnsafeGluePathError,
    ensure_run_task_for_reuse,
    prepare_run_task,
    preview_run_task,
    read_task_snapshot_name,
    reuse_run_task,
    run_glue,
)
from .paths import PipelinePaths, derive_paths
from .downstream_selection import (
    load_downstream_selection,
    prepare_downstream_selection,
    preview_downstream_selection,
)
from .portable_storage_io import write_json_file
from .portable_storage_paths import ArchiveError
from .review import describe_pipeline_review, run_pipeline_review
from .revision import (
    describe_pipeline_revision,
    retain_revision_output_tail,
    revision_failure_event_detail,
    run_pipeline_revision,
)
from .portable_publication import portable_replacement_pending
from .revision_publication import (
    initial_review_must_be_preserved,
    recover_incomplete_promotion,
)
from .resume_config import (
    ResumeConfigError,
    read_run_idea_cycle_budget,
    write_resume_policy,
)
from .safe_fs import ensure_directory, entry_exists_at, open_directory
from .task_snapshot import (
    reseal_run_task,
    task_snapshot_is_complete,
    task_snapshot_publication_is_complete,
)

TASK_RUNTIME_CACHE_HANDOFF_ENV = "AGENTIC_TASK_RUNTIME_CACHE_HANDOFF"
_DOWNSTREAM_PHASES = {"ablations", "final_evaluation", "posthoc", "paper", "review", "revision"}

# Human-readable phase names for terminal and event output.
PHASE_LABELS = {
    "task_drafting": "Task drafting",
    "investigation": "Investigation",
    "ablations": "Ablations",
    "final_evaluation": "Final evaluation",
    "posthoc": "Post-hoc analysis",
    "paper": "Paper",
    "review": "Review",
    "revision": "Revision",
}


def phase_label(phase: str) -> str:
    """Return the display name for a pipeline phase id."""
    return PHASE_LABELS.get(phase, phase.replace("_", " ").capitalize())
from .phases import (
    build_ablations_cmd,
    build_final_cmd,
    build_investigation_cmd,
    build_paper_cmd,
    build_paper_cmd_without_final,
    build_paper_cmd_with_posthoc,
    build_posthoc_cmd,
    build_task_drafting_cmd,
    forward_paper_output,
    run_cmd,
)

logger = logging.getLogger(__name__)

_CONFIGS_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs")


def _quote_comma_separated_phases(arg: str) -> str:
    value = arg.split("=", 1)[1]
    already_quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}
    if "," not in value or already_quoted or "${" in value:
        return arg
    return f"execution.phases='{value}'"


def _write_investigation_completion(path: Path, payload: dict[str, object]) -> bool:
    try:
        write_json_file(path, payload)
    except (ArchiveError, OSError) as exc:
        logger.error("Could not persist investigation completion %s: %s", path, exc)
        return False
    return True


class _InvestigationWorkspaceExistsError(UnsafeGluePathError):
    """Raised when an exclusive fresh-workspace claim finds any entry."""


def _claim_task_drafting_workspace(paths: PipelinePaths) -> tuple[int, int]:
    """Claim or safely reopen the retryable task-drafting output directory."""
    try:
        return _claim_task_drafting_workspace_impl(paths)
    except UnsafeGluePathError:
        raise
    except OSError as exc:
        raise UnsafeGluePathError("could not safely claim task drafting output") from exc


def _claim_task_drafting_workspace_impl(paths: PipelinePaths) -> tuple[int, int]:
    envelope = Path(os.path.abspath(paths.run_dir))
    workspace = Path(os.path.abspath(paths.task_drafting_dir))
    if workspace.parent != envelope or workspace.name != "task_drafting":
        raise UnsafeGluePathError("task drafting output must be a direct run-envelope child")
    with ensure_directory(envelope, role="task drafting run envelope") as envelope_fd:
        envelope_stat = os.fstat(envelope_fd)
        if envelope_stat.st_uid != os.getuid():
            raise UnsafeGluePathError("task drafting run envelope must be owned by this user")
        if stat.S_IMODE(envelope_stat.st_mode) != 0o700:
            os.fchmod(envelope_fd, 0o700)
        try:
            os.mkdir(workspace.name, 0o700, dir_fd=envelope_fd)
        except FileExistsError:
            pass
        workspace_fd = os.open(
            workspace.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=envelope_fd,
        )
        try:
            workspace_stat = os.fstat(workspace_fd)
            if workspace_stat.st_uid != os.getuid():
                raise UnsafeGluePathError("task drafting output must be owned by this user")
            if stat.S_IMODE(workspace_stat.st_mode) != 0o700:
                os.fchmod(workspace_fd, 0o700)
            return workspace_stat.st_dev, workspace_stat.st_ino
        finally:
            os.close(workspace_fd)


def _verify_task_drafting_workspace(
    paths: PipelinePaths,
    expected_identity: tuple[int, int],
) -> None:
    with open_directory(paths.task_drafting_dir, role="task drafting output") as workspace_fd:
        workspace_stat = os.fstat(workspace_fd)
        if (workspace_stat.st_dev, workspace_stat.st_ino) != expected_identity:
            raise UnsafeGluePathError("task drafting output changed during execution")


def _prepare_investigation_workspace(paths: PipelinePaths) -> None:
    """Atomically claim a fresh agent-writable investigation workspace."""
    envelope = Path(os.path.abspath(paths.run_dir))
    workspace = Path(os.path.abspath(paths.investigation_workspace_dir))
    if workspace.parent != envelope or workspace.name != "investigation":
        raise UnsafeGluePathError("investigation workspace must be a direct run-envelope child")

    try:
        with ensure_directory(envelope, role="investigation run envelope") as envelope_fd:
            envelope_stat = os.fstat(envelope_fd)
            if envelope_stat.st_uid != os.getuid():
                raise UnsafeGluePathError("investigation run envelope must be owned by this user")
            if stat.S_IMODE(envelope_stat.st_mode) != 0o700:
                os.fchmod(envelope_fd, 0o700)
            try:
                os.mkdir(workspace.name, 0o700, dir_fd=envelope_fd)
            except FileExistsError as exc:
                raise _InvestigationWorkspaceExistsError(
                    f"fresh investigation workspace already exists: {workspace}"
                ) from exc
            try:
                workspace_fd = os.open(
                    workspace.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=envelope_fd,
                )
            except OSError as exc:
                raise UnsafeGluePathError(
                    f"could not safely open investigation workspace: {workspace}"
                ) from exc
            try:
                workspace_stat = os.fstat(workspace_fd)
                if workspace_stat.st_uid != os.getuid():
                    raise UnsafeGluePathError("investigation workspace must be owned by this user")
                if stat.S_IMODE(workspace_stat.st_mode) != 0o700:
                    os.fchmod(workspace_fd, 0o700)
            finally:
                os.close(workspace_fd)
    except UnsafeGluePathError:
        raise
    except OSError as exc:
        raise UnsafeGluePathError(
            f"could not safely prepare investigation workspace: {workspace}"
        ) from exc


def _reuse_investigation_workspace(paths: PipelinePaths) -> None:
    """Validate an existing agent-writable workspace for resume."""
    try:
        with open_directory(
            paths.investigation_workspace_dir,
            role="resume investigation workspace",
        ) as workspace_fd:
            workspace_stat = os.fstat(workspace_fd)
            if workspace_stat.st_uid != os.getuid():
                raise UnsafeGluePathError("investigation workspace must be owned by this user")
            if stat.S_IMODE(workspace_stat.st_mode) != 0o700:
                os.fchmod(workspace_fd, 0o700)
    except FileNotFoundError as exc:
        raise UnsafeGluePathError("resume investigation workspace does not exist") from exc


def _override_is_explicit(arguments: list[str], key: str) -> bool:
    return any(
        argument.partition("=")[0].lstrip("+") == key for argument in arguments if "=" in argument
    )


def _apply_recorded_idea_cycle_budget(cfg: PipelineConfig, *, explicit: bool) -> str | None:
    """Resume with the idea-cycle budget the run recorded unless one was given explicitly.

    Returns an error message when the recorded budget cannot be trusted. A run that
    recorded no budget keeps the configured value; an explicit override always wins
    and only warns when it disagrees with the record.
    """
    run_dir = cfg.paths.run_dir
    if not run_dir or "${" in run_dir:
        return None
    try:
        recorded = read_run_idea_cycle_budget(Path(run_dir).expanduser())
    except ResumeConfigError as exc:
        return (
            f"could not recover the resumed run's idea-cycle budget ({exc}); "
            "pass investigation.execution.max_idea_cycles explicitly"
        )
    configured = cfg.investigation.max_idea_cycles
    if recorded is None:
        logger.warning(
            "Resumed run recorded no idea-cycle budget; keeping "
            "investigation.execution.max_idea_cycles=%s",
            configured,
        )
    elif explicit and recorded != configured:
        logger.warning(
            "investigation.execution.max_idea_cycles=%s overrides the resumed run's recorded "
            "budget of %d",
            configured,
            recorded,
        )
    elif not explicit and recorded != configured:
        logger.info(
            "Resuming with the recorded idea-cycle budget %d instead of the configured %s",
            recorded,
            configured,
        )
        cfg.investigation.max_idea_cycles = recorded
    return None


def _is_posthoc_config_error(error: str) -> bool:
    return error.startswith(("posthoc.", "paths.posthoc_dir"))


def _pipeline_journal_root(paths: PipelinePaths, selected: set[str]) -> Path | None:
    """Choose an existing run root without claiming a fresh child output."""
    roots = {phase: paths.run_dir for phase in PIPELINE_PHASES}
    for phase in PIPELINE_PHASES:
        if phase not in selected:
            continue
        root = roots[phase]
        if phase in {"task_drafting", "investigation", "ablations"} or root.is_dir():
            return root
    return None


def _checkpoint_primary_root(cfg: PipelineConfig, paths: PipelinePaths, selected: set[str]) -> Path:
    """Return the stable root shared by checkpoints for this pipeline run."""
    if cfg.versioning.run_root is not None:
        return Path(cfg.versioning.run_root).expanduser()
    return paths.run_dir


def _phase_output_roots(paths: PipelinePaths, phase: str) -> tuple[Path, ...]:
    return {
        "task_drafting": (paths.run_dir,),
        "investigation": (paths.run_dir,),
        "ablations": (paths.run_dir, paths.ablations_dir),
        "final_evaluation": (paths.run_dir, paths.final_dir),
        "posthoc": (paths.run_dir, paths.posthoc_dir),
        "paper": (paths.run_dir, paths.paper_dir),
        "review": (paths.run_dir, paths.paper_dir),
        "revision": (paths.run_dir, paths.paper_dir, paths.revision_dir),
    }[phase]


def _safe_checkpoint_root(path: Path) -> Path:
    """Normalize a root without following an agent-controlled symlink."""
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        with open_directory(absolute, role="checkpoint root"):
            pass
    except FileNotFoundError:
        # Missing outputs are normal at pipeline start. Existing ancestors were
        # still opened without following symlinks.
        pass
    return absolute


def _checkpoint_roots(cfg: PipelineConfig, paths: PipelinePaths) -> tuple[Path, ...]:
    """Cover the primary run and any selected outputs outside of it."""
    selected = set(parse_pipeline_phases(cfg.execution.phases))
    candidates = [_checkpoint_primary_root(cfg, paths, selected)]
    candidates.extend(
        root
        for item in PIPELINE_PHASES
        if item in selected
        for root in _phase_output_roots(paths, item)
    )

    roots: list[Path] = []
    for candidate in candidates:
        try:
            resolved = _safe_checkpoint_root(candidate)
        except UnsafeGluePathError:
            logger.warning("Skipping an unsafe checkpoint root")
            continue
        if resolved in roots or any(resolved.is_relative_to(root) for root in roots):
            continue
        contained = [root for root in roots if root.is_relative_to(resolved)]
        if contained:
            insert_at = min(roots.index(root) for root in contained)
            roots = [root for root in roots if root not in contained]
            roots.insert(insert_at, resolved)
            continue
        roots.append(resolved)
    return tuple(roots)


def _checkpoint_topology_errors(
    cfg: PipelineConfig, paths: PipelinePaths, selected: set[str]
) -> list[str]:
    """Reject root layouts that can nest one checkpoint sidecar in another."""
    if not cfg.versioning.enabled:
        return []
    outputs = [
        Path(os.path.abspath(root.expanduser()))
        for phase in PIPELINE_PHASES
        if phase in selected
        for root in _phase_output_roots(paths, phase)
    ]
    explicit = (
        Path(os.path.abspath(Path(cfg.versioning.run_root).expanduser()))
        if cfg.versioning.run_root is not None
        else None
    )
    if explicit is not None and any(
        explicit != output and explicit.is_relative_to(output) for output in outputs
    ):
        return ["versioning.run_root must not be inside a selected phase output"]
    if "investigation" not in selected:
        return []
    investigation = Path(os.path.abspath(paths.run_dir.expanduser()))
    roots = outputs + ([explicit] if explicit is not None else [])
    if any(root != investigation and investigation.is_relative_to(root) for root in roots):
        return ["selected checkpoint roots must not contain paths.run_dir"]
    return []


def _task_drafting_topology_errors(paths: PipelinePaths, selected: set[str]) -> list[str]:
    """Keep immutable task sources outside the agent-writable drafting output."""
    if "task_drafting" not in selected:
        return []
    source = paths.source_task_dir.resolve(strict=False)
    output = paths.task_drafting_dir.resolve(strict=False)
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        return ["task source and task drafting output must not overlap"]
    return []


def _recover_pending_revision_publication(
    paths: PipelinePaths, selected: set[str]
) -> bool:
    """Complete a validated publication journal before downstream path checks."""
    if not selected.intersection({"paper", "review", "revision"}) or not portable_replacement_pending(
        paths.paper_dir
    ):
        return False
    logger.warning("Recovering interrupted revision publication: %s", paths.paper_dir)
    recover_incomplete_promotion(paths.paper_dir)
    return True


def _validated_drafted_scaffold(task_dir: Path) -> bool:
    """Verify the harness-authored validation receipt against current scaffold bytes."""
    if missing_task_scaffold_files(task_dir, require_contract_v1=True):
        return False
    try:
        load_task_contract(task_dir / "task_contract.json")
        report = load_task_alignment_report(
            task_dir / ".agentic_task_validation.json",
            scaffold_dir=task_dir,
        )
    except (OSError, TaskAlignmentError, TaskContractError, ValueError):
        return False
    return report.verdict == "pass"


def _checkpoint_settings(cfg: PipelineConfig) -> CheckpointSettings:
    return CheckpointSettings(
        enabled=cfg.versioning.enabled,
        git_dir_name=cfg.versioning.git_dir_name,
        max_file_bytes=cfg.versioning.max_file_bytes,
        max_total_bytes=cfg.versioning.max_total_bytes,
        timeout_seconds=cfg.versioning.timeout_seconds,
        tag_checkpoints=cfg.versioning.tag_checkpoints,
        final_include_environments=cfg.versioning.final_include_environments,
    )


def _pipeline_checkpoint(
    cfg: PipelineConfig,
    paths: PipelinePaths,
    *,
    label: str,
    phase: str,
    event: str,
    final: bool = False,
    suppress_roots: frozenset[Path] = frozenset(),
) -> frozenset[Path]:
    """Checkpoint the run root. Non-gating: never changes the pipeline's rc."""
    if not cfg.versioning.enabled:
        return frozenset()
    settings = _checkpoint_settings(cfg)
    degraded_roots: set[Path] = set()
    try:
        run_roots = _checkpoint_roots(cfg, paths)
    except Exception as exc:
        logger.warning(
            "Checkpoint '%s' degraded while resolving roots: %s", label, type(exc).__name__
        )
        return frozenset()
    for run_root in run_roots:
        if final and run_root in suppress_roots:
            logger.warning(
                "Checkpoint '%s' skipped for a root after its prior checkpoint degraded",
                label,
            )
            try:
                log_event(
                    "run_checkpoint",
                    label=label,
                    run_root=report_sensitive_path(str(run_root)),
                    status="skipped",
                    reason="previous_checkpoint_degraded",
                )
            except Exception as exc:
                logger.warning(
                    "Checkpoint '%s' skip event degraded: %s",
                    label,
                    type(exc).__name__,
                )
            continue
        if cfg.execution.dry_run:
            try:
                description = describe_checkpoint(
                    run_root=run_root, label=label, settings=settings, final=final
                )
            except Exception as exc:
                logger.warning(
                    "Checkpoint '%s' degraded during preview: %s",
                    label,
                    type(exc).__name__,
                )
                continue
            logger.info("  %s", description)
            continue
        try:
            result = write_checkpoint(
                run_root=run_root,
                label=label,
                phase=phase,
                event=event,
                settings=settings,
                run_id=paths.run_id,
                agent=cfg.agent.name,
                final=final,
                pin_run_root=True,
            )
            if result.status == "degraded":
                degraded_roots.add(run_root)
                logger.warning("Checkpoint '%s' degraded: %s", result.label, result.error)
            elif result.status == "committed":
                logger.info(
                    "Checkpoint '%s' committed as %s",
                    result.label,
                    result.tag or result.commit,
                )
            if (
                result.status in {"committed", "degraded"}
                or result.skipped_sensitive
                or result.skipped_oversize
            ):
                log_event(
                    "run_checkpoint",
                    label=result.label,
                    run_root=report_sensitive_path(str(run_root)),
                    status=result.status,
                    commit=result.commit,
                    tag=result.tag,
                    files=result.files,
                    skipped_oversize=list(result.skipped_oversize),
                    skipped_sensitive=list(result.skipped_sensitive),
                    error=result.error,
                )
        except Exception as exc:
            degraded_roots.add(run_root)
            logger.warning(
                "Checkpoint '%s' degraded for one root: %s",
                label,
                type(exc).__name__,
            )
    return frozenset(degraded_roots)


def _finish_pipeline(
    *,
    return_code: int,
    status: str,
    pipeline_started_at: float,
    outcome: str | None = None,
    cfg: PipelineConfig | None = None,
    paths: PipelinePaths | None = None,
    previous_checkpoint_degraded_roots: frozenset[Path] = frozenset(),
) -> int:
    """Close the journal, then capture it in the final full-directory checkpoint.

    ``cfg``/``paths`` are optional so the early returns that fire before paths
    are resolved keep working; those correctly take no final checkpoint.
    """
    fields: dict[str, Any] = {
        "status": status,
        "return_code": return_code,
        **_pipeline_elapsed_fields(pipeline_started_at),
    }
    if outcome is not None:
        fields["outcome"] = outcome
    log_event("pipeline_end", **fields)
    if cfg is not None and paths is not None:
        _pipeline_checkpoint(
            cfg,
            paths,
            label="final",
            phase="pipeline",
            event="pipeline_end",
            final=True,
            suppress_roots=previous_checkpoint_degraded_roots,
        )
    return return_code


def _pipeline_elapsed_fields(pipeline_started_at: float) -> dict[str, str | float]:
    elapsed_seconds = max(0.0, time.monotonic() - pipeline_started_at)
    return {
        "elapsed": format_elapsed(elapsed_seconds),
        "elapsed_seconds": round(elapsed_seconds, 1),
    }


def _verified_final_outcome(paths: PipelinePaths) -> str:
    program_dir, _ = resolve_frozen_incumbent(paths.investigation_workspace_dir.resolve())
    manifest = verify_final_manifest(
        paths.final_dir,
        task_dir=paths.task_dir,
        program_dir=program_dir,
    )
    return "qualified" if manifest["correct"] is True else "not_qualified"


def _log_paper_final_outcome_override(
    cfg: PipelineConfig, phase: str, final_outcome: str | None
) -> None:
    if (
        final_outcome == "not_qualified"
        and cfg.paper.final_outcome_policy == "allow_not_qualified"
    ):
        log_event(
            "paper_final_outcome_override",
            phase_name=phase,
            final_outcome_policy=cfg.paper.final_outcome_policy,
            final_outcome=final_outcome,
        )


def _normalize_positional_task_dir(arguments: list[str]) -> list[str]:
    """Translate a leading bare task-directory argument into ``task.dir=<path>``.

    ``agsci run examples/circpack key=value`` is shorthand for
    ``agsci run task.dir=examples/circpack key=value``. Only one positional
    argument is accepted, and it cannot be combined with an explicit ``task.dir``.
    """
    positional = [
        argument for argument in arguments if "=" not in argument and not argument.startswith("-")
    ]
    if not positional:
        return list(arguments)
    if len(positional) > 1:
        raise ValueError(
            "only one positional task directory is accepted; got "
            + ", ".join(repr(argument) for argument in positional)
        )
    if _override_is_explicit(arguments, "task.dir"):
        raise ValueError(
            f"positional task directory {positional[0]!r} conflicts with an explicit task.dir="
        )
    task_dir = positional[0]
    if "${" in task_dir:
        raise ValueError(f"positional task directory {task_dir!r} must not contain interpolations")
    return [
        f"task.dir={json.dumps(task_dir)}" if argument == task_dir else argument
        for argument in arguments
    ]


def _normalize_logging_flags(arguments: list[str]) -> list[str]:
    """Translate convenient logging flags into Hydra overrides."""
    normalized: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--log-format":
            if index + 1 >= len(arguments):
                raise ValueError("--log-format requires human or json")
            normalized.append(f"logging.format={arguments[index + 1]}")
            index += 2
            continue
        if argument.startswith("--log-format="):
            normalized.append(f"logging.format={argument.split('=', 1)[1]}")
        elif argument == "--verbose":
            normalized.append("logging.verbose=true")
        elif argument == "--heartbeat-seconds":
            if index + 1 >= len(arguments):
                raise ValueError("--heartbeat-seconds requires a non-negative integer")
            normalized.append(f"logging.heartbeat_seconds={arguments[index + 1]}")
            index += 2
            continue
        elif argument.startswith("--heartbeat-seconds="):
            normalized.append(f"logging.heartbeat_seconds={argument.split('=', 1)[1]}")
        else:
            normalized.append(argument)
        index += 1
    return normalized


def _run_pipeline(
    argv: list[str] | None,
    image_leases: list[DockerImageLease],
    runtime_cache_tasks: list[Path],
) -> int:
    """CLI entry point. Returns a process exit code."""
    pipeline_started_at = time.monotonic()
    configure_standard_logging(level=logging.INFO)

    if argv is None:
        argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        print_pipeline_help()
        return 0

    try:
        argv = _normalize_positional_task_dir(_normalize_logging_flags(argv))
    except ValueError as exc:
        logger.error("Config error: %s", exc)
        return 1

    # Pre-parse values whose valid syntax Hydra interprets as another type.
    # Timestamps look numeric, and comma-separated phases look like sweeps.
    hydra_args: list[str] = []
    dry_run_flag = False
    for arg in argv:
        if arg == "--dry-run":
            dry_run_flag = True
        elif arg.startswith("run_id="):
            run_id = arg.split("=", 1)[1].strip()
            hydra_value = run_id
            is_quoted = len(run_id) >= 2 and run_id[0] == run_id[-1] and run_id[0] in {"'", '"'}
            is_null = run_id.lower() == "null" or run_id == "~"
            if run_id == "~":
                hydra_value = "null"
            elif not is_quoted and not is_null and "${" not in run_id:
                hydra_value = json.dumps(run_id)
            hydra_args.append(f"run_id={hydra_value}")
        elif arg.startswith("execution.phases="):
            hydra_args.append(_quote_comma_separated_phases(arg))
        else:
            hydra_args.append(arg)
    if dry_run_flag:
        hydra_args.append("execution.dry_run=true")
    task_dir_is_explicit = _override_is_explicit(hydra_args, "task.dir")
    phases_are_explicit = _override_is_explicit(hydra_args, "execution.phases")

    try:
        cfg = load_pipeline_config(overrides=hydra_args)
    except (HydraException, OmegaConfBaseException, ValueError) as exc:
        logger.error("Config error: %s", config_error_message(exc))
        return 1
    try:
        selected = parse_pipeline_phases(cfg.execution.phases)
    except ValueError:
        selected = set()
    else:
        if cfg.investigation.resume and not phases_are_explicit:
            selected.discard("task_drafting")
            cfg.execution.phases = ",".join(
                phase for phase in PIPELINE_PHASES if phase in selected
            )
        if (
            cfg.investigation.draft_task_if_missing
            and not cfg.investigation.resume
            and "task_drafting" not in selected
        ):
            logger.warning(
                "investigation.execution.draft_task_if_missing is deprecated; "
                "selecting task_drafting phase"
            )
            selected.add("task_drafting")
            cfg.execution.phases = ",".join(
                phase for phase in PIPELINE_PHASES if phase in selected
            )
    if "investigation" in selected and cfg.investigation.resume:
        budget_error = _apply_recorded_idea_cycle_budget(
            cfg,
            explicit=_override_is_explicit(hydra_args, "investigation.execution.max_idea_cycles"),
        )
        if budget_error is not None:
            logger.error("Config error: %s", budget_error)
            return 1
    preflight_degraded_phases: set[str] = set()
    errors = validate_pipeline_config(cfg)
    if "posthoc" in selected:
        posthoc_errors = [error for error in errors if _is_posthoc_config_error(error)]
        if posthoc_errors:
            for error in posthoc_errors:
                logger.warning("Posthoc best-effort config degraded: %s", error)
            errors = [error for error in errors if error not in posthoc_errors]
            selected.remove("posthoc")
            preflight_degraded_phases.add("posthoc")
            cfg.paths.posthoc_dir = None
            cfg.execution.phases = ",".join(phase for phase in PIPELINE_PHASES if phase in selected)
    if errors:
        for e in errors:
            logger.error("Config error: %s", e)
        return 1
    configure_standard_logging(
        level=logging.INFO if cfg.logging.verbose or cfg.execution.dry_run else logging.WARNING
    )
    if not selected:
        logger.warning("Pipeline completed with only a degraded best-effort posthoc phase.")
        return 0
    try:
        if cfg.execution.dry_run:
            preview_runtime_storage()
        else:
            initialize_runtime_storage()
    except (OSError, ValueError) as exc:
        logger.error("Temporary storage initialization failed: %s", exc)
        return 1
    paths = derive_paths(cfg)
    cfg.run_id = paths.run_id
    investigation_outcome = (
        None
        if "investigation" in selected or not selected.intersection(_DOWNSTREAM_PHASES)
        else completed_investigation_outcome(paths.investigation_workspace_dir)
    )
    terminal_no_winner = investigation_outcome == "completed_no_winner"
    try:
        terminal_downstream_only = (
            terminal_no_winner
            and selected <= _DOWNSTREAM_PHASES
            and not downstream_study_available(cfg, paths.investigation_workspace_dir)
        )
    except (OSError, ValueError) as exc:
        logger.error("Invalid downstream selection: %s", exc)
        return 1
    if not cfg.execution.dry_run:
        runtime_cache_tasks.append(paths.task_dir)
    drafting_path_errors = _task_drafting_topology_errors(paths, selected)
    if drafting_path_errors:
        for error in drafting_path_errors:
            logger.error("Config error: %s", error)
        return 1
    checkpoint_errors = _checkpoint_topology_errors(cfg, paths, selected)
    if checkpoint_errors:
        for error in checkpoint_errors:
            logger.error("Config error: %s", error)
        return 1
    if not cfg.execution.dry_run and not terminal_downstream_only:
        try:
            _recover_pending_revision_publication(paths, selected)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Revision publication recovery failed: %s", exc)
            return 1
    if selected.intersection({"review", "revision"}) and not terminal_downstream_only:
        review_path_errors = validate_review_output_path(
            paths.run_dir,
            paths.paper_dir,
        )
        if review_path_errors:
            for error in review_path_errors:
                logger.error("Config error: %s", error)
            return 1
    if not cfg.execution.dry_run and not terminal_downstream_only:
        try:
            _pin_evaluation_images(cfg, selected, paths, image_leases)
        except ValueError as exc:
            logger.error("Container image preflight failed: %s", exc)
            return 1
        gpu_error = prepare_gpu_allocation(cfg, selected, os.environ)
        if gpu_error is not None:
            logger.error("GPU allocation preflight failed: %s", gpu_error)
            return 1
    run_task_available = False
    try:
        if _run_task_reuse_required(cfg, selected) and not terminal_downstream_only:
            run_task_available = ensure_run_task_for_reuse(
                paths,
                dry_run=cfg.execution.dry_run,
                source_is_explicit=task_dir_is_explicit,
            )
        else:
            run_task_available = False
            if paths.task_dir.is_dir() and not paths.task_dir.is_symlink():
                try:
                    run_task_available = task_snapshot_is_complete(paths.task_dir)
                except UnsafeGluePathError:
                    if (
                        "task_drafting" not in selected
                        or not task_snapshot_publication_is_complete(paths.task_dir)
                        or not _validated_drafted_scaffold(paths.task_dir)
                    ):
                        raise
                    logger.warning("retrying task drafting from an unsealed task snapshot")
                    run_task_available = True
        paths = _apply_stored_task_name(cfg, paths, selected)
    except (UnsafeGluePathError, ValueError) as exc:
        logger.error("Could not prepare run-local task: %s", exc)
        return 1
    journal_root = _pipeline_journal_root(paths, selected)
    configure_event_logging(
        log_format=cfg.logging.format,
        verbose=cfg.logging.verbose,
        heartbeat_seconds=cfg.logging.heartbeat_seconds,
        results_dir=journal_root or Path.cwd(),
        worker_index=0,
        source="pipeline",
        renderer="pipeline",
        journal_name="pipeline_events.jsonl",
        journal_enabled=journal_root is not None and not cfg.execution.dry_run,
        pipeline_started_at=pipeline_started_at,
    )
    log_event(
        "pipeline_start",
        task=paths.task_name,
        run_id=paths.run_id,
        phases=[phase for phase in PIPELINE_PHASES if phase in selected],
        dry_run=cfg.execution.dry_run,
        investigation_selected="investigation" in selected,
        show_run_header=cfg.logging.show_run_header,
        **{
            "agent.name": cfg.agent.name,
            "agent.model": cfg.agent.model,
            "agent.reasoning_effort": cfg.agent.reasoning_effort,
            "paths.task_dir": paths.task_dir,
            "paths.results_dir": paths.run_dir,
            "execution.max_idea_cycles": cfg.investigation.max_idea_cycles,
            "execution.max_program_evals_per_attempt": (
                cfg.investigation.max_program_evals_per_attempt
            ),
            "execution.workers": cfg.investigation.workers,
            "container.enabled": cfg.container.enabled,
        },
    )
    previous_checkpoint_degraded_roots = _pipeline_checkpoint(
        cfg, paths, label="pipeline start", phase="pipeline", event="pipeline_start"
    )
    logger.info("  run_id:      %s", paths.run_id)
    if cfg.logging.verbose:
        _log_banner(cfg, paths, selected)

    first_failure_rc: int | None = None
    degraded_phases = set(preflight_degraded_phases)
    phase_succeeded: dict[str, bool] = {}
    required_final_failed = False
    required_final_not_qualified = False
    allow_not_qualified = cfg.paper.final_outcome_policy == "allow_not_qualified"
    final_outcome: str | None = None
    investigation_workspace_ready = False
    downstream_selection: dict[str, Any] | None = None
    downstream_selection_checked = False
    drafted_dependencies_invalid = False
    selected_order = [phase for phase in PIPELINE_PHASES if phase in selected]
    container_phases = {
        "task_drafting",
        "investigation",
        "ablations",
        "posthoc",
        "paper",
        "final_evaluation",
        "revision",
    }
    for phase in PIPELINE_PHASES:
        if phase not in selected:
            continue
        if phase in _DOWNSTREAM_PHASES and investigation_outcome == "completed_no_winner":
            if not downstream_selection_checked and cfg.execution.no_winner_policy != "skip":
                try:
                    if cfg.execution.dry_run:
                        downstream_selection = preview_downstream_selection(
                            paths.investigation_workspace_dir
                        )
                    else:
                        existing_selection = load_downstream_selection(
                            paths.investigation_workspace_dir
                        )
                        if existing_selection is None and (
                            (paths.final_dir / "final_manifest.json").exists()
                            or (paths.final_dir / "final_manifest.json").is_symlink()
                            or (paths.final_dir / "results").exists()
                        ):
                            raise ValueError("cannot select a fallback after final evaluation")
                        downstream_selection = prepare_downstream_selection(
                            paths.investigation_workspace_dir
                        )
                except (OSError, ValueError) as exc:
                    logger.error("Could not freeze downstream study: %s", exc)
                    first_failure_rc = first_failure_rc or 2
                    log_event("phase_end", phase_name=phase, label=phase_label(phase),
                              status="failed", reason=str(exc))
                    break
                downstream_selection_checked = True
                if downstream_selection is not None:
                    log_event(
                        "downstream_selection",
                        selection_kind=downstream_selection["selection_kind"],
                        combined_score=downstream_selection["combined_score"],
                        baseline_score=downstream_selection.get("baseline_score"),
                        source_attempt_id=downstream_selection.get("source_attempt_id"),
                        program_tree_sha256=downstream_selection.get("program_tree_sha256"),
                        investigation_outcome=investigation_outcome,
                        dry_run=cfg.execution.dry_run,
                    )
            if downstream_selection is None:
                reason = "investigation completed without a qualifying winner"
                logger.warning("Skipping %s: %s; no downstream study selected.", phase, reason)
                log_event(
                    "phase_end",
                    phase_name=phase,
                    label=phase_label(phase),
                    status="skipped",
                    reason=reason,
                )
                phase_succeeded[phase] = False
                continue
        if (
            phase == "posthoc"
            and "ablations" in selected
            and not phase_succeeded.get("ablations", False)
        ):
            logger.warning(
                "Skipping posthoc because selected ablations did not succeed; "
                "paper will continue without posthoc evidence."
            )
            degraded_phases.add(phase)
            phase_succeeded[phase] = False
            log_event(
                "phase_end",
                phase_name=phase,
                label=phase_label(phase),
                status="degraded",
                reason="selected ablations did not succeed",
            )
            previous_checkpoint_degraded_roots = _pipeline_checkpoint(
                cfg, paths, label="posthoc phase", phase=phase, event="phase_end"
            )
            continue
        if (
            phase in {"paper", "review", "revision"}
            and task_requires_final_evaluation(paths.task_dir)
            and (not cfg.execution.dry_run or "final_evaluation" not in selected)
            and final_outcome is None
            and not required_final_failed
        ):
            try:
                final_outcome = _verified_final_outcome(paths)
            except FinalEvaluationError as exc:
                logger.error("Could not verify required final evaluation: %s", exc)
                required_final_failed = True
                if first_failure_rc is None:
                    first_failure_rc = 2
            else:
                required_final_not_qualified = final_outcome == "not_qualified"
                phase_succeeded["final_evaluation"] = True
        if phase == "paper" and task_requires_final_evaluation(paths.task_dir) and (
            (
                required_final_not_qualified
                and downstream_selection is None
                and not allow_not_qualified
            )
            or not phase_succeeded.get("final_evaluation", False)
        ):
            if required_final_not_qualified:
                reason = "required final evaluation did not qualify"
                logger.warning("Skipping paper because the final evaluation did not qualify.")
            else:
                reason = "required final evaluation did not succeed"
                logger.error(
                    "Skipping paper because the task requires a successful final evaluation."
                )
            log_event(
                "phase_end",
                phase_name=phase,
                label=phase_label(phase),
                status="skipped",
                reason=reason,
            )
            phase_succeeded[phase] = False
            if not required_final_not_qualified and first_failure_rc is None:
                first_failure_rc = 2
            if cfg.execution.fail_fast and not required_final_not_qualified:
                break
            continue
        if phase in {"review", "revision"} and (
            required_final_failed
            or (
                required_final_not_qualified
                and downstream_selection is None
                and not allow_not_qualified
            )
        ):
            if required_final_not_qualified:
                reason = "required final evaluation did not qualify"
                logger.warning(
                    "Skipping %s because the final evaluation did not qualify.", phase
                )
            else:
                reason = "required final evaluation failed"
                logger.error(
                    "Skipping %s because the task's required final evaluation failed.", phase
                )
            log_event(
                "phase_end",
                phase_name=phase,
                label=phase_label(phase),
                status="skipped",
                reason=reason,
            )
            phase_succeeded[phase] = False
            continue
        dependency = {
            "investigation": "task_drafting",
            "ablations": "investigation",
            "final_evaluation": "investigation",
            "posthoc": "investigation",
            "paper": "investigation",
            "review": "paper",
            "revision": "review",
        }.get(phase)
        if (
            dependency is not None
            and dependency in selected
            and not phase_succeeded.get(dependency, False)
        ):
            reason = f"dependency {dependency} failed"
            logger.error("Skipping %s: %s.", phase, reason)
            log_event(
                "phase_end",
                phase_name=phase,
                label=phase_label(phase),
                status="skipped",
                reason=reason,
            )
            phase_succeeded[phase] = False
            continue
        if phase in {"review", "revision"}:
            _log_paper_final_outcome_override(cfg, phase, final_outcome)
        phase_started_at = time.monotonic()
        log_event(
            "phase_start",
            phase_name=phase,
            label=phase_label(phase),
            current=selected_order.index(phase) + 1,
            total=len(selected_order),
        )
        image_preflight_failed = False
        if (
            cfg.container.enabled
            and phase in container_phases
            and not cfg.execution.dry_run
        ):
            try:
                _verify_image_leases(image_leases)
            except ValueError as exc:
                logger.error("Container image lease preflight failed: %s", exc)
                image_preflight_failed = True
        if image_preflight_failed:
            rc = 1
            if phase == "posthoc":
                phase_succeeded[phase] = False
                degraded_phases.add(phase)
                log_event(
                    "phase_end",
                    phase_name=phase,
                    label=phase_label(phase),
                    status="degraded",
                    elapsed=f"{time.monotonic() - phase_started_at:.1f}s",
                    return_code=rc,
                    reason="container image lease unavailable",
                )
                previous_checkpoint_degraded_roots = _pipeline_checkpoint(
                    cfg, paths, label="posthoc phase", phase=phase, event="phase_end"
                )
                continue
        elif phase in {"task_drafting", "investigation"}:
            task_drafting_identity: tuple[int, int] | None = None
            try:
                if run_task_available:
                    pass
                elif not cfg.investigation.resume and cfg.execution.dry_run:
                    preview_run_task(paths)
                elif not cfg.investigation.resume:
                    prepare_run_task(paths)
                    run_task_available = True
                if (
                    phase == "investigation"
                    and not cfg.execution.dry_run
                    and not investigation_workspace_ready
                ):
                    if cfg.investigation.resume:
                        _reuse_investigation_workspace(paths)
                    else:
                        _prepare_investigation_workspace(paths)
                        write_resume_policy(
                            paths.run_dir,
                            max_idea_cycles=cfg.investigation.max_idea_cycles,
                        )
                    investigation_workspace_ready = True
                    if not _write_investigation_completion(
                        investigation_completion_receipt_path(
                            paths.investigation_workspace_dir
                        ),
                        {"schema_version": 1, "status": "running"},
                    ):
                        raise UnsafeGluePathError("could not invalidate investigation completion")
                if phase == "task_drafting" and not cfg.execution.dry_run:
                    task_drafting_identity = _claim_task_drafting_workspace(paths)
            except (ResumeConfigError, UnsafeGluePathError, ValueError) as exc:
                logger.error("Could not prepare run-local task or investigation workspace: %s", exc)
                log_event(
                    "phase_end",
                    phase_name=phase,
                    label=phase_label(phase),
                    status="failed",
                    elapsed=f"{time.monotonic() - phase_started_at:.1f}s",
                )
                return _finish_pipeline(
                    return_code=1,
                    status="failed",
                    pipeline_started_at=pipeline_started_at,
                    cfg=cfg,
                    paths=paths,
                    previous_checkpoint_degraded_roots=(
                        previous_checkpoint_degraded_roots
                    ),
                )
            command = (
                build_task_drafting_cmd(cfg, paths)
                if phase == "task_drafting"
                else build_investigation_cmd(cfg, paths)
            )
            env_overrides = (
                {
                    PIPELINE_STARTED_AT_ENV: repr(pipeline_started_at),
                    TASK_RUNTIME_CACHE_HANDOFF_ENV: "1",
                }
                if phase == "task_drafting"
                else None
            )
            rc = run_cmd(
                command,
                dry_run=cfg.execution.dry_run,
                env_overrides=env_overrides,
            )
            if (
                phase == "task_drafting" and not cfg.execution.dry_run
                and rc not in HOST_FAILURES
            ):
                try:
                    assert task_drafting_identity is not None
                    _verify_task_drafting_workspace(paths, task_drafting_identity)
                    if not paths.task_dir.is_dir():
                        raise UnsafeGluePathError("task drafting did not publish a task snapshot")
                    if rc == 0:
                        discard_managed_task_runtime(paths.task_dir)
                        reseal_run_task(paths)
                        if (
                            "paper" in selected
                            and "final_evaluation" not in selected
                            and task_requires_final_evaluation(paths.task_dir)
                        ):
                            logger.error(
                                "The drafted task requires final evaluation before paper; "
                                "include final in execution.phases. The published task is "
                                "preserved; no downstream work was started."
                            )
                            drafted_dependencies_invalid = True
                            rc = 2
                except (OSError, UnsafeGluePathError) as exc:
                    logger.error("Could not publish drafted task snapshot: %s", exc)
                    rc = 1
            if phase == "investigation" and rc == 0 and not cfg.execution.dry_run:
                completion_errors = validate_completed_investigation(
                    paths.investigation_workspace_dir
                )
                if completion_errors:
                    completion_payload = investigation_completion_payload(
                        paths.investigation_workspace_dir
                    )
                    if completion_payload is not None and (
                        completion_payload["outcome"] == "completed_no_winner"
                    ):
                        completion_recorded = _write_investigation_completion(
                            investigation_completion_receipt_path(
                                paths.investigation_workspace_dir
                            ),
                            completion_payload,
                        )
                        if completion_recorded:
                            investigation_outcome = "completed_no_winner"
                            logger.warning(
                                "Investigation completed without a scored correct winner."
                            )
                        else:
                            rc = 2
                    else:
                        for error in completion_errors:
                            logger.error("Investigation validation failed: %s", error)
                        rc = 2
                else:
                    completion_payload = investigation_completion_payload(
                        paths.investigation_workspace_dir
                    )
                    if completion_payload is None:
                        logger.error("Investigation completion state is malformed.")
                        rc = 2
                    else:
                        if not _write_investigation_completion(
                            investigation_completion_receipt_path(
                                paths.investigation_workspace_dir
                            ),
                            completion_payload,
                        ):
                            rc = 2
                        else:
                            investigation_outcome = "winner"
        elif phase == "ablations":
            if "investigation" not in selected and run_task_available:
                try:
                    reuse_run_task(paths)
                except UnsafeGluePathError as exc:
                    logger.error("Could not reuse run-local task: %s", exc)
                    log_event(
                        "phase_end",
                        phase_name=phase,
                        label=phase_label(phase),
                        status="failed",
                        elapsed=f"{time.monotonic() - phase_started_at:.1f}s",
                    )
                    return _finish_pipeline(
                        return_code=1,
                        status="failed",
                        pipeline_started_at=pipeline_started_at,
                        cfg=cfg,
                        paths=paths,
                        previous_checkpoint_degraded_roots=(
                            previous_checkpoint_degraded_roots
                        ),
                    )
            rc = run_cmd(build_ablations_cmd(cfg, paths), dry_run=cfg.execution.dry_run)
        elif phase == "final_evaluation":
            rc = run_cmd(
                build_final_cmd(
                    cfg,
                    paths,
                    image_recovery_ref=_image_lease_tag(
                        cfg.container.image,
                        image_leases,
                    ),
                ),
                dry_run=cfg.execution.dry_run,
            )
            if rc == 0 and not cfg.execution.dry_run:
                try:
                    final_outcome = _verified_final_outcome(paths)
                except FinalEvaluationError as exc:
                    logger.error("Could not verify completed final evaluation: %s", exc)
                    rc = 1
        elif phase == "posthoc":
            rc = 0
            prepared_output_identity: tuple[int, int] | None = None
            try:
                if "investigation" not in selected:
                    ensure_run_task_for_reuse(
                        paths,
                        dry_run=cfg.execution.dry_run,
                        source_is_explicit=task_dir_is_explicit,
                    )
                    if not cfg.execution.dry_run:
                        reuse_run_task(paths)
                if cfg.container.enabled and not cfg.execution.dry_run:
                    cfg.container.image = resolve_immutable_image(cfg.container.image)
                prepared_output_identity = _prepare_posthoc_output(
                    paths, dry_run=cfg.execution.dry_run
                )
            except (UnsafeGluePathError, ValueError) as exc:
                logger.warning("Posthoc preflight degraded: %s", exc)
                rc = 2
            if rc == 0:
                try:
                    if prepared_output_identity is not None:
                        _verify_posthoc_output_identity(paths, prepared_output_identity)
                except UnsafeGluePathError as exc:
                    logger.warning("Posthoc launch preflight degraded: %s", exc)
                    rc = 2
                if rc == 0:
                    rc = run_cmd(build_posthoc_cmd(cfg, paths), dry_run=cfg.execution.dry_run)
            if prepared_output_identity is not None:
                try:
                    _verify_posthoc_output_identity(paths, prepared_output_identity)
                except UnsafeGluePathError as exc:
                    logger.warning("Posthoc output identity degraded after launch: %s", exc)
                    prepared_output_identity = None
                    rc = 2
            posthoc_ready = cfg.execution.dry_run or (
                rc == 0 and _current_posthoc_artifacts_available(paths)
            )
            phase_succeeded[phase] = posthoc_ready
            if not posthoc_ready:
                degraded_phases.add(phase)
                reason = "command_failed" if rc != 0 else "no_valid_analyses"
                if prepared_output_identity is not None:
                    _record_posthoc_status(paths, status="degraded", reason=reason, return_code=rc)
                logger.warning(
                    "Post-hoc analysis best-effort phase degraded (%s, rc=%d); paper will continue.",
                    reason,
                    rc,
                )
            elif not cfg.execution.dry_run:
                _record_posthoc_status(paths, status="completed", reason=None, return_code=rc)
            log_event(
                "phase_end",
                phase_name=phase,
                label=phase_label(phase),
                status="completed" if posthoc_ready else "degraded",
                elapsed=f"{time.monotonic() - phase_started_at:.1f}s",
                return_code=rc,
            )
            previous_checkpoint_degraded_roots = _pipeline_checkpoint(
                cfg,
                paths,
                label="posthoc phase",
                phase=phase,
                event="phase_end",
            )
            continue
        elif phase == "paper":
            paper_requires_final = task_requires_final_evaluation(paths.task_dir)
            if (
                not paper_requires_final
                and final_outcome is None
                and not cfg.execution.dry_run
                and (
                    (paths.final_dir / "final_manifest.json").exists()
                    or (paths.final_dir / "final_manifest.json").is_symlink()
                )
            ):
                try:
                    final_outcome = _verified_final_outcome(paths)
                except FinalEvaluationError as exc:
                    logger.warning("Ignoring invalid optional final evaluation: %s", exc)
            try:
                run_glue(paths, cfg, dry_run=cfg.execution.dry_run)
            except UnsafeGluePathError as exc:
                logger.error("Could not prepare paper inputs: %s", exc)
                log_event(
                    "phase_end",
                    phase_name=phase,
                    label=phase_label(phase),
                    status="failed",
                    elapsed=f"{time.monotonic() - phase_started_at:.1f}s",
                )
                return _finish_pipeline(
                    return_code=1,
                    status="failed",
                    pipeline_started_at=pipeline_started_at,
                    cfg=cfg,
                    paths=paths,
                    previous_checkpoint_degraded_roots=(
                        previous_checkpoint_degraded_roots
                    ),
                )
            if task_requires_final_evaluation(
                paths.paper_task_dir
            ) and not phase_succeeded.get("final_evaluation", False):
                logger.error("Contracted paper phase requires a successful final phase.")
                rc = 2
                phase_succeeded[phase] = False
                log_event(
                    "phase_end",
                    phase_name=phase,
                    label=phase_label(phase),
                    status="failed",
                    reason="contracted task requires final phase",
                )
                if first_failure_rc is None:
                    first_failure_rc = rc
                if cfg.execution.fail_fast:
                    break
                continue
            include_posthoc = phase_succeeded.get("posthoc", False)
            if "posthoc" not in selected and cfg.paths.posthoc_dir:
                input_errors = validate_posthoc_paper_input(paths.posthoc_dir)
                if input_errors:
                    for error in input_errors:
                        logger.warning("Ignoring unavailable posthoc input: %s", error)
                    degraded_phases.add("posthoc")
                    log_event(
                        "optional_input_degraded",
                        phase_name="posthoc",
                        consumer_phase=phase,
                        reasons=input_errors,
                    )
                else:
                    include_posthoc = True
            include_final_evaluation = (
                paper_requires_final or final_outcome == "qualified"
                or (
                    (downstream_selection is not None or allow_not_qualified)
                    and final_outcome == "not_qualified"
                )
            )
            if not include_final_evaluation:
                paper_cmd = build_paper_cmd_without_final(
                    cfg,
                    paths,
                    posthoc_dir=paths.posthoc_dir if include_posthoc else None,
                )
            else:
                paper_cmd = (
                    build_paper_cmd_with_posthoc(cfg, paths)
                    if include_posthoc
                    else build_paper_cmd(cfg, paths)
                )
            _log_paper_final_outcome_override(cfg, phase, final_outcome)
            rc = run_cmd(
                paper_cmd,
                dry_run=cfg.execution.dry_run,
                output_line_fn=lambda line: forward_paper_output(line, event_fn=log_event),
            )
        elif phase == "review":
            if cfg.execution.dry_run:
                logger.info(
                    "Review preview — %s",
                    describe_pipeline_review(cfg=cfg, paths=paths),
                )
                rc = 0
            else:
                try:
                    review_input_errors = validate_review_input(
                        paths.run_dir,
                        paths.paper_dir,
                    )
                    if review_input_errors:
                        raise ValueError("; ".join(review_input_errors))
                    if "revision" in selected and initial_review_must_be_preserved(
                        paths.paper_dir
                    ):
                        review_result = FinalAiReviewResult(
                            status="completed",
                            review_path=paths.paper_dir / "ai_review.before_revision.json",
                            log_path=paths.paper_dir / "ai_review.before_revision.log",
                        )
                    elif "revision" in selected:
                        review_result = run_pipeline_review(
                            cfg=cfg,
                            paths=paths,
                            artifact_stem="ai_review.before_revision",
                            manifest_key="before_revision",
                        )
                    else:
                        review_result = run_pipeline_review(cfg=cfg, paths=paths)
                except Exception as exc:
                    logger.warning("Review phase degraded: %s", exc)
                    degraded_phases.add(phase)
                    phase_succeeded[phase] = False
                    log_event(
                        "phase_end",
                        phase_name=phase,
                        label=phase_label(phase),
                        status="degraded",
                        elapsed=f"{time.monotonic() - phase_started_at:.1f}s",
                        reason=str(exc),
                    )
                    previous_checkpoint_degraded_roots = _pipeline_checkpoint(
                        cfg, paths, label="review phase", phase=phase, event="phase_end"
                    )
                    continue
                if review_result.status == "failed":
                    logger.warning("Review phase degraded: %s", review_result.error or "failed")
                    degraded_phases.add(phase)
                    phase_succeeded[phase] = False
                    log_event(
                        "phase_end",
                        phase_name=phase,
                        label=phase_label(phase),
                        status="degraded",
                        elapsed=f"{time.monotonic() - phase_started_at:.1f}s",
                        reason=review_result.error,
                        log_path=review_result.log_path,
                    )
                    previous_checkpoint_degraded_roots = _pipeline_checkpoint(
                        cfg, paths, label="review phase", phase=phase, event="phase_end"
                    )
                    continue
                log_event(
                    "review_result",
                    overall_score=getattr(review_result, "overall_score", None),
                    decision=getattr(review_result, "decision", None),
                    review_path=review_result.review_path,
                )
                rc = 0
        else:
            if cfg.execution.dry_run:
                logger.info(
                    "Revision preview — %s",
                    describe_pipeline_revision(cfg=cfg, paths=paths),
                )
                rc = 0
            else:
                revision_error: str | None
                revision_path: Path | None
                try:

                    def execute_revision(command: list[str]) -> tuple[str, int]:
                        output_tail = ""

                        def capture_revision_event(event: str, **fields: Any) -> None:
                            nonlocal output_tail
                            if fields.get("state") == "failed":
                                output_tail = retain_revision_output_tail(
                                    output_tail,
                                    revision_failure_event_detail(fields),
                                )
                            log_event(event, **fields)

                        def capture_revision_output(line: str) -> str:
                            nonlocal output_tail
                            retained = forward_paper_output(line, event_fn=capture_revision_event)
                            if retained:
                                output_tail = retain_revision_output_tail(output_tail, retained)
                            return retained

                        return_code = run_cmd(
                            command,
                            dry_run=False,
                            output_line_fn=capture_revision_output,
                        )
                        return output_tail, return_code

                    revision_result = run_pipeline_revision(
                        cfg=cfg,
                        paths=paths,
                        execute_command=execute_revision,
                        review_runner=run_pipeline_review,
                    )
                except Exception as exc:
                    revision_error = str(exc)
                    revision_status = "failed"
                    revision_path = None
                else:
                    revision_error = revision_result.error
                    revision_status = revision_result.status
                    revision_path = revision_result.revision_dir
                if revision_status != "completed":
                    logger.warning(
                        "Revision phase degraded: %s", revision_error or revision_status
                    )
                    degraded_phases.add(phase)
                    phase_succeeded[phase] = False
                    log_event(
                        "phase_end",
                        phase_name=phase,
                        label=phase_label(phase),
                        status="degraded",
                        elapsed=f"{time.monotonic() - phase_started_at:.1f}s",
                        reason=revision_error,
                        revision_dir=revision_path,
                    )
                    previous_checkpoint_degraded_roots = _pipeline_checkpoint(
                        cfg, paths, label="revision phase", phase=phase, event="phase_end"
                    )
                    continue
                log_event(
                    "revision_result",
                    revision_dir=revision_result.revision_dir,
                    before_review_path=revision_result.before_review_path,
                    after_review_path=revision_result.after_review_path,
                    **revision_review_fields(
                        revision_result.before_review_path, revision_result.after_review_path
                    ),
                )
                rc = 0

        final_evidence_required = phase == "final_evaluation" and task_requires_final_evaluation(
            paths.task_dir
        )
        host_failure = HOST_FAILURES.get(rc) if phase in {"task_drafting", "investigation"} else None
        if rc != 0 and final_evidence_required:
            required_final_failed = True
        if final_evidence_required and final_outcome == "not_qualified":
            required_final_not_qualified = True
        optional_phase_degraded = rc != 0 and (
            phase == "ablations"
            or (phase == "final_evaluation" and not final_evidence_required)
        )
        phase_succeeded[phase] = rc == 0
        phase_status = "completed" if rc == 0 else "failed"
        if optional_phase_degraded:
            phase_status = "degraded"
        phase_fields: dict[str, Any] = {
            "phase_name": phase,
            "label": phase_label(phase),
            "status": phase_status,
            "elapsed": f"{time.monotonic() - phase_started_at:.1f}s",
            "return_code": rc,
        }
        if phase == "final_evaluation" and rc == 0 and final_outcome is not None:
            phase_fields["outcome"] = final_outcome
        if phase == "investigation" and rc == 0 and investigation_outcome is not None:
            phase_fields["outcome"] = investigation_outcome
        if host_failure is not None:
            phase_fields["failure_category"], phase_fields["failure_scope"] = host_failure
        log_event("phase_end", **phase_fields)
        previous_checkpoint_degraded_roots = _pipeline_checkpoint(
            cfg,
            paths,
            label=f"{phase} phase",
            phase=phase,
            event="phase_end",
        )
        if optional_phase_degraded:
            degraded_phases.add(phase)
            if phase == "ablations":
                logger.warning(
                    "Ablations best-effort phase degraded (rc=%d); paper will continue.",
                    rc,
                )
            else:
                logger.warning(
                    "Final evaluation best-effort phase degraded (rc=%d); "
                    "remaining selected phases will continue.",
                    rc,
                )
            continue
        if rc != 0:
            if first_failure_rc is None:
                first_failure_rc = rc
            logger.error("%s failed (rc=%d).", phase_label(phase), rc)
            _log_resume_hint(phase, cfg, paths)
            if cfg.execution.fail_fast or drafted_dependencies_invalid or host_failure is not None:
                return _finish_pipeline(
                    return_code=rc,
                    status="failed",
                    pipeline_started_at=pipeline_started_at,
                    cfg=cfg,
                    paths=paths,
                    previous_checkpoint_degraded_roots=(
                        previous_checkpoint_degraded_roots
                    ),
                )

    if first_failure_rc is not None:
        logger.error("Pipeline finished with failures (first rc=%d).", first_failure_rc)
        return _finish_pipeline(
            return_code=first_failure_rc,
            status="failed",
            pipeline_started_at=pipeline_started_at,
            cfg=cfg,
            paths=paths,
            previous_checkpoint_degraded_roots=previous_checkpoint_degraded_roots,
        )

    logger.info("Pipeline complete. Paper output dir: %s", paths.paper_dir)
    if phase_succeeded.get("paper", False):
        logger.info("Expected paper PDF: %s", paths.paper_dir / "paper.pdf")
    if phase_succeeded.get("review", False):
        logger.info("Expected AI review: %s", paths.paper_dir / "ai_review.json")
    pipeline_status = "completed_degraded" if degraded_phases else "completed"
    if degraded_phases:
        logger.warning(
            "Pipeline completed with degraded optional phases: %s",
            ", ".join(sorted(degraded_phases)),
        )
    return _finish_pipeline(
        return_code=0,
        status=pipeline_status,
        pipeline_started_at=pipeline_started_at,
        outcome=(
            "completed_no_winner"
            if investigation_outcome == "completed_no_winner"
            else "not_qualified" if final_outcome == "not_qualified" else None
        ),
        cfg=cfg,
        paths=paths,
        previous_checkpoint_degraded_roots=previous_checkpoint_degraded_roots,
    )


def _main(argv: list[str] | None = None) -> int:
    image_leases: list[DockerImageLease] = []
    runtime_cache_tasks: list[Path] = []
    try:
        return _run_pipeline(argv, image_leases, runtime_cache_tasks)
    finally:
        _release_image_leases(image_leases)
        for task_dir in runtime_cache_tasks:
            try:
                discard_task_runtime_install_cache(task_dir)
            except (OSError, ValueError) as exc:
                logger.warning("Could not remove qualified runtime cache: %s", exc)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and always release structured logging resources."""
    pipeline_started_at = time.monotonic()
    try:
        return _main(argv)
    except KeyboardInterrupt:
        if event_logging_configured():
            log_event(
                "pipeline_end",
                status="interrupted",
                return_code=130,
                **_pipeline_elapsed_fields(pipeline_started_at),
            )
        return 130
    except Exception as exc:
        if event_logging_configured():
            log_event(
                "pipeline_end",
                status="failed",
                error_type=type(exc).__name__,
                **_pipeline_elapsed_fields(pipeline_started_at),
            )
        raise
    finally:
        shutdown_event_logging()


def _apply_stored_task_name(
    cfg: PipelineConfig,
    paths: PipelinePaths,
    selected: set[str],
) -> PipelinePaths:
    snapshot_dir = _task_name_snapshot_dir(cfg, paths, selected)
    if snapshot_dir is None:
        return paths
    stored_task_name = read_task_snapshot_name(snapshot_dir)
    if stored_task_name is None:
        return paths
    return derive_paths(cfg, task_name=stored_task_name)


def _run_task_reuse_required(cfg: PipelineConfig, selected: set[str]) -> bool:
    if "investigation" in selected:
        return cfg.investigation.resume
    return bool(selected.intersection({"ablations", "final_evaluation", "paper", "revision"}))


def _task_name_snapshot_dir(
    cfg: PipelineConfig,
    paths: PipelinePaths,
    selected: set[str],
) -> Path | None:
    if "investigation" in selected:
        return paths.task_dir if cfg.investigation.resume else None
    if "ablations" in selected:
        return paths.task_dir
    if "final_evaluation" in selected:
        return paths.task_dir
    if "paper" in selected:
        return paths.paper_task_dir
    if "revision" in selected:
        return paths.paper_task_dir
    return None


GPU_COMPUTE_PHASES = frozenset({"task_drafting", "investigation", "ablations", "final_evaluation"})


def prepare_gpu_allocation(
    cfg: PipelineConfig,
    selected: Iterable[str],
    environment: MutableMapping[str, str],
    *,
    discover: Callable[..., str] = discover_gpu_selector,
) -> str | None:
    """Settle ``AGENT_CONTAINER_GPUS`` against ``container.gpu_count`` before compute phases.

    The run's GPU pool is ``container.gpu_count`` per container times the parallel
    investigation workers (one when the investigation phase is not selected). An
    explicit selector must name exactly that many GPUs; without one the first that
    many ``nvidia-smi`` reports are discovered and exported (plus the ablation sandbox
    selector when unset). With several workers the pool is exported as
    ``AGENT_CONTAINER_GPU_POOL`` for the parallel launcher to slice, and
    ``AGENT_CONTAINER_GPUS`` narrows to the first slice so every other phase still
    sees exactly ``container.gpu_count`` GPUs. Returns an error message, or ``None``.
    """
    if not GPU_COMPUTE_PHASES.intersection(selected):
        return None
    gpu_count = cfg.container.gpu_count
    workers = cfg.investigation.workers if "investigation" in selected else 1
    raw_selector = environment.get(GPU_SELECTOR_ENV)
    discovered = not (raw_selector or "").strip()
    try:
        settled = settle_gpu_pool(raw_selector, gpu_count, workers, discover=discover)
    except GpuSelectorError as exc:
        return str(exc)
    if settled is None:
        return None
    pool, first_slice = settled
    if discovered:
        environment[GPU_SELECTOR_ENV] = pool
        environment.setdefault("VARIANT_EVAL_SANDBOX_GPUS", first_slice)
        logger.info(
            "Discovered %d GPU(s) for container.gpu_count=%d x %d worker(s): %s",
            gpu_count * workers, gpu_count, workers, pool,
        )
    if workers > 1:
        environment[GPU_POOL_ENV] = pool
        environment[GPU_SELECTOR_ENV] = first_slice
        logger.info(
            "GPU pool %s is sliced %d per worker; phases outside the investigation use %s",
            pool, gpu_count, first_slice,
        )
    return None


def _pin_evaluation_images(
    cfg: PipelineConfig,
    selected: set[str],
    paths: PipelinePaths,
    image_leases: list[DockerImageLease],
) -> None:
    """Resolve immutable agent and evaluator images before selected phases start."""
    container_phases = {
        "task_drafting",
        "investigation",
        "ablations",
        "posthoc",
        "paper",
        "final_evaluation",
        "revision",
    }
    draft_selected = "task_drafting" in selected
    agent_image_ref = cfg.container.image
    agent_image_id: str | None = None
    lease_scope = str(paths.run_dir.resolve())
    if cfg.container.enabled and selected.intersection(container_phases):
        agent_lease = acquire_image_lease(
            agent_image_ref,
            lease_scope=lease_scope,
            role="agent",
        )
        image_leases.append(agent_lease)
        agent_image_id = agent_lease.image_id
        cfg.container.image = agent_image_id

    if "ablations" not in selected or not cfg.container.enabled:
        return  # native ablations evaluate on the host, without a sandbox image
    sandbox_ref = cfg.ablations.sandbox_image or agent_image_ref
    if agent_image_id is not None and sandbox_ref == agent_image_ref:
        sandbox_image_id = agent_image_id
    else:
        sandbox_lease = acquire_image_lease(
            sandbox_ref,
            lease_scope=lease_scope,
            role="sandbox",
        )
        image_leases.append(sandbox_lease)
        sandbox_image_id = sandbox_lease.image_id
    if draft_selected and agent_image_id is not None and sandbox_image_id != agent_image_id:
        raise ValueError("task drafting and ablations must use the same immutable Docker image")
    cfg.ablations.sandbox_image = sandbox_image_id


def _verify_image_leases(image_leases: list[DockerImageLease]) -> None:
    for lease in image_leases:
        lease.verify()


def _image_lease_tag(image_id: str, image_leases: list[DockerImageLease]) -> str | None:
    return next(
        (lease.tag for lease in image_leases if lease.image_id == image_id),
        None,
    )


def _release_image_leases(image_leases: list[DockerImageLease]) -> None:
    for lease in reversed(image_leases):
        try:
            lease.release()
        except Exception as exc:
            logger.warning("Docker image lease cleanup degraded for %s: %s", lease.tag, exc)


def _log_banner(cfg: PipelineConfig, paths: PipelinePaths, selected: set[str]) -> None:
    logger.info("Pipeline run")
    logger.info("  task:        %s (%s)", paths.task_name, cfg.task.dir)
    logger.info("  run task:    %s", paths.task_dir)
    logger.info("  agent:       %s (%s)", cfg.agent.name, cfg.agent.model or "<default>")
    logger.info("  run_id:      %s", paths.run_id)
    logger.info("  phases:      %s", [phase for phase in PIPELINE_PHASES if phase in selected])
    logger.info("  dry_run:     %s", cfg.execution.dry_run)
    logger.info("  run envelope:          %s", paths.run_dir)
    logger.info("  investigation workspace: %s", paths.investigation_workspace_dir)
    logger.info("  ablations dir:    %s", paths.ablations_dir)
    logger.info("  posthoc dir:      %s", paths.posthoc_dir)
    logger.info("  final dir:        %s", paths.final_dir)
    logger.info("  paper dir:        %s", paths.paper_dir)


def _log_resume_hint(phase: str, cfg: PipelineConfig, paths: PipelinePaths) -> None:
    start = PIPELINE_PHASES.index(phase)
    selected = parse_pipeline_phases(cfg.execution.phases)
    remaining = ",".join(
        candidate for candidate in PIPELINE_PHASES[start:] if candidate in selected
    )
    logger.error("  Partial outputs from the %s phase may be present.", phase)
    if phase == "investigation":
        restart_target = (
            "a new run_id and a new paths.run_dir" if cfg.paths.run_dir else "a new run_id"
        )
        pruning = cfg.investigation.prune_every_n_cycles
        resume_mode_supported = (
            cfg.investigation.workers == 1 and pruning == 0
        ) or 0 < pruning < cfg.investigation.max_idea_cycles
        if not resume_mode_supported:
            logger.error(
                "  This investigation run cannot be resumed; restart with %s.",
                restart_target,
            )
            return
        state_errors = validate_investigation_resume_state(paths.investigation_workspace_dir)
        if state_errors:
            logger.error(
                "  No usable investigation checkpoint was written; fix the "
                "original error and restart with %s.",
                restart_target,
            )
            return
    if phase == "final_evaluation":
        invocation_path = paths.final_dir / "invocation.json"
        invocation_retained = invocation_path.exists() or invocation_path.is_symlink()
        if invocation_retained:
            logger.error(
                "  The final evaluator's once-only invocation was retained; "
                "a same-run retry is intentionally refused."
            )
            logger.error("    run_id=%s", paths.run_id)
            logger.error("    paths.final_evaluation_dir=%s", paths.final_dir)
            logger.error(
                "  Resolve the retained evaluator/container state, then start a new pipeline run."
            )
            return

    logger.error("  Rerun the original command with these overrides:")
    logger.error("    execution.phases=%s", remaining)
    logger.error("    run_id=%s", paths.run_id)
    if phase == "investigation":
        logger.error("    investigation.execution.resume=true")
        logger.error(
            "    investigation.execution.max_idea_cycles=%d",
            cfg.investigation.max_idea_cycles,
        )
        logger.error("    investigation.execution.workers=%d", cfg.investigation.workers)
        logger.error(
            "    investigation.execution.prune_every_n_cycles=%d",
            cfg.investigation.prune_every_n_cycles,
        )
        logger.error(
            "    investigation.execution.max_program_evals_per_attempt=%d",
            cfg.investigation.max_program_evals_per_attempt,
        )
    logger.error("    paths.run_dir=%s", paths.run_dir)
    logger.error("    paths.ablations_dir=%s", paths.ablations_dir)
    logger.error("    paths.posthoc_dir=%s", paths.posthoc_dir)
    logger.error("    paths.final_evaluation_dir=%s", paths.final_dir)
    logger.error("    paths.paper_dir=%s", paths.paper_dir)


def _current_posthoc_artifacts_available(paths: PipelinePaths) -> bool:
    """Return whether the current command produced paper-consumable analyses."""
    from src.orchestrator.paper_data_posthoc import collect_posthoc_analyses

    try:
        analyses = collect_posthoc_analyses(
            paths.ablations_dir,
            posthoc_dir=paths.posthoc_dir,
        )
    except (OSError, UnicodeError, ValueError, RecursionError):
        logger.exception("Could not inspect posthoc artifacts: %s", paths.posthoc_dir)
        return False
    return bool(analyses)


def _prepare_posthoc_output(paths: PipelinePaths, *, dry_run: bool) -> tuple[int, int] | None:
    """Create a fresh, contained post-hoc run root without following symlinks."""
    run_dir = Path(os.path.abspath(paths.run_dir))
    output = Path(os.path.abspath(paths.posthoc_dir))
    if output.parent != run_dir or output.name != "posthoc":
        raise UnsafeGluePathError(
            f"posthoc output must be the run-envelope posthoc directory: {run_dir / 'posthoc'}"
        )
    if dry_run:
        return None
    with ensure_directory(run_dir, role="posthoc output parent") as parent_fd:
        parent_stat = os.fstat(parent_fd)
        if parent_stat.st_uid != os.getuid() or parent_stat.st_mode & 0o022:
            raise UnsafeGluePathError(
                f"posthoc output parent must be owner-controlled and not writable by others: "
                f"{run_dir}"
            )
        if entry_exists_at(parent_fd, output.name):
            raise UnsafeGluePathError(f"fresh posthoc output already exists: {output}")
        try:
            os.mkdir(output.name, 0o700, dir_fd=parent_fd)
        except OSError as exc:
            raise UnsafeGluePathError(f"could not create posthoc output: {output}") from exc
    with open_directory(output, role="fresh posthoc output") as output_fd:
        output_stat = os.fstat(output_fd)
        if output_stat.st_uid != os.getuid() or output_stat.st_mode & 0o077:
            raise UnsafeGluePathError(f"fresh posthoc output is not owner-private: {output}")
        return output_stat.st_dev, output_stat.st_ino


def _verify_posthoc_output_identity(
    paths: PipelinePaths, expected_identity: tuple[int, int]
) -> None:
    """Reject replacement of the prepared output immediately before launch."""
    with open_directory(paths.posthoc_dir, role="fresh posthoc output") as output_fd:
        output_stat = os.fstat(output_fd)
        actual_identity = output_stat.st_dev, output_stat.st_ino
        if actual_identity != expected_identity:
            raise UnsafeGluePathError(
                f"fresh posthoc output changed after preparation: {paths.posthoc_dir}"
            )


def _record_posthoc_status(
    paths: PipelinePaths,
    *,
    status: str,
    reason: str | None,
    return_code: int,
) -> None:
    """Best-effort integration status record; never blocks paper writing."""
    output_dir = paths.posthoc_dir
    temporary_name = f".integration_status.{secrets.token_hex(8)}.tmp"
    try:
        payload = {
            "status": status,
            "reason": reason,
            "return_code": return_code,
            "best_effort": True,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with open_directory(output_dir, role="posthoc output") as output_fd:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=output_fd,
            )
            try:
                with os.fdopen(temporary_fd, "w", encoding="utf-8") as temporary_file:
                    temporary_fd = -1
                    temporary_file.write(json.dumps(payload, indent=2) + "\n")
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                os.replace(
                    temporary_name,
                    "integration_status.json",
                    src_dir_fd=output_fd,
                    dst_dir_fd=output_fd,
                )
            finally:
                if temporary_fd >= 0:
                    os.close(temporary_fd)
                try:
                    os.unlink(temporary_name, dir_fd=output_fd)
                except FileNotFoundError:
                    pass
    except (OSError, UnsafeGluePathError) as exc:
        logger.warning("Could not record posthoc integration status: %s", exc)


def load_pipeline_config(overrides: list[str] | None = None) -> PipelineConfig:
    """Compose ``configs/pipeline_config.yaml`` onto the :class:`PipelineConfig` schema.

    The dataclass is the schema: Hydra type-checks every override against it,
    rejects unknown keys, and ``OmegaConf.to_object`` returns the dataclass
    with interpolations resolved. There is no per-key code here.
    """
    register_pipeline_schema()
    for override in overrides or []:
        if override.startswith("+"):
            # Hydra's append syntax bypasses struct checking and to_object drops
            # keys the schema does not declare, so an appended key would vanish.
            raise ValueError(
                f"Unsupported override {override!r}: every pipeline key is declared, "
                "so keys cannot be appended (drop the leading '+')"
            )
    with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base="1.3"):
        raw_cfg = compose(config_name="pipeline_config", overrides=overrides or [])
    schema = OmegaConf.to_object(raw_cfg)
    sections = {spec.name: getattr(schema, spec.name) for spec in dataclasses.fields(PipelineConfig)}
    for name, phase_cls in PHASE_CLASSES.items():
        sections[name] = phase_cls(flatten_mapping(sections[name]))
    return PipelineConfig(**sections)


def config_error_message(exc: BaseException) -> str:
    """Hydra wraps the OmegaConf validation error; surface both messages."""
    message = str(exc).strip()
    cause = exc.__cause__
    if cause is not None and str(cause).strip():
        cause_text = str(cause).strip().splitlines()[0]
        if cause_text not in message:
            message = f"{message}: {cause_text}"
    return message


def print_pipeline_help() -> None:
    """Print every config key with its type, default, and help text."""
    rows = describe_pipeline_config()
    key_width = max(len(row.key) for row in rows)
    type_width = max(len(row.type_name) for row in rows)
    print("Usage: agsci run [task_dir] [--dry-run] [--help] key=value ...")
    print("       python -m src.pipeline.cli [task_dir] [--dry-run] [--help] key=value ...")
    print()
    print("A bare leading argument names the task directory (shorthand for task.dir=<path>).")
    print("Every key below is a Hydra override; unknown keys are rejected.")
    print("Defaults come from src/pipeline/config.py; configs/pipeline_config.yaml only")
    print("states values that differ from a downstream stage's own default.")
    print("preset=<name> applies a checked-in task preset (configs/preset/<name>.yaml);")
    print("explicit key=value overrides still take precedence over the preset.")
    print()
    for row in rows:
        line = (
            f"  {row.key.ljust(key_width)}  {row.type_name.ljust(type_width)}"
            f"  = {row.default}"
        )
        if row.help:
            line += f"  # {row.help}"
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())

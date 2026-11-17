"""Prepare and validate a portable task-local evaluator runtime."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import tomllib
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any
from pathlib import Path

from src.final_evaluation.candidate_broker import (
    CandidateBroker,
    CandidateBrokerError,
    CandidateBrokerInfrastructureError,
    CandidateContainerCleanupError,
    ContainerMetadataCleanupError,
)
from src.final_evaluation.runner import (
    FinalEvaluationError,
    GpuAllocationInvalidError,
    _broker_recovery_root,
    _candidate_command,
    _ensure_broker_recovery_root,
    _evaluation_command,
    _reject_legacy_broker_directory,
)

from .checkpoint_security import redact_sensitive_text
from .storage_failures import storage_failure_category
from .evaluator_outputs import EvaluatorOutputContractError, validate_evaluator_outputs
from .evaluator_replay import EvaluationOutcome, replay_mismatch
from .evaluator_tree import (
    MANAGED_RUNTIME_MANIFEST_FILENAME,
    TASK_SNAPSHOT_METADATA_FILENAME,
    managed_evaluator_tree_digest,
    task_snapshot_metadata_present,
)
from .container_images import resolve_immutable_image
from .container_gpus import (
    GPU_ALLOCATION_FAILURE_CATEGORIES,
    GPU_ALLOCATION_BUSY_CATEGORY,
    GPU_ALLOCATION_INVALID_CATEGORY,
    GPU_ALLOCATION_QUARANTINED_CATEGORY,
    GPU_ALLOCATION_LEASE_ROOT,
    GpuAllocationLease,
    GpuLeaseError,
    GpuLeaseQuarantined,
    GpuLeaseTimeout,
    acquire_gpu_allocation_lease,
)
from .program_spec import entrypoint_relpath_from_spec
from .preparation_progress import preparation_step
from .reset_sequence import (
    declared_reset_sequence_matches,
    reset_sequence_diagnostic,
)
from .task_contract import (
    TASK_CONTRACT_FILENAME,
    TaskContract,
    TaskContractError,
    load_task_contract,
)
from .bounded_process import (
    BoundedProcessError,
    ProcessCleanupWitness,
    ProcessOutputLimitError,
    ProcessTimeoutError,
    run_bounded_process,
)
from .task_runtime_cache import (
    TaskRuntimeInstallCache,
    TaskRuntimeInstallCacheDeadlineExceeded,
    TaskRuntimeInstallCacheKey,
)
from .task_runtime_archive import archive_baseline_results, validate_baseline_results_dir
from .runtime_build_coordination import (
    RuntimeBuildLease,
    RuntimeBuildLeaseError,
    RuntimeBuildLeaseTimeout,
    acquire_runtime_build_lease,
)
from .task_alignment import task_tree_fingerprint
from .runtime_output import (
    TaskRuntimeOutputEvidence, failed_evaluation_evidence, output_limit_evidence, timeout_evidence,
)
from .qualification_usage import (
    measure_qualification_usage,
    observe_qualification_broker,
    retain_qualification_usage,
)

logger = logging.getLogger(__name__)

_PINNED_REQUIREMENT = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?"
    r"==[A-Za-z0-9][A-Za-z0-9.!+_-]*"
)
_PACKAGE_NAME_SEPARATOR = re.compile(r"[-_.]+")
_CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}\Z")
_RUNTIME_CID_STAGE = (
    r"(?:runtime-create|tooling-build|tooling-install|"
    r"runtime-install|runtime-evaluation)"
)
_SENSITIVE_ENV_TOKENS = (
    "AUTH",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "PROXY",
    "SECRET",
    "TOKEN",
)
RUNTIME_MANIFEST_FILENAME = MANAGED_RUNTIME_MANIFEST_FILENAME
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLING_BUILD_REQUIREMENTS = ("setuptools==80.9.0", "wheel==0.45.1")
_RUNTIME_LOCK_POLL_SECONDS = 0.05
_CONTAINER_CLEANUP_RESERVE_FRACTION = 0.1
_MAX_CONTAINER_CLEANUP_SECONDS = 30.0
_BROKER_TEARDOWN_RESERVE_SECONDS = 21.0
_TASK_BROKER_PREFIXES = {
    "runtime-search": "draft-search",
    "final-smoke": "draft-final",
}


def _tooling_package_dirs() -> tuple[Path, ...]:
    """Return the harness packages copied into drafted task runtimes.

    Read from ``[tool.setuptools].packages`` in ``pyproject.toml`` so the sealed
    install always finds every package the project declares; ``src`` itself is
    copied separately.
    """
    try:
        data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        packages = data["tool"]["setuptools"]["packages"]
    except (OSError, UnicodeError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("pyproject.toml does not declare [tool.setuptools].packages") from exc
    if not isinstance(packages, list) or not all(isinstance(name, str) for name in packages):
        raise ValueError("pyproject.toml [tool.setuptools].packages must be a list of strings")
    return tuple(Path(*name.split(".")) for name in packages if name != "src")


class TaskRuntimePreparationResult(str, Enum):
    READY = "ready"
    AUTHORED_FAILURE = "authored_failure"
    DEPENDENCY_FAILURE = "dependency_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class TaskRuntimeEvaluationRequirement(str, Enum):
    """Scientific result requirement for baseline runtime validation."""

    CORRECT_RESULT = "correct_result"
    VALID_RESULT = "valid_result"
    FEASIBLE_RESULT = "feasible_result"


@dataclass(frozen=True, slots=True)
class TaskRuntimeFailureEvidence:
    """Allowlisted scalar evidence for a runtime timeout."""

    deadline_owner: str
    contract_wall_clock_seconds: int
    evaluator_timeout_seconds: int
    candidate_rpc_timeout_seconds: int
    broker_teardown_reserve_seconds: float
    container_cleanup_reserve_seconds: float
    candidate_rpc_calls: int
    candidate_calls_before_first_reset: int
    candidate_reset_count: int
    candidate_peer_disconnects: int


@dataclass(frozen=True, slots=True)
class TaskRuntimePreparationReport:
    """Bounded, structured outcome from task runtime preparation."""

    result: TaskRuntimePreparationResult
    failure_stage: str | None
    failure_category: str | None
    diagnostic: str
    task_fingerprint: str
    elapsed_seconds: float
    failure_evidence: TaskRuntimeFailureEvidence | None = None
    output_evidence: TaskRuntimeOutputEvidence | None = None
    compute_usage: dict[str, object] | None = None

    @property
    def ready(self) -> bool:
        return self.result is TaskRuntimePreparationResult.READY


class _RuntimeReportDiagnosticHandler(logging.Handler):
    """Retain a private, sanitized tail of runtime errors."""

    def __init__(self, max_bytes: int = 8 * 1024) -> None:
        super().__init__(logging.ERROR)
        self._max_bytes = max_bytes
        self._tail = bytearray()
        self._failure_evidence: TaskRuntimeFailureEvidence | None = None
        self._output_evidence: TaskRuntimeOutputEvidence | None = None
        self.storage_failure: tuple[str, str] | None = None
        self.gpu_failure_category: str | None = None

    def emit(self, record: logging.LogRecord) -> None:
        gpu_category = getattr(record, "task_runtime_gpu_failure", None)
        if isinstance(gpu_category, str) and gpu_category in GPU_ALLOCATION_FAILURE_CATEGORIES:
            self.gpu_failure_category = gpu_category
        try:
            message = redact_sensitive_text(record.getMessage())
            encoded = (message + "\n").encode("utf-8", errors="replace")
        except Exception:
            return
        evidence = getattr(record, "task_runtime_failure_evidence", None)
        if isinstance(evidence, TaskRuntimeFailureEvidence):
            self._failure_evidence = evidence
        output_evidence = getattr(record, "task_runtime_output_evidence", None)
        if isinstance(output_evidence, TaskRuntimeOutputEvidence):
            self._output_evidence = output_evidence
        storage_failure = getattr(record, "task_runtime_storage_failure", None)
        if storage_failure is not None and self.storage_failure is None:
            self.storage_failure = storage_failure
        self._tail.extend(encoded)
        excess = len(self._tail) - self._max_bytes
        if excess > 0:
            del self._tail[:excess]

    def diagnostic(self) -> str:
        return self._tail.decode("utf-8", errors="ignore").strip()

    def failure_evidence(self) -> TaskRuntimeFailureEvidence | None:
        return self._failure_evidence

    def output_evidence(self) -> TaskRuntimeOutputEvidence | None:
        return self._output_evidence


def _runtime_failure_fields(
    result: TaskRuntimePreparationResult,
) -> tuple[str | None, str | None]:
    if result is TaskRuntimePreparationResult.AUTHORED_FAILURE:
        return "baseline_validation", "runtime_baseline_validation"
    if result is TaskRuntimePreparationResult.DEPENDENCY_FAILURE:
        return "dependency_install", "runtime_dependencies"
    if result is TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE:
        return "runtime_preparation", "runtime_bootstrap"
    return None, None


def _safe_task_fingerprint(task_dir: Path) -> str:
    try:
        return task_tree_fingerprint(task_dir)
    except (OSError, ValueError):
        return ""


def runtime_preparation_report_from_result(
    result: TaskRuntimePreparationResult,
    *,
    task_dir: Path,
    diagnostic: str = "",
    elapsed_seconds: float = 0.0,
    failure_evidence: TaskRuntimeFailureEvidence | None = None,
    output_evidence: TaskRuntimeOutputEvidence | None = None,
) -> TaskRuntimePreparationReport:
    """Build a report for compatibility callers that only retained the result enum."""
    failure_stage, failure_category = _runtime_failure_fields(result)
    task_fingerprint = _safe_task_fingerprint(task_dir)
    if output_evidence is not None:
        output_evidence = replace(output_evidence, task_fingerprint=task_fingerprint)
    if (
        output_evidence is not None
        and result is not TaskRuntimePreparationResult.READY
        and output_evidence.failure_category != "timeout"
    ):
        failure_stage, failure_category = output_evidence.stage, output_evidence.failure_category
    return TaskRuntimePreparationReport(
        result=result,
        failure_stage=failure_stage,
        failure_category=failure_category,
        diagnostic=diagnostic if result is not TaskRuntimePreparationResult.READY else "",
        task_fingerprint=task_fingerprint,
        elapsed_seconds=elapsed_seconds,
        failure_evidence=failure_evidence,
        output_evidence=output_evidence,
    )


def _draft_validation_reset_sequence_matches(
    observed_reset_seeds: tuple[object, ...],
    declared_reset_seeds: tuple[int, ...],
) -> bool:
    """Validate an explicit task-drafting smoke reset sequence."""
    return declared_reset_sequence_matches(observed_reset_seeds, declared_reset_seeds)


def _candidate_call_sequence_diagnostic(
    *,
    calls: int,
    calls_before_first_reset: int,
    observed_reset_seeds: tuple[object, ...],
    expected_primary_reset_seeds: tuple[int, ...],
    peer_disconnects: int,
    context: str,
) -> str | None:
    """Return bounded feedback for every candidate transcript violation."""
    failures: list[str] = []
    if calls < 1:
        failures.append("made no candidate calls")
    if calls_before_first_reset:
        noun = "call" if calls_before_first_reset == 1 else "calls"
        failures.append(f"made {calls_before_first_reset} {noun} before the first reset")
    if not declared_reset_sequence_matches(
        observed_reset_seeds,
        expected_primary_reset_seeds,
    ):
        failures.append(
            reset_sequence_diagnostic(
                observed_reset_seeds,
                expected_primary_reset_seeds,
                context=context,
            )
        )
    if peer_disconnects:
        noun = "time" if peer_disconnects == 1 else "times"
        failures.append(f"disconnected {peer_disconnects} {noun} before receiving a response")
    return "; ".join(failures) or None


def discard_managed_task_runtime(task_dir: Path) -> None:
    """Remove a harness-owned runtime without following child symlinks."""
    runtime = task_dir / ".venv"
    manifest = task_dir / RUNTIME_MANIFEST_FILENAME
    if runtime.exists() or runtime.is_symlink():
        metadata = runtime.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"unsafe managed runtime directory: {runtime}")
        for directory, directory_names, file_names in os.walk(runtime, followlinks=False):
            current = Path(directory)
            for name in (*directory_names, *file_names):
                mode = (current / name).lstat().st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
                    raise ValueError(f"unsafe managed runtime entry: {current / name}")
        shutil.rmtree(runtime)
    if manifest.exists() or manifest.is_symlink():
        mode = manifest.lstat().st_mode
        if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            raise ValueError(f"unsafe managed runtime manifest: {manifest}")
        manifest.unlink()


def runtime_preparation_failure_is_repairable(
    result: TaskRuntimePreparationResult,
) -> bool:
    """Return whether another scoped drafting pass can repair this failure."""
    return result in {
        TaskRuntimePreparationResult.AUTHORED_FAILURE,
        TaskRuntimePreparationResult.DEPENDENCY_FAILURE,
    }


class TaskRuntimeDeadlineExceeded(TimeoutError):
    """The shared runtime-preparation deadline has no remaining budget."""

    def __init__(
        self,
        message: str,
        *,
        evidence: TaskRuntimeFailureEvidence | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_evidence = evidence


def _nonnegative_runtime_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _contracted_timeout_evidence(
    *,
    deadline_owner: str,
    contract_wall_clock_seconds: int,
    evaluator_timeout_seconds: int,
    candidate_rpc_timeout_seconds: int,
    container_cleanup_reserve_seconds: float,
    broker: object | None,
) -> TaskRuntimeFailureEvidence:
    reset_seeds = getattr(broker, "reset_seeds", ())
    reset_count = len(reset_seeds) if isinstance(reset_seeds, tuple) else 0
    return TaskRuntimeFailureEvidence(
        deadline_owner=deadline_owner,
        contract_wall_clock_seconds=contract_wall_clock_seconds,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
        candidate_rpc_timeout_seconds=candidate_rpc_timeout_seconds,
        broker_teardown_reserve_seconds=_BROKER_TEARDOWN_RESERVE_SECONDS,
        container_cleanup_reserve_seconds=container_cleanup_reserve_seconds,
        candidate_rpc_calls=_nonnegative_runtime_count(getattr(broker, "calls", 0)),
        candidate_calls_before_first_reset=_nonnegative_runtime_count(
            getattr(broker, "calls_before_first_reset", 0)
        ),
        candidate_reset_count=reset_count,
        candidate_peer_disconnects=_nonnegative_runtime_count(
            getattr(broker, "peer_disconnects", 0)
        ),
    )


class _TaskRuntimeContractDeadlineExceeded(TaskRuntimeDeadlineExceeded):
    """The task-authored production-search ceiling was exhausted."""


class _TaskRuntimeCleanupDeadlineExceeded(TaskRuntimeDeadlineExceeded):
    """A timed-out runtime command could not complete container cleanup."""


class _CheckedCommandStatus(str, Enum):
    SUCCEEDED = "succeeded"
    COMMAND_FAILED = "command_failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


@dataclass(frozen=True, slots=True)
class _BaselineEvaluationResult:
    status: _CheckedCommandStatus
    combined_score: float | None = None
    correct: bool | None = None
    feasible: bool | None = None


@dataclass(frozen=True, slots=True)
class _CheckedCommandResult:
    status: _CheckedCommandStatus
    process: subprocess.CompletedProcess[str] | None
    cleanup_succeeded: bool = True


@dataclass(slots=True)
class _CommandCleanupWitness:
    verified: bool = False
    unexpected_failure: BaseException | None = None


@contextmanager
def _preserve_unexpected_failure(witness: _CommandCleanupWitness) -> Iterator[None]:
    try:
        yield
    except BaseException as exc:
        if witness.unexpected_failure is None and not isinstance(
            exc, (TaskRuntimeDeadlineExceeded, CandidateBrokerError),
        ):
            witness.unexpected_failure = exc
        raise


def _requirements_digest(requirements_path: Path) -> str:
    content = requirements_path.read_bytes() if requirements_path.is_file() else b""
    return hashlib.sha256(content).hexdigest()


def _evaluator_digest(evaluator_path: Path) -> str:
    return hashlib.sha256(evaluator_path.read_bytes()).hexdigest()


def _tooling_digest() -> str:
    digest = hashlib.sha256()
    paths = [_REPO_ROOT / "pyproject.toml", _REPO_ROOT / "src/__init__.py"]
    for relative_dir in _tooling_package_dirs():
        paths.extend(sorted((_REPO_ROOT / relative_dir).glob("*.py")))
    for path in paths:
        digest.update(path.relative_to(_REPO_ROOT).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _is_safe_regular_file(path: Path) -> bool:
    return path.exists() and not path.is_symlink() and path.is_file()


def _validate_requirements(requirements_path: Path) -> bool:
    if requirements_path.is_symlink() or (
        requirements_path.exists() and not _is_safe_regular_file(requirements_path)
    ):
        logger.error("Drafted requirements are not a safe regular file: %s", requirements_path)
        return False
    try:
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        logger.error("Failed to read drafted requirements: %s", exc)
        return False
    invalid = [
        line.strip()
        for line in lines
        if line.strip()
        and not line.lstrip().startswith("#")
        and _PINNED_REQUIREMENT.fullmatch(line.strip()) is None
    ]
    if invalid:
        logger.error(
            "Drafted requirements must contain only pinned package==version entries: %s",
            invalid,
        )
        return False
    return True


def _normalized_dependency(name: str, version: str) -> tuple[str, str]:
    base_name, separator, extras_value = name.partition("[")
    normalized_name = _PACKAGE_NAME_SEPARATOR.sub("-", base_name).lower()
    if separator:
        extras = extras_value.removesuffix("]").split(",")
        normalized_extras = sorted(
            {_PACKAGE_NAME_SEPARATOR.sub("-", extra).lower() for extra in extras}
        )
        normalized_name += f"[{','.join(normalized_extras)}]"
    return normalized_name, version


def _requirements_match_contract(requirements_path: Path, contract: TaskContract) -> bool:
    if not _is_safe_regular_file(requirements_path):
        logger.error("Contracted task requires a safe pinned requirements.txt")
        return False
    declared = {
        _normalized_dependency(dependency.name, dependency.version)
        for dependency in contract.provenance.dependencies
    }
    installed: set[tuple[str, str]] = set()
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        name, version = requirement.split("==", 1)
        installed.add(_normalized_dependency(name, version))
    if installed != declared:
        logger.error(
            "requirements.txt and task contract dependencies differ: installed=%s declared=%s",
            sorted(installed),
            sorted(declared),
        )
        return False
    return True


def _installation_environment(
    base_environment: Mapping[str, str], temporary_home: Path, task_dir: Path
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in base_environment.items()
        if not any(token in key.upper() for token in _SENSITIVE_ENV_TOKENS)
    }
    environment.update(
        HOME=str(temporary_home),
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        TASK_SOURCE_DIR=str(task_dir),
    )
    return environment


def _docker_command(
    *,
    task_dir: Path,
    image_id: str,
    network: str,
    inner_command: list[str],
    cid_path: Path,
    runtime_dir: Path | None = None,
    runtime_writable: bool = False,
    results_dir: Path | None = None,
    cuda_visible_devices: str | None = None,
    sandbox_gpus: str | None = None,
    search_evaluation: bool = False,
    draft_smoke: bool = False,
    qualification: bool = False,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--cidfile",
        str(cid_path),
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "4096",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--network",
        network,
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=4g",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "TASK_SOURCE_DIR=/task",
        "--volume",
        f"{task_dir}:/task:ro",
    ]
    if task_snapshot_metadata_present(task_dir):
        command.extend(
            (
                "--mount",
                "type=bind,src=/dev/null,"
                f"dst=/task/{TASK_SNAPSHOT_METADATA_FILENAME},readonly",
            )
        )
    if search_evaluation and (task_dir / TASK_CONTRACT_FILENAME).is_file():
        try:
            contract = load_task_contract(task_dir / TASK_CONTRACT_FILENAME)
        except TaskContractError as exc:
            raise ValueError(f"invalid task contract for search runtime: {exc}") from exc
        command.extend(
            (
                "--tmpfs",
                f"/task/{contract.resources.final_root}:ro,noexec,nosuid,nodev,size=64k,nr_inodes=16",
                "--mount",
                (f"type=bind,src=/dev/null,dst=/task/{contract.evaluators.final},readonly"),
            )
        )
    if draft_smoke:
        command.extend(("--env", "AGENTIC_TASK_DRAFT_SMOKE=1"))
    if qualification:
        command.extend(("--env", "AGENTIC_TASK_QUALIFICATION=1"))
    if runtime_dir is not None:
        mode = "rw" if runtime_writable else "ro"
        command.extend(("--volume", f"{runtime_dir}:/runtime:{mode}"))
    if runtime_writable:
        command.extend(
            (
                "--env",
                "TMPDIR=/runtime/.tmp",
                "--env",
                "PIP_NO_CACHE_DIR=1",
            )
        )
    if results_dir is not None:
        command.extend(("--volume", f"{results_dir}:/results:rw"))
    if cuda_visible_devices is not None:
        command.extend(("--env", f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}"))
    if sandbox_gpus is not None:
        command.extend(("--gpus", sandbox_gpus))
    command.append(image_id)
    command.extend(inner_command)
    return command


def _remaining_runtime_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TaskRuntimeDeadlineExceeded("task runtime preparation deadline exceeded")
    return remaining


def _runtime_execution_deadline(deadline: float) -> float:
    remaining = _remaining_runtime_seconds(deadline)
    cleanup_reserve = min(
        _MAX_CONTAINER_CLEANUP_SECONDS,
        remaining * _CONTAINER_CLEANUP_RESERVE_FRACTION,
    )
    return deadline - cleanup_reserve


def _remove_container_before_deadline(
    cid_path: Path, deadline: float, *, cleanup_witness: _CommandCleanupWitness | None = None,
) -> bool:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    if cleanup_witness is None:
        return _force_remove_container(
            cid_path, timeout_seconds=min(_MAX_CONTAINER_CLEANUP_SECONDS, remaining),
        )
    return _force_remove_container(
        cid_path,
        timeout_seconds=min(_MAX_CONTAINER_CLEANUP_SECONDS, remaining),
        cleanup_witness=cleanup_witness,
    )


def _task_generation_broker_directory(identity_dir: Path, phase: str) -> Path:
    prefix = _TASK_BROKER_PREFIXES.get(phase)
    if prefix is None:
        raise ValueError(f"unsupported task-generation broker phase: {phase}")
    identity = hashlib.sha256(os.fsencode(identity_dir.resolve())).hexdigest()[:32]
    return _broker_recovery_root() / f"{prefix}-{identity}"


def _legacy_task_generation_broker_directory(identity_dir: Path, phase: str) -> Path:
    if phase not in _TASK_BROKER_PREFIXES:
        raise ValueError(f"unsupported task-generation broker phase: {phase}")
    identity = hashlib.sha256(os.fsencode(identity_dir.resolve())).hexdigest()[:32]
    return _broker_recovery_root() / f"task-{phase}-{identity}"


def _prepare_task_broker_directory(
    identity_dir: Path,
    phase: str,
    *,
    deadline: float,
) -> Path:
    try:
        root = _ensure_broker_recovery_root()
    except FinalEvaluationError as exc:
        raise CandidateBrokerInfrastructureError(
            "task-generation broker recovery root is unavailable"
        ) from exc
    legacy_dir = _legacy_task_generation_broker_directory(identity_dir, phase)
    broker_dir = _task_generation_broker_directory(identity_dir, phase)
    if legacy_dir.parent != root or broker_dir.parent != root:
        raise CandidateBrokerInfrastructureError("task-generation broker identity is unsafe")
    _recover_task_broker_directory(legacy_dir, deadline=deadline)
    _recover_task_broker_directory(broker_dir, deadline=deadline)
    return broker_dir


def _recover_task_broker_directory(broker_dir: Path, *, deadline: float) -> None:
    try:
        _reject_legacy_broker_directory(broker_dir)
    except FinalEvaluationError as exc:
        raise CandidateBrokerInfrastructureError(str(exc)) from exc
    try:
        metadata = broker_dir.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise CandidateBrokerInfrastructureError(
            "task-generation broker recovery directory is unsafe"
        )
    for name in ("candidate.cid", "evaluator.cid"):
        cid_path = broker_dir / name
        if not (cid_path.exists() or cid_path.is_symlink()):
            continue
        timeout = min(_MAX_CONTAINER_CLEANUP_SECONDS, _remaining_runtime_seconds(deadline))
        if not _force_remove_container(cid_path, timeout_seconds=timeout):
            raise CandidateContainerCleanupError(
                "task-generation container cleanup failed; recovery metadata retained"
            )
        cid_path.unlink(missing_ok=True)
    socket_path = broker_dir / "candidate.sock"
    try:
        socket_metadata = socket_path.lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISSOCK(socket_metadata.st_mode) or socket_metadata.st_uid != os.getuid():
            raise CandidateBrokerInfrastructureError(
                "task-generation broker recovery socket is unsafe"
            )
        socket_path.unlink()
    scoring_mask = broker_dir / "scoring-mask"
    if scoring_mask.exists() or scoring_mask.is_symlink():
        mask_metadata = scoring_mask.lstat()
        if (
            not stat.S_ISDIR(mask_metadata.st_mode)
            or mask_metadata.st_uid != os.getuid()
            or stat.S_IMODE(mask_metadata.st_mode) != 0o700
        ):
            raise CandidateBrokerInfrastructureError(
                "task-generation broker scoring mask is unsafe"
            )
        shutil.rmtree(scoring_mask)
    try:
        broker_dir.rmdir()
    except OSError as exc:
        raise CandidateBrokerInfrastructureError(
            "task-generation broker recovery directory contains unexpected entries"
        ) from exc


@contextmanager
def _runtime_build_lock(task_dir: Path, deadline: float | None = None) -> Iterator[None]:
    lock_path = task_dir.parent / f".{task_dir.name}.runtime-build.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
        ):
            raise OSError(f"task runtime build lock is unsafe: {lock_path}")
        if deadline is None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        else:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = _remaining_runtime_seconds(deadline)
                    time.sleep(min(_RUNTIME_LOCK_POLL_SECONDS, remaining))
        yield
    finally:
        os.close(descriptor)


def _remove_stale_runtime_builds(task_dir: Path, deadline: float | None = None) -> None:
    prefix = f".{task_dir.name}.runtime-build-"
    parent_descriptor = os.open(
        task_dir.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for name in os.listdir(parent_descriptor):
            if deadline is not None:
                _remaining_runtime_seconds(deadline)
            if not name.startswith(prefix):
                continue
            try:
                directory_descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_descriptor,
                )
            except OSError as exc:
                raise OSError(f"stale task runtime build is unsafe: {name}") from exc
            metadata = os.fstat(directory_descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_mode & 0o077
            ):
                os.close(directory_descriptor)
                raise OSError(f"stale task runtime build is unsafe: {name}")
            quarantine = f"{prefix}cleanup-{secrets.token_hex(8)}"
            try:
                os.rename(
                    name,
                    quarantine,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                quarantined = os.stat(
                    quarantine,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (quarantined.st_dev, quarantined.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    os.rename(
                        quarantine,
                        name,
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                    )
                    raise OSError(f"stale task runtime build changed before cleanup: {name}")
                shutil.rmtree(quarantine, dir_fd=parent_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        os.close(parent_descriptor)


def _remove_stale_runtime_containers(task_dir: Path, deadline: float | None = None) -> None:
    cid_pattern = re.compile(
        rf"\.{re.escape(task_dir.name)}\.{_RUNTIME_CID_STAGE}-[0-9a-f]{{32}}\.cid\Z"
    )
    for entry in os.scandir(task_dir.parent):
        if deadline is not None:
            _remaining_runtime_seconds(deadline)
        if cid_pattern.fullmatch(entry.name) is None:
            continue
        metadata = entry.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o022
        ):
            raise OSError(f"stale task runtime CID file is unsafe: {entry.name}")
        cid_path = task_dir.parent / entry.name
        timeout = 30.0 if deadline is None else _remaining_runtime_seconds(deadline)
        if not _force_remove_container(cid_path, timeout_seconds=timeout):
            if deadline is not None:
                _remaining_runtime_seconds(deadline)
            raise OSError(f"could not stop stale task runtime container: {entry.name}")
        cid_path.unlink(missing_ok=True)


def _force_remove_container(
    cid_path: Path, *, timeout_seconds: float = 30.0,
    cleanup_witness: _CommandCleanupWitness | None = None,
) -> bool:
    if cleanup_witness is not None:
        cleanup_witness.verified = False
    try:
        descriptor = os.open(
            cid_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError:
        return True
    except OSError as exc:
        logger.error("Failed to read task runtime container ID: %s", exc)
        return False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            logger.error("Task runtime container ID file is unsafe: %s", cid_path)
            return False
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw_container_id = handle.read(129)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw_container_id) > 128:
        logger.error("Task runtime container ID file is too large: %s", cid_path)
        return False
    try:
        container_id = raw_container_id.decode("ascii").strip()
    except UnicodeDecodeError:
        logger.error("Task runtime container ID is not ASCII: %s", cid_path)
        return False
    if not container_id:
        return True
    if _CONTAINER_ID.fullmatch(container_id) is None:
        logger.error("Task runtime container ID is invalid: %s", cid_path)
        return False
    try:
        result = subprocess.run(
            ["docker", "rm", "--force", container_id],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.error("Failed to force-remove task runtime container %s: %s", container_id, exc)
        return False
    if result.returncode != 0 and "no such container" not in result.stderr.lower():
        logger.error(
            "Failed to force-remove task runtime container %s: %s",
            container_id,
            result.stderr.strip(),
        )
        return False
    if cleanup_witness is not None:
        cleanup_witness.verified = True
    return True


def _log_runtime_exception(message: str, error: BaseException, *, stage: str) -> None:
    category = storage_failure_category(error)
    if category is not None:
        logger.error(
            "Task runtime infrastructure failure: %s during %s", category, stage,
            extra={"task_runtime_storage_failure": (stage, category)},
        )
    else:
        logger.error(message, error)


def _log_failed_evaluation(
    process: subprocess.CompletedProcess[str] | None,
    *,
    environment: Mapping[str, str],
    stage: str = "baseline_validation",
) -> None:
    if process is None:
        return
    evidence = failed_evaluation_evidence(
        process.stderr or "", stage=stage, environment=environment,
    )
    if not process.stderr:
        logger.error(
            "Task runtime %s exited with status %d without writing a diagnostic to stderr",
            stage, process.returncode,
            extra={"task_runtime_output_evidence": evidence},
        )
        return
    logger.error(
        "Task runtime %s failed (rc=%d); evaluator stderr (untrusted diagnostic): %s",
        stage, process.returncode, evidence.stderr or "<empty>",
        extra={"task_runtime_output_evidence": evidence},
    )


def _log_runtime_timeout_evidence(evidence: TaskRuntimeOutputEvidence) -> None:
    try:
        logger.error(
            "Task runtime %s exceeded its execution deadline; "
            "captured stderr is retained as untrusted repair evidence",
            evidence.stage, extra={"task_runtime_output_evidence": evidence},
        )
    except Exception:
        # A secondary logging failure must not skip container cleanup or replace the timeout.
        pass


def _run_checked(
    command: list[str],
    *,
    task_dir: Path,
    deadline: float,
    run: Callable[..., subprocess.CompletedProcess[str]],
    environment: Mapping[str, str],
    cid_path: Path | None = None,
    execution_deadline: float | None = None,
    stage: str = "runtime_preparation",
    cleanup_witness: _CommandCleanupWitness | None = None,
) -> _CheckedCommandResult:
    if execution_deadline is None:
        execution_deadline = _runtime_execution_deadline(deadline)
    else:
        execution_deadline = min(execution_deadline, deadline)
    timeout_seconds = _remaining_runtime_seconds(execution_deadline)
    remove_cid_file = True
    primary_failure: BaseException | None = None
    result: subprocess.CompletedProcess[str] | None = None
    process_cleanup = ProcessCleanupWitness()
    try:
        if run is subprocess.run:
            bounded = run_bounded_process(
                command,
                cwd=task_dir,
                env=environment,
                timeout_seconds=timeout_seconds,
                child_umask=0o077 if cid_path is not None else None,
                cleanup_witness=process_cleanup,
            )
            result = subprocess.CompletedProcess(
                command,
                bounded.returncode,
                bounded.stdout.decode("utf-8", errors="replace"),
                bounded.stderr.decode("utf-8", errors="replace"),
            )
        else:
            result = run(
                command,
                cwd=str(task_dir),
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        _remaining_runtime_seconds(execution_deadline)
    except (subprocess.TimeoutExpired, ProcessTimeoutError) as exc:
        primary_failure = exc
        evidence = timeout_evidence(exc, stage=stage, environment=environment)
        _log_runtime_timeout_evidence(evidence)
        if cid_path is not None:
            remove_cid_file = _remove_container_before_deadline(cid_path, deadline)
            if not remove_cid_file:
                raise _TaskRuntimeCleanupDeadlineExceeded(
                    "task runtime deadline exceeded and container cleanup did not complete"
                ) from exc
        raise TaskRuntimeDeadlineExceeded("task runtime preparation deadline exceeded") from exc
    except TaskRuntimeDeadlineExceeded as exc:
        primary_failure = exc
        if result is not None:
            evidence = replace(
                failed_evaluation_evidence(
                    result.stderr or "", stage=stage, environment=environment,
                ),
                failure_category="timeout",
            )
            _log_runtime_timeout_evidence(evidence)
        if cid_path is not None:
            remove_cid_file = _remove_container_before_deadline(cid_path, deadline)
            if not remove_cid_file:
                raise _TaskRuntimeCleanupDeadlineExceeded(
                    "task runtime deadline exceeded and container cleanup did not complete"
                ) from exc
        raise
    except ProcessOutputLimitError as exc:
        primary_failure = exc
        if cid_path is not None:
            remove_cid_file = _remove_container_before_deadline(cid_path, deadline)
        evidence = output_limit_evidence(
            exc,
            stage=stage,
            task_fingerprint="",
            environment=environment,
        )
        logger.error(
            "Task runtime %s exceeded its combined console stdout/stderr limit "
            "(%d bytes; observed %d bytes). This is separate from candidate RPC wire limits.",
            stage,
            exc.max_output_bytes,
            exc.observed_output_bytes,
            extra={"task_runtime_output_evidence": evidence},
        )
        return _CheckedCommandResult(
            _CheckedCommandStatus.COMMAND_FAILED, None, cleanup_succeeded=remove_cid_file
        )
    except (OSError, BoundedProcessError) as exc:
        primary_failure = exc
        if cid_path is not None:
            remove_cid_file = _remove_container_before_deadline(cid_path, deadline)
        _log_runtime_exception("Task runtime command failed to launch: %s", exc, stage=stage)
        return _CheckedCommandResult(
            _CheckedCommandStatus.INFRASTRUCTURE_FAILED, None, cleanup_succeeded=remove_cid_file
        )
    except BaseException as exc:
        primary_failure = exc
        if cleanup_witness is not None:
            cleanup_witness.unexpected_failure = exc
        if cid_path is not None:
            if cleanup_witness is not None and process_cleanup.reaped:
                remove_cid_file = _remove_container_before_deadline(
                    cid_path, deadline, cleanup_witness=cleanup_witness,
                )
            else:
                remove_cid_file = _remove_container_before_deadline(cid_path, deadline)
        raise
    finally:
        if cid_path is not None and remove_cid_file:
            try:
                cid_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                if primary_failure is None and (result is None or result.returncode == 0):
                    _log_runtime_exception(
                        "Task runtime container-id cleanup failed: %s", cleanup_error,
                        stage="container_cleanup",
                    )
                    raise ContainerMetadataCleanupError(
                        "task runtime container-id cleanup failed; recovery metadata retained"
                    ) from cleanup_error
                logger.error(
                    "Task runtime secondary container-id cleanup failure: %s; "
                    "primary failure retained",
                    storage_failure_category(cleanup_error) or type(cleanup_error).__name__,
                )
    assert result is not None
    if result.returncode != 0:
        _log_failed_evaluation(result, environment=environment, stage=stage)
        status = (
            _CheckedCommandStatus.INFRASTRUCTURE_FAILED
            if result.returncode in {125, 126, 127}
            else _CheckedCommandStatus.COMMAND_FAILED
        )
        return _CheckedCommandResult(status, result)
    return _CheckedCommandResult(_CheckedCommandStatus.SUCCEEDED, result)


def _run_checked_with_progress(
    command: list[str],
    *,
    task_dir: Path,
    deadline: float,
    run: Callable[..., subprocess.CompletedProcess[str]],
    environment: Mapping[str, str],
    cid_path: Path | None = None,
    execution_deadline: float | None = None,
    stage: str,
    cleanup_witness: _CommandCleanupWitness | None = None,
) -> _CheckedCommandResult:
    step_id = {
        "runtime_create": "runtime_create",
        "tooling_build": "tooling_build",
        "tooling_install": "tooling_install",
        "dependency_install": "dependency_install",
        "baseline_validation": "baseline_evaluation",
    }[stage]
    with preparation_step(step_id) as step:
        result = _run_checked(
            command, task_dir=task_dir, deadline=deadline, run=run,
            environment=environment, cid_path=cid_path, execution_deadline=execution_deadline,
            stage=stage, cleanup_witness=cleanup_witness,
        )
        if result.status is not _CheckedCommandStatus.SUCCEEDED:
            step.fail()
        return result


def _load_runtime_manifest(task_dir: Path) -> dict[str, object] | None:
    manifest_path = task_dir / RUNTIME_MANIFEST_FILENAME
    if not _is_safe_regular_file(manifest_path):
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _runtime_is_compatible(
    task_dir: Path,
    *,
    image_id: str,
    requirements_sha256: str,
    tooling_sha256: str,
    evaluator_sha256: str,
    evaluator_tree_sha256: str,
    gpu_evaluation: bool,
    cuda_visible_devices: str | None,
) -> bool:
    venv_dir = task_dir / ".venv"
    python_path = venv_dir / "bin" / "python"
    if venv_dir.is_symlink() or python_path.is_symlink() or not python_path.is_file():
        return False
    manifest = _load_runtime_manifest(task_dir)
    return bool(
        manifest
        and manifest.get("schema_version") == 1
        and manifest.get("requirements_sha256") == requirements_sha256
        and manifest.get("tooling_sha256") == tooling_sha256
        and manifest.get("evaluator_sha256") == evaluator_sha256
        and manifest.get("evaluator_tree_sha256") == evaluator_tree_sha256
        and manifest.get("gpu_evaluation") is gpu_evaluation
        and manifest.get("cuda_visible_devices") == cuda_visible_devices
        and manifest.get("image_id") == image_id
    )


def _manifest_baseline_satisfies(
    manifest: dict[str, object] | None,
    requirement: TaskRuntimeEvaluationRequirement,
) -> bool:
    """Whether a compatible runtime's manifest already proves the baseline for *requirement*."""
    if not manifest:
        return False
    score = manifest.get("baseline_combined_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        return False
    correct = manifest.get("baseline_correct")
    if not isinstance(correct, bool):
        return False
    if requirement is TaskRuntimeEvaluationRequirement.CORRECT_RESULT:
        return correct
    if requirement is TaskRuntimeEvaluationRequirement.FEASIBLE_RESULT:
        return manifest.get("baseline_feasible") is True
    return True


def _write_runtime_manifest(task_dir: Path, payload: dict[str, object]) -> None:
    manifest_path = task_dir / RUNTIME_MANIFEST_FILENAME
    if manifest_path.is_symlink():
        raise ValueError(f"refusing symlinked runtime manifest: {manifest_path}")
    temporary_path = task_dir / f".{RUNTIME_MANIFEST_FILENAME}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary_path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _valid_evaluation_tree(
    results_dir: Path,
    *,
    requirement: TaskRuntimeEvaluationRequirement = (
        TaskRuntimeEvaluationRequirement.CORRECT_RESULT
    ),
) -> bool:
    qualification_path = results_dir / "qualification.json"
    if requirement is TaskRuntimeEvaluationRequirement.FEASIBLE_RESULT:
        if _validated_evaluation_feasibility(results_dir) is not True:
            logger.error("Offline baseline did not produce a feasible qualification result")
            return False
    elif qualification_path.exists() or qualification_path.is_symlink():
        logger.error("Offline baseline produced qualification.json outside qualification")
        return False
    result_bytes = 0
    for path in results_dir.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            logger.error("Offline baseline produced unsafe result entry: %s", path)
            return False
        if path.is_file():
            result_bytes += path.stat().st_size
            if result_bytes > 64 * 1024 * 1024:
                logger.error("Offline baseline results exceed 64 MiB")
                return False
    for filename in ("metrics.json", "correct.json"):
        path = results_dir / filename
        if path.is_symlink() or not path.is_file():
            logger.error("Offline baseline evaluation did not produce %s", path)
            return False
        if path.stat().st_size > 1024 * 1024:
            logger.error("Offline baseline result exceeds 1 MiB: %s", path)
            return False
    return True


def _validated_evaluation_score(
    results_dir: Path,
    *,
    requirement: TaskRuntimeEvaluationRequirement = (
        TaskRuntimeEvaluationRequirement.CORRECT_RESULT
    ),
) -> float | None:
    if not _valid_evaluation_tree(results_dir, requirement=requirement):
        return None
    try:
        metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
        correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.error("Offline baseline results are invalid: %s", exc)
        return None
    score = metrics.get("combined_score") if isinstance(metrics, dict) else None
    correct_value = correct.get("correct") if isinstance(correct, dict) else None
    if not (
        not isinstance(score, bool)
        and isinstance(score, (int, float))
        and math.isfinite(float(score))
        and isinstance(correct_value, bool)
        and (
            requirement is not TaskRuntimeEvaluationRequirement.CORRECT_RESULT
            or correct_value is True
        )
    ):
        return None
    return float(score)


def _validated_drafting_evaluation_score(results_dir: Path) -> float | None:
    """Validate drafting output shape while preserving scientific correctness."""
    return _validated_evaluation_score(
        results_dir,
        requirement=TaskRuntimeEvaluationRequirement.VALID_RESULT,
    )


def _validated_evaluation_correctness(results_dir: Path) -> bool | None:
    """Read the correctness bit retained by a separately validated result."""
    try:
        correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeError, json.JSONDecodeError):
        return None
    value = correct.get("correct") if isinstance(correct, dict) else None
    return value if isinstance(value, bool) else None


def _validated_evaluation_feasibility(results_dir: Path) -> bool | None:
    """Read one exact qualification-only feasibility result."""
    path = results_dir / "qualification.json"
    if (
        not _is_safe_regular_file(path)
        or path.stat(follow_symlinks=False).st_size > 1024 * 1024
    ):
        return None

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate qualification result key: {key}")
            payload[key] = value
        return payload

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "feasible"}:
        return None
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        return None
    if not isinstance(payload["feasible"], bool):
        return None
    return payload["feasible"]


def _validated_contracted_baseline_score(
    results_dir: Path,
    contract: TaskContract,
    *,
    requirement: TaskRuntimeEvaluationRequirement = (
        TaskRuntimeEvaluationRequirement.CORRECT_RESULT
    ),
) -> float | None:
    try:
        metrics, correct = validate_evaluator_outputs(
            results_dir, tuple(metric.name for metric in contract.metrics.search),
        )
    except EvaluatorOutputContractError as error:
        logger.error("Contracted baseline evaluator output invalid: %s", error)
        return None
    if requirement is TaskRuntimeEvaluationRequirement.CORRECT_RESULT and not correct:
        logger.error("Contracted baseline requires correct=true")
        return None
    if not _valid_evaluation_tree(results_dir, requirement=requirement):
        return None
    return float(metrics["combined_score"])


@contextmanager
def _baseline_results_directory(retained_results_dir: Path | None) -> Iterator[Path]:
    """Retain explicitly requested outputs without replacing an existing path."""
    if retained_results_dir is None:
        with tempfile.TemporaryDirectory(prefix="agentic-task-eval-") as temporary:
            yield Path(temporary)
        return
    if not retained_results_dir.is_absolute():
        raise ValueError("retained results path must be absolute")
    parent = retained_results_dir.parent
    if parent.resolve(strict=True) != parent:
        raise ValueError("retained results parent must be canonical without symlinks")
    metadata = parent.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("retained results parent must be a directory")
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError("retained results parent changed during creation")
        os.mkdir(retained_results_dir.name, mode=0o700, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    yield retained_results_dir


@measure_qualification_usage
def _run_contracted_baseline(
    *,
    task_dir: Path,
    runtime_venv: Path,
    image_id: str,
    contract: TaskContract,
    entrypoint: str,
    deadline: float,
    environment: Mapping[str, str],
    run: Callable[..., subprocess.CompletedProcess[str]],
    draft_smoke: bool = False,
    baseline_results_dir: Path | None = None,
    baseline_artifacts_dir: Path | None = None,
    evaluation_requirement: TaskRuntimeEvaluationRequirement = (
        TaskRuntimeEvaluationRequirement.CORRECT_RESULT
    ),
    retained_results_dir: Path | None = None,
) -> _BaselineEvaluationResult:
    if baseline_results_dir is not None:
        try:
            if draft_smoke:
                raise ValueError("baseline archives require production evaluation")
            baseline_results_dir = validate_baseline_results_dir(
                baseline_results_dir, task_dir=task_dir,
            )
        except (OSError, ValueError) as exc:
            _log_runtime_exception("Baseline archive preflight failed: %s", exc,
                                   stage="baseline_archive")
            return _BaselineEvaluationResult(_CheckedCommandStatus.INFRASTRUCTURE_FAILED)
    contract_wall_clock_seconds = contract.compute.search.wall_clock_seconds
    contract_deadline = time.monotonic() + contract_wall_clock_seconds
    contract_deadline_is_limiting = contract_deadline <= deadline
    effective_deadline = min(deadline, contract_deadline)
    evaluation_deadline = effective_deadline - _BROKER_TEARDOWN_RESERVE_SECONDS
    try:
        broker_dir = _prepare_task_broker_directory(
            task_dir,
            "runtime-search",
            deadline=evaluation_deadline,
        )
    except CandidateBrokerInfrastructureError as exc:
        logger.error("Contracted baseline recovery failed: %s", exc)
        return _BaselineEvaluationResult(_CheckedCommandStatus.INFRASTRUCTURE_FAILED)
    candidate_rpc_timeout = max(1, math.ceil(_remaining_runtime_seconds(evaluation_deadline)))
    evaluator_execution_deadline = _runtime_execution_deadline(evaluation_deadline)
    evaluator_timeout = max(
        1,
        math.ceil(_remaining_runtime_seconds(evaluator_execution_deadline)),
    )
    container_cleanup_reserve_seconds = evaluation_deadline - evaluator_execution_deadline
    with _baseline_results_directory(retained_results_dir) as results_dir:
        evaluator_cid = broker_dir / "evaluator.cid"
        try:
            evaluator_command = _evaluation_command(
                task_dir=task_dir,
                program_dir=task_dir / "program_0",
                entrypoint=task_dir / "program_0" / entrypoint,
                evaluator=task_dir / contract.evaluators.search,
                results_dir=results_dir,
                python_bin=runtime_venv / "bin" / "python",
                container_image=image_id,
                gpu_count=contract.compute.search.evaluator_gpu_count,
                cpu_cores=contract.compute.search.cpu_cores,
                memory_gb=contract.compute.search.memory_gb,
                environment=environment,
                broker_dir=broker_dir,
                candidate_editable_paths=contract.candidate.editable_paths,
                public_artifact_bundle=contract.resources.public_artifact_bundle,
                rpc_timeout_seconds=candidate_rpc_timeout,
                plane="search",
                runtime_venv_dir=runtime_venv,
                draft_smoke=draft_smoke,
                qualification=(
                    evaluation_requirement is TaskRuntimeEvaluationRequirement.FEASIBLE_RESULT
                ),
            )
            candidate_command = _candidate_command(
                task_dir=task_dir,
                program_dir=task_dir / "program_0",
                entrypoint=task_dir / "program_0" / entrypoint,
                container_image=image_id,
                contract=contract,
                environment=environment,
                broker_dir=broker_dir,
                plane="search",
                runtime_venv_dir=runtime_venv,
            )
        except GpuAllocationInvalidError:
            logger.error(
                GPU_ALLOCATION_INVALID_CATEGORY,
                extra={"task_runtime_gpu_failure": GPU_ALLOCATION_INVALID_CATEGORY},
            )
            return _BaselineEvaluationResult(_CheckedCommandStatus.INFRASTRUCTURE_FAILED)
        except (OSError, ValueError, FinalEvaluationError) as exc:
            logger.error("Contracted baseline command construction failed: %s", exc)
            return _BaselineEvaluationResult(_CheckedCommandStatus.INFRASTRUCTURE_FAILED)
        expected_primary_reset_seeds = (
            contract.protocol.search_smoke_reset_sequence
            if draft_smoke
            else contract.protocol.search_reset_sequence
        )
        broker: CandidateBroker | None = None
        evaluator_cleanup = _CommandCleanupWitness()
        try:
            gpu_count = contract.compute.search.gpu_count
            gpu_lease: GpuAllocationLease | None = (
                acquire_gpu_allocation_lease(
                    environment.get("AGENT_CONTAINER_GPUS"),
                    gpu_count,
                    lease_root=GPU_ALLOCATION_LEASE_ROOT,
                    deadline=evaluator_execution_deadline,
                )
                if gpu_count
                else None
            )
            allocation = gpu_lease if gpu_lease is not None else nullcontext()
            with allocation:
                try:
                    if (
                        gpu_lease is not None
                        and evaluator_execution_deadline <= time.monotonic()
                    ):
                        logger.error(
                            GPU_ALLOCATION_BUSY_CATEGORY,
                            extra={"task_runtime_gpu_failure": GPU_ALLOCATION_BUSY_CATEGORY},
                        )
                        return _BaselineEvaluationResult(
                            _CheckedCommandStatus.INFRASTRUCTURE_FAILED
                        )
                    if gpu_lease is not None:
                        gpu_lease.prepare_runtime()
                        evaluator_command = gpu_lease.wrap_command(evaluator_command)
                        candidate_command = gpu_lease.wrap_command(candidate_command)
                    broker = CandidateBroker(
                        candidate_command,
                        broker_dir,
                        rpc_timeout_seconds=_remaining_runtime_seconds(evaluation_deadline),
                        reset_timeout_seconds=_remaining_runtime_seconds(evaluation_deadline),
                        max_rpc_calls=contract.compute.search.max_candidate_rpc_calls,
                        expected_reset_seeds=expected_primary_reset_seeds,
                    )
                    with broker, _preserve_unexpected_failure(evaluator_cleanup):
                        observe_qualification_broker(broker)
                        evaluation = _run_checked_with_progress(
                            evaluator_command,
                            task_dir=task_dir,
                            deadline=evaluation_deadline,
                            run=run,
                            environment=environment,
                            cid_path=evaluator_cid,
                            execution_deadline=evaluator_execution_deadline,
                            stage="baseline_validation",
                            cleanup_witness=evaluator_cleanup,
                        )
                        if not evaluation.cleanup_succeeded and gpu_lease is not None:
                            gpu_lease.quarantine()
                except BaseException as exc:
                    original_failure = evaluator_cleanup.unexpected_failure
                    if original_failure is not None:
                        cleanup_verified = (
                            evaluator_cleanup.verified
                            and broker is not None
                            and broker.shutdown_verified
                            and not isinstance(exc, CandidateContainerCleanupError)
                        )
                        if gpu_lease is not None and not cleanup_verified:
                            gpu_lease.quarantine()
                        if original_failure is exc:
                            raise
                        raise original_failure from exc
                    if gpu_lease is not None and (
                        isinstance(exc, (
                            _TaskRuntimeCleanupDeadlineExceeded, CandidateContainerCleanupError,
                        ))
                        or not isinstance(exc, (TaskRuntimeDeadlineExceeded, CandidateBrokerError))
                    ):
                        gpu_lease.quarantine()
                    raise
        except GpuLeaseTimeout:
            logger.error(
                GPU_ALLOCATION_BUSY_CATEGORY,
                extra={"task_runtime_gpu_failure": GPU_ALLOCATION_BUSY_CATEGORY},
            )
            return _BaselineEvaluationResult(_CheckedCommandStatus.INFRASTRUCTURE_FAILED)
        except GpuLeaseQuarantined:
            logger.error(
                "%s: automatic recovery could not establish safe reuse; inspect ownership",
                GPU_ALLOCATION_QUARANTINED_CATEGORY,
                extra={"task_runtime_gpu_failure": GPU_ALLOCATION_QUARANTINED_CATEGORY},
            )
            return _BaselineEvaluationResult(_CheckedCommandStatus.INFRASTRUCTURE_FAILED)
        except GpuLeaseError:
            logger.error(
                GPU_ALLOCATION_INVALID_CATEGORY,
                extra={"task_runtime_gpu_failure": GPU_ALLOCATION_INVALID_CATEGORY},
            )
            return _BaselineEvaluationResult(_CheckedCommandStatus.INFRASTRUCTURE_FAILED)
        except CandidateBrokerInfrastructureError as exc:
            logger.error("Contracted baseline container cleanup failed: %s", exc)
            return _BaselineEvaluationResult(_CheckedCommandStatus.INFRASTRUCTURE_FAILED)
        except _TaskRuntimeCleanupDeadlineExceeded as exc:
            evidence = _contracted_timeout_evidence(
                deadline_owner=(
                    "task_contract" if contract_deadline_is_limiting else "task_drafting"
                ),
                contract_wall_clock_seconds=contract_wall_clock_seconds,
                evaluator_timeout_seconds=evaluator_timeout,
                candidate_rpc_timeout_seconds=candidate_rpc_timeout,
                container_cleanup_reserve_seconds=container_cleanup_reserve_seconds,
                broker=broker,
            )
            raise _TaskRuntimeCleanupDeadlineExceeded(
                str(exc),
                evidence=evidence,
            ) from exc
        except TaskRuntimeDeadlineExceeded as exc:
            evidence = _contracted_timeout_evidence(
                deadline_owner=(
                    "task_contract" if contract_deadline_is_limiting else "task_drafting"
                ),
                contract_wall_clock_seconds=contract_wall_clock_seconds,
                evaluator_timeout_seconds=evaluator_timeout,
                candidate_rpc_timeout_seconds=candidate_rpc_timeout,
                container_cleanup_reserve_seconds=container_cleanup_reserve_seconds,
                broker=broker,
            )
            if contract_deadline_is_limiting:
                raise _TaskRuntimeContractDeadlineExceeded(
                    "Production search exceeded the task-authored "
                    f"{contract_wall_clock_seconds}-second wall-clock ceiling; calibrate or "
                    "reduce the production workload so training, rollouts, result writing, "
                    "and cleanup complete within the declared budget.",
                    evidence=evidence,
                ) from exc
            raise TaskRuntimeDeadlineExceeded(str(exc), evidence=evidence) from exc
        except CandidateBrokerError as exc:
            logger.error("Contracted baseline candidate protocol failed: %s", exc)
            return _BaselineEvaluationResult(_CheckedCommandStatus.COMMAND_FAILED)
        if evaluation.status is not _CheckedCommandStatus.SUCCEEDED:
            return _BaselineEvaluationResult(evaluation.status)
        _remaining_runtime_seconds(effective_deadline)
        if evaluation.process is None or evaluation.process.returncode != 0:
            return _BaselineEvaluationResult(_CheckedCommandStatus.COMMAND_FAILED)
        call_sequence_diagnostic = _candidate_call_sequence_diagnostic(
            calls=broker.calls,
            calls_before_first_reset=broker.calls_before_first_reset,
            observed_reset_seeds=broker.reset_seeds,
            expected_primary_reset_seeds=expected_primary_reset_seeds,
            peer_disconnects=broker.peer_disconnects,
            context="Draft search smoke" if draft_smoke else "Search evaluation",
        )
        if call_sequence_diagnostic is not None:
            _log_failed_evaluation(evaluation.process, environment=environment)
            logger.error(
                "Contracted baseline candidate call sequence was invalid: %s",
                call_sequence_diagnostic,
            )
            return _BaselineEvaluationResult(_CheckedCommandStatus.COMMAND_FAILED)
        baseline_score = _validated_contracted_baseline_score(
            results_dir,
            contract,
            requirement=evaluation_requirement,
        )
        if baseline_score is None:
            _log_failed_evaluation(evaluation.process, environment=environment)
            logger.error("Contracted baseline results failed validation")
            return _BaselineEvaluationResult(_CheckedCommandStatus.COMMAND_FAILED)
        baseline_correct = _validated_evaluation_correctness(results_dir)
        if baseline_correct is None:
            _log_failed_evaluation(evaluation.process, environment=environment)
            logger.error("Contracted baseline correctness could not be retained")
            return _BaselineEvaluationResult(_CheckedCommandStatus.COMMAND_FAILED)
        baseline_feasible = None
        if evaluation_requirement is TaskRuntimeEvaluationRequirement.FEASIBLE_RESULT:
            baseline_feasible = _validated_evaluation_feasibility(results_dir)
            if baseline_feasible is not True:
                _log_failed_evaluation(evaluation.process, environment=environment)
                logger.error("Contracted baseline feasibility could not be retained")
                return _BaselineEvaluationResult(_CheckedCommandStatus.COMMAND_FAILED)
        if baseline_results_dir is not None:
            try:
                archive_baseline_results(results_dir, baseline_results_dir, task_dir=task_dir)
            except (OSError, ValueError) as exc:
                _log_runtime_exception("Baseline result archive failed: %s", exc,
                                       stage="baseline_archive")
                return _BaselineEvaluationResult(_CheckedCommandStatus.INFRASTRUCTURE_FAILED)
        if baseline_artifacts_dir is not None:
            if not evaluation.cleanup_succeeded or not broker.shutdown_verified:
                logger.error("Baseline artifacts require verified evaluator and candidate cleanup")
                return _BaselineEvaluationResult(_CheckedCommandStatus.INFRASTRUCTURE_FAILED)
            from .baseline_artifacts import archive_baseline_artifacts

            try:
                archive_baseline_artifacts(
                    results_dir, baseline_artifacts_dir,
                    metric_names=tuple(metric.name for metric in contract.metrics.search),
                    combined_score=baseline_score, correct=baseline_correct,
                    feasible=baseline_feasible,
                )
            except (OSError, ValueError) as exc:
                logger.error("Could not archive validated baseline artifacts: %s", exc)
                return _BaselineEvaluationResult(_CheckedCommandStatus.INFRASTRUCTURE_FAILED)
    return _BaselineEvaluationResult(
        _CheckedCommandStatus.SUCCEEDED,
        baseline_score,
        baseline_correct,
        baseline_feasible,
    )


def _evaluation_outcome(result: _BaselineEvaluationResult) -> EvaluationOutcome:
    assert result.combined_score is not None and result.correct is not None
    return EvaluationOutcome(result.combined_score, result.correct)


def _replay_contracted_smoke_baseline(
    first: _BaselineEvaluationResult,
    evaluation_options: Mapping[str, Any],
) -> TaskRuntimePreparationResult | None:
    """Evaluate the unchanged baseline again; a different result is an authored failure."""
    replay = _run_contracted_baseline(**evaluation_options)
    if replay.status is _CheckedCommandStatus.INFRASTRUCTURE_FAILED:
        return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
    if replay.status is not _CheckedCommandStatus.SUCCEEDED:
        logger.error("Replayed baseline evaluation failed validation")
        return TaskRuntimePreparationResult.AUTHORED_FAILURE
    mismatch = replay_mismatch(_evaluation_outcome(first), _evaluation_outcome(replay))
    if mismatch is not None:
        logger.error("Baseline replay failed: %s", mismatch)
        return TaskRuntimePreparationResult.AUTHORED_FAILURE
    return None


def prepare_task_runtime(
    task_dir: Path,
    timeout_seconds: int = 1800,
    *,
    containerize: bool = False,
    container_image: str = "",
    container_network: str = "bridge",
    python_bin: str = "python3",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    base_environment: Mapping[str, str] = os.environ,
    evaluator_tree_sha256: str | None = None,
    gpu_evaluation: bool = False,
    draft_smoke: bool = False,
    install_cache_dir: Path | None = None,
    baseline_results_dir: Path | None = None,
    baseline_artifacts_dir: Path | None = None,
    evaluation_requirement: TaskRuntimeEvaluationRequirement = (
        TaskRuntimeEvaluationRequirement.CORRECT_RESULT
    ),
    retained_results_dir: Path | None = None,
) -> bool:
    """Serialize runtime preparation and reclaim interrupted build trees."""
    return (
        prepare_task_runtime_result(
            task_dir,
            timeout_seconds,
            containerize=containerize,
            container_image=container_image,
            container_network=container_network,
            python_bin=python_bin,
            run=run,
            base_environment=base_environment,
            evaluator_tree_sha256=evaluator_tree_sha256,
            gpu_evaluation=gpu_evaluation,
            draft_smoke=draft_smoke,
            install_cache_dir=install_cache_dir,
            baseline_results_dir=baseline_results_dir,
            evaluation_requirement=evaluation_requirement,
            retained_results_dir=retained_results_dir,
            baseline_artifacts_dir=baseline_artifacts_dir,
        )
        is TaskRuntimePreparationResult.READY
    )


@retain_qualification_usage
def prepare_task_runtime_report(
    task_dir: Path,
    timeout_seconds: int = 1800,
    *,
    containerize: bool = False,
    container_image: str = "",
    container_network: str = "bridge",
    python_bin: str = "python3",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    base_environment: Mapping[str, str] = os.environ,
    evaluator_tree_sha256: str | None = None,
    gpu_evaluation: bool = False,
    draft_smoke: bool = False,
    install_cache_dir: Path | None = None,
    baseline_results_dir: Path | None = None,
    baseline_artifacts_dir: Path | None = None,
    evaluation_requirement: TaskRuntimeEvaluationRequirement = (
        TaskRuntimeEvaluationRequirement.CORRECT_RESULT
    ),
    retained_results_dir: Path | None = None,
) -> TaskRuntimePreparationReport:
    """Prepare a runtime and retain its typed failure and bounded diagnostic."""
    started_at = time.monotonic()
    diagnostic_handler = _RuntimeReportDiagnosticHandler()
    logger.addHandler(diagnostic_handler)
    try:
        result = prepare_task_runtime_result(
            task_dir,
            timeout_seconds,
            containerize=containerize,
            container_image=container_image,
            container_network=container_network,
            python_bin=python_bin,
            run=run,
            base_environment=base_environment,
            evaluator_tree_sha256=evaluator_tree_sha256,
            gpu_evaluation=gpu_evaluation,
            draft_smoke=draft_smoke,
            install_cache_dir=install_cache_dir,
            baseline_results_dir=baseline_results_dir,
            evaluation_requirement=evaluation_requirement,
            retained_results_dir=retained_results_dir,
            baseline_artifacts_dir=baseline_artifacts_dir,
        )
    finally:
        logger.removeHandler(diagnostic_handler)
    report = runtime_preparation_report_from_result(
        result,
        task_dir=task_dir,
        diagnostic=diagnostic_handler.diagnostic(),
        elapsed_seconds=time.monotonic() - started_at,
        failure_evidence=diagnostic_handler.failure_evidence(),
        output_evidence=diagnostic_handler.output_evidence(),
    )
    if (
        result is TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
        and diagnostic_handler.gpu_failure_category is not None
    ):
        report = replace(
            report, failure_stage="resource_selection",
            failure_category=diagnostic_handler.gpu_failure_category,
        )
    if diagnostic_handler.storage_failure is not None:
        stage, category = diagnostic_handler.storage_failure
        report = replace(
            report, result=TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE,
            failure_stage=stage, failure_category=category,
        )
    return report


def prepare_task_runtime_result(
    task_dir: Path,
    timeout_seconds: int = 1800,
    *,
    containerize: bool = False,
    container_image: str = "",
    container_network: str = "bridge",
    python_bin: str = "python3",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    base_environment: Mapping[str, str] = os.environ,
    evaluator_tree_sha256: str | None = None,
    gpu_evaluation: bool = False,
    draft_smoke: bool = False,
    install_cache_dir: Path | None = None,
    baseline_results_dir: Path | None = None,
    baseline_artifacts_dir: Path | None = None,
    evaluation_requirement: TaskRuntimeEvaluationRequirement = (
        TaskRuntimeEvaluationRequirement.CORRECT_RESULT
    ),
    retained_results_dir: Path | None = None,
) -> TaskRuntimePreparationResult:
    """Prepare a runtime and retain whether a failure was authored or operational."""
    task_dir = task_dir.resolve()
    if baseline_results_dir is not None:
        try:
            if draft_smoke or not (task_dir / TASK_CONTRACT_FILENAME).is_file():
                raise ValueError("baseline archives require contracted production evaluation")
            baseline_results_dir = validate_baseline_results_dir(
                baseline_results_dir, task_dir=task_dir,
            )
        except (OSError, ValueError) as exc:
            _log_runtime_exception("Baseline archive preflight failed: %s", exc,
                                   stage="baseline_archive")
            return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
    try:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise TaskRuntimeDeadlineExceeded(
                "task runtime preparation timeout must be positive and finite"
            )
        deadline = time.monotonic() + timeout_seconds
        with _runtime_build_lock(task_dir, deadline):
            _remove_stale_runtime_containers(task_dir, deadline)
            _remove_stale_runtime_builds(task_dir, deadline)
            return _prepare_task_runtime(
                task_dir,
                deadline=deadline,
                containerize=containerize,
                container_image=container_image,
                container_network=container_network,
                python_bin=python_bin,
                run=run,
                base_environment=base_environment,
                evaluator_tree_sha256=evaluator_tree_sha256,
                gpu_evaluation=gpu_evaluation,
                draft_smoke=draft_smoke,
                install_cache_dir=install_cache_dir,
                baseline_results_dir=baseline_results_dir,
                evaluation_requirement=evaluation_requirement,
                retained_results_dir=retained_results_dir,
                baseline_artifacts_dir=baseline_artifacts_dir,
            )
    except _TaskRuntimeContractDeadlineExceeded as exc:
        logger.error(
            "Task runtime baseline validation failed: %s",
            exc,
            extra={"task_runtime_failure_evidence": exc.failure_evidence},
        )
        return TaskRuntimePreparationResult.AUTHORED_FAILURE
    except TaskRuntimeDeadlineExceeded as exc:
        logger.error(
            "Task runtime preparation timed out: %s",
            exc,
            extra={"task_runtime_failure_evidence": exc.failure_evidence},
        )
        return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
    except OSError as exc:
        _log_runtime_exception("Task runtime staging failed: %s", exc, stage="runtime_staging")
        return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE


def _prepare_task_runtime(
    task_dir: Path,
    *,
    deadline: float,
    containerize: bool = False,
    container_image: str = "",
    container_network: str = "bridge",
    python_bin: str = "python3",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    base_environment: Mapping[str, str] = os.environ,
    evaluator_tree_sha256: str | None = None,
    gpu_evaluation: bool = False,
    draft_smoke: bool = False,
    install_cache_dir: Path | None = None,
    baseline_results_dir: Path | None = None,
    baseline_artifacts_dir: Path | None = None,
    evaluation_requirement: TaskRuntimeEvaluationRequirement = (
        TaskRuntimeEvaluationRequirement.CORRECT_RESULT
    ),
    retained_results_dir: Path | None = None,
) -> TaskRuntimePreparationResult:
    """Build a managed venv and prove the baseline in the same sealed image."""
    del python_bin  # Host execution is intentionally unsupported for drafted code.
    if baseline_artifacts_dir is not None and (
        draft_smoke
        or baseline_artifacts_dir.resolve().is_relative_to(task_dir.resolve())
        or task_dir.resolve().is_relative_to(baseline_artifacts_dir.resolve())
    ):
        logger.error("Baseline artifacts require a production run and an external output directory")
        return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
    if not containerize:
        logger.error("Drafted task runtime preparation requires container.enabled=true")
        return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE

    sandbox_gpus: str | None = None
    cuda_visible_devices: str | None = ""
    if gpu_evaluation:
        sandbox_gpus = base_environment.get("AGENT_CONTAINER_GPUS", "").strip()
        if not sandbox_gpus:
            logger.error("GPU task evaluation requires an explicit AGENT_CONTAINER_GPUS selector")
            return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
        cuda_visible_devices = base_environment.get("CUDA_VISIBLE_DEVICES")

    task_dir = task_dir.resolve()
    _remaining_runtime_seconds(deadline)
    current_evaluator_tree_sha256 = managed_evaluator_tree_digest(task_dir)
    _remaining_runtime_seconds(deadline)
    if evaluator_tree_sha256 is not None and current_evaluator_tree_sha256 != evaluator_tree_sha256:
        logger.error("Task evaluator tree changed before runtime preparation")
        return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
    evaluator_tree_sha256 = current_evaluator_tree_sha256
    requirements_path = task_dir / "requirements.txt"
    if requirements_path.exists() and not _validate_requirements(requirements_path):
        return TaskRuntimePreparationResult.AUTHORED_FAILURE
    contract_path = task_dir / TASK_CONTRACT_FILENAME
    contract: TaskContract | None = None
    if contract_path.is_file():
        try:
            contract = load_task_contract(contract_path)
        except TaskContractError as exc:
            logger.error("Invalid contracted baseline task: %s", exc)
            return TaskRuntimePreparationResult.AUTHORED_FAILURE
        if not _requirements_match_contract(requirements_path, contract):
            return TaskRuntimePreparationResult.AUTHORED_FAILURE
    if baseline_artifacts_dir is not None and contract is None:
        logger.error("Baseline artifact archival requires a contracted task")
        return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
    requirements_sha256 = _requirements_digest(requirements_path)
    _remaining_runtime_seconds(deadline)
    tooling_sha256 = _tooling_digest()
    _remaining_runtime_seconds(deadline)
    evaluator_path = task_dir / "evaluate.py"
    if not _is_safe_regular_file(evaluator_path):
        logger.error("Task runtime requires a safe evaluator: %s", evaluator_path)
        return TaskRuntimePreparationResult.AUTHORED_FAILURE
    evaluator_sha256 = _evaluator_digest(evaluator_path)
    _remaining_runtime_seconds(deadline)
    try:

        def run_image_inspection(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            kwargs["timeout"] = _remaining_runtime_seconds(deadline)
            try:
                return run(command, **kwargs)
            except subprocess.TimeoutExpired as exc:
                raise TaskRuntimeDeadlineExceeded(
                    "task runtime preparation deadline exceeded during image inspection"
                ) from exc

        image_id = resolve_immutable_image(container_image, run=run_image_inspection)
        _remaining_runtime_seconds(deadline)
    except ValueError as exc:
        logger.error("Task runtime image preflight failed: %s", exc)
        return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE

    venv_dir = task_dir / ".venv"
    manifest_path = task_dir / RUNTIME_MANIFEST_FILENAME
    if manifest_path.is_symlink() or venv_dir.is_symlink():
        logger.error("Refusing symlinked task runtime metadata or directory")
        return TaskRuntimePreparationResult.AUTHORED_FAILURE
    runtime_manifest = _load_runtime_manifest(task_dir)
    if venv_dir.exists() and runtime_manifest is None:
        logger.error("Refusing unmanaged task runtime: %s", venv_dir)
        return TaskRuntimePreparationResult.AUTHORED_FAILURE
    if venv_dir.exists() and not _is_safe_regular_file(venv_dir / "bin" / "python"):
        logger.error("Refusing unsafe task runtime: %s", venv_dir)
        return TaskRuntimePreparationResult.AUTHORED_FAILURE
    task_runtime_compatible = venv_dir.exists() and _runtime_is_compatible(
        task_dir,
        image_id=image_id,
        requirements_sha256=requirements_sha256,
        tooling_sha256=tooling_sha256,
        evaluator_sha256=evaluator_sha256,
        evaluator_tree_sha256=evaluator_tree_sha256,
        gpu_evaluation=gpu_evaluation,
        cuda_visible_devices=cuda_visible_devices,
    )
    reuse_runtime = install_cache_dir is None and task_runtime_compatible
    if venv_dir.exists() and not reuse_runtime:
        logger.info("Rebuilding stale managed task runtime: %s", venv_dir)
    fresh_baseline_requested = (
        draft_smoke
        or baseline_results_dir is not None
        or baseline_artifacts_dir is not None
        or retained_results_dir is not None
    )
    if (
        reuse_runtime
        and not fresh_baseline_requested
        and _manifest_baseline_satisfies(runtime_manifest, evaluation_requirement)
    ):
        with preparation_step("baseline_reuse"):
            logger.info("Reusing validated task runtime and baseline: %s", venv_dir)
        return TaskRuntimePreparationResult.READY

    install_cache: TaskRuntimeInstallCache | None = None
    install_cache_key: TaskRuntimeInstallCacheKey | None = None
    cached_runtime_root: Path | None = None
    if install_cache_dir is not None:
        install_cache_key = TaskRuntimeInstallCacheKey(
            image_id=image_id,
            requirements_sha256=requirements_sha256,
            tooling_sha256=tooling_sha256,
        )
        try:
            install_cache = TaskRuntimeInstallCache(install_cache_dir)
            if not reuse_runtime:
                with preparation_step("runtime_cache"):
                    cached_runtime_root = install_cache.lookup(
                        install_cache_key,
                        deadline=deadline,
                    )
        except TaskRuntimeInstallCacheDeadlineExceeded as exc:
            raise TaskRuntimeDeadlineExceeded(
                "task runtime install cache lookup deadline exceeded"
            ) from exc
        except (OSError, ValueError) as exc:
            _log_runtime_exception(
                "Task runtime install cache validation failed: %s", exc, stage="cache_validation",
            )
            return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE

    build_lease: RuntimeBuildLease | None = None
    if not reuse_runtime and cached_runtime_root is None:
        try:
            with preparation_step("runtime_capacity"):
                build_lease = acquire_runtime_build_lease(deadline=deadline)
            if install_cache is not None and install_cache_key is not None:
                with preparation_step("runtime_cache"):
                    cached_runtime_root = install_cache.lookup(
                        install_cache_key,
                        deadline=deadline,
                    )
        except (RuntimeBuildLeaseTimeout, TaskRuntimeInstallCacheDeadlineExceeded) as exc:
            if build_lease is not None:
                build_lease.release()
            raise TaskRuntimeDeadlineExceeded(
                "task runtime build capacity remained unavailable until the deadline"
            ) from exc
        except (RuntimeBuildLeaseError, OSError, ValueError) as exc:
            if build_lease is not None:
                build_lease.release()
            _log_runtime_exception(
                "Task runtime build coordination failed: %s", exc, stage="build_coordination",
            )
            return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
        if cached_runtime_root is not None:
            build_lease.release()
            build_lease = None

    cache_owned_build = (
        install_cache is not None and install_cache_key is not None and cached_runtime_root is None
    )
    if cache_owned_build:
        assert install_cache is not None
        try:
            build_root = install_cache.create_build_root(deadline=deadline)
        except TaskRuntimeInstallCacheDeadlineExceeded as exc:
            assert build_lease is not None
            build_lease.release()
            build_lease = None
            raise TaskRuntimeDeadlineExceeded(
                "task runtime install cache build staging deadline exceeded"
            ) from exc
        except (OSError, ValueError) as exc:
            assert build_lease is not None
            build_lease.release()
            build_lease = None
            _log_runtime_exception(
                "Task runtime install cache build staging failed: %s", exc, stage="cache_staging",
            )
            return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
    else:
        build_root = task_dir.parent / f".{task_dir.name}.runtime-build-{uuid.uuid4().hex}"
        try:
            build_root.mkdir(mode=0o700)
        except BaseException:
            if build_lease is not None:
                build_lease.release()
                build_lease = None
            raise
    cleanup_build_root = True
    publication_root = build_root
    cleanup_publication_root = True
    staged_venv = build_root / ".venv"
    tooling_source = build_root / "tooling-source"
    runtime_root = task_dir if reuse_runtime else cached_runtime_root or build_root
    runtime_python = "/task/.venv/bin/python" if reuse_runtime else "/runtime/.venv/bin/python"
    try:
        with tempfile.TemporaryDirectory(prefix="agentic-task-runtime-") as temp_home:
            temp_path = Path(temp_home)
            environment = _installation_environment(base_environment, temp_path, task_dir)
            if not reuse_runtime and cached_runtime_root is None:
                (build_root / ".tmp").mkdir(mode=0o700)
                tooling_source.mkdir()
                shutil.copy2(_REPO_ROOT / "pyproject.toml", tooling_source)
                (tooling_source / "src").mkdir()
                shutil.copy2(_REPO_ROOT / "src" / "__init__.py", tooling_source / "src")
                for relative_dir in _tooling_package_dirs():
                    _remaining_runtime_seconds(deadline)
                    shutil.copytree(
                        _REPO_ROOT / relative_dir,
                        tooling_source / relative_dir,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    )
                create_cid = task_dir.parent / (
                    f".{task_dir.name}.runtime-create-{uuid.uuid4().hex}.cid"
                )
                create_command = _docker_command(
                    task_dir=task_dir,
                    image_id=image_id,
                    network=container_network,
                    runtime_dir=build_root,
                    runtime_writable=True,
                    cid_path=create_cid,
                    inner_command=["python3", "-m", "venv", "--copies", "/runtime/.venv"],
                )
                if (
                    _run_checked_with_progress(
                        create_command,
                        task_dir=task_dir,
                        deadline=deadline,
                        run=run,
                        environment=environment,
                        cid_path=create_cid,
                        stage="runtime_create",
                    ).status
                    is not _CheckedCommandStatus.SUCCEEDED
                ):
                    return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE

                tooling_build_cid = task_dir.parent / (
                    f".{task_dir.name}.tooling-build-{uuid.uuid4().hex}.cid"
                )
                tooling_build_command = _docker_command(
                    task_dir=task_dir,
                    image_id=image_id,
                    network=container_network,
                    runtime_dir=build_root,
                    runtime_writable=True,
                    cid_path=tooling_build_cid,
                    inner_command=[
                        "/runtime/.venv/bin/python",
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--no-input",
                        *_TOOLING_BUILD_REQUIREMENTS,
                    ],
                )
                if (
                    _run_checked_with_progress(
                        tooling_build_command,
                        task_dir=task_dir,
                        deadline=deadline,
                        run=run,
                        environment=environment,
                        cid_path=tooling_build_cid,
                        stage="tooling_build",
                    ).status
                    is not _CheckedCommandStatus.SUCCEEDED
                ):
                    return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE

                tooling_install_cid = task_dir.parent / (
                    f".{task_dir.name}.tooling-install-{uuid.uuid4().hex}.cid"
                )
                tooling_install_command = _docker_command(
                    task_dir=task_dir,
                    image_id=image_id,
                    network=container_network,
                    runtime_dir=build_root,
                    runtime_writable=True,
                    cid_path=tooling_install_cid,
                    inner_command=[
                        "/runtime/.venv/bin/python",
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--no-input",
                        "--no-deps",
                        "--no-build-isolation",
                        "/runtime/tooling-source",
                    ],
                )
                if (
                    _run_checked_with_progress(
                        tooling_install_command,
                        task_dir=task_dir,
                        deadline=deadline,
                        run=run,
                        environment=environment,
                        cid_path=tooling_install_cid,
                        stage="tooling_install",
                    ).status
                    is not _CheckedCommandStatus.SUCCEEDED
                ):
                    return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE

                if requirements_path.is_file():
                    install_cid = task_dir.parent / (
                        f".{task_dir.name}.runtime-install-{uuid.uuid4().hex}.cid"
                    )
                    install_command = _docker_command(
                        task_dir=task_dir,
                        image_id=image_id,
                        network=container_network,
                        runtime_dir=build_root,
                        runtime_writable=True,
                        cid_path=install_cid,
                        inner_command=[
                            "/runtime/.venv/bin/python",
                            "-m",
                            "pip",
                            "install",
                            "--disable-pip-version-check",
                            "--no-input",
                            "--only-binary=:all:",
                            "--requirement",
                            "/task/requirements.txt",
                        ],
                    )
                    install_result = _run_checked_with_progress(
                        install_command,
                        task_dir=task_dir,
                        deadline=deadline,
                        run=run,
                        environment=environment,
                        cid_path=install_cid,
                        stage="dependency_install",
                    )
                    if install_result.status is _CheckedCommandStatus.INFRASTRUCTURE_FAILED:
                        return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
                    if install_result.status is not _CheckedCommandStatus.SUCCEEDED:
                        return TaskRuntimePreparationResult.DEPENDENCY_FAILURE

                if install_cache is not None and install_cache_key is not None:
                    try:
                        with preparation_step("runtime_publish"):
                            cached_runtime_root = install_cache.store_owned(
                                install_cache_key,
                                staged_venv,
                                deadline=deadline,
                            )
                    except TaskRuntimeInstallCacheDeadlineExceeded as exc:
                        cleanup_build_root = not staged_venv.exists()
                        raise TaskRuntimeDeadlineExceeded(
                            "task runtime install cache publication deadline exceeded"
                        ) from exc
                    except (OSError, ValueError):
                        cleanup_build_root = not staged_venv.exists()
                        raise
                    runtime_root = cached_runtime_root

                assert build_lease is not None
                build_lease.release()
                build_lease = None

            entrypoint = entrypoint_relpath_from_spec(task_dir / "program_0")
            if entrypoint is None:
                logger.error("Drafted baseline has no Python entrypoint")
                return TaskRuntimePreparationResult.AUTHORED_FAILURE
            baseline_score: float | None = None
            baseline_correct: bool | None = None
            baseline_feasible: bool | None = None
            if contract is not None:
                evaluation_options: dict[str, Any] = dict(
                    task_dir=task_dir,
                    runtime_venv=runtime_root / ".venv",
                    image_id=image_id,
                    contract=contract,
                    entrypoint=entrypoint,
                    deadline=deadline,
                    environment=environment,
                    run=run,
                    draft_smoke=draft_smoke,
                    evaluation_requirement=evaluation_requirement,
                )
                baseline_result = _run_contracted_baseline(
                    **evaluation_options,
                    baseline_results_dir=baseline_results_dir,
                    retained_results_dir=retained_results_dir,
                    baseline_artifacts_dir=baseline_artifacts_dir,
                )
                if baseline_result.status is _CheckedCommandStatus.INFRASTRUCTURE_FAILED:
                    return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
                if baseline_result.status is not _CheckedCommandStatus.SUCCEEDED:
                    logger.error("Offline baseline evaluation failed validation")
                    return TaskRuntimePreparationResult.AUTHORED_FAILURE
                if draft_smoke:
                    replay_failure = _replay_contracted_smoke_baseline(
                        baseline_result, evaluation_options,
                    )
                    if replay_failure is not None:
                        return replay_failure
                baseline_score = baseline_result.combined_score
                baseline_correct = baseline_result.correct
                baseline_feasible = baseline_result.feasible
            else:
                with _baseline_results_directory(retained_results_dir) as results_dir:
                    evaluation_cid = task_dir.parent / (
                        f".{task_dir.name}.runtime-evaluation-{uuid.uuid4().hex}.cid"
                    )
                    evaluation_command = _docker_command(
                        task_dir=task_dir,
                        image_id=image_id,
                        network="none",
                        runtime_dir=None if reuse_runtime else runtime_root,
                        results_dir=results_dir,
                        cuda_visible_devices=cuda_visible_devices,
                        sandbox_gpus=sandbox_gpus,
                        cid_path=evaluation_cid,
                        search_evaluation=True,
                        draft_smoke=draft_smoke,
                        qualification=(
                            evaluation_requirement
                            is TaskRuntimeEvaluationRequirement.FEASIBLE_RESULT
                        ),
                        inner_command=[
                            runtime_python,
                            "/task/evaluate.py",
                            "--program_path",
                            f"/task/program_0/{entrypoint}",
                            "--results_dir",
                            "/results",
                        ],
                    )
                    evaluation_result = _run_checked_with_progress(
                        evaluation_command,
                        task_dir=task_dir,
                        deadline=deadline,
                        run=run,
                        environment=environment,
                        cid_path=evaluation_cid,
                        stage="baseline_validation",
                    )
                    baseline_score = None
                    if evaluation_result.status is _CheckedCommandStatus.SUCCEEDED:
                        baseline_score = _validated_evaluation_score(
                            results_dir,
                            requirement=evaluation_requirement,
                        )
                        if baseline_score is not None:
                            baseline_correct = _validated_evaluation_correctness(results_dir)
                            if (
                                evaluation_requirement
                                is TaskRuntimeEvaluationRequirement.FEASIBLE_RESULT
                            ):
                                baseline_feasible = _validated_evaluation_feasibility(results_dir)
                                if baseline_feasible is not True:
                                    baseline_score = None
                    if evaluation_result.status is _CheckedCommandStatus.INFRASTRUCTURE_FAILED:
                        return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
                    if (
                        evaluation_result.status is not _CheckedCommandStatus.SUCCEEDED
                        or baseline_score is None
                    ):
                        if evaluation_result.status is _CheckedCommandStatus.SUCCEEDED:
                            _log_failed_evaluation(evaluation_result.process, environment=environment)
                        logger.error("Offline baseline evaluation failed validation")
                        return TaskRuntimePreparationResult.AUTHORED_FAILURE

            if (
                baseline_score is None
                or baseline_correct is None
                or (
                    evaluation_requirement is TaskRuntimeEvaluationRequirement.FEASIBLE_RESULT
                    and baseline_feasible is not True
                )
            ):
                logger.error("Offline baseline evaluation did not retain a validated result")
                return TaskRuntimePreparationResult.AUTHORED_FAILURE

            if managed_evaluator_tree_digest(task_dir) != evaluator_tree_sha256:
                logger.error("Task evaluator tree changed during runtime preparation")
                return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
            _remaining_runtime_seconds(deadline)

        if cache_owned_build:
            publication_root = task_dir.parent / (
                f".{task_dir.name}.runtime-build-{uuid.uuid4().hex}"
            )
            publication_root.mkdir(mode=0o700)
        backup_venv = publication_root / ".previous-venv"
        published_new_runtime = False
        try:
            if not reuse_runtime:
                if venv_dir.exists():
                    venv_dir.replace(backup_venv)
                if install_cache is not None and install_cache_key is not None:
                    try:
                        with preparation_step("runtime_copy"):
                            install_cache.materialize(
                                install_cache_key,
                                venv_dir,
                                deadline=deadline,
                            )
                    except TaskRuntimeInstallCacheDeadlineExceeded as exc:
                        raise TaskRuntimeDeadlineExceeded(
                            "task runtime install cache materialization deadline exceeded"
                        ) from exc
                else:
                    staged_venv.replace(venv_dir)
                published_new_runtime = True
            _write_runtime_manifest(
                task_dir,
                {
                    "image_id": image_id,
                    "baseline_combined_score": baseline_score,
                    "baseline_correct": baseline_correct,
                    "baseline_feasible": baseline_feasible,
                    "evaluator_sha256": evaluator_sha256,
                    "evaluator_tree_sha256": evaluator_tree_sha256,
                    "gpu_evaluation": gpu_evaluation,
                    "cuda_visible_devices": cuda_visible_devices,
                    "sandbox_gpus": sandbox_gpus,
                    "requirements_sha256": requirements_sha256,
                    "schema_version": 1,
                    "tooling_sha256": tooling_sha256,
                },
            )
        except BaseException as publication_error:
            if storage_failure_category(publication_error) is not None:
                _log_runtime_exception(
                    "Task runtime publication failed: %s", publication_error,
                    stage="runtime_publication",
                )
            try:
                if published_new_runtime and venv_dir.exists():
                    venv_dir.replace(publication_root / ".failed-new-venv")
                if backup_venv.exists():
                    backup_venv.replace(venv_dir)
            except (OSError, ValueError) as rollback_error:
                if publication_root == build_root:
                    cleanup_build_root = False
                else:
                    cleanup_publication_root = False
                publication_error.add_note("Task runtime rollback failed; recovery state retained")
                _log_runtime_exception(
                    "Task runtime rollback failed: %s", rollback_error, stage="runtime_rollback",
                )
                raise publication_error from rollback_error
            raise
        if backup_venv.exists():
            shutil.rmtree(backup_venv)
    except TaskRuntimeDeadlineExceeded:
        raise
    except ContainerMetadataCleanupError as exc:
        logger.error("Task runtime metadata cleanup failed: %s", exc)
        return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
    except (OSError, ValueError) as exc:
        _log_runtime_exception(
            "Task runtime publication failed: %s", exc, stage="runtime_publication",
        )
        return TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
    finally:
        try:
            if cleanup_build_root:
                shutil.rmtree(build_root, ignore_errors=True)
            else:
                logger.error("Task runtime recovery state retained: %s", build_root)
            if publication_root != build_root:
                if cleanup_publication_root:
                    shutil.rmtree(publication_root, ignore_errors=True)
                else:
                    logger.error("Task runtime recovery state retained: %s", publication_root)
        finally:
            if build_lease is not None:
                build_lease.release()

    logger.info("Task runtime ready and baseline validated: %s", venv_dir)
    return TaskRuntimePreparationResult.READY

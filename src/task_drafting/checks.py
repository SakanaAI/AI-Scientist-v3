"""Production-topology validation checks shared by drafting and repair agents."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NoReturn

from src.final_evaluation.candidate_broker import (
    CandidateBroker,
    CandidateBrokerError,
    CandidateBrokerInfrastructureError,
    CandidateContainerCleanupError,
    ContainerMetadataCleanupError,
)
from src.final_evaluation.runner import _candidate_command, _evaluation_command
from src.task_drafting.contract_sync import ContractSyncError, synchronize_declared_digests
from src.task_drafting.gate import TaskGateReport, sanitize_diagnostic
from src.task_drafting.harness_capability_policy import (
    unavailable_harness_environment_findings,
)
from src.task_drafting.production_canary import run_production_entry_canaries
from src.task_drafting.publication import (
    RuntimeResourceValidationDeadlineExceeded,
    validate_staged_runtime_resources,
)
from src.task_drafting.publication_fs import TaskPublicationError
from src.task_drafting.rpc_transport_policy import isolated_rpc_transport_findings
from src.investigation.preparation_progress import preparation_step
from src.investigation.bounded_process import (
    BoundedProcessError,
    ProcessOutputLimitError,
    ProcessTimeoutError,
    run_bounded_process,
)
from src.investigation.container_images import (
    ContainerImageInfrastructureError,
    ContainerImageResolutionError,
    resolve_immutable_image,
)
from src.investigation.container_gpus import (
    GPU_ALLOCATION_FAILURE_CATEGORIES,
    GPU_ALLOCATION_BUSY_CATEGORY,
    GPU_ALLOCATION_INVALID_CATEGORY,
    GPU_ALLOCATION_LEASE_ROOT,
    GPU_ALLOCATION_QUARANTINED_CATEGORY,
    GpuAllocationLease,
    GpuLeaseError,
    GpuLeaseQuarantined,
    GpuLeaseTimeout,
    acquire_gpu_allocation_lease,
)
from src.investigation.evaluator_replay import EvaluationOutcome, replay_mismatch
from src.investigation.evaluator_tree import managed_evaluator_tree_digest
from src.investigation.program_spec import entrypoint_relpath_from_spec
from src.investigation.reset_sequence import reset_sequence_diagnostic
from src.investigation.task_alignment import task_tree_fingerprint
from src.investigation.task_contract import TaskContract, TaskContractError, load_task_contract
from src.investigation.task_contract_lint import task_contract_preflight_findings
from src.investigation.task_runtime import (
    TaskRuntimeDeadlineExceeded,
    TaskRuntimeEvaluationRequirement,
    TaskRuntimePreparationResult,
    _draft_validation_reset_sequence_matches,
    _evaluator_digest,
    _installation_environment,
    _requirements_digest,
    _prepare_task_broker_directory,
    _remaining_runtime_seconds,
    _remove_container_before_deadline,
    _runtime_is_compatible,
    _runtime_execution_deadline,
    _tooling_digest,
    _validated_drafting_evaluation_score,
    prepare_task_runtime_result,
    runtime_preparation_failure_is_repairable,
)


@dataclass(frozen=True)
class GateSettings:
    task_dir: Path
    source_task_dir: Path
    containerize: bool
    container_image: str
    container_network: str
    timeout_seconds: int
    install_cache_dir: Path | None = None
    production_entry_timeout_seconds: int = 300
    acceptance_timeout_seconds: int | None = None
    max_runtime_bytes: int = 16 * 1024 * 1024 * 1024


class DraftGateFailure(RuntimeError):
    def __init__(
        self,
        status: str,
        category: str,
        component: str,
        stage: str,
        diagnostic: str = "",
        exit_code: int = 3,
    ) -> None:
        super().__init__(diagnostic or category)
        self.status = status
        self.category = category
        self.component = component
        self.stage = stage
        self.diagnostic = diagnostic
        self.exit_code = exit_code


def _prepared_runtime_available(settings: GateSettings, evaluator_tree_sha256: str) -> bool:
    try:
        contract = load_task_contract(settings.task_dir / "task_contract.json")
        image_id = resolve_immutable_image(settings.container_image)
        gpu_evaluation = contract.compute.search.accelerator == "gpu"
        return _runtime_is_compatible(
            settings.task_dir,
            image_id=image_id,
            requirements_sha256=_requirements_digest(
                settings.task_dir / "requirements.txt"
            ),
            tooling_sha256=_tooling_digest(),
            evaluator_sha256=_evaluator_digest(
                settings.task_dir / contract.evaluators.search
            ),
            evaluator_tree_sha256=evaluator_tree_sha256,
            gpu_evaluation=gpu_evaluation,
            cuda_visible_devices=(
                os.environ.get("CUDA_VISIBLE_DEVICES") if gpu_evaluation else ""
            ),
        )
    except (OSError, TaskContractError, ValueError):
        return False


def _cleanup_failure_diagnostic(exc: BaseException) -> str:
    if isinstance(exc, (CandidateContainerCleanupError, ContainerMetadataCleanupError)):
        return sanitize_diagnostic(str(exc), max_bytes=4096)
    return "Container runtime failed; cleanup details unavailable."


class _RuntimeDiagnosticHandler(logging.Handler):
    """Retain a bounded tail of host runtime errors for scaffold repair."""

    def __init__(self, max_bytes: int = 8 * 1024) -> None:
        super().__init__(logging.ERROR)
        self._max_bytes = max_bytes
        self._tail = bytearray()
        self.gpu_failure_category: str | None = None

    def emit(self, record: logging.LogRecord) -> None:
        gpu_category = getattr(record, "task_runtime_gpu_failure", None)
        if isinstance(gpu_category, str) and gpu_category in GPU_ALLOCATION_FAILURE_CATEGORIES:
            self.gpu_failure_category = gpu_category
        try:
            message = sanitize_diagnostic(record.getMessage(), max_bytes=self._max_bytes)
            encoded = (message + "\n").encode("utf-8", errors="replace")
        except Exception:
            return
        self._tail.extend(encoded)
        excess = len(self._tail) - self._max_bytes
        if excess > 0:
            del self._tail[:excess]

    def diagnostic(self) -> str:
        return self._tail.decode("utf-8", errors="replace").strip()


def _authored(category: str, component: str, stage: str, diagnostic: str = "") -> NoReturn:
    raise DraftGateFailure("authored_failure", category, component, stage, diagnostic)


def _infrastructure(
    category: str,
    component: str,
    stage: str,
    diagnostic: str = "",
) -> NoReturn:
    raise DraftGateFailure(
        "infrastructure_failure",
        category,
        component,
        stage,
        diagnostic,
        exit_code=4,
    )


def _reset_sequence_diagnostic(
    observed_reset_seeds: tuple[object, ...],
    declared_reset_seeds: tuple[int, ...],
) -> str:
    """Describe a smoke reset mismatch without rendering arbitrary JSON values."""
    return reset_sequence_diagnostic(
        observed_reset_seeds,
        declared_reset_seeds,
        context="Draft final smoke",
    )


def _safe_scaffold_fingerprint(task_dir: Path) -> str:
    try:
        return task_tree_fingerprint(task_dir)
    except Exception:
        return "sha256:" + "0" * 64


def _require_score(results_dir: Path, plane: str, contract: object) -> EvaluationOutcome:
    try:
        metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
        correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _authored(f"{plane}_invalid_json", f"{plane}_evaluator", "results")
    declared_metrics = (
        contract.metrics.search if plane == "search" else contract.metrics.final  # type: ignore[attr-defined]
    )
    declared = {metric.name for metric in declared_metrics}
    search_score = (
        _validated_drafting_evaluation_score(results_dir) if plane == "search" else 0.0
    )
    if (
        not isinstance(metrics, dict)
        or set(metrics) != declared
        or not isinstance(correct, dict)
        or set(correct) != {"correct"}
        or not isinstance(correct["correct"], bool)
        or search_score is None
        or not math.isfinite(search_score)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in metrics.values()
        )
    ):
        _authored(f"{plane}_invalid_results", f"{plane}_evaluator", "results")
    return EvaluationOutcome(search_score, correct["correct"])


def _native_smoke(settings: GateSettings, evaluator: str, entrypoint: str, plane: str) -> None:
    safe_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AGENTIC_CANDIDATE_")
        and not any(
            token in key.upper()
            for token in ("AUTH", "CREDENTIAL", "KEY", "PASSWORD", "PROXY", "SECRET", "TOKEN")
        )
    }
    safe_environment.update(
        AGENTIC_TASK_DRAFT_SMOKE="1",
        HOME=tempfile.gettempdir(),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        TASK_EVALUATION_PLANE=plane,
        TASK_SOURCE_DIR=str(settings.task_dir),
    )
    first = _native_evaluation(settings, evaluator, entrypoint, plane, safe_environment)
    if plane != "search":
        return
    mismatch = replay_mismatch(
        first, _native_evaluation(settings, evaluator, entrypoint, plane, safe_environment),
    )
    if mismatch is not None:
        _authored("search_nondeterministic", "search_evaluator", "execution", mismatch)


def _native_evaluation(
    settings: GateSettings,
    evaluator: str,
    entrypoint: str,
    plane: str,
    safe_environment: dict[str, str],
) -> EvaluationOutcome:
    with tempfile.TemporaryDirectory(prefix=f"agentic-task-{plane}-smoke-") as output:
        try:
            result = run_bounded_process(
                [
                    sys.executable,
                    str(settings.task_dir / evaluator),
                    "--program_path",
                    str(settings.task_dir / "program_0" / entrypoint),
                    "--results_dir",
                    output,
                ],
                cwd=settings.task_dir,
                env=safe_environment,
                timeout_seconds=settings.timeout_seconds,
            )
        except ProcessTimeoutError:
            _authored(f"{plane}_timeout", f"{plane}_evaluator", "execution")
        except ProcessOutputLimitError:
            _authored(f"{plane}_excessive_output", f"{plane}_evaluator", "execution")
        except (OSError, BoundedProcessError):
            _infrastructure(f"{plane}_process_runtime", f"{plane}_evaluator", "execution")
        if result.returncode != 0:
            diagnostic = (
                result.stderr.decode("utf-8", errors="replace") if plane == "search" else ""
            )
            _authored(
                f"{plane}_nonzero_exit",
                f"{plane}_evaluator",
                "execution",
                sanitize_diagnostic(diagnostic),
            )
        contract = load_task_contract(settings.task_dir / "task_contract.json")
        return _require_score(Path(output), plane, contract)


def _prepare_runtime(settings: GateSettings, evaluator_digest: str) -> None:
    contract = load_task_contract(settings.task_dir / "task_contract.json")
    if (
        contract.compute.search.accelerator == "gpu"
        and not os.environ.get("AGENT_CONTAINER_GPUS", "").strip()
    ):
        _infrastructure("gpu_selector_missing", "harness", "resource_selection")
    try:
        container_image = resolve_immutable_image(settings.container_image)
    except ContainerImageInfrastructureError as exc:
        _infrastructure("runtime_bootstrap", "runtime", "container_start", str(exc))
    except ContainerImageResolutionError:
        _infrastructure("container_image_resolution", "harness", "container_start")
    runtime_logger = logging.getLogger("src.investigation.task_runtime")
    diagnostic_handler = _RuntimeDiagnosticHandler()
    runtime_logger.addHandler(diagnostic_handler)
    try:
        result = prepare_task_runtime_result(
            settings.task_dir,
            timeout_seconds=settings.timeout_seconds,
            containerize=True,
            container_image=container_image,
            container_network=settings.container_network,
            evaluator_tree_sha256=evaluator_digest,
            gpu_evaluation=contract.compute.search.accelerator == "gpu",
            draft_smoke=True,
            install_cache_dir=settings.install_cache_dir,
            evaluation_requirement=TaskRuntimeEvaluationRequirement.VALID_RESULT,
        )
    finally:
        runtime_logger.removeHandler(diagnostic_handler)
    if runtime_preparation_failure_is_repairable(result):
        category = (
            "runtime_dependencies"
            if result is TaskRuntimePreparationResult.DEPENDENCY_FAILURE
            else "runtime_baseline_validation"
        )
        _authored(
            category,
            "runtime",
            "bootstrap",
            diagnostic_handler.diagnostic(),
        )
    if result is TaskRuntimePreparationResult.DEPENDENCY_FAILURE:
        _infrastructure(
            "runtime_harness_dependencies",
            "runtime",
            "bootstrap",
            diagnostic_handler.diagnostic(),
        )
    if result is not TaskRuntimePreparationResult.READY:
        _infrastructure(
            diagnostic_handler.gpu_failure_category or "runtime_bootstrap",
            "runtime",
            "resource_selection" if diagnostic_handler.gpu_failure_category else "bootstrap",
            diagnostic_handler.diagnostic(),
        )


def _final_topology_smoke(
    settings: GateSettings,
    evaluator_digest: str,
    entrypoint: str,
    *,
    deadline: float | None = None,
) -> None:
    contract = load_task_contract(settings.task_dir / "task_contract.json")
    if (
        contract.compute.final.accelerator == "gpu"
        and not os.environ.get("AGENT_CONTAINER_GPUS", "").strip()
    ):
        _infrastructure("gpu_selector_missing", "harness", "resource_selection")
    try:
        image_id = resolve_immutable_image(settings.container_image)
    except ValueError:
        _infrastructure("container_image_resolution", "harness", "container_start")
    smoke_deadline = min(
        math.inf if deadline is None else deadline,
        time.monotonic() + settings.timeout_seconds,
    )
    evaluation_deadline = _runtime_execution_deadline(smoke_deadline)
    gpu_lease: GpuAllocationLease | None = None
    try:
        if contract.compute.final.gpu_count:
            gpu_lease = acquire_gpu_allocation_lease(
                os.environ.get("AGENT_CONTAINER_GPUS"),
                contract.compute.final.gpu_count,
                lease_root=GPU_ALLOCATION_LEASE_ROOT,
                deadline=min(evaluation_deadline, time.monotonic() + 30.0),
            )
    except GpuLeaseTimeout:
        _infrastructure(GPU_ALLOCATION_BUSY_CATEGORY, "harness", "resource_selection")
    except GpuLeaseQuarantined:
        _infrastructure(
            GPU_ALLOCATION_QUARANTINED_CATEGORY,
            "harness",
            "resource_selection",
        )
    except GpuLeaseError:
        _infrastructure(
            GPU_ALLOCATION_INVALID_CATEGORY,
            "harness",
            "resource_selection",
        )
    allocation = gpu_lease if gpu_lease is not None else nullcontext()
    try:
        with allocation:
            if gpu_lease is not None and evaluation_deadline <= time.monotonic():
                _infrastructure(
                    GPU_ALLOCATION_BUSY_CATEGORY,
                    "harness",
                    "resource_selection",
                )
            _run_final_topology_with_allocation(
                settings=settings,
                contract=contract,
                image_id=image_id,
                entrypoint=entrypoint,
                smoke_deadline=smoke_deadline,
                evaluation_deadline=evaluation_deadline,
                gpu_lease=gpu_lease,
            )
    except GpuLeaseQuarantined:
        _infrastructure(GPU_ALLOCATION_QUARANTINED_CATEGORY, "harness", "resource_selection")
    except GpuLeaseError:
        _infrastructure(GPU_ALLOCATION_INVALID_CATEGORY, "harness", "resource_selection")


def _run_final_topology_with_allocation(
    *,
    settings: GateSettings,
    contract: TaskContract,
    image_id: str,
    entrypoint: str,
    smoke_deadline: float,
    evaluation_deadline: float,
    gpu_lease: GpuAllocationLease | None,
) -> None:
    try:
        broker_dir = _prepare_task_broker_directory(
            settings.source_task_dir,
            "final-smoke",
            deadline=evaluation_deadline,
        )
    except (CandidateBrokerInfrastructureError, TaskRuntimeDeadlineExceeded) as exc:
        if gpu_lease is not None and (
            isinstance(exc, (TaskRuntimeDeadlineExceeded, CandidateContainerCleanupError))
        ):
            gpu_lease.quarantine()
        _infrastructure(
            "final_container_recovery", "broker", "container_start", _cleanup_failure_diagnostic(exc)
        )
    effective_timeout = max(1, math.ceil(_remaining_runtime_seconds(evaluation_deadline)))
    with tempfile.TemporaryDirectory(prefix="agentic-task-final-smoke-") as output:
        results_dir = Path(output)
        evaluator_path = settings.task_dir / contract.evaluators.final
        program_dir = settings.task_dir / "program_0"
        entrypoint_path = program_dir / entrypoint
        command = _evaluation_command(
            task_dir=settings.task_dir,
            program_dir=program_dir,
            entrypoint=entrypoint_path,
            evaluator=evaluator_path,
            results_dir=results_dir,
            python_bin=settings.task_dir / ".venv/bin/python",
            container_image=image_id,
            gpu_count=contract.compute.final.evaluator_gpu_count,
            cpu_cores=contract.compute.final.cpu_cores,
            memory_gb=contract.compute.final.memory_gb,
            environment=os.environ,
            broker_dir=broker_dir,
            candidate_editable_paths=contract.candidate.editable_paths,
            sealed_task_paths=(
                contract.evaluators.search,
                contract.resources.search_root,
                *(
                    artifact.path
                    for artifact in contract.provenance.artifacts
                    if artifact.role == "search_helper"
                ),
            ),
            rpc_timeout_seconds=effective_timeout,
            public_artifact_bundle=contract.resources.public_artifact_bundle,
            draft_smoke=True,
        )
        candidate_command = _candidate_command(
            task_dir=settings.task_dir,
            program_dir=program_dir,
            entrypoint=entrypoint_path,
            container_image=image_id,
            contract=contract,
            environment=os.environ,
            broker_dir=broker_dir,
        )
        if gpu_lease is not None:
            gpu_lease.prepare_runtime()
            command = gpu_lease.wrap_command(command)
            candidate_command = gpu_lease.wrap_command(candidate_command)
        evaluator_cleanup_verified = True
        evaluator_launch_attempted = False
        try:
            with tempfile.TemporaryDirectory(prefix="agentic-task-smoke-home-") as home:
                environment = _installation_environment(os.environ, Path(home), settings.task_dir)
                with CandidateBroker(
                    candidate_command,
                    broker_dir,
                    rpc_timeout_seconds=_remaining_runtime_seconds(evaluation_deadline),
                    reset_timeout_seconds=_remaining_runtime_seconds(evaluation_deadline),
                    max_rpc_calls=contract.compute.final.max_candidate_rpc_calls,
                    expected_reset_seeds=contract.protocol.final_smoke_reset_sequence,
                ) as broker:
                    completed_normally = False
                    evaluator_cleanup_verified = False
                    evaluator_launch_attempted = True
                    try:
                        result = run_bounded_process(
                            command,
                            cwd=settings.task_dir,
                            env=environment,
                            timeout_seconds=_remaining_runtime_seconds(evaluation_deadline),
                        )
                        completed_normally = True
                    except ProcessTimeoutError:
                        _authored("final_timeout", "final_evaluator", "execution")
                    except ProcessOutputLimitError:
                        _authored("final_excessive_output", "final_evaluator", "execution")
                    except (OSError, BoundedProcessError):
                        _infrastructure("final_process_runtime", "final_evaluator", "execution")
                    finally:
                        evaluator_cid = broker_dir / "evaluator.cid"
                        cid_observed = evaluator_cid.is_file()
                        try:
                            evaluator_removed = _remove_container_before_deadline(
                                evaluator_cid, smoke_deadline
                            )
                        except OSError:
                            if gpu_lease is not None:
                                gpu_lease.quarantine()
                            _infrastructure(
                                "evaluator_container_cleanup", "harness", "cleanup"
                            )
                        evaluator_cleanup_verified = evaluator_removed and (
                            completed_normally or cid_observed
                        )
                        if evaluator_removed:
                            try:
                                evaluator_cid.unlink(missing_ok=True)
                            except OSError as exc:
                                raise ContainerMetadataCleanupError(
                                    "evaluator container-id cleanup failed; recovery metadata retained"
                                ) from exc
        except CandidateBrokerInfrastructureError as exc:
            if gpu_lease is not None and (
                isinstance(exc, CandidateContainerCleanupError)
                or (evaluator_launch_attempted and not evaluator_cleanup_verified)
            ):
                gpu_lease.quarantine()
            _infrastructure(
                "candidate_container_runtime", "candidate", "container_start",
                _cleanup_failure_diagnostic(exc),
            )
        except CandidateBrokerError:
            _authored("candidate_protocol_failure", "candidate", "candidate_rpc")
        except BaseException:
            if (
                gpu_lease is not None
                and evaluator_launch_attempted
                and not evaluator_cleanup_verified
            ):
                gpu_lease.quarantine()
            raise
        if not evaluator_cleanup_verified:
            if gpu_lease is not None:
                gpu_lease.quarantine()
            _infrastructure("evaluator_container_cleanup", "harness", "cleanup")
        if result.returncode in {125, 126, 127}:
            _infrastructure("final_container_runtime", "final_evaluator", "container_start")
        if topology_category := _final_topology_category(
            returncode=result.returncode,
            broker_calls=broker.calls,
        ):
            _authored(topology_category, "final_evaluator", "candidate_rpc")
        if broker.calls_before_first_reset:
            _authored("candidate_call_before_reset", "candidate", "candidate_rpc")
        if not _draft_validation_reset_sequence_matches(
            broker.reset_seeds, contract.protocol.final_smoke_reset_sequence
        ):
            _authored(
                "candidate_reset_sequence",
                "candidate",
                "candidate_rpc",
                _reset_sequence_diagnostic(
                    broker.reset_seeds,
                    contract.protocol.final_smoke_reset_sequence,
                ),
            )
        if broker.peer_disconnects:
            _authored("candidate_peer_disconnect", "candidate", "candidate_rpc")
        _require_score(results_dir, "final", contract)


def _final_topology_category(*, returncode: int, broker_calls: int) -> str | None:
    if returncode != 0:
        return "final_evaluator_nonzero"
    if broker_calls < 1:
        return "final_candidate_not_called"
    return None


def run_gate(operation: str, settings: GateSettings) -> TaskGateReport:
    if operation == "acceptance-smoke":
        started = time.monotonic()
        acceptance_timeout = min(
            7170,
            2 * settings.timeout_seconds + settings.production_entry_timeout_seconds,
        )
        if settings.acceptance_timeout_seconds is not None:
            acceptance_timeout = min(acceptance_timeout, settings.acceptance_timeout_seconds)
        acceptance_deadline = started + acceptance_timeout
        try:
            with preparation_step("resources"):
                validate_staged_runtime_resources(
                    settings.task_dir,
                    max_runtime_bytes=settings.max_runtime_bytes,
                    deadline=acceptance_deadline,
                )
        except RuntimeResourceValidationDeadlineExceeded:
            return TaskGateReport(
                1,
                "infrastructure_failure",
                "runtime",
                "harness",
                "resource-validation",
                "acceptance_deadline_exceeded",
                _safe_scaffold_fingerprint(settings.task_dir),
                4,
                time.monotonic() - started,
                "Runtime resource validation exhausted the acceptance deadline.",
            )
        except TaskPublicationError as exc:
            diagnostic = sanitize_diagnostic(str(exc))
            category = (
                "runtime_resources_exceed_limit"
                if diagnostic.startswith("runtime_resources_exceed_limit:")
                else "runtime_resources_invalid"
            )
            return TaskGateReport.authored_failure(
                gate="runtime",
                component="runtime",
                stage="resource-validation",
                category=category,
                scaffold_fingerprint=_safe_scaffold_fingerprint(settings.task_dir),
                exit_code=3,
                elapsed_seconds=time.monotonic() - started,
                diagnostic=diagnostic,
            )
        operations = ["contract", "search-smoke", "final-smoke"]
        if settings.containerize:
            operations.append("production-entry")
        for subgate in operations:
            remaining = acceptance_deadline - time.monotonic()
            if remaining <= 0:
                return TaskGateReport(
                    1,
                    "infrastructure_failure",
                    subgate,
                    "harness",
                    "deadline",
                    "acceptance_deadline_exceeded",
                    _safe_scaffold_fingerprint(settings.task_dir),
                    4,
                    time.monotonic() - started,
                    "Acceptance gate exhausted its shared deadline.",
                )
            bounded_seconds = max(1, math.ceil(remaining))
            subgate_settings = replace(
                settings,
                timeout_seconds=min(settings.timeout_seconds, bounded_seconds),
                production_entry_timeout_seconds=min(
                    settings.production_entry_timeout_seconds, bounded_seconds
                ),
            )
            with preparation_step(subgate.replace("-", "_")) as progress:
                report = _run_gate(
                    subgate,
                    subgate_settings,
                    exact_contract=subgate == "contract",
                    deadline=acceptance_deadline,
                )
                if report.status != "pass":
                    progress.fail()
            if report.status != "pass":
                return report
        return TaskGateReport.passed(
            operation,
            task_tree_fingerprint(settings.task_dir, deadline=acceptance_deadline),
            time.monotonic() - started,
        )

    return _run_gate(operation, settings)


def _run_gate(
    operation: str,
    settings: GateSettings,
    *,
    exact_contract: bool = False,
    deadline: float | None = None,
) -> TaskGateReport:
    started = time.monotonic()
    pass_diagnostic = ""
    try:
        canonical_task = settings.source_task_dir / "task.md"
        staged_task = settings.task_dir / "task.md"
        if exact_contract and staged_task.read_bytes() != canonical_task.read_bytes():
            _authored(
                "canonical_task_modified",
                "contract",
                "contract",
                "task.md must match the canonical source task exactly",
            )
        synchronize_declared_digests(
            settings.task_dir / "task_contract.json",
            canonical_task=settings.task_dir / "task.md",
        )
        before = task_tree_fingerprint(settings.task_dir, deadline=deadline)
        contract = load_task_contract(settings.task_dir / "task_contract.json")
        preflight_findings = (
            task_contract_preflight_findings(
                settings.task_dir / "task_contract.json",
                requirements_path=settings.task_dir / "requirements.txt",
                require_explicit_rpc_budgets=True,
            )
            if exact_contract
            else ()
        )
        if preflight_findings:
            _authored(
                "contract_preflight_invalid",
                "contract",
                "contract",
                "\n".join(preflight_findings),
            )
        entrypoint = entrypoint_relpath_from_spec(settings.task_dir / "program_0")
        if entrypoint is None:
            _authored(
                "candidate_entrypoint_invalid",
                "candidate",
                "contract",
                "program_0/program_spec.toml must declare an existing regular entrypoint "
                "inside program_0 (default: program_0/src/program.py)",
            )
        transport_findings = isolated_rpc_transport_findings(settings.task_dir)
        if transport_findings:
            _authored(
                "rpc_transport_invalid",
                "candidate",
                "contract",
                "\n".join(transport_findings),
            )
        capability_findings = unavailable_harness_environment_findings(
            settings.task_dir,
            (
                ("search_evaluator", contract.evaluators.search),
                ("final_evaluator", contract.evaluators.final),
            ),
        )
        if capability_findings:
            finding = capability_findings[0]
            _authored(
                "harness_capability_invalid",
                finding.component,
                "contract",
                "\n".join(item.diagnostic for item in capability_findings),
            )
        if operation != "contract":
            digest = managed_evaluator_tree_digest(settings.task_dir)
            if not settings.containerize:
                if operation == "production-entry":
                    _infrastructure(
                        "production_entry_requires_container",
                        "harness",
                        "production-entry",
                    )
                if operation in {"runtime", "search-smoke"}:
                    _native_smoke(settings, contract.evaluators.search, entrypoint, "search")
                if operation == "final-smoke":
                    _native_smoke(settings, contract.evaluators.final, entrypoint, "final")
            else:
                if operation == "production-entry":
                    if not _prepared_runtime_available(settings, digest):
                        _infrastructure(
                            "production_entry_runtime_missing",
                            "runtime",
                            "production-entry",
                        )
                else:
                    runtime_ready = _prepared_runtime_available(settings, digest)
                    if operation != "final-smoke" or not runtime_ready:
                        _prepare_runtime(settings, digest)
                if operation == "final-smoke":
                    _final_topology_smoke(
                        settings,
                        digest,
                        entrypoint,
                        deadline=deadline,
                    )
                if operation == "production-entry":
                    canary = run_production_entry_canaries(
                        task_dir=settings.task_dir,
                        source_task_dir=settings.source_task_dir,
                        container_image=settings.container_image,
                        caller_deadline=min(
                            deadline or math.inf,
                            time.monotonic() + settings.production_entry_timeout_seconds,
                        ),
                        max_duration_seconds=settings.production_entry_timeout_seconds,
                    )
                    pass_diagnostic = canary.diagnostic
                    if canary.status == "authored_failure":
                        _authored(
                            canary.category,
                            canary.component,
                            canary.stage,
                            canary.diagnostic,
                        )
                    if canary.status == "infrastructure_failure":
                        raise DraftGateFailure(
                            "infrastructure_failure",
                            canary.category,
                            canary.component,
                            canary.stage,
                            canary.diagnostic,
                            exit_code=4,
                        )
        after = task_tree_fingerprint(settings.task_dir, deadline=deadline)
        if before != after:
            _authored("evaluator_modified_scaffold", "harness", operation)
    except DraftGateFailure as exc:
        return TaskGateReport(
            1,
            exc.status,  # type: ignore[arg-type]
            operation,
            exc.component,
            exc.stage,
            exc.category,
            _safe_scaffold_fingerprint(settings.task_dir),
            exc.exit_code,
            time.monotonic() - started,
            sanitize_diagnostic(exc.diagnostic),
        )
    except (ContractSyncError, TaskContractError) as exc:
        return TaskGateReport.authored_failure(
            gate=operation,
            component="contract",
            stage="contract",
            category="contract_invalid",
            scaffold_fingerprint=_safe_scaffold_fingerprint(settings.task_dir),
            exit_code=3,
            elapsed_seconds=time.monotonic() - started,
            diagnostic=sanitize_diagnostic(str(exc)),
        )
    except Exception:
        return TaskGateReport(
            1,
            "infrastructure_failure",
            operation,
            "harness",
            operation,
            "unexpected_harness_failure",
            _safe_scaffold_fingerprint(settings.task_dir),
            4,
            time.monotonic() - started,
            "Harness gate execution failed.",
        )
    return TaskGateReport.passed(
        operation,
        task_tree_fingerprint(settings.task_dir, deadline=deadline),
        time.monotonic() - started,
        diagnostic=pass_diagnostic,
    )


def _settings_from_args(args: argparse.Namespace) -> GateSettings:
    return GateSettings(
        task_dir=args.task_dir.resolve(strict=True),
        source_task_dir=args.source_task_dir.resolve(strict=True),
        containerize=args.containerize,
        container_image=args.container_image,
        container_network=args.container_network,
        timeout_seconds=args.timeout_seconds,
        install_cache_dir=(
            args.install_cache_dir.resolve(strict=True) if args.install_cache_dir else None
        ),
        production_entry_timeout_seconds=args.production_entry_timeout_seconds,
        acceptance_timeout_seconds=args.acceptance_timeout_seconds,
        max_runtime_bytes=args.max_runtime_bytes,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run a task drafting validation gate")
    parser.add_argument(
        "operation",
        choices=(
            "contract",
            "runtime",
            "search-smoke",
            "final-smoke",
            "production-entry",
            "acceptance-smoke",
        ),
    )
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--source-task-dir", type=Path, required=True)
    parser.add_argument("--containerize", action="store_true")
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--container-network", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--production-entry-timeout-seconds", type=int, default=300)
    parser.add_argument("--acceptance-timeout-seconds", type=int)
    parser.add_argument("--max-runtime-bytes", type=int, required=True)
    parser.add_argument("--install-cache-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 86400:
        parser.error("--timeout-seconds must be from 1 to 86400")
    if not 30 <= args.production_entry_timeout_seconds <= 1800:
        parser.error("--production-entry-timeout-seconds must be from 30 to 1800")
    if args.acceptance_timeout_seconds is not None and not (
        1 <= args.acceptance_timeout_seconds <= 604800
    ):
        parser.error("--acceptance-timeout-seconds must be from 1 to 604800")
    if not 1 <= args.max_runtime_bytes <= 64 * 1024 * 1024 * 1024:
        parser.error("--max-runtime-bytes must be from 1 byte to 64 GiB")
    report = run_gate(args.operation, _settings_from_args(args))
    print(json.dumps(report.to_payload(), ensure_ascii=False, sort_keys=True))
    if report.status == "pass":
        return 0
    return 3 if report.status == "authored_failure" else 4


if __name__ == "__main__":
    raise SystemExit(_main())

"""Evaluate a variant candidate natively, for runs with ``container.enabled=false``.

The Docker sandbox in ``host_eval`` cannot run tasks that need a compiler, JIT kernels, or data
outside the task (the same tasks that force a native campaign). This path runs the sealed
evaluator on the host in the environment native investigation gives it (the parent environment
without credential-like variables or Python path overrides, the real HOME), with a private
TMPDIR, and keeps the sandbox's output contract: the evaluator writes into a private scratch
directory, the host canonicalizes the results, copies only the promotion allowlist into
``.harness_results``, validates it, and promotes it into the candidate directory.

Containment is shared with native investigation (``run_contained``): the harness becomes a child
subreaper, so descendants that start their own sessions (torchrun workers do) are stopped on
exit, timeout, and SIGINT/SIGTERM. Agents and evaluators share the user's permissions in native
mode; the sealed evaluator digest, checked by the caller before and after every evaluation, is
what detects tampering. Contract tasks never run here.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.investigation.evaluate_program import credential_free_environment, run_contained
from src.investigation.task_contract import TASK_CONTRACT_FILENAME

from .host_eval import (
    HostEvaluationResult,
    SandboxEvaluationError,
    SandboxEvaluationTimeout,
    _load_mapping,
    _promote_results,
    _utc_now,
    _validate_candidate,
    _validate_staged_results,
    _write_json,
    discard_evaluation_artifacts,
    resolve_host_python,
)
from .sandbox_outputs import canonicalize_results, copy_promotable_results

logger = logging.getLogger(__name__)

DIAGNOSTIC_TAIL_BYTES = 64 * 1024
LOG_TAIL_BYTES = 1024 * 1024
_STDOUT_LOG = "harness_evaluate_stdout.log"


def native_evaluation_python(*, task_source_dir: Path | None, venv_dir: Path | None) -> Path:
    """The task's own venv when it has one, else this interpreter (the native harness default)."""
    try:
        return resolve_host_python(task_source_dir=task_source_dir, venv_dir=venv_dir)
    except ValueError:
        fallback = Path(sys.executable).absolute()
        logger.warning("No task venv found; native variant evaluation uses %s", fallback)
        return fallback


def is_contract_task(task_source_dir: Path) -> bool:
    return (task_source_dir / TASK_CONTRACT_FILENAME).exists() or (
        task_source_dir / TASK_CONTRACT_FILENAME
    ).is_symlink()


def evaluate_variant_program_natively(
    *,
    task_bundle_dir: Path,
    evaluator_source_dir: Path,
    task_source_dir: Path,
    program_index: int,
    python_bin: Path,
    timeout_seconds: int,
    environment: Mapping[str, str] | None = None,
) -> HostEvaluationResult:
    """Run the sealed evaluator on one candidate on the host and promote its results.

    Raises ``SandboxEvaluationError`` (non-zero exit, with the output tail as diagnostic),
    ``SandboxEvaluationTimeout``, or ``SystemExit(128 + signum)`` when the harness is stopped;
    every failure discards the candidate's evaluation artifacts.
    """
    task_bundle_dir = task_bundle_dir.resolve()
    evaluate_py = evaluator_source_dir.resolve() / "evaluate.py"
    task_source_dir = task_source_dir.resolve()
    if not evaluate_py.is_file():
        raise ValueError(f"missing sealed evaluator: {evaluate_py}")
    if not python_bin.is_file():
        raise ValueError(f"missing evaluator Python: {python_bin}")
    if is_contract_task(task_source_dir):
        raise ValueError("contract tasks evaluate only in the sealed Docker sandbox")
    program_dir, entrypoint_rel = _validate_candidate(task_bundle_dir, program_index)
    environment = dict(environment or {})
    scratch = Path(tempfile.mkdtemp(prefix="variant-native-eval-"))
    try:
        return _evaluate_in(
            scratch,
            program_dir=program_dir,
            command=_command(python_bin, evaluate_py, program_dir / entrypoint_rel, scratch, environment),
            environment=_evaluator_environment(scratch, environment, task_source_dir, python_bin),
            timeout_seconds=timeout_seconds,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)  # never masks the result or a stop signal


def _command(
    python_bin: Path, evaluate_py: Path, entrypoint: Path, scratch: Path, environment: Mapping[str, str]
) -> list[str]:
    command = [
        str(python_bin),
        str(evaluate_py),
        "--program_path",
        str(entrypoint),
        "--results_dir",
        str(scratch / "results"),
    ]
    replications = environment.get("NUM_REPLICATIONS", "1")
    if replications.isdigit() and int(replications) > 1:
        command += ["--num_replications", replications]
    return command


def _evaluate_in(
    scratch: Path,
    *,
    program_dir: Path,
    command: list[str],
    environment: dict[str, str],
    timeout_seconds: int,
) -> HostEvaluationResult:
    results, log_path = scratch / "results", scratch / "evaluator.log"
    results.mkdir(mode=0o700)
    manifest_path = program_dir / "evaluation_manifest.json"
    manifest: dict[str, Any] = {
        "producer": "native_host_eval",
        "command": command,
        "execution_boundary": "native",
        "program_id": program_dir.name,
        "started_at": _utc_now(),
    }
    _write_json(manifest_path, manifest)
    try:
        with open(log_path, "wb") as log:
            run = run_contained(
                command, cwd=scratch, environment=environment, timeout_seconds=timeout_seconds, stdout=log
            )
        diagnostic = _tail(log_path, DIAGNOSTIC_TAIL_BYTES)
        if run.signum is not None:
            raise SystemExit(128 + run.signum)
        if run.timed_out:
            raise SandboxEvaluationTimeout(timeout_seconds, command)
        assert run.returncode is not None
        if run.returncode != 0:
            raise SandboxEvaluationError(run.returncode, command, diagnostic)
        _promote(results, program_dir, environment, log_path)
        metrics = _load_mapping(program_dir / "metrics.json")
        correct_payload = _load_mapping(program_dir / "correct.json")
        if not metrics or not correct_payload:
            raise RuntimeError(f"native evaluation of {program_dir.name} produced invalid artifacts")
        manifest.update(finished_at=_utc_now(), returncode=run.returncode)
        _write_json(manifest_path, manifest)
    except BaseException:
        discard_evaluation_artifacts(program_dir)
        raise
    return HostEvaluationResult(
        returncode=run.returncode,
        correct=correct_payload.get("correct") is True,
        metrics=metrics,
        diagnostic=diagnostic,
        command=tuple(command),
    )


def _evaluator_environment(
    scratch: Path,
    environment: Mapping[str, str],
    task_source_dir: Path,
    python_bin: Path,
) -> dict[str, str]:
    temporary = scratch / "tmp"
    temporary.mkdir(mode=0o700)
    env = credential_free_environment()
    env.update(environment)
    env.update(
        TMPDIR=str(temporary),
        PYTHONNOUSERSITE="1",
        TASK_SOURCE_DIR=str(task_source_dir),
        VENV_DIR=str(python_bin.parent.parent),
    )
    return env


def _promote(results: Path, program_dir: Path, environment: Mapping[str, str], log_path: Path) -> None:
    """The sandbox's promotion contract: canonicalize, copy the allowlist, validate, promote.

    The harness's own log tail is written only into the fresh, harness-owned staging directory,
    never through a path the evaluator could have planted.
    """
    canonicalize_results(results, program_dir.name, environment)
    staged = program_dir / ".harness_results"
    staged.mkdir(mode=0o700)
    copy_promotable_results(results, staged)
    staged_log = staged / _STDOUT_LOG
    staged_log.unlink(missing_ok=True)
    descriptor = os.open(staged_log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(_tail(log_path, LOG_TAIL_BYTES).encode("utf-8"))
    _validate_staged_results(staged, program_dir, environment)
    _promote_results(staged, program_dir)


def _tail(path: Path, limit: int) -> str:
    try:
        with open(path, "rb") as source:
            source.seek(max(0, path.stat().st_size - limit))
            return source.read().decode("utf-8", errors="replace")
    except OSError:
        return ""

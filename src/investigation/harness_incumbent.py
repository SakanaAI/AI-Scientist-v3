"""Trusted incumbent selection for harness-evaluated campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .files import read_json, resolve_path
from .program_clone import _NONOVERRIDABLE_EXCLUDES
from .program_spec import entrypoint_relpath_from_spec

_MAX_ARTIFACT_BYTES = 1024 * 1024


def _finite_score(raw_score: object) -> float | None:
    if (
        isinstance(raw_score, bool)
        or not isinstance(raw_score, (int, float))
        or not math.isfinite(float(raw_score))
    ):
        return None
    return float(raw_score)


_SOURCE_OMITTED_DIRS = frozenset({"__pycache__"})


def _program_source_records(program_dir: Path) -> dict[str, str] | None:
    """Digest every source file under *program_dir*; ``None`` if the tree has symlinks."""
    records: dict[str, str] = {}
    for path in sorted(program_dir.rglob("*")):
        relative = path.relative_to(program_dir)
        if relative.parts[0] in _NONOVERRIDABLE_EXCLUDES or any(
            part in _SOURCE_OMITTED_DIRS for part in relative.parts
        ):
            continue
        if path.is_symlink():
            return None
        if path.is_file():
            records[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records


def seed_matches_validated_baseline(seed_program_dir: Path, baseline_program_dir: Path) -> bool:
    """Whether a run's seed is the task's validated baseline, by path or by identical sources.

    Parallel coordinators hand workers a frozen copy of ``program_0``; the copy earned the
    manifest's baseline score just like the original, so identical sources are accepted.
    """
    seed = resolve_path(seed_program_dir)
    baseline = resolve_path(baseline_program_dir)
    if seed == baseline:
        return True
    if not seed.is_dir() or not baseline.is_dir():
        return False
    try:
        seed_records = _program_source_records(seed)
        baseline_records = _program_source_records(baseline)
    except OSError:
        return False
    return bool(seed_records) and seed_records == baseline_records


def runtime_baseline_score(task_dir: Path) -> float | None:
    manifest = read_json(task_dir / ".agentic_runtime.json")
    return _finite_score(manifest.get("baseline_combined_score") if manifest else None)


def runtime_baseline_correct(task_dir: Path) -> bool | None:
    """Return retained baseline correctness, treating legacy strict manifests as true."""
    manifest = read_json(task_dir / ".agentic_runtime.json")
    if not manifest or runtime_baseline_score(task_dir) is None:
        return None
    value = manifest.get("baseline_correct", True)
    return value if isinstance(value, bool) else None


@dataclass(frozen=True, slots=True)
class _TrustedProgramEvaluation:
    score: float
    correct: bool


def _read_harness_json(path: Path) -> dict[str, Any] | None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_ARTIFACT_BYTES:
            return None
        chunks: list[bytes] = []
        remaining = _MAX_ARTIFACT_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > _MAX_ARTIFACT_BYTES:
            return None
        parsed = json.loads(payload.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    finally:
        os.close(descriptor)


def _sealed_program_evaluation(
    program_dir: Path,
    *,
    baseline_program_dir: Path,
    baseline_score: float | None,
    baseline_correct: bool | None,
    trust_native: bool = False,
) -> _TrustedProgramEvaluation | None:
    program_dir = resolve_path(program_dir)
    if program_dir == resolve_path(baseline_program_dir):
        validated_baseline = _finite_score(baseline_score)
        if validated_baseline is not None and isinstance(baseline_correct, bool):
            return _TrustedProgramEvaluation(validated_baseline, baseline_correct)
    manifest = _read_harness_json(program_dir / "evaluation_manifest.json")
    if not manifest or not (
        (_is_sandbox_manifest(manifest) or (trust_native and _is_native_manifest(manifest)))
        and manifest.get("program_id") == program_dir.name
        and manifest.get("exit_code") == 0
        and isinstance(manifest.get("command"), list)
        and bool(manifest["command"])
        and isinstance(manifest.get("started_at"), str)
        and isinstance(manifest.get("finished_at"), str)
    ):
        return None
    metrics = _read_harness_json(program_dir / "metrics.json")
    correct = _read_harness_json(program_dir / "correct.json")
    if not metrics or not correct or correct.get("correct") is not True:
        return None
    score = _finite_score(metrics.get("combined_score"))
    return _TrustedProgramEvaluation(score, True) if score is not None else None


def _is_sandbox_manifest(manifest: dict[str, Any]) -> bool:
    return (
        manifest.get("producer") == "sandbox_host_eval"
        and manifest.get("execution_boundary") == "sandbox"
        and manifest.get("returncode") == 0
    )


def _is_native_manifest(manifest: dict[str, Any]) -> bool:
    """A native run's own harness evaluation (``harness_eval`` direct, no sandbox)."""
    return manifest.get("producer") == "harness_eval"


def _legacy_program_dir(best_manifest: dict[str, Any]) -> Path | None:
    program_dir_raw = best_manifest.get("program_dir")
    if isinstance(program_dir_raw, str) and program_dir_raw.strip():
        return resolve_path(program_dir_raw)
    program_path_raw = best_manifest.get("program_path")
    if not isinstance(program_path_raw, str) or not program_path_raw.strip():
        return None
    program_path = resolve_path(program_path_raw)
    return program_path.parent.parent if program_path.parent.name == "src" else program_path.parent


def reconcile_harness_incumbent(
    *,
    task_dir: Path,
    results_dir: Path,
    seed_program_dir: Path,
    baseline_score: float | None,
    baseline_correct: bool | None,
    best_manifest: dict[str, Any],
    outer_state: dict[str, Any],
    validated_baseline_program_dir: Path | None = None,
    trust_native: bool = False,
) -> bool:
    """Pair the incumbent score with the program that earned it.

    ``trust_native`` (native runs only, ``container.enabled=false``) also accepts evaluations the
    harness ran directly on the host, since such runs have no runtime or sandbox to vouch for one.
    """
    baseline_program_dir = resolve_path(
        validated_baseline_program_dir or task_dir / "program_0"
    )
    seed_program_dir = resolve_path(seed_program_dir)
    results_dir = resolve_path(results_dir)
    seed_evaluation = _sealed_program_evaluation(
        seed_program_dir,
        baseline_program_dir=baseline_program_dir,
        baseline_score=baseline_score,
        baseline_correct=baseline_correct,
        trust_native=trust_native,
    )
    persisted_program = _legacy_program_dir(best_manifest)
    persisted_program_is_managed = persisted_program is not None and (
        persisted_program == seed_program_dir or persisted_program.is_relative_to(results_dir)
    )
    persisted_evaluation = (
        _sealed_program_evaluation(
            persisted_program,
            baseline_program_dir=baseline_program_dir,
            baseline_score=baseline_score,
            baseline_correct=baseline_correct,
            trust_native=trust_native,
        )
        if persisted_program_is_managed and persisted_program is not None
        else None
    )

    if persisted_evaluation is None and seed_evaluation is None:
        return False
    if persisted_evaluation is not None and (
        seed_evaluation is None
        or persisted_evaluation.correct > seed_evaluation.correct
        or (
            persisted_evaluation.correct == seed_evaluation.correct
            and persisted_evaluation.score >= seed_evaluation.score
        )
    ):
        assert persisted_program is not None
        incumbent_program = persisted_program
        incumbent_evaluation = persisted_evaluation
        source_attempt_id = best_manifest.get("source_attempt_id")
    else:
        assert seed_evaluation is not None
        incumbent_program = seed_program_dir
        incumbent_evaluation = seed_evaluation
        source_attempt_id = "bootstrap"

    best_manifest.update(
        program_dir=str(incumbent_program),
        entrypoint_path=(
            entrypoint_relpath_from_spec(incumbent_program, required=False) or "src/program.py"
        ),
        combined_score=incumbent_evaluation.score,
        correct=incumbent_evaluation.correct,
        source_attempt_id=source_attempt_id,
    )
    outer_state.update(
        best_program_dir=str(incumbent_program),
        best_combined_score=incumbent_evaluation.score,
        best_correct=incumbent_evaluation.correct,
    )
    return True

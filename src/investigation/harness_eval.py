"""Harness-owned single evaluation for agent-created candidate programs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluate_program import credential_free_environment, run_contained
from .harness_feedback import evaluation_feedback, search_metric_feedback
from .harness_sources import read_candidate_sources as _read_candidate_sources
from .evaluator_outputs import EvaluatorOutputContractError
from .task_failure import TASK_CONTRACT_EXIT_CODE, TASK_CONTRACT_FAILURE, write_task_failure
from .runtime_output import failed_evaluation_evidence
from .program_spec import entrypoint_relpath_from_spec
from .rubric_checks import (
    COMPARATORS,
    MAX_HARNESS_RUBRICS,
    candidate_source_tokens,
    missing_source_patterns,
    rubric_metric_check,
)

EVALUATOR_ARTIFACTS = (
    "metrics.json",
    "correct.json",
    "rubric_status.json",
    "stdout.log",
    "stderr.log",
    "evaluation_manifest.json",
    "harness_evaluate_stdout.log",
    "harness_evaluate_stderr.log",
)

@dataclass(frozen=True)
class SealedHarnessConfig:
    """Immutable container boundary for a managed task evaluation."""

    task_source_dir: Path
    sandbox_image: str
    sandbox_gpus: str | None
    environment: Mapping[str, str]
    gpu_evaluation: bool = False


def _evaluation_environment(
    environment: Mapping[str, str], *, gpu_evaluation: bool
) -> dict[str, str]:
    child_environment = dict(environment)
    if not gpu_evaluation:
        child_environment["CUDA_VISIBLE_DEVICES"] = ""
    return child_environment


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _target_rubrics(path: Path) -> list[dict[str, Any]]:
    raw = _load_json(path, [])
    if isinstance(raw, dict):
        raw = raw.get("target_rubrics", [])
    if not isinstance(raw, list):
        return []
    if len(raw) > MAX_HARNESS_RUBRICS:
        raise SystemExit(f"Harness supports at most {MAX_HARNESS_RUBRICS} target rubrics")
    return [item for item in raw if isinstance(item, dict)]


def _correct_value(path: Path) -> bool:
    raw = _load_json(path, {})
    if isinstance(raw, dict):
        return bool(raw.get("correct"))
    return bool(raw)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric_key_for_check(metric: str, metrics: dict[str, Any]) -> str:
    if metric in metrics:
        return metric
    if metric == "score" and "combined_score" in metrics:
        return "combined_score"
    return metric


def _source_contains_patterns(rubric: dict[str, Any]) -> list[str] | None:
    verification = rubric.get("verification")
    if not isinstance(verification, dict) or verification.get("type") != "source_contains":
        return None
    patterns = verification.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        return None
    if not all(isinstance(pattern, str) and pattern for pattern in patterns):
        return None
    return patterns


def _rubric_status(
    *,
    rubric: dict[str, Any],
    metrics: dict[str, Any],
    correct: bool,
    exit_code: int,
    missing_candidate_patterns: set[str],
) -> tuple[bool, str]:
    base_evidence = (
        f"harness_evaluated_once exit={exit_code}; "
        f"correct={correct}; combined_score={metrics.get('combined_score')}"
    )
    if exit_code != 0:
        return False, f"{base_evidence}; criterion not assessed because evaluation failed"
    if not correct:
        return False, f"{base_evidence}; criterion not assessed because scientific acceptance failed"

    rubric_type = str(rubric.get("type", "")).strip().lower()
    source_patterns = _source_contains_patterns(rubric)
    if source_patterns is not None:
        missing_patterns = [
            pattern for pattern in source_patterns if pattern in missing_candidate_patterns
        ]
        if missing_patterns:
            return False, f"{base_evidence}; missing source pattern {missing_patterns[0]!r}"
        return True, f"{base_evidence}; source_contains={len(source_patterns)} patterns"
    metric_check = rubric_metric_check(rubric)
    if metric_check is None:
        reason = "no metric threshold found"
        if rubric_type and rubric_type != "metric":
            reason = f"harness cannot verify {rubric_type} rubric"
        return False, f"{base_evidence}; {reason}"

    metric, comparator, threshold = metric_check
    metric_key = _metric_key_for_check(metric, metrics)
    raw_actual = metrics.get(metric_key)
    actual = (
        float(raw_actual)
        if isinstance(raw_actual, (int, float)) and not isinstance(raw_actual, bool)
        else None
    )
    if actual is None:
        return False, f"{base_evidence}; missing numeric metric {metric_key}"

    passed = COMPARATORS[comparator](actual, threshold)
    evidence = f"{base_evidence}; {metric_key}={actual:g} {comparator} {threshold:g}"
    return passed, evidence


def _append_memory(
    *,
    md_path: Path,
    jsonl_path: Path,
    attempt_id: str,
    idea_id: str,
    metrics: dict[str, Any],
    correct: bool,
    feedback: str,
) -> None:
    timestamp = _utc_now()
    score = metrics.get("combined_score")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"### {timestamp} | step=evaluate | scope=attempt\n"
            "- summary: harness evaluated program_1 exactly once\n"
            f"- evidence: combined_score={score}, correct={correct}; {feedback}\n"
            "- next_action: generate next idea\n"
            f"- idea_id: {idea_id}\n"
            f"- attempt_id: {attempt_id}\n"
            "- program_id: program_1\n\n"
        )

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": timestamp,
        "step": "evaluate",
        "scope": "attempt",
        "summary": "harness evaluated program_1 exactly once",
        "evidence": f"combined_score={score}, correct={correct}; {feedback}",
        "next_action": "generate next idea",
        "idea_id": idea_id,
        "attempt_id": attempt_id,
        "program_id": "program_1",
        "metrics": metrics,
    }
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _ensure_agent_only_created_program_1(experiment_dir: Path) -> Path:
    program_1 = experiment_dir / "program_1"
    if program_1.is_symlink() or not program_1.is_dir():
        raise SystemExit("Harness-evaluated attempt requires agent-created program_1")

    for child in experiment_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("program_"):
            continue
        if child.name not in {"program_0", "program_1"}:
            raise SystemExit(
                "Harness-evaluated attempt forbids extra candidate directories: "
                f"{child.name}"
            )

    for artifact in EVALUATOR_ARTIFACTS:
        path = program_1 / artifact
        if path.exists() or path.is_symlink():
            raise SystemExit(f"agent-created evaluator artifact is forbidden: {path}")

    entrypoint_rel = entrypoint_relpath_from_spec(program_1)
    if entrypoint_rel is None:
        raise SystemExit("program_1 must contain a valid program_spec.toml entrypoint")
    entrypoint = program_1 / entrypoint_rel
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise SystemExit(f"program_1 entrypoint does not exist: {entrypoint}")
    return entrypoint


def _run_sealed_evaluation(
    *,
    experiment_dir: Path,
    attempt_id: str,
    idea_id: str,
    python_bin: str,
    timeout_seconds: int,
    config: SealedHarnessConfig,
    task_failure_fd: int | None = None,
) -> int:
    from src.variants_pipeline.host_eval import (
        SandboxEvaluationError,
        SandboxEvaluationTimeout,
        evaluate_variant_program,
    )

    if config.gpu_evaluation and not config.sandbox_gpus:
        raise SystemExit(
            "GPU harness evaluation requires an explicit Docker GPU selector"
        )
    environment = _evaluation_environment(
        config.environment,
        gpu_evaluation=config.gpu_evaluation,
    )
    sandbox_gpus = config.sandbox_gpus if config.gpu_evaluation else None
    try:
        result = evaluate_variant_program(
            task_bundle_dir=experiment_dir,
            evaluator_source_dir=config.task_source_dir,
            task_source_dir=config.task_source_dir,
            program_index=1,
            python_bin=Path(python_bin),
            sandbox_image=config.sandbox_image,
            timeout_seconds=timeout_seconds,
            environment=environment,
            sandbox_gpus=sandbox_gpus,
        )
    except EvaluatorOutputContractError as exc:
        # A receipt is authoritative; the public manifest and process exit are not.
        if task_failure_fd is not None:
            write_task_failure(task_failure_fd, attempt_id, idea_id)
        _write_sealed_evaluation_artifacts(
            program_dir=experiment_dir / "program_1", attempt_id=attempt_id,
            idea_id=idea_id, command=[], returncode=TASK_CONTRACT_EXIT_CODE,
            diagnostic=failed_evaluation_evidence(
                str(exc), stage="sealed_evaluation", environment=config.environment,
            ).stderr,
            failure_category=TASK_CONTRACT_FAILURE,
        )
        return TASK_CONTRACT_EXIT_CODE
    except SandboxEvaluationError as exc:
        _write_sealed_evaluation_artifacts(
            program_dir=experiment_dir / "program_1",
            attempt_id=attempt_id,
            idea_id=idea_id,
            command=list(exc.command),
            returncode=exc.returncode,
            diagnostic=exc.diagnostic,
        )
        return exc.returncode
    except SandboxEvaluationTimeout as exc:
        _write_sealed_evaluation_artifacts(
            program_dir=experiment_dir / "program_1",
            attempt_id=attempt_id,
            idea_id=idea_id,
            command=list(exc.command),
            returncode=124,
            diagnostic=str(exc),
            timeout_seconds=exc.timeout_seconds,
        )
        return 124
    _write_sealed_evaluation_artifacts(
        program_dir=experiment_dir / "program_1",
        attempt_id=attempt_id,
        idea_id=idea_id,
        command=list(result.command),
        returncode=result.returncode,
        diagnostic=result.diagnostic,
    )
    return result.returncode


def _write_sealed_evaluation_artifacts(
    *,
    program_dir: Path,
    attempt_id: str,
    idea_id: str,
    command: list[str],
    returncode: int,
    diagnostic: str,
    timeout_seconds: int | None = None,
    failure_category: str | None = None,
) -> None:
    manifest_path = program_dir / "evaluation_manifest.json"
    existing = _load_json(manifest_path, {})
    manifest = existing if isinstance(existing, dict) else {}
    manifest.pop("failure_category", None)
    manifest.pop("failure_scope", None)
    manifest.update(
        producer="sandbox_host_eval",
        execution_boundary="sandbox",
        attempt_id=attempt_id,
        idea_id=idea_id,
        program_id="program_1",
        command=command,
        started_at=manifest.get("started_at") or _utc_now(),
        finished_at=_utc_now(),
        exit_code=returncode,
        returncode=returncode,
    )
    if timeout_seconds is not None:
        manifest["timeout_seconds"] = timeout_seconds
    if failure_category is not None:
        manifest.update(failure_category=failure_category, failure_scope="task")
    _write_json(manifest_path, manifest)
    (program_dir / "harness_evaluate_stdout.log").write_text("", encoding="utf-8")
    (program_dir / "harness_evaluate_stderr.log").write_text(
        diagnostic, encoding="utf-8"
    )


def run_harness_evaluation(
    *,
    experiment_dir: Path,
    attempt_id: str,
    idea_id: str,
    rubrics_path: Path,
    python_bin: str,
    global_memory_md: Path,
    global_memory_jsonl: Path,
    timeout_seconds: int,
    sealed_config: SealedHarnessConfig | None = None,
    environment: Mapping[str, str] = os.environ,
    gpu_evaluation: bool = False,
    task_failure_fd: int | None = None,
) -> int:
    experiment_dir = experiment_dir.resolve()
    evaluate_py = experiment_dir / "evaluate.py"
    if not evaluate_py.is_file():
        raise SystemExit(f"Missing evaluator: {evaluate_py}")

    entrypoint = _ensure_agent_only_created_program_1(experiment_dir)
    program_dir = experiment_dir / "program_1"
    target_rubrics = _target_rubrics(rubrics_path)
    candidate_tokens = candidate_source_tokens(_read_candidate_sources(program_dir))
    source_patterns = [
        pattern
        for rubric in target_rubrics
        for pattern in (_source_contains_patterns(rubric) or [])
    ]
    missing_candidate_patterns = set(missing_source_patterns(candidate_tokens, source_patterns))
    if sealed_config is not None:
        returncode = _run_sealed_evaluation(
            experiment_dir=experiment_dir,
            attempt_id=attempt_id,
            idea_id=idea_id,
            python_bin=python_bin,
            timeout_seconds=timeout_seconds,
            config=sealed_config,
            task_failure_fd=task_failure_fd,
        )
    else:
        returncode = _run_direct_evaluation(
            experiment_dir=experiment_dir,
            evaluate_py=evaluate_py,
            entrypoint=entrypoint,
            program_dir=program_dir,
            attempt_id=attempt_id,
            idea_id=idea_id,
            python_bin=python_bin,
            timeout_seconds=timeout_seconds,
            environment=_native_evaluation_environment(
                environment,
                gpu_evaluation=gpu_evaluation,
            ),
        )

    missing_artifacts = [
        name for name in ("metrics.json", "correct.json") if not (program_dir / name).is_file()
    ]
    if returncode == 0 and missing_artifacts:
        raise SystemExit(
            "harness evaluation completed without required artifacts: "
            + ", ".join(missing_artifacts)
        )

    metrics = _load_json(program_dir / "metrics.json", {})
    if not isinstance(metrics, dict):
        metrics = {}
    correct = _correct_value(program_dir / "correct.json")
    rubrics = []
    for item in target_rubrics:
        rubric_id = item.get("id")
        if not isinstance(rubric_id, str) or not rubric_id.strip():
            continue
        passed, evidence = _rubric_status(
            rubric=item,
            metrics=metrics,
            correct=correct,
            exit_code=returncode,
            missing_candidate_patterns=missing_candidate_patterns,
        )
        assessment = (
            "not_assessed" if returncode != 0 or not correct else "passed" if passed else "failed"
        )
        rubrics.append({
            "id": rubric_id, "pass": passed, "assessment": assessment, "evidence": evidence,
        })
    memory_metrics: dict[str, Any] = {
        **search_metric_feedback(
            sealed_config.task_source_dir if sealed_config is not None else experiment_dir,
            metrics,
        ),
        "combined_score": metrics.get("combined_score"),
        "correct": correct,
    }
    status = {
        "attempt_id": attempt_id,
        "idea_id": idea_id,
        "overall_pass": bool(rubrics) and all(item["pass"] for item in rubrics),
        "rubrics": rubrics,
        "metrics": memory_metrics,
        "next_recommendation": "generate a new idea",
    }
    _write_json(program_dir / "rubric_status.json", status)
    _write_json(experiment_dir / "rubric_status.json", status)
    _append_memory(
        md_path=global_memory_md,
        jsonl_path=global_memory_jsonl,
        attempt_id=attempt_id,
        idea_id=idea_id,
        metrics=memory_metrics,
        correct=correct,
        feedback=evaluation_feedback(memory_metrics, returncode),
    )
    return returncode


def _native_evaluation_environment(
    environment: Mapping[str, str], *, gpu_evaluation: bool
) -> dict[str, str]:
    """Native evaluations of attempts and of the seed share one environment: the caller's,
    minus Python path overrides and credential-like variables (as for native variants)."""
    return _evaluation_environment(
        credential_free_environment(environment), gpu_evaluation=gpu_evaluation
    )


def _open_new_log(path: Path) -> int:
    """A fresh harness log in an agent-writable directory, never written through a planted link."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)


def _run_direct_evaluation(
    *,
    experiment_dir: Path,
    evaluate_py: Path,
    entrypoint: Path,
    program_dir: Path,
    attempt_id: str,
    idea_id: str,
    python_bin: str,
    timeout_seconds: int,
    environment: Mapping[str, str],
) -> int:
    """Run the evaluator on the host in its own session and leave no descendant behind.

    Output (stdout and stderr) streams to ``harness_evaluate_stdout.log`` instead of memory, so a
    timed-out multi-hour evaluation still leaves its log. A timeout or a stop signal is recorded
    in the manifest and raises ``SystemExit``, as before.
    """
    manifest_path = program_dir / "evaluation_manifest.json"
    command = [
        python_bin,
        str(evaluate_py),
        "--program_path",
        str(entrypoint),
        "--results_dir",
        str(program_dir),
    ]
    manifest: dict[str, Any] = {
        "producer": "harness_eval",
        "attempt_id": attempt_id,
        "idea_id": idea_id,
        "program_id": program_dir.name,
        "command": command,
        "started_at": _utc_now(),
    }
    _write_json(manifest_path, manifest)
    stderr_note = _open_new_log(program_dir / "harness_evaluate_stderr.log")
    with os.fdopen(stderr_note, "w", encoding="utf-8") as note:
        note.write("stderr is merged into harness_evaluate_stdout.log\n")
    with os.fdopen(_open_new_log(program_dir / "harness_evaluate_stdout.log"), "wb") as log:
        run = run_contained(
            command,
            cwd=experiment_dir,
            environment=dict(environment),
            timeout_seconds=timeout_seconds,
            stdout=log,
        )
    if run.timed_out:
        manifest.update(finished_at=_utc_now(), exit_code=124, timeout_seconds=timeout_seconds)
        _write_json(manifest_path, manifest)
        raise SystemExit(f"harness evaluation timed out after {timeout_seconds}s")
    if run.signum is not None:
        manifest.update(finished_at=_utc_now(), exit_code=128 + run.signum)
        _write_json(manifest_path, manifest)
        raise SystemExit(128 + run.signum)
    assert run.returncode is not None
    manifest.update(finished_at=_utc_now(), exit_code=run.returncode)
    _write_json(manifest_path, manifest)
    return run.returncode


def bootstrap_native_incumbent(
    *,
    program_dir: Path,
    task_dir: Path,
    python_bin: str,
    timeout_seconds: int,
    environment: Mapping[str, str],
    gpu_evaluation: bool,
) -> int:
    """Evaluate a native run's frozen seed exactly as native attempts are evaluated.

    Native runs have no managed runtime or sandbox to vouch for the baseline, so the harness
    earns the incumbent score itself before the first cycle. Like an attempt (``run.sh``), the
    evaluator is a copy of the task's ``evaluate.py`` next to the program, runs from that
    directory with ``TASK_SOURCE_DIR`` set to the task, and is contained and logged the same way.
    Returns the evaluator's exit code; a timeout or stop signal raises ``SystemExit``.
    """
    program_dir = program_dir.resolve()
    task_dir = task_dir.resolve()
    entrypoint_rel = entrypoint_relpath_from_spec(program_dir)
    if entrypoint_rel is None:
        raise SystemExit(f"Seed program has no valid program_spec.toml: {program_dir}")
    experiment_dir = program_dir.parent
    evaluate_py = experiment_dir / "evaluate.py"
    shutil.copyfile(task_dir / "evaluate.py", evaluate_py)
    return _run_direct_evaluation(
        experiment_dir=experiment_dir,
        evaluate_py=evaluate_py,
        entrypoint=program_dir / entrypoint_rel,
        program_dir=program_dir,
        attempt_id="bootstrap",
        idea_id="bootstrap",
        python_bin=python_bin,
        timeout_seconds=timeout_seconds,
        environment=_native_evaluation_environment(
            {**environment, "TASK_SOURCE_DIR": str(task_dir)}, gpu_evaluation=gpu_evaluation
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--idea-id", required=True)
    parser.add_argument("--rubrics-file", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--global-memory-md", required=True)
    parser.add_argument("--global-memory-jsonl", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--task-source-dir")
    parser.add_argument("--sandbox-image")
    parser.add_argument("--sandbox-gpus")
    parser.add_argument("--gpu-evaluation", action="store_true")
    parser.add_argument("--task-failure-fd", type=int)
    args = parser.parse_args(argv)

    if args.task_failure_fd is not None and (
        args.task_failure_fd < 3 or not args.task_source_dir
    ):
        parser.error("--task-failure-fd requires a private descriptor and sealed evaluation")

    if bool(args.task_source_dir) != bool(args.sandbox_image):
        parser.error("--task-source-dir and --sandbox-image must be provided together")
    if args.sandbox_gpus and not args.sandbox_image:
        parser.error("--sandbox-gpus requires --sandbox-image")
    if args.sandbox_gpus and not args.gpu_evaluation:
        parser.error("--sandbox-gpus requires --gpu-evaluation")
    if args.gpu_evaluation and args.sandbox_image and not args.sandbox_gpus:
        parser.error("containerized --gpu-evaluation requires --sandbox-gpus")
    sealed_config = None
    if args.task_source_dir:
        sealed_config = SealedHarnessConfig(
            task_source_dir=Path(args.task_source_dir),
            sandbox_image=args.sandbox_image,
            sandbox_gpus=args.sandbox_gpus,
            environment=dict(os.environ),
            gpu_evaluation=args.gpu_evaluation,
        )

    return run_harness_evaluation(
        experiment_dir=Path(args.experiment_dir),
        attempt_id=args.attempt_id,
        idea_id=args.idea_id,
        rubrics_path=Path(args.rubrics_file),
        python_bin=args.python_bin,
        global_memory_md=Path(args.global_memory_md),
        global_memory_jsonl=Path(args.global_memory_jsonl),
        timeout_seconds=args.timeout_seconds,
        sealed_config=sealed_config,
        environment=os.environ,
        gpu_evaluation=args.gpu_evaluation,
        task_failure_fd=args.task_failure_fd,
    )


if __name__ == "__main__":
    raise SystemExit(main())

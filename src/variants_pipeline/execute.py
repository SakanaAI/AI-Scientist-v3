"""Stage 5: author variants in containers and evaluate them on the host."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.investigation.container_images import resolve_immutable_image
from src.investigation.evaluator_tree import managed_evaluator_tree_digest
from src.investigation.files import evaluation_results_complete
from src.final_evaluation.candidate_broker import CandidateBrokerInfrastructureError
from src.investigation.safe_staging import (
    atomic_write_bytes,
    read_text as safe_read_text,
    remove_path,
)

from .agent_prompt import stage_agent_prompt
from .constants import ANALYSIS_CSV_SCHEMAS, MAX_EVALS_PER_VARIANT, REQUIRED_RUBRIC_IDS
from .host_eval import (
    HostEvaluationResult,
    discard_evaluation_artifacts,
    evaluate_variant_program,
    resolve_host_python,
)
from .models import VariantRunResult, VariantSpec
from .native_eval import evaluate_variant_program_natively, native_evaluation_python
from .qualification_summary import publish_summary_qualification
from .reference_preflight import validate_reference_workspace
from .rubric_contract import (
    build_rubric_contract,
    canonicalize_variant_rubric_status,
    load_rubric_status,
    validate_variant_rubric_status,
)

logger = logging.getLogger(__name__)

# Maximum number of times to re-launch a variant agent that edited but didn't
# evaluate.  Each retry gets a pointed error context so the agent knows what it
# missed on the previous attempt.
_MAX_INCOMPLETE_RETRIES = 2
_MAX_VERIFICATION_ERROR_LENGTH = 256
_SAFE_RUBRIC_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")

PopenFactory = Callable[..., subprocess.Popen[str]]
RegisterFn = Callable[[subprocess.Popen[str]], None]
UnregisterFn = Callable[[subprocess.Popen[str]], None]
TerminateFn = Callable[..., None]
EnsureTextFn = Callable[[Optional[str | bytes]], str]
_DIAGNOSTIC_LOG_CHARS = 4000
ContainerEnvFn = Callable[..., dict[str, str]]
EventFn = Callable[..., None]


def resolve_sandbox_image(
    image_reference: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Resolve a local Docker reference to the immutable ID used for evaluation."""
    return resolve_immutable_image(image_reference, run=run)


def execute_variant_agents(
    *,
    variant_specs: list[VariantSpec],
    variants_run_dir: Path,
    containerize: bool,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    timeout_seconds: int,
    evaluation_timeout_seconds: int | None = None,
    timeout_grace_seconds: int,
    max_evals_per_variant: int = MAX_EVALS_PER_VARIANT,
    sandbox_image: str | None = None,
    num_replications: int = 1,
    task_source_dir: Path | None = None,
    evaluator_source_dir: Path | None = None,
    evaluator_source_digest: str | None = None,
    venv_dir: Path | None = None,
    container_image: str = "",
    container_network: str = "bridge",
    container_auth_root: Path | None = None,
    container_env_file: Path | None = None,
    container_session_name: str | None = None,
    container_mount_root: Path | None = None,
    script_path: Path | None = None,
    popen_factory: PopenFactory | None = None,
    register_subprocess: RegisterFn | None = None,
    unregister_subprocess: UnregisterFn | None = None,
    terminate_process_group: TerminateFn | None = None,
    container_execution_env: ContainerEnvFn | None = None,
    ensure_text: EnsureTextFn | None = None,
    log_event_fn: EventFn | None = None,
) -> list[VariantRunResult]:
    """Run variant authoring and host-owned evaluation sequentially."""
    if evaluation_timeout_seconds is not None and (
        type(evaluation_timeout_seconds) is not int or evaluation_timeout_seconds < 1
    ):
        raise ValueError("evaluation_timeout_seconds must be a positive integer or None")
    results: list[VariantRunResult] = []
    if variant_specs and evaluator_source_digest is None:
        raise ValueError("variant execution requires a sealed evaluator digest")
    trusted_evaluator_source = (
        evaluator_source_dir or variants_run_dir / "_retrofitted_bundle"
    ).resolve()
    if evaluator_source_digest is not None:
        if managed_evaluator_tree_digest(trusted_evaluator_source) != evaluator_source_digest:
            raise ValueError("sealed evaluator source tree changed before execution")
    resolved_sandbox_image = ""
    if variant_specs and containerize:  # native runs (containerize=False) evaluate on the host
        resolved_sandbox_image = resolve_sandbox_image(
            sandbox_image or os.environ.get("VARIANT_EVAL_SANDBOX_IMAGE", "")
        )

    for variant_index, spec in enumerate(variant_specs, start=1):
        if evaluator_source_digest is not None:
            if managed_evaluator_tree_digest(trusted_evaluator_source) != evaluator_source_digest:
                raise ValueError("sealed evaluator source tree changed during execution")
        logger.info("Executing variant: %s (%s)", spec.variant_id, spec.kind)
        if log_event_fn is not None:
            log_event_fn(
                "variant_start",
                variant_id=spec.variant_id,
                title=spec.title,
                kind=spec.kind,
                current=variant_index,
                total=len(variant_specs),
            )
        result = _run_single_variant(
            spec=spec,
            agent=agent,
            agent_model=agent_model,
            agent_profile=agent_profile,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            evaluation_timeout_seconds=evaluation_timeout_seconds,
            timeout_grace_seconds=timeout_grace_seconds,
            max_evals=max_evals_per_variant,
            sandbox_image=resolved_sandbox_image,
            num_replications=num_replications,
            evaluator_source_dir=trusted_evaluator_source,
            evaluator_source_digest=evaluator_source_digest,
            task_source_dir=task_source_dir,
            venv_dir=venv_dir,
            containerize=containerize,
            container_image=container_image,
            container_network=container_network,
            container_auth_root=container_auth_root,
            container_env_file=container_env_file,
            container_session_name=container_session_name,
            container_mount_root=container_mount_root,
            script_path=script_path,
            popen_factory=popen_factory,
            register_subprocess=register_subprocess,
            unregister_subprocess=unregister_subprocess,
            terminate_process_group=terminate_process_group,
            container_execution_env=container_execution_env,
            ensure_text=ensure_text,
        )
        results.append(result)
        if log_event_fn is not None:
            log_event_fn(
                "variant_result",
                variant_id=spec.variant_id,
                title=spec.title,
                kind=spec.kind,
                status=result.status,
                eval_count=result.eval_count,
                rubric_passed=result.rubric_passed,
                exit_code=result.exit_code,
            )
        logger.info(
            "Variant %s: status=%s, rubric_passed=%s, programs=%s",
            spec.variant_id,
            result.status,
            result.rubric_passed,
            result.eval_count,
        )
        if spec.kind == "control" and not result.rubric_passed:
            logger.error(
                "Control %s failed verification; aborting remaining variants",
                spec.variant_id,
            )
            break

    if log_event_fn is not None:
        summary = _variant_result_summary(results)
        log_event_fn("variants_summary", **summary)

    return results


def _variant_result_summary(results: list[VariantRunResult]) -> dict[str, int]:
    return {
        "total": len(results),
        "passed": sum(result.rubric_passed for result in results),
        "failed": sum(result.status == "failed" for result in results),
        "timeout": sum(result.status == "timeout" for result in results),
    }


def _run_single_variant(
    *,
    spec: VariantSpec,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    timeout_seconds: int,
    evaluation_timeout_seconds: int | None = None,
    timeout_grace_seconds: int,
    max_evals: int,
    sandbox_image: str | None = None,
    num_replications: int,
    evaluator_source_dir: Path,
    evaluator_source_digest: str | None = None,
    task_source_dir: Path | None,
    venv_dir: Path | None,
    containerize: bool,
    container_image: str,
    container_network: str,
    container_auth_root: Path | None,
    container_env_file: Path | None,
    container_session_name: str | None,
    container_mount_root: Path | None,
    script_path: Path | None,
    popen_factory: PopenFactory | None,
    register_subprocess: RegisterFn | None,
    unregister_subprocess: UnregisterFn | None,
    terminate_process_group: TerminateFn | None,
    container_execution_env: ContainerEnvFn | None,
    ensure_text: EnsureTextFn | None,
) -> VariantRunResult:
    """Author candidates in isolation, then evaluate them on the host."""
    from src.investigation.script_runners import _execute_script_command

    variant_dir = spec.variant_dir.resolve()
    task_bundle_dir = variant_dir / "task_bundle"

    model = agent_model.strip() if isinstance(agent_model, str) else ""
    profile = agent_profile.strip() if isinstance(agent_profile, str) else ""
    effort = reasoning_effort.strip() if isinstance(reasoning_effort, str) else ""

    _update_manifest_status(
        variant_dir,
        "running",
        agent=agent,
        agent_model=model or None,
        agent_profile=profile or None,
        reasoning_effort=effort or None,
    )

    # --- resolve defaults ---
    if popen_factory is None:
        popen_factory = subprocess.Popen
    if register_subprocess is None:
        register_subprocess = lambda p: None  # noqa: E731
    if unregister_subprocess is None:
        unregister_subprocess = lambda p: None  # noqa: E731
    if terminate_process_group is None:
        from src.investigation.runtime import terminate_active_process_group

        terminate_process_group = terminate_active_process_group
    if ensure_text is None:
        ensure_text = lambda t: (
            t if isinstance(t, str) else (t.decode("utf-8", errors="replace") if t else "")
        )  # noqa: E731
    if container_execution_env is None:
        from src.investigation.runtime import container_execution_env as _cee

        container_execution_env = _cee
    if script_path is None:
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "variant_agent.sh"

    rubrics_file = variant_dir / "target_rubrics.json"
    env = container_execution_env(
        containerize=containerize,
        container_session_name=container_session_name,
        container_mount_root=container_mount_root,
        sanitize_ambient_controls=True,
    )
    env["VARIANT_ID"] = spec.variant_id
    env["VARIANT_KIND"] = spec.kind
    env["BASE_PROGRAM"] = spec.base_program
    if num_replications > 1:
        env["NUM_REPLICATIONS"] = str(num_replications)

    host_python = (resolve_host_python if containerize else native_evaluation_python)(
        task_source_dir=task_source_dir,
        venv_dir=venv_dir,
    )
    host_env = {
        key: os.environ[key]
        for key in (
            "AGENTIC_PUBLIC_ASSETS_DIR",
            "CUDA_VISIBLE_DEVICES",
            "NCCL_DEBUG",
            "VARIANT_EVAL_MEMORY_LIMIT",
            "XLA_PYTHON_CLIENT_ALLOCATOR",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
        )
        if key in os.environ
    }
    host_env.update(
        VARIANT_ID=spec.variant_id,
        VARIANT_KIND=spec.kind,
        BASE_PROGRAM=spec.base_program,
        AGENT_NAME=agent,
    )
    if num_replications > 1:
        host_env["NUM_REPLICATIONS"] = str(num_replications)
    expected_rubric_ids = [
        rubric["id"]
        for rubric in build_rubric_contract(
            spec.variant_id,
            spec.kind,
            spec.base_program,
        )["rubrics"]
    ]
    target_rubric_ids = _target_rubric_ids(rubrics_file, expected_rubric_ids)
    validate_reference_workspace(task_bundle_dir, spec.variant_id, spec.base_program)
    evaluator_source_dir = evaluator_source_dir.resolve()
    if not (evaluator_source_dir / "evaluate.py").is_file():
        raise ValueError(f"sealed retrofitted evaluator is missing: {evaluator_source_dir}")
    resolved_sandbox_image = (sandbox_image or _required_sandbox_image()) if containerize else ""
    output_parts: list[str] = []
    exit_code = 1
    eval_count = 0
    rubric_passed = False
    timed_out = False
    prompt_staging_failed = False

    def run_agent(
        prompt: str,
        prompt_filename: str,
        skill_step: str,
        label: str,
    ) -> tuple[str, int]:
        nonlocal prompt_staging_failed
        try:
            staged_prompt = stage_agent_prompt(task_bundle_dir, prompt_filename, prompt)
        except OSError as error:
            prompt_staging_failed = True
            message = f"Failed to stage variant agent prompt: {error}"
            logger.error("Variant %s: %s", spec.variant_id, message)
            return message, 1

        cmd = [
            "bash",
            str(script_path),
            "--agent",
            agent,
            "--work-dir",
            str(task_bundle_dir),
            "--prompt-file",
            staged_prompt,
            "--stage-skills",
            "--skill-step",
            skill_step,
            "--max-program-evals",
            str(max_evals),
            "--variant-id",
            spec.variant_id,
            "--attempt-id",
            f"variant_{spec.variant_id}",
        ]
        if rubrics_file.exists():
            cmd.extend(["--rubrics-file", str(rubrics_file)])
        if task_source_dir is not None:
            cmd.extend(["--task-source-dir", str(task_source_dir)])
        if venv_dir is not None:
            cmd.extend(["--venv-dir", str(venv_dir)])
        if model:
            cmd.extend(["--agent-model", model])
        if profile:
            cmd.extend(["--agent-profile", profile])
        if effort:
            cmd.extend(["--reasoning-effort", effort])
        if containerize:
            cmd.append("--containerize")
            cmd.extend(["--container-image", container_image])
            cmd.extend(["--container-network", container_network])
            auth_root = container_auth_root if container_auth_root is not None else Path.home()
            cmd.extend(["--container-auth-root", str(auth_root)])
            if container_env_file is not None:
                cmd.extend(["--container-env-file", str(container_env_file)])
        return _execute_script_command(
            cmd=cmd,
            timeout_seconds=timeout_seconds,
            timeout_grace_seconds=timeout_grace_seconds,
            script_label=label,
            missing_output=f"[variants] variant_agent.sh not found for {spec.variant_id}\n",
            popen_factory=popen_factory,
            register_subprocess=register_subprocess,
            unregister_subprocess=unregister_subprocess,
            terminate_process_group=terminate_process_group,
            ensure_text=ensure_text,
            env=env,
        )

    first_program_index = 0 if spec.kind == "control" else 1
    last_program_index = (
        first_program_index if spec.kind == "control" else first_program_index + max_evals - 1
    )
    for program_index in range(first_program_index, last_program_index + 1):
        if program_index > 0:
            created = False
            for edit_attempt in range(1, _MAX_INCOMPLETE_RETRIES + 2):
                prompt = _build_host_edit_prompt(
                    spec=spec,
                    task_bundle_dir=task_bundle_dir,
                    program_index=program_index,
                    edit_attempt=edit_attempt,
                )
                agent_output, exit_code = run_agent(
                    prompt,
                    f"variant_agent_prompt_edit_{program_index:02d}_attempt_{edit_attempt:02d}.md",
                    "variant_edit",
                    f"variant_{spec.variant_id}_edit_{program_index}",
                )
                output_parts.append(agent_output)
                if prompt_staging_failed:
                    break
                if exit_code == 124:
                    timed_out = True
                    break
                created = (task_bundle_dir / f"program_{program_index}").is_dir()
                if created:
                    break
            if timed_out or not created:
                break

        evaluation_id = f"{spec.variant_id}_eval_{eval_count + 1}"
        env["EVALUATION_ID"] = evaluation_id
        host_env["EVALUATION_ID"] = evaluation_id
        logger.info(
            "Host-evaluating %s program_%d with %s",
            spec.variant_id,
            program_index,
            host_python,
        )
        evaluation_completed = False
        try:
            _verify_evaluator_source(
                evaluator_source_dir,
                evaluator_source_digest,
            )
            if not containerize and task_source_dir is None:
                raise ValueError("native variant evaluation needs the task source directory")
            evaluation_kwargs: dict[str, Any] = dict(
                task_bundle_dir=task_bundle_dir,
                evaluator_source_dir=evaluator_source_dir,
                task_source_dir=task_source_dir or task_bundle_dir,
                program_index=program_index,
                python_bin=host_python,
                timeout_seconds=(
                    timeout_seconds if evaluation_timeout_seconds is None
                    else evaluation_timeout_seconds
                ),
                environment=host_env,
            )
            if containerize:
                evaluation = evaluate_variant_program(
                    **evaluation_kwargs,
                    sandbox_image=resolved_sandbox_image,
                    sandbox_gpus=os.environ.get("VARIANT_EVAL_SANDBOX_GPUS"),
                )
            else:
                evaluation = evaluate_variant_program_natively(**evaluation_kwargs)
            evaluation_completed = True
            _verify_evaluator_source(
                evaluator_source_dir,
                evaluator_source_digest,
            )
        except CandidateBrokerInfrastructureError:
            raise
        except TimeoutError as exc:
            output_parts.append(str(exc))
            exit_code = 124
            timed_out = True
            break
        except (OSError, RuntimeError, ValueError) as exc:
            if evaluation_completed:
                try:
                    discard_evaluation_artifacts(
                        task_bundle_dir / f"program_{program_index}"
                    )
                except OSError as cleanup_error:
                    logger.error(
                        "Variant %s evaluation cleanup failed: %s",
                        spec.variant_id,
                        cleanup_error,
                    )
            diagnostic = str(getattr(exc, "diagnostic", "") or "")[-_DIAGNOSTIC_LOG_CHARS:]
            output_parts.append(f"{exc}\n{diagnostic}" if diagnostic else str(exc))
            logger.error(
                "Variant %s host evaluation failed: %s%s",
                spec.variant_id,
                exc,
                f"\n--- evaluator output (tail) ---\n{diagnostic}" if diagnostic else "",
            )
            exit_code = 1
            break

        eval_count += 1
        exit_code = evaluation.returncode
        if exit_code != 0:
            logger.error(
                "Variant %s evaluator exited with code %d",
                spec.variant_id,
                exit_code,
            )
            break
        verification_error: str | None = None
        rubric_valid = False
        canonical_rubric: dict[str, object] | None = None
        for verify_attempt in range(1, _MAX_INCOMPLETE_RETRIES + 2):
            _clear_rubric_status(task_bundle_dir, program_index)
            _write_rubric_scaffold(
                task_bundle_dir,
                program_index,
                spec.variant_id,
                target_rubric_ids,
                evaluation,
            )
            verification_prompt = _build_host_verify_prompt(
                spec=spec,
                program_index=program_index,
                evaluation_correct=evaluation.correct,
                target_rubric_ids=target_rubric_ids,
                previous_error=verification_error,
            )
            agent_output, verify_exit = run_agent(
                verification_prompt,
                f"variant_agent_prompt_verify_{program_index:02d}_attempt_{verify_attempt:02d}.md",
                "variant_verify",
                f"variant_{spec.variant_id}_verify_{program_index}",
            )
            output_parts.append(agent_output)
            if prompt_staging_failed:
                exit_code = 1
                break
            if verify_exit == 124:
                exit_code = 124
                timed_out = True
                break
            if verify_exit != 0:
                exit_code = verify_exit
                verification_error = _verification_error_context(
                    f"verifier exited with code {verify_exit}"
                )
                logger.warning(
                    "Variant %s verifier exited with code %d",
                    spec.variant_id,
                    verify_exit,
                )
                continue
            rubric = load_rubric_status(
                task_bundle_dir / f"program_{program_index}" / "rubric_status.json"
            )
            canonical_rubric, rubric_error = canonicalize_variant_rubric_status(
                rubric,
                variant_id=spec.variant_id,
                attempt_id=_variant_attempt_id(spec.variant_id, program_index),
                idea_id=spec.variant_id,
                required_rubric_ids=target_rubric_ids,
                combined_score=evaluation.metrics.get("combined_score"),
                correct=evaluation.correct,
            )
            rubric_valid = canonical_rubric is not None
            if canonical_rubric is not None:
                _write_canonical_rubric_status(
                    task_bundle_dir,
                    program_index,
                    canonical_rubric,
                )
            rubric_passed = bool(
                evaluation.correct
                and rubric_valid
                and canonical_rubric
                and canonical_rubric["overall_pass"] is True
            )
            if rubric_valid:
                break
            verification_error = _verification_error_context(
                rubric_error or "invalid rubric_status.json"
            )
            logger.warning(
                "Variant %s verifier attempt %d wrote invalid rubric status: %s",
                spec.variant_id,
                verify_attempt,
                rubric_error,
            )
        if not rubric_valid:
            _write_canonical_rubric_status(
                task_bundle_dir,
                program_index,
                _failed_rubric_status(
                    spec.variant_id,
                    program_index,
                    target_rubric_ids,
                    evaluation,
                    verification_error,
                ),
            )
        if prompt_staging_failed or timed_out or rubric_passed:
            break

    status = "timeout" if timed_out else ("completed" if rubric_passed else "failed")
    _update_manifest_status(variant_dir, status, eval_count=eval_count)

    return VariantRunResult(
        variant_id=spec.variant_id,
        status=status,
        eval_count=eval_count,
        rubric_passed=rubric_passed,
        exit_code=exit_code,
        output="\n".join(part for part in output_parts if part),
        analysis_files=_check_analysis_files(task_bundle_dir),
    )


def _verify_evaluator_source(
    evaluator_source_dir: Path,
    expected_digest: str | None,
) -> None:
    if expected_digest is None:
        return
    if managed_evaluator_tree_digest(evaluator_source_dir) != expected_digest:
        raise ValueError("sealed evaluator source tree changed before evaluation")


def _build_host_edit_prompt(
    *,
    spec: VariantSpec,
    task_bundle_dir: Path,
    program_index: int,
    edit_attempt: int,
) -> str:
    previous_index = program_index - 1
    previous_status = _previous_rubric_summary(
        task_bundle_dir,
        previous_index,
        spec,
    )
    return f"""You are authoring one scientific comparison candidate.
Use only the edit-program and memory-log skills. Do not evaluate, verify, run Python,
run tests, install packages, or create evaluator artifacts; the host harness owns those steps.

Variant specification:
{json.dumps(spec.to_dict(), indent=2)}

Create exactly ./program_{program_index}/ by cloning ./program_{previous_index}/.
Implement only the declared variant change. Preserve the CSV analysis infrastructure.
This is edit attempt {edit_attempt}. Previous rubric evidence, if any:
{previous_status or "(none; first candidate)"}

Stop immediately after the candidate files are complete.
"""


def _build_host_verify_prompt(
    *,
    spec: VariantSpec,
    program_index: int,
    evaluation_correct: bool,
    target_rubric_ids: list[str],
    previous_error: str | None,
) -> str:
    return f"""The host harness has evaluated ./program_{program_index}/.
Use only verify-rubrics and memory-log. Do not edit code, run evaluate.py, run tests,
install packages, or create another program directory.

Variant ID: {spec.variant_id}
Variant kind: {spec.kind}
Host correctness: {evaluation_correct}

Inspect ./program_{program_index}/metrics.json, correct.json, source, and analysis files.
The host has pre-created ./program_{program_index}/rubric_status.json with the exact
required rubric IDs. Update only each rubric's boolean `pass` and non-empty string
`evidence`, plus optional `next_recommendation`. Do not replace it with a metrics
summary, add/remove rubric IDs, or write ./rubric_status.json; the host owns identity,
metrics, aggregate `overall_pass`, and the workspace mirror.

Required rubric IDs:
{json.dumps(target_rubric_ids, indent=2)}

Previous host validation error: {previous_error or "(none; first verification attempt)"}
Then stop.
"""


def _target_rubric_ids(
    path: Path,
    expected_rubric_ids: list[str] | None = None,
) -> list[str]:
    expected = expected_rubric_ids or list(REQUIRED_RUBRIC_IDS)
    try:
        payload = json.loads(safe_read_text(path.parent, path.name))
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid target_rubrics.json: {path}") from exc
    if not isinstance(payload, list):
        raise ValueError("target_rubrics.json must contain a list")
    rubric_ids = [
        item["id"] for item in payload if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if (
        len(rubric_ids) != len(payload)
        or len(set(rubric_ids)) != len(rubric_ids)
        or any(_SAFE_RUBRIC_ID.fullmatch(rubric_id) is None for rubric_id in rubric_ids)
        or set(rubric_ids) != set(expected)
    ):
        raise ValueError("target_rubrics.json does not match the trusted rubric contract")
    return expected


def _clear_rubric_status(task_bundle_dir: Path, program_index: int) -> None:
    for relative_path in (
        "rubric_status.json",
        f"program_{program_index}/rubric_status.json",
    ):
        remove_path(task_bundle_dir, relative_path)


def _verification_error_context(error: str) -> str:
    single_line = " ".join(error.split())
    return single_line[:_MAX_VERIFICATION_ERROR_LENGTH]


def _previous_rubric_summary(
    task_bundle_dir: Path,
    program_index: int,
    spec: VariantSpec,
) -> str:
    status = load_rubric_status(task_bundle_dir / f"program_{program_index}" / "rubric_status.json")
    if status is None:
        return "(none; first candidate)"
    trusted_ids = [
        rubric["id"]
        for rubric in build_rubric_contract(
            spec.variant_id,
            spec.kind,
            spec.base_program,
        )["rubrics"]
    ]
    entries = status.get("rubrics")
    by_id: dict[object, object] = {}
    if isinstance(entries, list):
        by_id = {entry.get("id"): entry.get("pass") for entry in entries if isinstance(entry, dict)}
    summary = {
        "overall_pass": status.get("overall_pass") is True,
        "failed_rubric_ids": [
            rubric_id for rubric_id in trusted_ids if by_id.get(rubric_id) is not True
        ],
    }
    return json.dumps(summary, sort_keys=True)


def _write_rubric_scaffold(
    task_bundle_dir: Path,
    program_index: int,
    variant_id: str,
    rubric_ids: list[str],
    evaluation: HostEvaluationResult,
) -> None:
    scaffold = {
        "variant_id": variant_id,
        "attempt_id": _variant_attempt_id(variant_id, program_index),
        "idea_id": variant_id,
        "overall_pass": False,
        "rubrics": [{"id": rubric_id, "pass": False, "evidence": ""} for rubric_id in rubric_ids],
        "metrics": {
            "combined_score": evaluation.metrics.get("combined_score"),
            "correct": evaluation.correct,
        },
        "next_recommendation": "",
    }
    _write_rubric_json(
        task_bundle_dir,
        f"program_{program_index}/rubric_status.json",
        scaffold,
    )


def _failed_rubric_status(
    variant_id: str,
    program_index: int,
    rubric_ids: list[str],
    evaluation: HostEvaluationResult,
    verification_error: str | None,
) -> dict[str, object]:
    context = verification_error or "verifier did not produce valid rubric judgments"
    return {
        "variant_id": variant_id,
        "attempt_id": _variant_attempt_id(variant_id, program_index),
        "idea_id": variant_id,
        "overall_pass": False,
        "rubrics": [
            {
                "id": rubric_id,
                "pass": False,
                "evidence": f"Verification incomplete: {context}",
            }
            for rubric_id in rubric_ids
        ],
        "metrics": {
            "combined_score": evaluation.metrics.get("combined_score"),
            "correct": evaluation.correct,
        },
        "next_recommendation": "Repair rubric verification before another evaluation.",
    }


def _variant_attempt_id(variant_id: str, program_index: int) -> str:
    return f"{variant_id}_attempt_{program_index:02d}"


def _write_canonical_rubric_status(
    task_bundle_dir: Path,
    program_index: int,
    rubric_status: dict[str, object],
) -> None:
    content = (json.dumps(rubric_status, indent=2, sort_keys=True) + "\n").encode()
    atomic_write_bytes(
        task_bundle_dir,
        f"program_{program_index}/rubric_status.json",
        content,
    )
    atomic_write_bytes(task_bundle_dir, "rubric_status.json", content)
    publish_summary_qualification(task_bundle_dir / f"program_{program_index}")


def _write_rubric_json(
    task_bundle_dir: Path,
    relative_path: str,
    payload: dict[str, object],
) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    atomic_write_bytes(task_bundle_dir, relative_path, content)


def _required_sandbox_image() -> str:
    image = os.environ.get("VARIANT_EVAL_SANDBOX_IMAGE", "").strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("VARIANT_EVAL_SANDBOX_IMAGE must be an immutable sha256 image ID")
    return image


def _inspect_program_results(
    task_bundle_dir: Path,
    variant_id: str,
    required_rubric_ids: list[str] | None = None,
) -> tuple[int, bool]:
    """Inspect program_N directories for results.

    Returns ``(eval_count, rubric_passed)`` where *eval_count* is the number of
    program directories with ``metrics.json`` and *rubric_passed* is True if any
    completely evaluated program has a host-canonical passing rubric status.
    """
    eval_count = 0
    rubric_passed = False
    required = required_rubric_ids or list(REQUIRED_RUBRIC_IDS)

    for entry in sorted(task_bundle_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not re.match(r"^program_\d+$", entry.name):
            continue

        idx = int(entry.name.split("_")[1])
        if idx == 0:
            continue  # program_0 is the seed, not counted as an eval attempt

        evaluation_complete = evaluation_results_complete(entry)

        # Count programs that have been evaluated completely.
        if (entry / "metrics.json").exists() and evaluation_complete:
            eval_count += 1

        # Check rubric status
        rs = load_rubric_status(entry / "rubric_status.json")
        if evaluation_complete and _canonical_rubric_passed(
            entry,
            idx,
            variant_id,
            required,
            rs,
        ):
            rubric_passed = True

    return eval_count, rubric_passed


def _canonical_rubric_passed(
    program_dir: Path,
    program_index: int,
    variant_id: str,
    required_rubric_ids: list[str],
    status: dict[str, object] | None,
) -> bool:
    if status is None:
        return False
    try:
        correct_payload = json.loads(safe_read_text(program_dir, "correct.json"))
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
        return False
    metrics = status.get("metrics")
    valid_rubrics, _ = validate_variant_rubric_status(
        status,
        variant_id,
        set(required_rubric_ids),
    )
    return bool(
        valid_rubrics
        and status.get("idea_id") == variant_id
        and status.get("attempt_id") == _variant_attempt_id(variant_id, program_index)
        and status.get("overall_pass") is True
        and isinstance(metrics, dict)
        and metrics.get("correct") is True
        and isinstance(correct_payload, dict)
        and correct_payload.get("correct") is True
    )


def _check_analysis_files(task_bundle_dir: Path) -> dict[str, bool]:
    """Check which analysis files exist across all program_N directories."""
    result: dict[str, bool] = {}
    import re

    for entry in sorted(task_bundle_dir.iterdir()):
        if not entry.is_dir() or not re.match(r"^program_\d+$", entry.name):
            continue
        analysis_dir = entry / "analysis"
        for filename in ANALYSIS_CSV_SCHEMAS:
            path = analysis_dir / filename
            if path.exists():
                result[filename] = True
            elif filename not in result:
                result[filename] = False
        rubric_path = entry / "rubric_status.json"
        if rubric_path.exists():
            result["rubric_status.json"] = True
        elif "rubric_status.json" not in result:
            result["rubric_status.json"] = False

    return result


def _update_manifest_status(
    variant_dir: Path,
    status: str,
    eval_count: int | None = None,
    *,
    agent: str | None = None,
    agent_model: str | None = None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
) -> None:
    """Update the ``run_manifest.json`` status fields."""
    manifest_path = variant_dir / "run_manifest.json"
    if not manifest_path.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    now = datetime.now(timezone.utc).isoformat()
    manifest["status"] = status
    if status == "running":
        manifest["started_at"] = now
        manifest["completed_at"] = None
    elif status in ("completed", "failed", "timeout"):
        manifest["completed_at"] = now
    if eval_count is not None:
        manifest["eval_count"] = eval_count
    if agent is not None:
        manifest.update(
            {
                "agent": agent,
                "agent_model": agent_model,
                "agent_profile": agent_profile,
                "reasoning_effort": reasoning_effort,
            }
        )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

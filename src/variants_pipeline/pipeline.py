"""Top-level orchestrator: chains the 9 stages of the variants pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any, Callable

from src.investigation.runtime import format_elapsed

from .aggregate import aggregate_variant_csvs
from .config import VariantsPipelineConfig
from .extract import extract_bundle
from .generate import generate_variant_candidates
from .materialize import materialize_variant_workspaces
from .models import VariantRunResult, VariantSelection, VariantSpec
from .native_eval import is_contract_task
from .retrofit import retrofit_program
from .select import validate_and_select_variants

logger = logging.getLogger(__name__)
EventFn = Callable[..., None]


def run_variants_pipeline(
    cfg: VariantsPipelineConfig,
    stages: set[int] | None = None,
    custom_baseline_paths: list[Path] | None = None,
    custom_only: bool = False,
    log_event_fn: EventFn | None = None,
) -> int:
    """Execute the variants pipeline.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    stages:
        Optional set of stage numbers to run (0-8).  ``None`` runs all stages.
        When a subset is specified, earlier stages that produce artifacts
        consumed by later stages will attempt to load from disk if available.
    custom_only:
        When ``True``, skip control_best and baseline_reference — only
        custom baselines are materialized and executed.

    Returns an exit code (0 = success).
    """
    if stages is None:
        stages = {0, 1, 2, 3, 4, 5, 6, 7, 8}
    cross_task_requested = 6 in stages and bool(cfg.paths.extra_tasks)
    agent_stages = stages.intersection({1, 2, 3, 5, 8})
    if cross_task_requested:
        agent_stages.add(6)
    if not cfg.container.enabled:
        if cross_task_requested:
            logger.error("Cross-task variants require container.enabled=true")
            return 1
        if is_contract_task(Path(cfg.paths.task_dir).expanduser()):
            logger.error(
                "Contract tasks require container.enabled=true: native agents could read "
                "the hidden final evaluator"
            )
            return 1
        if agent_stages:
            logger.info(
                "Native variants: agents and evaluations run on the host (container.enabled=false)"
            )

    task_dir = Path(cfg.paths.task_dir).expanduser().resolve()
    results_dir = Path(cfg.paths.results_dir).expanduser().resolve()
    output_dir = _resolve_output_dir(cfg)

    sandbox_image: str | None = None
    evaluator_source_digest: str | None = None
    if 5 in stages:
        from .execute import resolve_sandbox_image
        from .runtime import prepare_evaluation_runtime
        from src.investigation.evaluator_tree import managed_evaluator_tree_digest

        try:
            evaluator_source_digest = managed_evaluator_tree_digest(task_dir)
        except (OSError, ValueError) as exc:
            logger.error(
                "Managed runtime evaluator preflight failed for %s: %s",
                task_dir,
                exc,
            )
            return 1
        if cfg.container.enabled:
            sandbox_image = resolve_sandbox_image(
                cfg.execution.sandbox_image or cfg.container.image
            )
            logger.info("Evaluation sandbox pinned: %s", sandbox_image)
            if not prepare_evaluation_runtime(
                task_dir,
                sandbox_image=sandbox_image,
                container_network=cfg.container.network,
                evaluator_tree_sha256=evaluator_source_digest,
            ):
                logger.error(
                    "Managed runtime preflight failed for primary task %s; "
                    "no variant stages were started",
                    task_dir,
                )
                return 1
        else:
            logger.info("Native evaluation: no sandbox image or managed runtime to prepare")

    logger.info("Variants pipeline starting: output_dir=%s", output_dir)

    generation_dir = output_dir / "generation"
    generation_dir.mkdir(parents=True, exist_ok=True)

    # --- common kwargs for agent invocation ---
    agent_kwargs: dict[str, Any] = {
        "agent": cfg.agent.name,
        "agent_model": cfg.agent.model,
        "agent_profile": cfg.agent.profile,
        "reasoning_effort": cfg.agent.reasoning_effort,
        "task_source_dir": task_dir,
        "containerize": cfg.container.enabled,
        "container_image": cfg.container.image,
        "container_network": cfg.container.network,
        "container_auth_root": Path(cfg.container.auth_root).expanduser(),
        "container_env_file": Path(cfg.container.env_file) if cfg.container.env_file else None,
    }

    # Intermediate artifacts (loaded from disk when stages are skipped)
    bundle_dir = output_dir / "_extracted_bundle"
    retrofitted_bundle_dir = output_dir / "_retrofitted_bundle"

    # --- Stage 0: Extract bundle ---
    baseline_override = (
        Path(cfg.paths.baseline_program_override).expanduser().resolve()
        if cfg.paths.baseline_program_override
        else None
    )

    if 0 in stages:
        stage_started_at = _log_stage_start(log_event_fn, 0, "Extract bundle")
        logger.info("=== Stage 0: Extract bundle ===")
        bundle_dir = extract_bundle(
            task_dir=task_dir,
            results_dir=results_dir,
            output_dir=output_dir,
            baseline_program_override=baseline_override,
        )
        _log_stage_end(log_event_fn, 0, "Extract bundle", stage_started_at)
    else:
        logger.info("=== Stage 0: Skipped (using existing bundle at %s) ===", bundle_dir)
        if not bundle_dir.is_dir():
            logger.error("No extracted bundle found at %s. Run stage 0 first.", bundle_dir)
            return 1

    # --- Stage 1: Retrofit program ---
    if 1 in stages:
        stage_started_at = _log_stage_start(log_event_fn, 1, "Retrofit program")
        logger.info("=== Stage 1: Retrofit program with CSV analysis ===")
        retrofit_report = retrofit_program(
            bundle_dir=bundle_dir,
            output_bundle_dir=retrofitted_bundle_dir,
            timeout_seconds=cfg.execution.retrofit_timeout_seconds,
            timeout_grace_seconds=cfg.execution.variant_timeout_grace_seconds,
            **agent_kwargs,
        )
        logger.info("Retrofit report: %s", json.dumps(retrofit_report, indent=2))
        from .retrofit import retrofit_bundle_complete

        if retrofit_report.get("exit_code") != 0 or not retrofit_bundle_complete(
            retrofitted_bundle_dir,
            bundle_dir,
            task_source_dir=task_dir,
        ):
            _log_stage_end(
                log_event_fn,
                1,
                "Retrofit program",
                stage_started_at,
                status="failed",
            )
            logger.error("Retrofit failed; stopping before variant generation")
            return 1
        _log_stage_end(log_event_fn, 1, "Retrofit program", stage_started_at)
    else:
        logger.info(
            "=== Stage 1: Skipped (using existing retrofit at %s) ===", retrofitted_bundle_dir
        )
        if not retrofitted_bundle_dir.is_dir():
            if stages.intersection({4, 5, 6, 8}):
                logger.error("No retrofitted bundle found at %s", retrofitted_bundle_dir)
                return 1
            retrofitted_bundle_dir = bundle_dir
        elif stages.intersection({4, 5, 6, 8}):
            from .retrofit import retrofit_bundle_complete

            if not retrofit_bundle_complete(
                retrofitted_bundle_dir, bundle_dir, task_source_dir=task_dir,
            ):
                logger.error(
                    "Retrofitted bundle is incomplete or untrusted: %s",
                    retrofitted_bundle_dir,
                )
                return 1

    # --- Stage 2: Generate variant candidates ---
    candidates = []
    if custom_only:
        logger.info("=== Stage 2: Skipped (--custom-only) ===")
    elif 2 in stages:
        stage_started_at = _log_stage_start(log_event_fn, 2, "Generate candidates")
        logger.info("=== Stage 2: Generate variant candidates ===")
        candidates = generate_variant_candidates(
            bundle_dir=bundle_dir,
            generation_dir=generation_dir,
            timeout_seconds=cfg.execution.generation_timeout_seconds,
            timeout_grace_seconds=cfg.execution.variant_timeout_grace_seconds,
            **agent_kwargs,
        )
        logger.info("Generated %d candidates", len(candidates))
        _log_stage_end(
            log_event_fn,
            2,
            "Generate candidates",
            stage_started_at,
            candidate_count=len(candidates),
        )
    else:
        logger.info("=== Stage 2: Skipped ===")
        candidates = _load_candidates_from_disk(generation_dir)
        logger.info("Loaded %d candidates from disk", len(candidates))

    # --- Stage 3: Validate and select variants ---
    selection = None
    if custom_only:
        logger.info("=== Stage 3: Skipped (--custom-only) ===")
    elif 3 in stages:
        stage_started_at = _log_stage_start(log_event_fn, 3, "Select variants")
        logger.info("=== Stage 3: Validate and select variants ===")
        selection = validate_and_select_variants(
            candidates=candidates,
            bundle_dir=bundle_dir,
            generation_dir=generation_dir,
            timeout_seconds=cfg.execution.selection_timeout_seconds,
            timeout_grace_seconds=cfg.execution.variant_timeout_grace_seconds,
            max_ablations=cfg.execution.max_ablations,
            max_extra_baselines=cfg.execution.max_extra_baselines,
            **agent_kwargs,
        )
        logger.info(
            "Selected %d variants (%d rejected)",
            len(selection.accepted),
            len(selection.rejected),
        )
        _log_stage_end(
            log_event_fn,
            3,
            "Select variants",
            stage_started_at,
            accepted_count=len(selection.accepted),
            rejected_count=len(selection.rejected),
        )
    else:
        logger.info("=== Stage 3: Skipped ===")
        selection = _load_selection_from_disk(generation_dir, candidates)

    # --- Inject custom baselines (between stages 3 and 4) ---
    if custom_baseline_paths:
        from .custom_baselines import load_custom_baselines

        custom_candidates = load_custom_baselines(custom_baseline_paths)
        if selection is None:
            selection = VariantSelection(
                accepted=[],
                rejected=[],
                controls=["control_best"],
            )

        if custom_only:
            # Discard all LLM-generated candidates — only custom baselines
            selection = VariantSelection(
                accepted=list(custom_candidates),
                rejected=[],
                controls=[],
            )
        else:
            # Custom baselines take precedence over LLM-generated with same ID
            existing_ids = {c.variant_id for c in selection.accepted}
            for custom in custom_candidates:
                if custom.variant_id in existing_ids:
                    selection.accepted = [
                        c for c in selection.accepted if c.variant_id != custom.variant_id
                    ]
                    logger.info(
                        "Custom baseline '%s' replaces LLM-generated candidate",
                        custom.variant_id,
                    )
                selection.accepted.append(custom)

        folder_count = sum(1 for c in custom_candidates if c.reference_dir is not None)
        logger.info(
            "Injected %d custom baseline(s) (%d folder, %d markdown): %s",
            len(custom_candidates),
            folder_count,
            len(custom_candidates) - folder_count,
            [c.variant_id for c in custom_candidates],
        )

    # --- Stage 4: Materialize variant workspaces ---
    variant_specs: list[VariantSpec] = []
    if 4 in stages:
        stage_started_at = _log_stage_start(log_event_fn, 4, "Materialize workspaces")
        logger.info("=== Stage 4: Materialize variant workspaces ===")
        variant_specs = materialize_variant_workspaces(
            bundle_dir=bundle_dir,
            retrofitted_bundle_dir=retrofitted_bundle_dir,
            variants_run_dir=output_dir,
            selection=selection,
            agent=cfg.agent.name,
            agent_model=cfg.agent.model,
            agent_profile=cfg.agent.profile,
            reasoning_effort=cfg.agent.reasoning_effort,
            max_evals_per_variant=cfg.execution.max_evals_per_variant,
            custom_only=custom_only,
            required_controls={"control_best"} if cross_task_requested else set(),
        )
        logger.info("Materialized %d variant workspaces", len(variant_specs))
        _log_stage_end(
            log_event_fn,
            4,
            "Materialize workspaces",
            stage_started_at,
            variant_count=len(variant_specs),
        )
    else:
        logger.info("=== Stage 4: Skipped ===")
        variant_specs = _load_variant_specs_from_disk(output_dir)
        logger.info("Loaded %d variant specs from disk", len(variant_specs))

    expected_control_dir = output_dir / "variants" / "control_best"
    control_specs = [spec for spec in variant_specs if spec.variant_id == "control_best"]
    if cross_task_requested and not (
        len(control_specs) == 1
        and control_specs[0].kind == "control"
        and control_specs[0].base_program == "best"
        and control_specs[0].variant_dir == expected_control_dir
    ):
        logger.error(
            "Cross-task stage requires a canonical control_best; rerun stage 4 to materialize it"
        )
        return 1

    if cross_task_requested and sandbox_image is None:
        from .execute import resolve_sandbox_image

        sandbox_image = resolve_sandbox_image(cfg.execution.sandbox_image or cfg.container.image)
        logger.info("Evaluation sandbox pinned: %s", sandbox_image)

    # --- Stage 5: Execute variant agents ---
    results: list[VariantRunResult] = []
    if 5 in stages:
        stage_started_at = _log_stage_start(log_event_fn, 5, "Execute variants")
        logger.info("=== Stage 5: Execute variant agents ===")
        from .execute import execute_variant_agents

        assert evaluator_source_digest is not None
        results = execute_variant_agents(
            variant_specs=variant_specs,
            variants_run_dir=output_dir,
            timeout_seconds=cfg.execution.variant_timeout_seconds,
            evaluation_timeout_seconds=cfg.execution.evaluation_timeout_seconds,
            timeout_grace_seconds=cfg.execution.variant_timeout_grace_seconds,
            max_evals_per_variant=cfg.execution.max_evals_per_variant,
            num_replications=cfg.execution.num_replications,
            sandbox_image=sandbox_image,
            evaluator_source_dir=task_dir,
            evaluator_source_digest=evaluator_source_digest,
            log_event_fn=log_event_fn,
            **agent_kwargs,
        )

        # Write summary
        _write_pipeline_summary(output_dir, variant_specs, results)

        passed = sum(1 for r in results if r.rubric_passed)
        failed = len(results) - passed
        logger.info(
            "Variants pipeline complete: %d passed, %d failed out of %d total",
            passed,
            failed,
            len(results),
        )
        failed_controls = _failed_control_ids(variant_specs, results)
        if failed_controls:
            _log_stage_end(
                log_event_fn,
                5,
                "Execute variants",
                stage_started_at,
                status="failed",
                passed=passed,
                failed=failed,
            )
            logger.error(
                "Required controls failed verification: %s; stopping before paper stages",
                ", ".join(failed_controls),
            )
            return 1
        if passed == 0:
            _log_stage_end(
                log_event_fn,
                5,
                "Execute variants",
                stage_started_at,
                status="failed",
                passed=passed,
                failed=failed,
            )
            logger.error("No variant produced a valid evaluation; stopping before paper stages")
            return 1
        _log_stage_end(
            log_event_fn,
            5,
            "Execute variants",
            stage_started_at,
            passed=passed,
            failed=failed,
        )

    else:
        logger.info("=== Stage 5: Skipped ===")
        downstream_requires_results = bool(stages.intersection({7, 8}) or cross_task_requested)
        if downstream_requires_results:
            summary_error = _validate_pipeline_summary(output_dir, variant_specs)
            if summary_error is not None:
                logger.error("Cannot resume downstream stages: %s", summary_error)
                return 1

    # --- Stage 6: Cross-task generalization ---
    if 6 in stages:
        extra_task_paths = [Path(p).expanduser().resolve() for p in cfg.paths.extra_tasks]
        if extra_task_paths:
            stage_started_at = _log_stage_start(log_event_fn, 6, "Cross-task evaluation")
            logger.info("=== Stage 6: Cross-task generalization ===")
            from .cross_task import prepare_and_execute_cross_task

            try:
                cross_task_results = prepare_and_execute_cross_task(
                    extra_task_paths=extra_task_paths,
                    variants_run_dir=output_dir,
                    variant_specs=variant_specs,
                    timeout_seconds=cfg.execution.cross_task_timeout_seconds,
                    evaluation_timeout_seconds=cfg.execution.evaluation_timeout_seconds,
                    timeout_grace_seconds=cfg.execution.variant_timeout_grace_seconds,
                    retrofit_timeout_seconds=cfg.execution.retrofit_timeout_seconds,
                    max_evals_per_variant=cfg.execution.max_evals_per_variant,
                    num_replications=cfg.execution.num_replications,
                    sandbox_image=sandbox_image,
                    **agent_kwargs,
                )
            except RuntimeError as exc:
                logger.error("Cross-task stage failed: %s", exc)
                return 1
            total_cross = sum(len(v) for v in cross_task_results.values())
            logger.info(
                "Cross-task generalization complete: %d tasks, %d total variants",
                len(cross_task_results),
                total_cross,
            )
            _log_stage_end(
                log_event_fn,
                6,
                "Cross-task evaluation",
                stage_started_at,
                task_count=len(cross_task_results),
                variant_count=total_cross,
            )
        else:
            logger.info("=== Stage 6: Skipped (no extra_tasks configured) ===")
    else:
        logger.info("=== Stage 6: Skipped ===")

    # --- Stage 7: Aggregate analysis CSVs ---
    if 7 in stages:
        stage_started_at = _log_stage_start(log_event_fn, 7, "Aggregate results")
        logger.info("=== Stage 7: Aggregate analysis CSVs ===")
        original_task_name = cfg.paths.task_name or Path(cfg.paths.task_dir).name or "original"
        aggregated = aggregate_variant_csvs(
            output_dir,
            original_task_name=original_task_name,
        )
        for task_name, task_files in aggregated.items():
            produced = [f for f, p in task_files.items() if p is not None]
            logger.info("Aggregated %s: %s", task_name, produced or "(none)")
        _log_stage_end(
            log_event_fn,
            7,
            "Aggregate results",
            stage_started_at,
            task_count=len(aggregated),
        )
    else:
        logger.info("=== Stage 7: Skipped ===")

    # --- Stage 8: Generate variants README ---
    if 8 in stages:
        stage_started_at = _log_stage_start(log_event_fn, 8, "Generate README")
        logger.info("=== Stage 8: Generate variants README ===")
        from .readme import generate_variants_readme

        if not variant_specs:
            variant_specs = _load_variant_specs_from_disk(output_dir)

        readme_path = generate_variants_readme(
            variant_specs=variant_specs,
            output_dir=output_dir,
            task_dir=task_dir,
            primary_task_name=cfg.paths.task_name,
            timeout_seconds=cfg.execution.readme_timeout_seconds,
            timeout_grace_seconds=cfg.execution.variant_timeout_grace_seconds,
            **agent_kwargs,
        )
        logger.info("Variants README written to %s", readme_path)
        _log_stage_end(log_event_fn, 8, "Generate README", stage_started_at)
    else:
        logger.info("=== Stage 8: Skipped ===")

    logger.info("Pipeline finished (stages: %s)", sorted(stages))
    return 0


def _log_stage_start(
    log_event_fn: EventFn | None,
    stage: int,
    label: str,
) -> float:
    started_at = time.monotonic()
    if log_event_fn is not None:
        log_event_fn(
            "stage_start",
            stage=stage,
            label=f"Stage {stage}: {label}",
            current=stage + 1,
            total=9,
        )
    return started_at


def _log_stage_end(
    log_event_fn: EventFn | None,
    stage: int,
    label: str,
    started_at: float,
    *,
    status: str = "completed",
    **fields: object,
) -> None:
    if log_event_fn is None:
        return
    log_event_fn(
        "stage_end",
        stage=stage,
        label=f"Stage {stage}: {label}",
        status=status,
        elapsed=format_elapsed(time.monotonic() - started_at),
        **fields,
    )


def _load_candidates_from_disk(generation_dir: Path) -> list:
    """Load candidates from a previous generation stage's variant_bundle.json."""
    from .generate import _parse_variant_bundle

    bundle_path = generation_dir / "variant_bundle.json"
    if not bundle_path.exists():
        logger.warning("No variant_bundle.json found; using empty candidates list")
        return []
    raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    _, candidates = _parse_variant_bundle(raw)
    return candidates


def _load_selection_from_disk(generation_dir: Path, candidates: list) -> VariantSelection:
    """Load selection from a previous selection stage's variant_selection.json."""
    sel_path = generation_dir / "variant_selection.json"
    if not sel_path.exists():
        logger.warning("No variant_selection.json found; using all candidates as accepted")
        return VariantSelection(accepted=candidates, rejected=[], controls=["control_best"])

    raw = json.loads(sel_path.read_text(encoding="utf-8"))
    accepted_ids = {e["variant_id"] for e in raw.get("accepted", [])}
    accepted = [c for c in candidates if c.variant_id in accepted_ids]
    rejected_ids = {e["variant_id"]: e.get("reason", "") for e in raw.get("rejected", [])}
    rejected = [
        (c, rejected_ids.get(c.variant_id, "")) for c in candidates if c.variant_id in rejected_ids
    ]
    controls = raw.get("controls", ["control_best"])
    return VariantSelection(accepted=accepted, rejected=rejected, controls=controls)


def _load_variant_specs_from_disk(output_dir: Path) -> list[VariantSpec]:
    """Load variant specs from existing variant directories."""
    variants_dir = output_dir / "variants"
    if variants_dir.is_symlink() or not variants_dir.is_dir():
        return []

    specs: list[VariantSpec] = []
    for vdir in sorted(variants_dir.iterdir()):
        if vdir.is_symlink() or not vdir.is_dir():
            continue
        data = _read_variant_spec_nofollow(vdir)
        if data is None:
            continue
        if data.get("variant_id") != vdir.name:
            logger.warning("Skipping mismatched variant ID in %s", vdir)
            continue
        try:
            specs.append(
                VariantSpec(
                    variant_id=data["variant_id"],
                    title=data.get("title", ""),
                    kind=data.get("kind", ""),
                    base_program=data.get("base_program", ""),
                    variant_dir=vdir,
                    hypothesis=data.get("hypothesis", ""),
                    relevance_reason=data.get("relevance_reason", ""),
                    feasibility_reason=data.get("feasibility_reason", ""),
                    target_components=data.get("target_components", []),
                    implementation_plan=data.get("implementation_plan", []),
                    expected_effect=data.get("expected_effect", ""),
                )
            )
        except KeyError as exc:
            logger.warning("Skipping invalid variant_spec.json in %s: %s", vdir, exc)

    return sorted(specs, key=lambda spec: spec.kind != "control")


def _read_variant_spec_nofollow(variant_dir: Path) -> dict[str, Any] | None:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        directory_fd = os.open(variant_dir, directory_flags)
    except OSError:
        return None
    try:
        try:
            file_fd = os.open("variant_spec.json", file_flags, dir_fd=directory_fd)
        except OSError:
            return None
        try:
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                return None
            with os.fdopen(file_fd, "r", encoding="utf-8") as handle:
                file_fd = -1
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        finally:
            if file_fd >= 0:
                os.close(file_fd)
    finally:
        os.close(directory_fd)
    return payload if isinstance(payload, dict) else None


def _resolve_output_dir(cfg: VariantsPipelineConfig) -> Path:
    """Resolve the output directory for this pipeline run."""
    if cfg.paths.output_dir:
        return Path(cfg.paths.output_dir).expanduser().resolve()

    task_name = cfg.paths.task_name or Path(cfg.paths.task_dir).name
    return (
        Path(cfg.paths.results_dir).expanduser().resolve()
        / "variants"
        / f"{task_name}_{cfg.agent.name}_{cfg.run_id}"
    )


def _write_pipeline_summary(
    output_dir: Path,
    variant_specs: list[VariantSpec],
    results: list[VariantRunResult],
) -> None:
    """Write a ``pipeline_summary.json`` to the output directory."""
    results_by_id = {r.variant_id: r for r in results}
    variant_entries: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": 1,
        "variant_specs_sha256": _variant_specs_digest(variant_specs),
        "total_variants": len(variant_specs),
        "total_completed": sum(1 for r in results if r.status == "completed"),
        "total_failed": sum(1 for r in results if r.status == "failed"),
        "total_timeout": sum(1 for r in results if r.status == "timeout"),
        "rubric_passed": sum(1 for r in results if r.rubric_passed),
        "variants": variant_entries,
    }

    for spec in variant_specs:
        result = results_by_id.get(spec.variant_id)
        entry: dict[str, Any] = {
            "variant_id": spec.variant_id,
            "kind": spec.kind,
            "title": spec.title,
        }
        if result:
            entry.update(
                {
                    "status": result.status,
                    "rubric_passed": result.rubric_passed,
                    "eval_count": result.eval_count,
                    "exit_code": result.exit_code,
                    "analysis_files": result.analysis_files,
                }
            )
        else:
            entry["status"] = "not_executed"
        variant_entries.append(entry)

    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Pipeline summary written to %s", summary_path)


def _validate_pipeline_summary(
    output_dir: Path,
    variant_specs: list[VariantSpec],
) -> str | None:
    """Validate a prior host-written execution summary and canonical artifacts."""
    summary_path = output_dir / "pipeline_summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        return "pipeline_summary.json is missing or unsafe"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "pipeline_summary.json is unreadable or malformed"
    if not isinstance(summary, dict) or summary.get("schema_version") != 1:
        return "pipeline_summary.json has an unsupported schema"
    if summary.get("variant_specs_sha256") != _variant_specs_digest(variant_specs):
        return "pipeline_summary.json does not match the materialized variants"
    entries = summary.get("variants")
    if not isinstance(entries, list):
        return "pipeline_summary.json has no valid variants list"
    entries_by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("variant_id"), str):
            return "pipeline_summary.json contains an invalid variant entry"
        variant_id = entry["variant_id"]
        if variant_id in entries_by_id:
            return f"pipeline_summary.json contains duplicate variant {variant_id!r}"
        entries_by_id[variant_id] = entry
    expected_ids = {spec.variant_id for spec in variant_specs}
    if set(entries_by_id) != expected_ids:
        return "pipeline_summary.json variant IDs are stale or incomplete"

    from .execute import _inspect_program_results
    from .rubric_contract import build_rubric_contract

    passed = 0
    for spec in variant_specs:
        entry = entries_by_id[spec.variant_id]
        if entry.get("status") not in {"completed", "failed", "timeout"}:
            return f"variant {spec.variant_id!r} has an invalid execution status"
        rubric_passed = entry.get("rubric_passed")
        if not isinstance(rubric_passed, bool):
            return f"variant {spec.variant_id!r} has no canonical rubric result"
        required_rubrics = [
            rubric["id"]
            for rubric in build_rubric_contract(
                spec.variant_id,
                spec.kind,
                spec.base_program,
            )["rubrics"]
        ]
        _, artifacts_passed = _inspect_program_results(
            spec.variant_dir / "task_bundle",
            spec.variant_id,
            required_rubrics,
        )
        if rubric_passed != artifacts_passed:
            return f"variant {spec.variant_id!r} summary disagrees with canonical artifacts"
        if spec.kind == "control" and not rubric_passed:
            return f"required control {spec.variant_id!r} failed verification"
        passed += int(rubric_passed)
    if passed == 0:
        return "no variant produced a valid evaluation"
    if summary.get("rubric_passed") != passed:
        return "pipeline_summary.json pass count is inconsistent"
    return None


def _variant_specs_digest(variant_specs: list[VariantSpec]) -> str:
    payload = json.dumps(
        [spec.to_dict() for spec in variant_specs],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _failed_control_ids(
    variant_specs: list[VariantSpec],
    results: list[VariantRunResult],
) -> list[str]:
    """Return controls that ran without producing a valid verification."""
    control_ids = {spec.variant_id for spec in variant_specs if spec.kind == "control"}
    return [
        result.variant_id
        for result in results
        if result.variant_id in control_ids and not result.rubric_passed
    ]

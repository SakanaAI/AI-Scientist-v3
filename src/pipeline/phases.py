"""Command builders and subprocess runner for the pipeline phases.

The builders are pure functions returning ``list[str]`` argv, so they are trivial
to unit-test (golden argv) and the dry-run path can print exact, copy-pasteable
commands. The stages are shelled out (not imported) so each keeps its own Hydra
init and process isolation.
"""

from __future__ import annotations

import logging
import os
import signal
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping

from src.orchestrator.paper_event_bridge import extract_transport_events
from src.investigation.runtime import terminate_active_process_group

from .config import (
    PaperConfig,
    PhasePassthrough,
    PipelineConfig,
    phase_uses_container,
    venue_file_for_format,
)
from .paper_workspace import workspace_validation_errors
from .paths import PipelinePaths

logger = logging.getLogger(__name__)

# Repo root = two levels up from this file (src/pipeline/stages.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STREAM_TERMINATION_GRACE_SECONDS = 5


def _bool(value: bool) -> str:
    """Render a bool as a Hydra/OmegaConf literal."""
    return "true" if value else "false"


def _hydra_quote(value: str) -> str:
    """Single-quote a value so Hydra's override parser treats it as a string.

    Needed for values containing characters special to the override grammar,
    e.g. a model like ``claude-opus-4-6[1m]`` (``[`` starts a list) or a
    timestamp like ``20260101_000000`` (underscore numeric separator -> int).
    We pass argv directly to ``subprocess.run`` (no shell), so the quotes must
    be part of the argv element itself for Hydra to see them.
    """
    encoded: list[str] = []
    index = 0
    while index < len(value):
        if value[index] == "\\":
            end = index
            while end < len(value) and value[end] == "\\":
                end += 1
            backslash_count = end - index
            if end == len(value):
                encoded.append("\\" * (2 * backslash_count))
            elif value[end] == "'":
                encoded.append("\\" * (2 * backslash_count + 1))
                encoded.append("'")
                end += 1
            else:
                encoded.append("\\" * backslash_count)
            index = end
            continue
        if value[index] == "'":
            encoded.append("\\'")
        else:
            encoded.append(value[index])
        index += 1
    return "'" + "".join(encoded) + "'"


def _hydra_string_override(key: str, value: object) -> str:
    """Build a Hydra override whose value round-trips as a string."""
    return f"{key}={_hydra_quote(str(value))}"


def _hydra_string_list_override(key: str, values: list[str]) -> str:
    """Build a Hydra list override with independently quoted string items."""
    encoded = ",".join(_hydra_quote(value) for value in values)
    return f"{key}=[{encoded}]"


def _render_hydra_override(key: str, value: object) -> str:
    """Render one forwarded stage value as a Hydra override."""
    if value is None:
        return f"{key}=null"
    if isinstance(value, bool):
        return f"{key}={_bool(value)}"
    if isinstance(value, (int, float)):
        return f"{key}={value}"
    if isinstance(value, str):
        return _hydra_string_override(key, value)
    if isinstance(value, list):
        return _hydra_string_list_override(key, [str(item) for item in value])
    raise TypeError(f"{key}: cannot forward a {type(value).__name__} value to a stage")


def hydra_overrides_for(
    phase: PhasePassthrough,
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> list[str]:
    """Overrides for every stage key the pipeline changes and does not own.

    Keys equal to the stage's own default are not repeated; the stage loads
    the same configuration either way.
    """
    forwarded = phase.forwarded_excluding(*exclude_prefixes)
    return [_render_hydra_override(key, value) for key, value in forwarded.items()]


def paper_flags_for(
    phase: PaperConfig,
    *,
    exclude: frozenset[str] = frozenset(),
) -> list[str]:
    """paper.sh flags for every mirrored key the pipeline changes and does not own.

    Keys paper.sh would otherwise read from the environment are always passed,
    first, so that a later flag (``--standalone-paper`` after ``--paper-profile``)
    keeps the precedence paper.sh gives it. ``exclude`` names keys a sibling
    script (paper_edit.sh) does not take or must not receive.
    """
    values = {key: phase.get(key) for key in sorted(phase.always_forwarded)}
    values.update(phase.forwarded)
    flags: list[str] = []
    for key, value in values.items():
        if key in exclude:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        elif value is not None:
            flags += [flag, str(value)]
    return flags


def _container_auth_root(cfg: PipelineConfig) -> str:
    """Expand home syntax once so every child phase receives the same path."""
    return str(Path(cfg.container.auth_root).expanduser())


def _container_env_file(cfg: PipelineConfig) -> str | None:
    """Expand home syntax once for consistent handling across child phases."""
    if cfg.container.env_file is None:
        return None
    if cfg.container.env_file == "":
        return ""
    return str(Path(str(cfg.container.env_file)).expanduser())


def _container_overrides(cfg: PipelineConfig) -> list[str]:
    """Hydra-style container overrides for the investigation and variants CLIs."""
    overrides = [f"container.enabled={_bool(cfg.container.enabled)}"]
    if cfg.container.enabled:
        overrides.append(_hydra_string_override("container.image", cfg.container.image))
        overrides.append(_hydra_string_override("container.network", cfg.container.network))
        overrides.append(_hydra_string_override("container.auth_root", _container_auth_root(cfg)))
        env_file = _container_env_file(cfg)
        if env_file:
            overrides.append(_hydra_string_override("container.env_file", env_file))
    return overrides


def _posthoc_container_overrides(cfg: PipelineConfig) -> list[str]:
    """Container overrides with the post-hoc phase's narrower network policy."""
    overrides = _container_overrides(cfg)
    network_prefix = "container.network="
    return [
        _hydra_string_override("container.network", cfg.posthoc.container_network)
        if override.startswith(network_prefix)
        else override
        for override in overrides
    ]


def _container_flags(cfg: PipelineConfig) -> list[str]:
    """paper.sh-style CLI flags mirroring the container config."""
    return _docker_flags(cfg) if cfg.container.enabled else []


def _paper_container_flags(cfg: PipelineConfig) -> list[str]:
    """paper.sh container flags; ``container.paper_enabled`` can differ from the run."""
    return _docker_flags(cfg) if phase_uses_container(cfg, "paper") else []


def _docker_flags(cfg: PipelineConfig) -> list[str]:
    flags = [
        "--containerize",
        "--container-image",
        cfg.container.image,
        "--container-network",
        cfg.container.network,
        "--container-auth-root",
        _container_auth_root(cfg),
    ]
    env_file = _container_env_file(cfg)
    if env_file is not None:
        flags += ["--container-env-file", env_file]
    return flags


def _logging_overrides(cfg: PipelineConfig) -> list[str]:
    """Forward non-default logging settings to Hydra child processes."""
    overrides = ["logging.show_run_header=false"]
    if cfg.logging.format != "human":
        overrides.append(f"logging.format={cfg.logging.format}")
    if cfg.logging.verbose:
        overrides.append("logging.verbose=true")
    if cfg.logging.heartbeat_seconds != 60:
        overrides.append(f"logging.heartbeat_seconds={cfg.logging.heartbeat_seconds}")
    return overrides


def _paper_logging_flags(cfg: PipelineConfig) -> list[str]:
    """Forward non-default logging settings to the paper shell entrypoint."""
    flags = ["--event-transport", "wrapper"]
    if cfg.logging.format != "human":
        flags += ["--log-format", cfg.logging.format]
    if cfg.logging.verbose:
        flags.append("--verbose")
    if cfg.logging.heartbeat_seconds != 60:
        flags += ["--heartbeat-seconds", str(cfg.logging.heartbeat_seconds)]
    return flags


def forward_paper_output(line: str, *, event_fn: Callable[..., None]) -> str:
    """Forward trusted paper events and return ordinary child output."""
    retained, records = extract_transport_events(line)
    for record in records:
        event_fn(
            "paper_gate",
            gate=record["gate"],
            state=record["state"],
            paper_session_id=record["session_id"],
            paper_timestamp=record["timestamp"],
            paper_fields=record["fields"],
        )
    if retained.startswith("Paper directory: "):
        return ""
    return retained


def _investigation_cmd_prefix(cfg: PipelineConfig, paths: PipelinePaths, results_dir: Path) -> list[str]:
    """The derived agent, path, and run-id overrides shared by both investigation CLI commands."""
    cmd = [
        sys.executable,
        "-m",
        "src.investigation.cli",
        _hydra_string_override("agent.name", cfg.agent.name),
    ]
    if cfg.agent.model:
        cmd.append(_hydra_string_override("agent.model", cfg.agent.model))
    if cfg.agent.profile is not None:
        cmd.append(_hydra_string_override("agent.profile", cfg.agent.profile))
    if cfg.agent.reasoning_effort is not None:
        cmd.append(_hydra_string_override("agent.reasoning_effort", cfg.agent.reasoning_effort))
    cmd += [
        _hydra_string_override("paths.task_dir", paths.task_dir),
        _hydra_string_override("paths.results_dir", results_dir),
        # Quote so the investigation CLI's Hydra keeps the timestamp as a string
        # (an unquoted value like 20260101_000000 parses as an int).
        _hydra_string_override("run_id", paths.run_id),
    ]
    return cmd


def build_investigation_cmd(cfg: PipelineConfig, paths: PipelinePaths) -> list[str]:
    """Build the investigation/scientist command (``src.investigation.cli``).

    Passes both an explicit ``paths.results_dir`` and ``run_id`` so the output
    location is known up front (no log parsing needed downstream). Every other
    investigation key the pipeline changes is forwarded generically; drafting budgets
    belong to the task drafting command only.
    """
    cmd = _investigation_cmd_prefix(cfg, paths, paths.investigation_workspace_dir)
    cmd += hydra_overrides_for(cfg.investigation, exclude_prefixes=("task_drafting.",))
    cmd.append("execution.draft_task_if_missing=false")
    cmd += _container_overrides(cfg)
    if cfg.container.gpu_count > 0:
        # Parallel workers slice the run's GPU pool by this width.
        cmd.append(f"container.gpu_count={cfg.container.gpu_count}")
    cmd += _logging_overrides(cfg)
    cmd += _versioning_overrides(cfg, paths)
    return cmd


def _ablation_deadline_overrides(cfg: PipelineConfig) -> list[str]:
    """Variant and cross-task deadlines; ``null`` keeps the variants default.

    Only a value that differs from the variants pipeline's own default is forwarded.
    """
    overrides: list[str] = []
    baseline = cfg.ablations.baseline
    for key in ("execution.variant_timeout_seconds", "execution.cross_task_timeout_seconds"):
        timeout = cfg.ablations.get(key)
        if timeout is None:
            continue
        if timeout != baseline[key]:
            overrides.append(f"{key}={timeout}")
    return overrides


_DRAFTING_DERIVED_KEYS = (
    "execution.resume",
    "execution.workers",
    "execution.run_timeout_seconds",
)


def build_task_drafting_cmd(cfg: PipelineConfig, paths: PipelinePaths) -> list[str]:
    """Build the task drafting/runtime command without starting ideation."""
    cmd = _investigation_cmd_prefix(cfg, paths, paths.task_drafting_output_dir)
    cmd += hydra_overrides_for(cfg.investigation, exclude_prefixes=_DRAFTING_DERIVED_KEYS)
    cmd += [
        "execution.workers=1",
        "execution.draft_task_if_missing=true",
        f"execution.run_timeout_seconds={cfg.task_drafting.timeout_seconds}",
        "execution.task_only=true",
    ]
    cmd += _container_overrides(cfg)
    cmd += _logging_overrides(cfg)
    cmd += _versioning_overrides(cfg, paths)
    return cmd


def _versioning_overrides(cfg: PipelineConfig, paths: PipelinePaths) -> list[str]:
    """Hydra-style checkpoint overrides for the investigation CLI.

    ``run_root`` pins the child to the pipeline's own run root, which is what
    makes parallel workers no-op: their ``paths.results_dir`` is a subtree.
    """
    overrides = [f"versioning.enabled={_bool(cfg.versioning.enabled)}"]
    if cfg.versioning.enabled:
        overrides.append(_hydra_string_override("versioning.run_root", paths.run_dir))
        overrides.append(
            _hydra_string_override("versioning.git_dir_name", cfg.versioning.git_dir_name)
        )
        overrides.append(f"versioning.max_file_bytes={cfg.versioning.max_file_bytes}")
        overrides.append(f"versioning.max_total_bytes={cfg.versioning.max_total_bytes}")
        overrides.append(f"versioning.timeout_seconds={cfg.versioning.timeout_seconds}")
        overrides.append(f"versioning.tag_checkpoints={_bool(cfg.versioning.tag_checkpoints)}")
        overrides.append(
            "versioning.final_include_environments="
            f"{_bool(cfg.versioning.final_include_environments)}"
        )
        overrides.append("versioning.finalize_on_exit=false")
    return overrides


def build_ablations_cmd(cfg: PipelineConfig, paths: PipelinePaths) -> list[str]:
    """Build the baselines and ablations command.

    An explicit ``paths.output_dir`` is always passed, so the run id does not
    need threading here (the variants config has no ``run_id`` key to override).
    """
    cmd = [sys.executable, "-m", "src.variants_pipeline.cli"]
    cmd += [
        _hydra_string_override("paths.task_dir", paths.task_dir),
        _hydra_string_override("paths.task_name", paths.task_name),
        _hydra_string_override("paths.results_dir", paths.investigation_workspace_dir),
        _hydra_string_override("paths.output_dir", paths.ablations_dir),
        _hydra_string_override("agent.name", cfg.agent.name),
    ]
    if cfg.agent.model:
        cmd.append(_hydra_string_override("agent.model", cfg.agent.model))
    if cfg.agent.profile is not None:
        cmd.append(_hydra_string_override("agent.profile", cfg.agent.profile))
    if cfg.agent.reasoning_effort is not None:
        cmd.append(_hydra_string_override("agent.reasoning_effort", cfg.agent.reasoning_effort))
    cmd += hydra_overrides_for(cfg.ablations)
    cmd += _ablation_deadline_overrides(cfg)
    cmd.append(
        _hydra_string_override(
            "execution.sandbox_image",
            cfg.ablations.sandbox_image or cfg.container.image,
        )
    )
    cmd += _container_overrides(cfg)
    cmd += _logging_overrides(cfg)
    return cmd


def build_final_cmd(
    cfg: PipelineConfig,
    paths: PipelinePaths,
    *,
    image_recovery_ref: str | None = None,
) -> list[str]:
    """Build the one-time contract-declared final evaluation command."""
    command = [
        sys.executable,
        "-m",
        "src.final_evaluation.cli",
        "--task-dir",
        str(paths.task_dir),
        "--investigation-results-dir",
        str(paths.investigation_workspace_dir),
        "--output-dir",
        str(paths.final_dir),
        "--python-bin",
        str(paths.task_dir / ".venv" / "bin" / "python"),
        "--container-image",
        cfg.container.image,
    ]
    if image_recovery_ref is not None:
        command.extend(("--container-image-recovery-ref", image_recovery_ref))
    command.extend(("--timeout-seconds", str(cfg.final.timeout_seconds)))
    return command


def build_posthoc_cmd(cfg: PipelineConfig, paths: PipelinePaths) -> list[str]:
    """Build artifact-only post-hoc analysis command."""
    cmd = [
        sys.executable,
        "-m",
        "src.posthoc_pipeline.cli",
        _hydra_string_override("paths.task_dir", paths.task_dir),
        _hydra_string_override("paths.results_dir", paths.investigation_workspace_dir),
        _hydra_string_override("paths.evidence_dir", paths.ablations_dir),
        _hydra_string_override("paths.output_dir", paths.posthoc_dir),
        _hydra_string_override("agent.name", cfg.agent.name),
    ]
    if cfg.agent.model:
        cmd.append(_hydra_string_override("agent.model", cfg.agent.model))
    if cfg.agent.profile is not None:
        cmd.append(_hydra_string_override("agent.profile", cfg.agent.profile))
    if cfg.agent.reasoning_effort is not None:
        cmd.append(_hydra_string_override("agent.reasoning_effort", cfg.agent.reasoning_effort))
    cmd.append(_hydra_string_override("run_id", paths.run_id))
    cmd += hydra_overrides_for(cfg.posthoc)
    cmd += [
        "execution.checkpoint_mode='optional'",
        "execution.retrain_if_checkpoint_missing=false",
    ]
    cmd += _posthoc_container_overrides(cfg)
    cmd += _logging_overrides(cfg)
    return cmd


def _venue_flags(cfg: PipelineConfig) -> list[str]:
    """Resolve the venue brief for paper.sh.

    ``None`` selects the brief matching ``paper.format``; an empty string opts
    out (paper.sh reads ``--venue-file ""`` as "no brief"); any other value is
    forwarded as an explicit path.
    """
    if cfg.paper.venue_file is None:
        return ["--venue-file", venue_file_for_format(cfg.paper.format)]
    return ["--venue-file", cfg.paper.venue_file]


def build_paper_cmd(cfg: PipelineConfig, paths: PipelinePaths) -> list[str]:
    """Build the paper-writing command."""
    return _build_paper_cmd(
        cfg,
        paths,
        posthoc_dir=None,
        final_evaluation_dir=paths.final_dir,
    )


def build_paper_cmd_with_posthoc(cfg: PipelineConfig, paths: PipelinePaths) -> list[str]:
    """Build paper command with a validated post-hoc run as input."""
    return _build_paper_cmd(
        cfg,
        paths,
        posthoc_dir=paths.posthoc_dir,
        final_evaluation_dir=paths.final_dir,
    )


def build_paper_cmd_without_final(
    cfg: PipelineConfig,
    paths: PipelinePaths,
    *,
    posthoc_dir: Path | None,
) -> list[str]:
    """Build a paper command that excludes optional non-qualifying final evidence."""
    command = _build_paper_cmd(
        cfg,
        paths,
        posthoc_dir=posthoc_dir,
        final_evaluation_dir=None,
    )
    command.append("--no-final-evaluation")
    return command


def _build_paper_cmd(
    cfg: PipelineConfig,
    paths: PipelinePaths,
    *,
    posthoc_dir: Path | None,
    final_evaluation_dir: Path | None,
) -> list[str]:
    """Build the paper-writing command."""
    cmd = [
        "bash",
        "src/scripts/paper.sh",
        "--agent",
        cfg.agent.name,
    ]
    if cfg.agent.model:
        cmd += ["--agent-model", cfg.agent.model]
    if cfg.agent.profile is not None:
        cmd += ["--agent-profile", cfg.agent.profile]
    if cfg.agent.reasoning_effort is not None:
        cmd += ["--reasoning-effort", cfg.agent.reasoning_effort]
    cmd += [
        "--task-dir",
        str(paths.task_dir),
        "--results-dir",
        str(paths.investigation_dir),
        "--ablations-dir",
        str(paths.ablations_dir),
        "--global-memory-dir",
        str(paths.investigation_dir),
        "--output-dir",
        str(paths.paper_dir),
        "--manifest-dir",
        str(paths.paper_dir),
    ]
    if final_evaluation_dir is not None:
        cmd += ["--final-evaluation-dir", str(final_evaluation_dir)]
    cmd += [
        "--contract-task-dir",
        str(paths.task_dir),
        "--investigation-results-dir",
        str(paths.investigation_dir),
    ]
    if posthoc_dir is not None:
        cmd += ["--posthoc-dir", str(posthoc_dir)]
    if paths.paper_dir.exists() and not workspace_validation_errors(
        output_dir=paths.paper_dir,
        task_dir=paths.task_dir,
        results_dir=paths.investigation_dir,
        global_memory_dir=paths.investigation_dir,
        ablations_dir=paths.ablations_dir,
    ):
        cmd.append("--resume-existing")
    cmd += paper_flags_for(cfg.paper)
    cmd += ["--final-outcome-policy", cfg.paper.final_outcome_policy]
    cmd += _venue_flags(cfg)
    cmd += _paper_container_flags(cfg)
    cmd += _paper_logging_flags(cfg)
    return cmd


def run_cmd(
    argv: list[str],
    *,
    dry_run: bool,
    cwd: Path = REPO_ROOT,
    env_overrides: Mapping[str, str] | None = None,
    output_line_fn: Callable[[str], str] | None = None,
) -> int:
    """Run *argv* from *cwd*. In dry-run mode, log the command and return 0."""
    logger.info("$ %s", shlex.join(argv))
    if dry_run:
        return 0
    environment = os.environ | dict(env_overrides) if env_overrides else None
    if output_line_fn is not None:
        streamed_proc = subprocess.Popen(  # noqa: S603 - trusted argv
            argv,
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        assert streamed_proc.stdout is not None
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        termination_started = False

        def terminate_streamed_process() -> None:
            nonlocal termination_started
            if termination_started:
                return
            termination_started = True
            terminate_active_process_group(
                streamed_proc,
                grace_seconds=_STREAM_TERMINATION_GRACE_SECONDS,
            )

        def handle_sigterm(signum: int, _frame: object) -> None:
            if termination_started:
                return
            terminate_streamed_process()
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, handle_sigterm)
        try:
            for line in streamed_proc.stdout:
                retained = output_line_fn(line)
                if retained:
                    sys.stdout.write(retained)
                    sys.stdout.flush()
            return streamed_proc.wait()
        except BaseException:
            terminate_streamed_process()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            streamed_proc.stdout.close()
    completed = subprocess.run(  # noqa: S603 - trusted argv
        argv,
        cwd=str(cwd),
        env=environment,
    )
    return completed.returncode

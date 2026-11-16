"""Phase command construction, forwarded stage config, presets, completion receipts, image leases.

The effective-stage-config golden fixture is regenerated (only when an effective stage config
must change; explain why in the commit body) with::

    PIPELINE_EFFECTIVE_CONFIG_REGEN=1 ./.venv/bin/pytest -q tests/pipeline/test_pipeline_phases.py -k effective_stage
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.errors import HydraException

import src.pipeline.cli as pipeline_cli
from src.investigation.container_images import DockerImageLease
from src.pipeline import phases
from src.pipeline.cli import (
    _image_lease_tag,
    _pin_evaluation_images,
    _release_image_leases,
    _verify_image_leases,
    load_pipeline_config,
)
from src.pipeline.config import (
    PipelineConfig,
    completed_investigation_outcome,
    investigation_completion_payload,
    investigation_completion_receipt_path,
    validate_completed_investigation,
    validate_completed_investigation_for_downstream,
    validate_pipeline_config,
)
from src.pipeline.paper_workspace import write_workspace_marker
from src.pipeline.paths import derive_paths
from src.pipeline.phases import (
    build_ablations_cmd,
    build_final_cmd,
    build_investigation_cmd,
    build_paper_cmd,
    build_paper_cmd_with_posthoc,
    build_paper_cmd_without_final,
    build_posthoc_cmd,
    build_task_drafting_cmd,
    forward_paper_output,
    run_cmd,
)
from src.variants_pipeline.cli import load_variants_config
from tests.paper_script_support import apply_flags

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20260101_000000"
CIRCPACK = "task.dir=examples/circpack"
AGENT_ID = "sha256:" + "a" * 64
SANDBOX_ID = "sha256:" + "b" * 64


# --------------------------------------------------------------------------- helpers


def _cfg(**kw: object) -> PipelineConfig:
    """Fully specified containerized config; ``kw`` keys are ``section.field`` or top-level."""
    cfg = PipelineConfig()
    cfg.task.dir = "examples/circpack"
    cfg.agent.name = "claude"
    cfg.agent.model = "claude-opus-4-6[1m]"
    cfg.agent.reasoning_effort = "high"
    cfg.run_id = RUN_ID
    cfg.container.enabled = True
    cfg.container.image = "agentic-scientist/agent-cli:latest"
    cfg.container.network = "host"
    cfg.container.auth_root = "/tmp/auth-root"
    cfg.container.env_file = ".env"
    for key, value in kw.items():
        head, _, tail = key.partition(".")
        if tail:
            setattr(getattr(cfg, head), tail, value)
        else:
            setattr(cfg, head, value)
    return cfg


def _loaded(*overrides: str) -> PipelineConfig:
    cfg = load_pipeline_config([CIRCPACK, *overrides])
    cfg.run_id = RUN_ID
    return cfg


def _flag(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _override(command: list[str], key: str) -> str:
    return next(arg for arg in command if arg.startswith(f"{key}="))


def _parsed(command: list[str], key: str) -> object:
    return OverridesParser.create().parse_override(_override(command, key)).value()


def _parse_all(command: list[str]) -> dict[str, object]:
    parsed = OverridesParser.create().parse_overrides([arg for arg in command if "=" in arg])
    return {override.key_or_group: override.value() for override in parsed}


@dataclass
class FakeLease:
    image_id: str
    tag: str = "pipeline:test"
    verify_error: ValueError | None = None
    release_error: Exception | None = None
    verified: bool = False
    released: bool = False

    def verify(self) -> None:
        self.verified = True
        if self.verify_error is not None:
            raise self.verify_error

    def release(self) -> None:
        self.released = True
        if self.release_error is not None:
            raise self.release_error


def _write_program(program_dir: Path) -> Path:
    entrypoint = program_dir / "src" / "program.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("", encoding="utf-8")
    return program_dir


def _write_manifest(workspace: Path, program_dir: Path, **fields: object) -> None:
    (workspace / "best_program_manifest.json").write_text(
        json.dumps({"program_dir": str(program_dir), "entrypoint_path": "src/program.py", **fields}),
        encoding="utf-8",
    )


def _write_receipt(workspace: Path, **override: object) -> dict[str, Any]:
    receipt = investigation_completion_payload(workspace)
    assert receipt is not None
    investigation_completion_receipt_path(workspace).write_text(
        json.dumps(receipt | override), encoding="utf-8"
    )
    return receipt


# --------------------------------------------------------------------------- investigation / drafting / final commands


def test_investigation_cmd_and_hydra_round_trip() -> None:
    # investigation command
    cfg = _cfg()
    cfg.investigation.max_idea_generation_attempts = 1
    cfg.investigation.idea_timeout_seconds = 600
    cfg.investigation.run_timeout_seconds = 600
    paths = derive_paths(cfg)
    assert build_investigation_cmd(cfg, paths) == [
        sys.executable,
        "-m",
        "src.investigation.cli",
        "agent.name='claude'",
        "agent.model='claude-opus-4-6[1m]'",
        "agent.reasoning_effort='high'",
        f"paths.task_dir='{paths.task_dir}'",
        f"paths.results_dir='{paths.investigation_workspace_dir}'",
        "run_id='20260101_000000'",
        # Only keys that differ from the investigation CLI's own defaults are forwarded.
        "execution.idea_timeout_seconds=600",
        "execution.max_idea_cycles=3",
        "execution.max_idea_generation_attempts=1",
        "execution.max_program_evals_per_attempt=1",
        "execution.run_timeout_seconds=600",
        "execution.draft_task_if_missing=false",
        "container.enabled=true",
        "container.image='agentic-scientist/agent-cli:latest'",
        "container.network='host'",
        "container.auth_root='/tmp/auth-root'",
        "container.env_file='.env'",
        "logging.show_run_header=false",
        "versioning.enabled=true",
        f"versioning.run_root='{paths.investigation_results_dir}'",
        "versioning.git_dir_name='.agentic_history'",
        "versioning.max_file_bytes=0",
        "versioning.max_total_bytes=17179869184",
        "versioning.timeout_seconds=600",
        "versioning.tag_checkpoints=true",
        "versioning.final_include_environments=false",
        "versioning.finalize_on_exit=false",
    ]

    # Values with special chars (``[1m]`` model, timestamp run_id) must survive Hydra's parser.
    cfg = _cfg()
    cmd = build_investigation_cmd(cfg, derive_paths(cfg))
    parsed = OverridesParser.create().parse_overrides(cmd[3:])  # raises on any bad override
    by_key = {p.key_or_group: p.value() for p in parsed}
    assert by_key["agent.model"] == "claude-opus-4-6[1m]"
    assert by_key["agent.reasoning_effort"] == "high"
    assert by_key["run_id"] == "20260101_000000"


def test_task_drafting_cmd_uses_separate_output_and_stops_before_ideation() -> None:
    cfg = _cfg()
    cfg.investigation.draft_task_if_missing = True
    cfg.investigation.workers = 8
    cfg.investigation.resume = True
    cfg.investigation.set("task_drafting.design_max_repairs", 2)
    cfg.investigation.set("task_drafting.workflow", "legacy")
    cfg.task_drafting.timeout_seconds = 100
    paths = derive_paths(cfg)

    cmd = build_task_drafting_cmd(cfg, paths)

    assert "execution.task_only=true" in cmd
    assert "logging.show_run_header=false" in cmd
    assert "execution.draft_task_if_missing=true" in cmd
    assert "execution.workers=1" in cmd
    assert "execution.workers=8" not in cmd
    assert "execution.resume=true" not in cmd
    assert "execution.run_timeout_seconds=100" in cmd
    assert f"paths.results_dir='{paths.task_drafting_dir}'" in cmd
    assert f"paths.results_dir='{paths.investigation_dir}'" not in cmd
    # Drafting budgets that differ from the CLI defaults reach the drafting command only.
    assert "task_drafting.design_max_repairs=2" in cmd
    assert "task_drafting.workflow='legacy'" in cmd
    assert not any(argument.startswith("task_drafting.contract_max_repairs") for argument in cmd)
    investigation_cmd = build_investigation_cmd(cfg, paths)
    assert not any(argument.startswith("task_drafting.") for argument in investigation_cmd)
    assert "execution.resume=true" in investigation_cmd
    assert "execution.workers=8" in investigation_cmd

    # the pipeline routes task drafting to its own command from a loaded config
    cfg = _loaded("investigation.execution.draft_task_if_missing=true")
    paths = derive_paths(cfg)
    command = build_task_drafting_cmd(cfg, paths)
    investigation = build_investigation_cmd(cfg, paths)
    assert cfg.investigation.draft_task_if_missing is True
    assert "execution.draft_task_if_missing=true" in command
    assert f"paths.results_dir='{paths.task_drafting_dir}'" in command
    assert "execution.draft_task_if_missing=false" in investigation


def test_final_cmd_uses_contract_evaluator_and_forwards_image_recovery_reference() -> None:
    cfg = _cfg()
    cfg.final.timeout_seconds = 900
    paths = derive_paths(cfg)
    assert build_final_cmd(cfg, paths) == [
        sys.executable, "-m", "src.final_evaluation.cli",
        "--task-dir", str(paths.task_dir),
        "--investigation-results-dir", str(paths.investigation_workspace_dir),
        "--output-dir", str(paths.final_output_dir),
        "--python-bin", str(paths.task_dir / ".venv" / "bin" / "python"),
        "--container-image", cfg.container.image,
        "--timeout-seconds", "900",
    ]

    # the owned image recovery reference is forwarded
    cfg = _cfg()
    cmd = build_final_cmd(cfg, derive_paths(cfg), image_recovery_ref="agentic-scientist/pipeline-lease:test")
    assert _flag(cmd, "--container-image-recovery-ref") == "agentic-scientist/pipeline-lease:test"


def test_investigation_cmd_forwards_flags_and_gpu_count_without_versioning_downstream() -> None:
    cases = [
        ("versioning_off", {"versioning.enabled": False}, ["versioning.enabled=false"], ["versioning.run_root"]),
        ("harness", {"investigation.harness_evaluated_attempts": True}, ["execution.harness_evaluated_attempts=true"], []),
        (
            "gpu_evaluation",
            {"investigation.harness_evaluated_attempts": True, "investigation.gpu_evaluation": True},
            ["execution.gpu_evaluation=true"], [],
        ),
        ("resume", {"investigation.resume": True}, ["execution.resume=true"], []),
        (
            "never_owns_task_drafting",
            {"investigation.draft_task_if_missing": True},
            ["execution.draft_task_if_missing=false", "logging.show_run_header=false"],
            ["execution.draft_task_if_missing=true"],
        ),
    ]
    for name, overrides, present, absent in cases:
        cfg = _cfg(**overrides)
        cmd = build_investigation_cmd(cfg, derive_paths(cfg))
        for item in present:
            assert item in cmd, name
        for prefix in absent:
            assert not any(item.startswith(prefix) for item in cmd), name

    # the gpu count is forwarded for worker slicing, to the investigation command only
    cfg = _cfg(**{"container.gpu_count": 2})
    assert "container.gpu_count=2" in build_investigation_cmd(cfg, derive_paths(cfg))
    assert not any(
        arg.startswith("container.gpu_count") for arg in build_task_drafting_cmd(cfg, derive_paths(cfg))
    )
    cpu = _cfg()
    assert not any(
        arg.startswith("container.gpu_count") for arg in build_investigation_cmd(cpu, derive_paths(cpu))
    )

    # downstream phase commands carry no versioning overrides: those CLIs have no versioning
    # key; the pipeline parent checkpoints them.
    cfg = _cfg()
    paths = derive_paths(cfg)
    for cmd in (build_ablations_cmd(cfg, paths), build_posthoc_cmd(cfg, paths)):
        assert not any(item.startswith("versioning.") for item in cmd)


def test_phase_commands_forward_expanded_paths_hydra_safely(tmp_path: Path, monkeypatch) -> None:
    # string overrides round-trip hydra grammar characters
    cfg = _cfg()
    cfg.task.dir = "examples/ada's-task[trial],v1\\"
    cfg.paths.run_dir = "/tmp/ada's-results[trial],v1\\"
    cfg.paths.ablations_dir = "/tmp/ada's-output[trial],v1\\"
    cfg.container.image = "registry/image[tag],v1\\"
    cfg.container.network = "ada's-network[trial],v1\\"
    cfg.container.env_file = "/tmp/ada's-env[trial],v1\\"
    paths = derive_paths(cfg)
    for command in (build_investigation_cmd(cfg, paths), build_ablations_cmd(cfg, paths)):
        by_key = _parse_all(command)
        assert by_key["paths.task_dir"] == str(paths.task_dir)
        assert by_key["paths.results_dir"] == str(paths.investigation_workspace_dir)
        assert by_key["container.image"] == cfg.container.image
        assert by_key["container.network"] == cfg.container.network
        assert by_key["container.env_file"] == cfg.container.env_file
    assert by_key["paths.output_dir"] == str(paths.ablations_output_dir)

    # the task path uses the expanded pipeline path
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = _cfg(**{"task.dir": "~/task"})
    paths = derive_paths(cfg)
    for command in (build_investigation_cmd(cfg, paths), build_ablations_cmd(cfg, paths)):
        assert _override(command, "paths.task_dir") == f"paths.task_dir='{paths.task_dir}'"

    # phase commands consume the nested investigation workspace
    cfg = _cfg(**{"paths.run_dir": str(tmp_path)})
    paths = derive_paths(cfg)
    investigation_dir = tmp_path / "investigation"
    assert f"paths.results_dir='{investigation_dir}'" in build_investigation_cmd(cfg, paths)
    assert f"paths.results_dir='{investigation_dir}'" in build_ablations_cmd(cfg, paths)
    assert f"paths.results_dir='{investigation_dir}'" in build_posthoc_cmd(cfg, paths)
    assert _flag(build_final_cmd(cfg, paths), "--investigation-results-dir") == str(investigation_dir)
    assert _flag(build_paper_cmd(cfg, paths), "--investigation-results-dir") == str(investigation_dir)


# --------------------------------------------------------------------------- ablations command


def test_ablations_cmd_forwards_steps_lists_and_hydra_safe_strings() -> None:
    # steps and custom-only forwarding
    cfg = _cfg()
    cfg.ablations.steps = "0-3"
    cfg.ablations.custom_only = True
    cfg.ablations.num_replications = 5
    cfg.ablations.max_evals_per_variant = 5
    paths = derive_paths(cfg)
    cmd = build_ablations_cmd(cfg, paths)
    assert cmd[:3] == [sys.executable, "-m", "src.variants_pipeline.cli"]
    assert "execution.stages='0-3'" in cmd
    assert "execution.custom_only=true" in cmd
    assert f"paths.output_dir='{paths.ablations_output_dir}'" in cmd
    assert "paths.task_name='circpack'" in cmd
    assert "execution.num_replications=5" in cmd
    assert "execution.max_evals_per_variant=5" in cmd
    # No run_id override for stage 2 (variants config has no run_id key).
    assert not any(c.startswith("run_id=") for c in cmd)

    # list forwarding
    cfg = _cfg()
    cfg.ablations.custom_baselines = ["/a/x.md", "/b/y"]
    cfg.ablations.extra_tasks = ["examples/grpo"]
    cmd = build_ablations_cmd(cfg, derive_paths(cfg))
    assert "paths.custom_baselines=['/a/x.md','/b/y']" in cmd
    assert "paths.extra_tasks=['examples/grpo']" in cmd

    # string lists round-trip hydra grammar characters
    cfg = _cfg()
    cfg.ablations.custom_baselines = ["/tmp/a,b[1]'\\", "/tmp/plain"]
    cfg.ablations.extra_tasks = ["examples/ada's-task[trial],v1\\"]
    by_key = _parse_all(build_ablations_cmd(cfg, derive_paths(cfg)))
    assert by_key["paths.custom_baselines"] == cfg.ablations.custom_baselines
    assert by_key["paths.extra_tasks"] == cfg.ablations.extra_tasks


def test_ablations_cmd_sandbox_image_and_evaluation_timeouts() -> None:
    # sandbox image reference
    for name, override, expected in [
        ("explicit_sandbox_image", "ablations.execution.sandbox_image=registry.example/evaluator:v2", "registry.example/evaluator:v2"),
        ("container_image_default", "container.image=registry.example/agent:v3", "registry.example/agent:v3"),
    ]:
        cfg = _loaded(override)
        assert f"execution.sandbox_image='{expected}'" in build_ablations_cmd(cfg, derive_paths(cfg)), name

    # evaluation timeouts are preserved through the downstream loader
    cases = [
        ([], (7200, 7200)),
        (["investigation.execution.run_timeout_seconds=10800"], (7200, 7200)),
        (
            [
                "investigation.execution.run_timeout_seconds=10800",
                "ablations.execution.variant_timeout_seconds=12000",
            ],
            (12000, 7200),
        ),
        (
            [
                "ablations.execution.variant_timeout_seconds=9100",
                "ablations.execution.cross_task_timeout_seconds=9200",
            ],
            (9100, 9200),
        ),
    ]
    for overrides, expected_timeouts in cases:
        cfg = load_pipeline_config([CIRCPACK, *overrides])
        downstream = load_variants_config(build_ablations_cmd(cfg, derive_paths(cfg))[3:])
        assert (
            downstream.execution.variant_timeout_seconds,
            downstream.execution.cross_task_timeout_seconds,
        ) == expected_timeouts, overrides


def test_ablations_rejects_invalid_timeout() -> None:
    for field in ("variant_timeout_seconds", "cross_task_timeout_seconds"):
        for value in ("0", "-1", "true", "1.5"):
            cfg = load_pipeline_config([CIRCPACK, f"ablations.execution.{field}={value}"])
            assert any(
                f"ablations.execution.{field}" in error for error in validate_pipeline_config(cfg)
            ), (field, value)


def test_native_ablations_reject_cross_task_and_contract_tasks_before_any_phase(tmp_path) -> None:
    native = ["container.enabled=false", 'execution.phases="investigation,ablations"']
    cfg = load_pipeline_config([CIRCPACK, *native])
    assert not [error for error in validate_pipeline_config(cfg) if "native" in error or "extra_tasks" in error]

    cfg = load_pipeline_config([CIRCPACK, *native, "ablations.paths.extra_tasks=[examples/grpo]"])
    assert any("extra_tasks" in error for error in validate_pipeline_config(cfg))

    contract_task = tmp_path / "contract_task"
    shutil.copytree("examples/circpack", contract_task)
    (contract_task / "task_contract.json").write_text("{}")
    cfg = load_pipeline_config([f"task.dir={contract_task}", *native])
    assert any("cannot run a contract task" in error for error in validate_pipeline_config(cfg))
    cfg = load_pipeline_config([f"task.dir={contract_task}", 'execution.phases="investigation,ablations"'])
    assert not any("cannot run a contract task" in error for error in validate_pipeline_config(cfg))


def test_native_harness_evaluation_rejects_parallel_workers_before_any_phase() -> None:
    def worker_errors(*overrides: str) -> list[str]:
        cfg = load_pipeline_config([
            CIRCPACK, "execution.phases=investigation", "investigation.execution.workers=2", *overrides,
        ])
        return [error for error in validate_pipeline_config(cfg) if "supports investigation.execution.workers=1" in error]

    assert worker_errors("container.enabled=false", "investigation.execution.harness_evaluated_attempts=true")
    assert not worker_errors("investigation.execution.harness_evaluated_attempts=true")
    assert not worker_errors("container.enabled=false", "investigation.execution.harness_evaluated_attempts=false")


def test_a_paper_phase_without_a_container_is_rejected_before_any_phase() -> None:
    def paper_errors(*overrides: str) -> list[str]:
        cfg = load_pipeline_config([CIRCPACK, 'execution.phases="investigation,paper"', *overrides])
        return [error for error in validate_pipeline_config(cfg) if "paper phase needs a container" in error]

    assert paper_errors("container.enabled=false")
    assert paper_errors("container.paper_enabled=false")
    assert not paper_errors()
    assert not paper_errors("container.enabled=false", "container.paper_enabled=true")
    cfg = load_pipeline_config([CIRCPACK, "container.enabled=false", "execution.phases=investigation"])
    assert not [error for error in validate_pipeline_config(cfg) if "paper" in error]


def test_paper_enabled_containerizes_only_the_paper_of_a_native_run(tmp_path) -> None:
    cfg = load_pipeline_config([
        CIRCPACK, "container.enabled=false", "container.paper_enabled=true",
        "container.image=agent:latest", f"paths.run_dir={tmp_path / 'run'}",
    ])
    paths = derive_paths(cfg)
    paper = build_paper_cmd(cfg, paths)
    assert paper[paper.index("--container-image") + 1] == "agent:latest"
    assert "--containerize" in paper
    investigation = build_investigation_cmd(cfg, paths)
    assert "container.enabled=false" in investigation
    assert not any(arg.startswith("container.image=") for arg in investigation)

    native = load_pipeline_config([CIRCPACK, "container.enabled=false", f"paths.run_dir={tmp_path / 'run'}"])
    assert "--containerize" not in build_paper_cmd(native, derive_paths(native))


def test_a_native_run_pins_the_agent_image_only_for_a_containerized_paper(monkeypatch) -> None:
    leased: list[str] = []

    class _Lease:
        image_id = "sha256:agent"

    def lease(ref: str, **_kwargs: object) -> _Lease:
        leased.append(ref)
        return _Lease()

    monkeypatch.setattr("src.pipeline.cli.acquire_image_lease", lease)
    native = ["container.enabled=false", "container.paper_enabled=true", "container.image=agent:latest"]
    cfg = load_pipeline_config([CIRCPACK, *native])
    _pin_evaluation_images(cfg, {"investigation", "ablations"}, derive_paths(cfg), [])
    assert leased == [] and cfg.container.image == "agent:latest"
    _pin_evaluation_images(cfg, {"investigation", "ablations", "paper"}, derive_paths(cfg), [])
    assert leased == ["agent:latest"] and cfg.container.image == "sha256:agent"
    assert cfg.ablations.sandbox_image is None  # native ablations still need no sandbox image


# --------------------------------------------------------------------------- paper / posthoc commands


def test_paper_cmd_variants() -> None:
    cfg = _cfg()
    paths = derive_paths(cfg)
    assert build_paper_cmd(cfg, paths) == [
        "bash", "src/scripts/paper.sh",
        "--agent", "claude",
        "--agent-model", "claude-opus-4-6[1m]",
        "--reasoning-effort", "high",
        "--task-dir", str(paths.task_dir),
        "--results-dir", str(paths.investigation_dir),
        "--ablations-dir", str(paths.ablations_dir),
        "--global-memory-dir", str(paths.investigation_dir),
        "--output-dir", str(paths.paper_dir),
        "--manifest-dir", str(paths.paper_dir),
        "--final-evaluation-dir", str(paths.final_dir),
        "--contract-task-dir", str(paths.task_dir),
        "--investigation-results-dir", str(paths.investigation_dir),
        # Environment-backed paper.sh flags come first and are always passed, then the
        # mirrored flags the pipeline changes from the script's defaults, in key order.
        "--paper-format", "iclr",
        "--paper-max-repair-rounds", "10",
        "--paper-profile", "default",
        "--paper-review-rounds", "0",
        "--paper-max-refs", "30",
        "--paper-min-refs", "25",
        "--standalone-paper",
        "--final-outcome-policy", "qualified_only",
        "--venue-file", "src/templates/paper_venues/iclr.md",
        "--containerize",
        "--container-image", "agentic-scientist/agent-cli:latest",
        "--container-network", "host",
        "--container-auth-root", "/tmp/auth-root",
        "--container-env-file", ".env",
        "--event-transport", "wrapper",
    ]

    # without final omits the final evaluation input
    cmd = build_paper_cmd_without_final(cfg, paths, posthoc_dir=paths.posthoc_dir)
    assert "--final-evaluation-dir" not in cmd
    assert "--no-final-evaluation" in cmd
    assert _flag(cmd, "--posthoc-dir") == str(paths.posthoc_dir)

    # with posthoc forwards the explicit run
    assert _flag(build_paper_cmd_with_posthoc(cfg, paths), "--posthoc-dir") == str(paths.posthoc_output_dir)

    # alife format
    cfg = _cfg(**{"paper.format": "alife"})
    command = build_paper_cmd(cfg, derive_paths(cfg))
    assert _flag(command, "--paper-format") == "alife"
    assert _flag(command, "--venue-file") == "src/templates/paper_venues/alife.md"


def test_posthoc_cmd_uses_artifact_only_mode_and_emits_network_override_once() -> None:
    cfg = _cfg()
    paths = derive_paths(cfg)
    cmd = build_posthoc_cmd(cfg, paths)
    assert cmd[:3] == [sys.executable, "-m", "src.posthoc_pipeline.cli"]
    assert f"paths.task_dir='{paths.task_dir}'" in cmd
    assert f"paths.results_dir='{paths.investigation_workspace_dir}'" in cmd
    assert f"paths.output_dir='{paths.posthoc_output_dir}'" in cmd
    assert f"paths.evidence_dir='{paths.ablations_output_dir}'" in cmd
    assert "execution.checkpoint_mode='optional'" in cmd
    assert "execution.retrain_if_checkpoint_missing=false" in cmd
    assert "container.network='bridge'" in cmd

    # the posthoc network override is emitted once
    cfg = _loaded("posthoc.container.network=host")
    assert build_posthoc_cmd(cfg, derive_paths(cfg)).count("container.network='host'") == 1


def test_paper_cmd_resumes_only_marked_existing_output(tmp_path: Path) -> None:
    cases = [
        ("marked", True, False, True),
        ("marked_with_sibling_ablations_alias", True, True, True),
        ("unmarked", False, False, False),
    ]
    for name, marker, alias, resumes in cases:
        cfg = _cfg(**{"paths.run_dir": str(tmp_path / name)})
        paths = derive_paths(cfg)
        paths.paper_dir.mkdir(parents=True)
        if marker:
            write_workspace_marker(
                output_dir=paths.paper_dir, task_dir=paths.task_dir,
                results_dir=paths.investigation_dir, global_memory_dir=paths.investigation_dir,
            )
        if alias:
            (paths.ablations_dir / "aggregated").mkdir(parents=True)
            (paths.paper_dir / "aggregated").symlink_to(paths.ablations_dir / "aggregated", target_is_directory=True)

        assert ("--resume-existing" in build_paper_cmd(cfg, paths)) is resumes, name


# --------------------------------------------------------------------------- agent / container forwarding


def test_container_settings_are_forwarded_to_every_phase(tmp_path: Path, monkeypatch) -> None:
    # container disabled omits flags
    cfg = _cfg(**{"container.enabled": False})
    assert "container.enabled=false" in build_investigation_cmd(cfg, derive_paths(cfg))
    assert "container.image='agentic-scientist/agent-cli:latest'" not in build_investigation_cmd(cfg, derive_paths(cfg))
    assert "--containerize" not in build_paper_cmd(cfg, derive_paths(cfg))

    # auth_root is forwarded to every phase and hydra-safe
    cfg = _cfg(**{"container.auth_root": '/tmp/rob\\\'s "auth"[profile]\\'})
    paths = derive_paths(cfg)
    assert _parsed(build_investigation_cmd(cfg, paths), "container.auth_root") == cfg.container.auth_root
    assert _parsed(build_ablations_cmd(cfg, paths), "container.auth_root") == cfg.container.auth_root
    assert _flag(build_paper_cmd(cfg, paths), "--container-auth-root") == cfg.container.auth_root

    # env_file expands for every phase
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = _cfg(**{"container.env_file": "~/.agent.env"})
    expected = str(tmp_path / ".agent.env")
    paths = derive_paths(cfg)
    for command in (build_investigation_cmd(cfg, paths), build_ablations_cmd(cfg, paths)):
        assert _parsed(command, "container.env_file") == expected
    assert _flag(build_paper_cmd(cfg, paths), "--container-env-file") == expected

    # an empty env_file disables the paper default
    cfg = _cfg(**{"container.env_file": ""})
    paths = derive_paths(cfg)
    for command in (build_investigation_cmd(cfg, paths), build_ablations_cmd(cfg, paths)):
        assert "." not in command
        assert not any(arg.startswith("container.env_file=") for arg in command)
    assert _flag(build_paper_cmd(cfg, paths), "--container-env-file") == ""


def test_agent_and_logging_settings_are_forwarded_to_every_phase() -> None:
    # no model omits model args
    cfg = _cfg(**{"agent.model": None})
    assert not any(c.startswith("agent.model=") for c in build_investigation_cmd(cfg, derive_paths(cfg)))
    assert "--agent-model" not in build_paper_cmd(cfg, derive_paths(cfg))

    # a non-claude agent does not inherit the claude model
    cfg = load_pipeline_config(["agent.name=codex"])
    paths = derive_paths(cfg)
    assert not any(arg.startswith("agent.model=") for arg in build_investigation_cmd(cfg, paths))
    assert not any(arg.startswith("agent.model=") for arg in build_ablations_cmd(cfg, paths))
    assert "--agent-model" not in build_paper_cmd(cfg, paths)

    # agent profile is forwarded to every phase and hydra-safe
    cfg = _cfg(**{"agent.name": "codex", "agent.profile": "ada's-profile[trial],v1\\"})
    paths = derive_paths(cfg)
    for command in (build_investigation_cmd(cfg, paths), build_ablations_cmd(cfg, paths)):
        assert _parsed(command, "agent.profile") == cfg.agent.profile
    assert _flag(build_paper_cmd(cfg, paths), "--agent-profile") == cfg.agent.profile

    # reasoning effort is forwarded to every phase
    cfg = _cfg(**{"agent.reasoning_effort": "xhigh"})
    paths = derive_paths(cfg)
    for command in (build_investigation_cmd(cfg, paths), build_ablations_cmd(cfg, paths)):
        assert "agent.reasoning_effort='xhigh'" in command
    assert _flag(build_paper_cmd(cfg, paths), "--reasoning-effort") == "xhigh"

    # logging configuration is forwarded to every phase
    cfg = _cfg()
    cfg.logging.format = "json"
    cfg.logging.verbose = True
    cfg.logging.heartbeat_seconds = 15
    paths = derive_paths(cfg)
    for command in (build_investigation_cmd(cfg, paths), build_ablations_cmd(cfg, paths), build_posthoc_cmd(cfg, paths)):
        assert "logging.format=json" in command
        assert "logging.verbose=true" in command
        assert "logging.heartbeat_seconds=15" in command
        assert "logging.show_run_header=false" in command
    paper = build_paper_cmd(cfg, paths)
    assert "--log-format" in paper
    assert _flag(paper, "--log-format") == "json"
    assert "--verbose" in paper
    assert "--heartbeat-seconds" in paper
    assert _flag(paper, "--heartbeat-seconds") == "15"
    assert _flag(paper, "--event-transport") == "wrapper"


# --------------------------------------------------------------------------- run_cmd / paper output forwarding


def test_run_cmd_dry_run_never_executes_and_merges_environment_overrides(monkeypatch) -> None:
    # dry run never executes
    called = {"n": 0}

    def _fake_run(*a: object, **k: object) -> None:
        called["n"] += 1

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.phases.subprocess.run", _fake_run)
        assert run_cmd(["echo", "hi"], dry_run=True, cwd=Path(".")) == 0
    assert called["n"] == 0

    # environment overrides are merged into the inherited environment
    captured: dict[str, object] = {}

    def fake_run(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setenv("EXISTING_PIPELINE_VALUE", "kept")
    monkeypatch.setattr("src.pipeline.phases.subprocess.run", fake_run)
    rc = run_cmd(
        ["echo", "hi"], dry_run=False, cwd=Path("."),
        env_overrides={"AGENTIC_PIPELINE_STARTED_AT_MONOTONIC": "12.5"},
    )
    assert rc == 0
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["EXISTING_PIPELINE_VALUE"] == "kept"
    assert environment["AGENTIC_PIPELINE_STARTED_AT_MONOTONIC"] == "12.5"


def _fake_streaming_process() -> object:
    return type("FakeProcess", (), {"stdout": io.StringIO("event\n"), "wait": lambda self: 0})()


def test_run_cmd_streams_transformed_stdout_and_bounds_child_cleanup(monkeypatch, capsys) -> None:
    # transformed stdout is streamed; a successful stream never terminates the child
    captured: dict[str, object] = {}

    class FakeProcess:
        stdout = io.StringIO("hidden event\nvisible output\n")

        def wait(self) -> int:
            return 7

        def terminate(self) -> None:
            raise AssertionError("successful streaming must not terminate the child")

    def fake_popen(*_args: object, **kwargs: object) -> FakeProcess:
        captured.update(kwargs)
        return FakeProcess()

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.phases.subprocess.Popen", fake_popen)
        rc = run_cmd(
            ["paper"], dry_run=False,
            output_line_fn=lambda line: "" if line.startswith("hidden") else line.upper(),
        )
    assert rc == 7
    assert capsys.readouterr().out == "VISIBLE OUTPUT\n"
    assert captured["stdout"] is not None
    assert captured["stderr"] is subprocess.STDOUT
    assert captured["text"] is True
    assert captured["start_new_session"] is True

    # a failing event callback bounds the streaming child cleanup
    cleanup: dict[str, object] = {}
    process = _fake_streaming_process()

    def terminate(child: object, *, grace_seconds: int) -> None:
        cleanup["child"] = child
        cleanup["grace_seconds"] = grace_seconds

    monkeypatch.setattr("src.pipeline.phases.subprocess.Popen", lambda *_a, **_k: process)
    monkeypatch.setattr("src.pipeline.phases.terminate_active_process_group", terminate)
    with pytest.raises(RuntimeError, match="event callback failed"):
        run_cmd(
            ["paper"], dry_run=False,
            output_line_fn=lambda _line: (_ for _ in ()).throw(RuntimeError("event callback failed")),
        )
    assert cleanup == {"child": process, "grace_seconds": 5}


def test_run_cmd_forwards_sigterm_to_streaming_process_group(monkeypatch) -> None:
    captured: dict[str, object] = {}
    handlers: dict[int, object] = {}
    previous_handler = object()
    process = _fake_streaming_process()

    def set_handler(signum: int, handler: object) -> None:
        handlers[signum] = handler

    def terminate(child: object, *, grace_seconds: int) -> None:
        captured["child"] = child
        captured["grace_seconds"] = grace_seconds
        nested_handler = handlers[signal.SIGTERM]
        assert callable(nested_handler)
        nested_handler(signal.SIGTERM, None)
        captured["cleanup_completed"] = True

    monkeypatch.setattr("src.pipeline.phases.subprocess.Popen", lambda *_a, **_k: process)
    monkeypatch.setattr("src.pipeline.phases.signal.getsignal", lambda _signal: previous_handler)
    monkeypatch.setattr("src.pipeline.phases.signal.signal", set_handler)
    monkeypatch.setattr("src.pipeline.phases.terminate_active_process_group", terminate)

    def deliver_sigterm(_line: str) -> str:
        handler = handlers[signal.SIGTERM]
        assert callable(handler)
        handler(signal.SIGTERM, None)
        return ""

    with pytest.raises(SystemExit, match="143"):
        run_cmd(["paper"], dry_run=False, output_line_fn=deliver_sigterm)

    assert captured == {"child": process, "grace_seconds": 5, "cleanup_completed": True}
    assert handlers[signal.SIGTERM] is previous_handler


def test_forward_paper_output_reemits_valid_event_and_hides_transport_lines() -> None:
    record = {
        "timestamp": "2026-09-02T10:00:00.000Z", "source": "paper", "event": "paper_gate",
        "session_id": "paper-test", "gate": "draft", "state": "completed", "fields": {"round": 1},
    }
    emitted: list[tuple[str, dict[str, object]]] = []

    retained = forward_paper_output(
        f"AGENTIC_SCIENTIST_PAPER_EVENT {json.dumps(record)}\n",
        event_fn=lambda event, **fields: emitted.append((event, fields)),
    )

    assert retained == ""
    assert emitted == [(
        "paper_gate",
        {
            "gate": "draft", "state": "completed", "paper_session_id": "paper-test",
            "paper_timestamp": "2026-09-02T10:00:00.000Z", "paper_fields": {"round": 1},
        },
    )]
    assert forward_paper_output("Paper directory: /tmp/paper\n", event_fn=lambda *_a: None) == ""
    assert forward_paper_output("visible warning\n", event_fn=lambda *_a: None) == "visible warning\n"


# --------------------------------------------------------------------------- stage key forwarding rules


def test_stage_key_forwarding_rules() -> None:
    # every built command contains only strings
    cfg = load_pipeline_config([
        "container.image=123", "container.network=456", "paths.run_dir=123", "paths.ablations_dir=456",
        "paths.paper_dir=789", "ablations.execution.stages=3", "run_id=20260101",
    ])
    paths = derive_paths(cfg)
    for command in (build_investigation_cmd(cfg, paths), build_ablations_cmd(cfg, paths), build_paper_cmd(cfg, paths)):
        assert all(isinstance(argument, str) for argument in command)

    # any stage key is reachable as <phase>.<key> with no pipeline plumbing
    cfg = _loaded(
        "investigation.execution.isolate_worker_memory=true", "ablations.execution.max_ablations=7",
        "posthoc.execution.max_attempts_per_analysis=1", "paper.paper_review_rounds=2",
    )
    paths = derive_paths(cfg)
    assert "execution.isolate_worker_memory=true" in build_investigation_cmd(cfg, paths)
    assert "execution.max_ablations=7" in build_ablations_cmd(cfg, paths)
    assert "execution.max_attempts_per_analysis=1" in build_posthoc_cmd(cfg, paths)
    assert _flag(build_paper_cmd(cfg, paths), "--paper-review-rounds") == "2"

    # a stage default is not forwarded
    cfg = _loaded()
    paths = derive_paths(cfg)
    investigation_cmd = build_investigation_cmd(cfg, paths)
    assert not any(argument.startswith("execution.workers=") for argument in investigation_cmd)
    assert "execution.max_idea_cycles=3" in investigation_cmd
    assert not any(argument.startswith("execution.bootstrap_timeout") for argument in build_posthoc_cmd(cfg, paths))
    paper_cmd = build_paper_cmd(cfg, paths)
    assert "--two-stage" not in paper_cmd
    assert "--skills" not in paper_cmd
    assert _flag(paper_cmd, "--final-outcome-policy") == "qualified_only"

    # paper.sh reads PAPER_FORMAT & co from the environment; the command must not rely on it.
    cfg = _loaded("paper.paper_format=iclr", "paper.paper_review_rounds=0")
    cmd = build_paper_cmd(cfg, derive_paths(cfg))
    assert _flag(cmd, "--paper-format") == "iclr"
    assert _flag(cmd, "--paper-review-rounds") == "0"
    assert _flag(cmd, "--paper-profile") == "default"
    assert _flag(cmd, "--paper-max-repair-rounds") == "10"

    # standalone paper wins over the default profile flag
    cfg = _loaded("paper.standalone_paper=true")
    cmd = build_paper_cmd(cfg, derive_paths(cfg))
    assert cmd.index("--paper-profile") < cmd.index("--standalone-paper")


def test_owned_keys_and_runtime_requirements_are_validated() -> None:
    # an owned key is a config error
    for override in [
        "investigation.paths.task_dir=/elsewhere",
        "investigation.run_id=abc",
        "investigation.paper.enabled=true",
        "ablations.paths.output_dir=/elsewhere",
        "posthoc.execution.checkpoint_mode=optional",
        "paper.output_dir=/elsewhere",
        "paper.event_transport=wrapper",
    ]:
        errors = validate_pipeline_config(load_pipeline_config([CIRCPACK, override]))
        key = override.split("=", 1)[0]
        assert f"{key} is set by the pipeline and cannot be overridden" in errors, errors

    # an owned key equal to the stage default is not a conflict
    cfg = load_pipeline_config([CIRCPACK, "investigation.container.enabled=false"])
    assert cfg.investigation.conflicting == []

    # task drafting requires a containerized runtime
    cfg = load_pipeline_config([CIRCPACK, "investigation.execution.draft_task_if_missing=true", "container.enabled=false"])
    assert any("requires container.enabled=true" in error for error in validate_pipeline_config(cfg))


# --------------------------------------------------------------------------- effective stage config golden


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline_effective_config" / "expected.json"
RUN_DIR = f"results/golden/circpack/claude/{RUN_ID}"

CASES: dict[str, list[str]] = {
    "defaults": [CIRCPACK],
    "run_task_sh": [
        CIRCPACK,
        "agent.name=claude",
        "agent.model=claude-opus-4-6",
        "agent.reasoning_effort=high",
        "container.image=agentic-scientist/agent-cli:abc123",
        "investigation.execution.max_idea_cycles=20",
        "investigation.execution.max_idea_generation_attempts=1",
        "investigation.execution.max_program_evals_per_attempt=2",
        "investigation.execution.idea_timeout_seconds=3000",
        "investigation.execution.run_timeout_seconds=21600",
        "investigation.execution.harness_evaluated_attempts=true",
        "ablations.execution.num_replications=1",
        "ablations.execution.max_evals_per_variant=1",
        "paper.paper_format=iclr",
        "paper.final_outcome_policy=allow_not_qualified",
        "versioning.final_include_environments=false",
        "execution.phases='task_drafting,investigation,ablations,final_evaluation,posthoc,paper'",
        "investigation.execution.gpu_evaluation=true",
        "container.gpu_count=4",
    ],
    "usage_smoke": [
        CIRCPACK,
        "agent.name=claude",
        "agent.model=claude-sonnet-5",
        "agent.reasoning_effort=high",
        "investigation.execution.max_idea_cycles=1",
        "investigation.execution.max_idea_generation_attempts=1",
        "investigation.execution.max_program_evals_per_attempt=1",
        "investigation.execution.idea_timeout_seconds=600",
        "investigation.execution.run_timeout_seconds=600",
        "ablations.execution.num_replications=1",
        "ablations.execution.max_evals_per_variant=1",
        "paper.paper_format=iclr",
        "execution.phases='task_drafting,investigation,ablations,final_evaluation,posthoc,paper,review'",
    ],
    "native_no_container": [
        CIRCPACK,
        "container.enabled=false",
        "execution.phases='investigation,ablations,posthoc,paper'",
    ],
    "resume_downstream": [
        CIRCPACK,
        "execution.phases='ablations,posthoc,paper'",
        f"paths.run_dir={RUN_DIR}",
    ],
    "paper_only_venue": [
        CIRCPACK,
        "execution.phases=paper",
        f"paths.run_dir={RUN_DIR}",
        "paper.venue_file=src/templates/paper_venues/iclr.md",
        "paper.standalone_paper=false",
        "paper.paper_format=iclr",
        "paper.paper_min_refs=10",
        "paper.paper_max_refs=12",
    ],
    "posthoc_and_ablation_knobs": [
        CIRCPACK,
        "posthoc.execution.primary_metric_key=loss",
        "posthoc.execution.max_selected_analyses=2",
        "posthoc.execution.primary_metric_tolerance=0.5",
        "posthoc.container.network=host",
        "ablations.execution.stages=0-3",
        "ablations.execution.custom_only=true",
        "ablations.paths.custom_baselines=[a.md,b.md]",
        "ablations.paths.extra_tasks=[examples/circpack]",
        "ablations.execution.sandbox_image=sandbox:1",
        "final.timeout_seconds=99",
    ],
    "drafting_logging_versioning": [
        CIRCPACK,
        "investigation.task_drafting.design_max_repairs=2",
        "investigation.task_drafting.max_runtime_bytes=1024",
        "task_drafting.timeout_seconds=100",
        "investigation.execution.resume=true",
        "investigation.execution.workers=2",
        "investigation.execution.prune_every_n_cycles=2",
        "logging.format=json",
        "logging.verbose=true",
        "logging.heartbeat_seconds=5",
        "versioning.enabled=false",
        "agent.model='claude-opus-4-6[1m]'",
        "agent.reasoning_effort=high",
        "container.network=bridge",
        "container.env_file=''",
        f"paths.run_dir={RUN_DIR}",
        "execution.phases='investigation,ablations,final_evaluation,posthoc,paper'",
    ],
}


def _normalize(value: Any) -> Any:
    """Make machine-specific values portable and JSON-comparable."""
    home = str(Path.home())
    if isinstance(value, str):
        return value.replace(home, "~")
    if isinstance(value, Path):
        return _normalize(str(value))
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = [_normalize(item) for item in value]
        return sorted(items) if isinstance(value, set) else items
    return value


def _parse_stage_spec(spec: str) -> list[int]:
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            result.update(range(int(low), int(high) + 1))
        else:
            result.add(int(part))
    return sorted(result)


def _strip_python(command: list[str], module: str) -> list[str]:
    assert command[0] == sys.executable
    assert command[1:3] == ["-m", module]
    return command[3:]


def _investigation_effective(command: list[str]) -> dict[str, Any]:
    from src.investigation.config import config_to_flat_dict
    from src.investigation.config_loader import load_config

    return config_to_flat_dict(load_config(_strip_python(command, "src.investigation.cli")))


def _variants_effective(command: list[str]) -> dict[str, Any]:
    argv = _strip_python(command, "src.variants_pipeline.cli")
    stages: list[int] | None = None
    custom_only = False
    overrides: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--stages":
            stages = _parse_stage_spec(argv[index + 1])
            index += 2
        elif argv[index] == "--custom-only":
            custom_only = True
            index += 1
        else:
            overrides.append(argv[index])
            index += 1
    config = dataclasses.asdict(load_variants_config(overrides))
    execution = config["execution"]
    if "stages" in execution:
        spec = execution.pop("stages")
        if spec is not None:
            stages = _parse_stage_spec(str(spec))
    if "custom_only" in execution:
        custom_only = bool(execution.pop("custom_only")) or custom_only
    return {"config": config, "stages": stages, "custom_only": custom_only}


def _posthoc_effective(command: list[str]) -> dict[str, Any]:
    from src.posthoc_pipeline.cli import load_posthoc_config

    return dataclasses.asdict(load_posthoc_config(_strip_python(command, "src.posthoc_pipeline.cli")))


def _paper_effective(command: list[str]) -> dict[str, str]:
    assert command[:2] == ["bash", "src/scripts/paper.sh"]
    return apply_flags(command[2:])


def effective_stage_configs(overrides: list[str]) -> dict[str, Any]:
    """Build every stage command for *overrides* and load it the stage's way."""
    cfg = load_pipeline_config(list(overrides))
    cfg.run_id = RUN_ID
    paths = derive_paths(cfg)
    return _normalize({
        "task_drafting": _investigation_effective(phases.build_task_drafting_cmd(cfg, paths)),
        "investigation": _investigation_effective(phases.build_investigation_cmd(cfg, paths)),
        "ablations": _variants_effective(phases.build_ablations_cmd(cfg, paths)),
        "final": _strip_python(phases.build_final_cmd(cfg, paths), "src.final_evaluation.cli"),
        "posthoc": _posthoc_effective(phases.build_posthoc_cmd(cfg, paths)),
        "paper_with_posthoc": _paper_effective(phases.build_paper_cmd_with_posthoc(cfg, paths)),
        "paper_without_final": _paper_effective(
            phases.build_paper_cmd_without_final(cfg, paths, posthoc_dir=None)
        ),
    })


def _load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_fixture_covers_every_case() -> None:
    if os.environ.get("PIPELINE_EFFECTIVE_CONFIG_REGEN") == "1":
        generated = {name: effective_stage_configs(overrides) for name, overrides in CASES.items()}
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_PATH.write_text(json.dumps(generated, indent=2, sort_keys=True) + "\n")
    assert set(_load_fixture()) == set(CASES)


@pytest.mark.parametrize("case", sorted(CASES))
def test_effective_stage_config_is_unchanged(case: str) -> None:
    expected = _load_fixture()[case]
    actual = effective_stage_configs(CASES[case])
    for stage in expected:
        assert actual[stage] == expected[stage], f"{case}: {stage} effective config changed"
    assert set(actual) == set(expected)


# --------------------------------------------------------------------------- task presets


PRESET_DIR = REPO_ROOT / "configs" / "preset"
EXAMPLES_DIR = REPO_ROOT / "examples"
SHARED_PROFILE = "paper"
TASK_PROFILES = sorted(path.stem for path in PRESET_DIR.glob("*.yaml") if path.stem != SHARED_PROFILE)
FULL_PHASES = "task_drafting,investigation,ablations,final_evaluation,posthoc,paper,review,revision"
# The campaign allocations, as container.gpu_count in each task preset.
GPU_COUNTS = {"grpo": 2}


def _preset(profile: str, *overrides: str):
    return load_pipeline_config([f"preset={profile}", f"task.dir=examples/{profile}", *overrides])


def test_every_drafted_task_profile_reproduces_campaign_settings() -> None:
    # every drafted task has a profile and a task directory
    assert set(TASK_PROFILES) == set(GPU_COUNTS)
    for profile in TASK_PROFILES:
        assert (EXAMPLES_DIR / profile / "task.md").is_file()
        text = (PRESET_DIR / f"{profile}.yaml").read_text()
        assert text.startswith("# @package _global_\n")
        assert "  - paper\n" in text

    # each profile reproduces the campaign settings
    for profile in TASK_PROFILES:
        cfg = _preset(profile)
        assert cfg.agent.reasoning_effort == "high", profile
        assert cfg.execution.phases == FULL_PHASES, profile
        assert cfg.investigation.get("execution.max_idea_cycles") == 20, profile
        assert cfg.investigation.get("execution.max_idea_generation_attempts") == 1, profile
        assert cfg.investigation.get("execution.max_program_evals_per_attempt") == 2, profile
        assert cfg.investigation.get("execution.idea_timeout_seconds") == 3000, profile
        assert cfg.investigation.get("execution.run_timeout_seconds") == 21600, profile
        assert cfg.investigation.get("execution.harness_evaluated_attempts") is True, profile
        assert cfg.investigation.get("execution.gpu_evaluation") is True, profile
        assert cfg.container.gpu_count == GPU_COUNTS[profile], profile
        assert cfg.ablations.get("execution.num_replications") == 1, profile
        assert cfg.ablations.get("execution.max_evals_per_variant") == 1, profile
        assert cfg.ablations.get("execution.variant_timeout_seconds") == 7200, profile
        assert cfg.ablations.get("execution.cross_task_timeout_seconds") == 7200, profile
        assert cfg.paper.get("paper_format") == "iclr", profile
        assert cfg.paper.get("final_outcome_policy") == "allow_not_qualified", profile
        assert cfg.versioning.final_include_environments is False, profile


def test_profile_precedence_defaults_and_unknown_profile() -> None:
    # CLI overrides take precedence over the profile
    cfg = _preset(
        "grpo",
        "investigation.execution.max_idea_cycles=7",
        "paper.final_outcome_policy=qualified_only",
        "investigation.execution.gpu_evaluation=false",
        "container.gpu_count=0",
        "execution.phases='paper,review'",
    )
    assert cfg.container.gpu_count == 0
    assert cfg.investigation.get("execution.max_idea_cycles") == 7
    assert cfg.paper.get("final_outcome_policy") == "qualified_only"
    assert cfg.investigation.get("execution.gpu_evaluation") is False
    assert cfg.execution.phases == "paper,review"
    # Untouched profile values survive alongside explicit overrides.
    assert cfg.paper.get("paper_format") == "iclr"

    # orchestrator defaults are unchanged without a profile
    cfg = load_pipeline_config(["task.dir=examples/circpack"])
    assert cfg.investigation.get("execution.max_idea_cycles") == 3
    assert cfg.ablations.get("execution.num_replications") == 5
    assert cfg.paper.get("paper_format") == "iclr"
    assert cfg.paper.get("final_outcome_policy") == "qualified_only"
    assert cfg.agent.reasoning_effort is None
    assert cfg.container.gpu_count == 0
    assert cfg.execution.phases == "task_drafting,investigation,ablations,final_evaluation,posthoc,paper"

    # an unknown profile is rejected
    with pytest.raises(HydraException, match="preset/not_a_profile"):
        load_pipeline_config(["preset=not_a_profile", "task.dir=examples/circpack"])


# --------------------------------------------------------------------------- completed-investigation receipts


def test_completed_investigation_outcome_from_manifest(tmp_path: Path) -> None:
    # a scored correct program is required for a winner
    workspace = tmp_path / "unscored"
    workspace.mkdir()
    _write_manifest(workspace, _write_program(workspace / "program_0"), combined_score=None, correct=None)
    errors = validate_completed_investigation(workspace)
    assert completed_investigation_outcome(workspace) is None
    assert validate_completed_investigation_for_downstream(workspace)
    _write_receipt(workspace)
    assert any("no scored correct program" in error for error in errors)
    assert completed_investigation_outcome(workspace) == "completed_no_winner"

    # a scored correct program is accepted
    workspace = tmp_path / "winner"
    workspace.mkdir()
    _write_manifest(workspace, _write_program(workspace / "program_1"), combined_score=0.25, correct=True)
    _write_receipt(workspace)
    assert validate_completed_investigation(workspace) == []
    assert completed_investigation_outcome(workspace) == "winner"

    # malformed no-winner state is rejected
    for index, (score, correct) in enumerate(
        [("bad", True), (float("nan"), True), (None, "yes"), (0.25, False)]
    ):
        workspace = tmp_path / f"malformed-{index}"
        workspace.mkdir()
        _write_manifest(workspace, _write_program(workspace / "program_0"), combined_score=score, correct=correct)
        assert completed_investigation_outcome(workspace) is None, (score, correct)
        assert validate_completed_investigation_for_downstream(workspace), (score, correct)


def test_completed_investigation_distinguishes_retained_baseline(tmp_path: Path) -> None:
    for source_attempt_id in ("bootstrap", "parallel_bootstrap"):
        for correct in (True, False):
            name = f"{source_attempt_id}-{correct}"
            workspace = tmp_path / name
            workspace.mkdir()
            _write_manifest(
                workspace, _write_program(workspace / "program_0"),
                combined_score=0.25, correct=correct, source_attempt_id=source_attempt_id,
            )
            errors = validate_completed_investigation(workspace)
            receipt = _write_receipt(workspace)
            assert any("no optimized winner" in error for error in errors), name
            assert receipt["outcome"] == "completed_no_winner", name
            assert completed_investigation_outcome(workspace) == "completed_no_winner", name
            assert validate_completed_investigation_for_downstream(workspace) == [], name

    # a retained baseline accepts a legacy winner receipt
    workspace = tmp_path / "legacy-receipt"
    workspace.mkdir()
    _write_manifest(
        workspace, _write_program(workspace / "program_0"),
        combined_score=0.25, correct=True, source_attempt_id="bootstrap",
    )
    _write_receipt(workspace, outcome="winner")
    assert completed_investigation_outcome(workspace) == "completed_no_winner"
    assert validate_completed_investigation_for_downstream(workspace) == []


def test_completed_investigation_rejects_escaping_program_and_unscored_anchor(tmp_path: Path) -> None:
    # a program outside the workspace (directly or via symlink) is rejected
    for escape in ("outside_workspace", "symlink"):
        root = tmp_path / escape
        workspace = root / "investigation"
        workspace.mkdir(parents=True)
        outside = _write_program(root / "outside")
        program_dir = outside
        if escape == "symlink":
            program_dir = workspace / "program_1"
            program_dir.symlink_to(outside, target_is_directory=True)
        _write_manifest(workspace, program_dir, combined_score=0.25, correct=True)
        errors = validate_completed_investigation(workspace)
        assert any("no usable best program" in error for error in errors), escape

    # ablations-only rejects an unscored investigation anchor
    root = tmp_path / "unscored"
    root.mkdir()
    workspace = _write_program(root / "investigation")
    _write_manifest(workspace, workspace, combined_score=None, correct=None)
    cfg = load_pipeline_config([CIRCPACK, "execution.phases='ablations,paper'", f"paths.run_dir={root}"])
    assert any("no scored correct program" in error for error in validate_pipeline_config(cfg))


# --------------------------------------------------------------------------- image leases


def _lease_cfg(tmp_path: Path) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.container.enabled = True
    cfg.container.image = "agent:latest"
    cfg.run_id = "test-run"
    cfg.paths.run_dir = str(tmp_path / "run")
    return cfg


def test_pipeline_pins_evaluation_images_only_when_needed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # a shared agent and sandbox image is pinned with one lease
    cfg = _lease_cfg(tmp_path)
    paths = derive_paths(cfg)
    leases: list[DockerImageLease] = []
    calls: list[tuple[str, str, str]] = []

    def acquire_shared(reference: str, *, lease_scope: str, role: str) -> DockerImageLease:
        calls.append((reference, lease_scope, role))
        return cast(DockerImageLease, FakeLease(AGENT_ID, "lease:agent"))

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.cli.acquire_image_lease", acquire_shared)
        _pin_evaluation_images(cfg, {"investigation", "ablations"}, paths, leases)
    assert calls == [("agent:latest", str(paths.run_dir.resolve()), "agent")]
    assert cfg.container.image == AGENT_ID
    assert cfg.ablations.sandbox_image == AGENT_ID
    assert len(leases) == 1

    # a distinct sandbox image gets its own lease
    cfg = _lease_cfg(tmp_path)
    cfg.ablations.sandbox_image = "sandbox:latest"
    leases = []

    def acquire_distinct(reference: str, *, lease_scope: str, role: str) -> DockerImageLease:
        image_id = AGENT_ID if role == "agent" else SANDBOX_ID
        return cast(DockerImageLease, FakeLease(image_id, f"lease:{role}"))

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.cli.acquire_image_lease", acquire_distinct)
        _pin_evaluation_images(cfg, {"investigation", "ablations"}, derive_paths(cfg), leases)
    assert cfg.container.image == AGENT_ID
    assert cfg.ablations.sandbox_image == SANDBOX_ID
    assert [lease.tag for lease in leases] == ["lease:agent", "lease:sandbox"]

    # the agent image is not resolved when containers are disabled or no phase needs it
    def unexpected_resolution(image: str) -> str:
        raise AssertionError(f"unexpected image resolution: {image}")

    monkeypatch.setattr("src.pipeline.cli.acquire_image_lease", unexpected_resolution)
    for name, enabled, selected in [
        ("containers_disabled", "false", {"investigation"}),
        ("native_ablations", "false", {"investigation", "ablations"}),
        ("review_only", "true", {"review"}),
    ]:
        cfg = load_pipeline_config([CIRCPACK, f"container.enabled={enabled}", "container.image=agent:latest"])
        _pin_evaluation_images(cfg, selected, derive_paths(cfg), [])
        assert cfg.container.image == "agent:latest", name
        assert cfg.ablations.sandbox_image is None, name


def test_image_leases_are_verified_routed_and_released_despite_cleanup_failures(
    monkeypatch, caplog
) -> None:
    # every active lease is verified
    fake_leases = [FakeLease(AGENT_ID, "lease:agent"), FakeLease(SANDBOX_ID, "lease:sandbox")]
    _verify_image_leases(cast(list[DockerImageLease], fake_leases))
    assert all(lease.verified for lease in fake_leases)

    # the owned agent lease tag is routed to final
    leases = cast(
        list[DockerImageLease], [FakeLease(AGENT_ID, "lease:agent"), FakeLease(SANDBOX_ID, "lease:sandbox")]
    )
    assert _image_lease_tag(AGENT_ID, leases) == "lease:agent"

    # all leases are released when one cleanup fails
    fake_leases = [
        FakeLease(AGENT_ID, "lease:agent", release_error=RuntimeError("daemon down")),
        FakeLease(SANDBOX_ID, "lease:sandbox"),
    ]
    _release_image_leases(cast(list[DockerImageLease], fake_leases))
    assert all(lease.released for lease in fake_leases)
    assert "daemon down" in caplog.text

    # a cleanup failure does not mask the pipeline result
    fake_lease = FakeLease(AGENT_ID, "lease:agent", release_error=RuntimeError("daemon down"))

    def run_pipeline(argv: list[str] | None, leases: list[DockerImageLease], runtime_cache_tasks: list[Path]) -> int:
        leases.append(cast(DockerImageLease, fake_lease))
        return 7

    monkeypatch.setattr(pipeline_cli, "_run_pipeline", run_pipeline)
    assert pipeline_cli._main([]) == 7
    assert fake_lease.released is True

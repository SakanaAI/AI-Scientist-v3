"""Checkpoint wiring at the entry points: pipeline CLI roots/events and investigation main_entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from src.investigation.config import InvestigationConfig
from src.investigation.main_entry import (
    _checkpoint_run_root,
    _checkpoint_settings,
    _log_checkpoint,
    _make_checkpoint_fn,
)
from src.investigation.parallel_helpers import build_parallel_worker_command
from src.investigation.run_checkpoint import CheckpointResult, checkpoint_git_dir
from src.pipeline import cli as pipeline_cli
from src.pipeline.config import PipelineConfig
from src.pipeline.paths import derive_paths

pytestmark = pytest.mark.usefixtures("stub_image_leases")

TOKEN = "sk-proj-abcdefghijklmnopqrstuvwxyz012345"


def _set_dotted(cfg: Any, **kw: object) -> Any:
    for key, value in kw.items():
        head, _, tail = key.partition(".")
        setattr(getattr(cfg, head), tail, value)
    return cfg


def _pipeline_cfg(tmp_path: Path | None = None, **kw: str) -> PipelineConfig:
    """PipelineConfig with dotted overrides; `{tmp}` in values expands to tmp_path."""
    if tmp_path is not None:
        kw = {key: value.format(tmp=tmp_path) for key, value in kw.items()}
    return _set_dotted(PipelineConfig(), **kw)


def _recorder(monkeypatch, status: str = "committed") -> list[dict[str, Any]]:
    """Record every checkpoint the pipeline takes, in order."""
    calls: list[dict[str, Any]] = []

    def fake_write_checkpoint(**kwargs: Any) -> CheckpointResult:
        calls.append(kwargs)
        return CheckpointResult(
            status=status,  # type: ignore[arg-type]
            label=str(kwargs["label"]),
            commit="0" * 40,
            error="git exploded" if status == "degraded" else None,
        )

    monkeypatch.setattr("src.pipeline.cli.write_checkpoint", fake_write_checkpoint)
    return calls


def _event_log(monkeypatch) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        pipeline_cli, "log_event", lambda event, **kwargs: events.append((event, kwargs))
    )
    return events


def _stub_phases(monkeypatch, tmp_path: Path, *, review_status: str = "completed") -> None:
    """Make every phase succeed without running anything real."""
    monkeypatch.setattr("src.pipeline.cli.run_cmd", lambda *a, **k: 0)
    monkeypatch.setattr("src.pipeline.cli.run_glue", lambda *a, **k: None)
    monkeypatch.setattr("src.pipeline.cli.prepare_run_task", lambda *a, **k: None)
    monkeypatch.setattr("src.pipeline.cli.reuse_run_task", lambda *a, **k: None)
    monkeypatch.setattr("src.pipeline.cli.ensure_run_task_for_reuse", lambda *a, **k: True)
    monkeypatch.setattr("src.pipeline.cli.validate_completed_investigation", lambda *a, **k: [])
    monkeypatch.setattr(
        "src.pipeline.cli.investigation_completion_payload",
        lambda *a, **k: {
            "schema_version": 1,
            "status": "completed",
            "outcome": "winner",
            "manifest_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr("src.pipeline.cli._current_posthoc_artifacts_available", lambda *a: True)
    monkeypatch.setattr("src.pipeline.cli._record_posthoc_status", lambda *a, **k: None)
    monkeypatch.setattr("src.pipeline.cli._prepare_posthoc_output", lambda *a, **k: None)
    monkeypatch.setattr("src.pipeline.cli._verify_posthoc_output_identity", lambda *a, **k: None)
    monkeypatch.setattr("src.pipeline.cli.validate_review_input", lambda *a, **k: [])
    monkeypatch.setattr("src.pipeline.cli.resolve_immutable_image", lambda *a, **k: "sha256:x")

    class _Review:
        status = review_status
        error = None if review_status == "completed" else "review blew up"
        log_path = tmp_path / "review.log"
        review_path = tmp_path / "ai_review.json"

    monkeypatch.setattr("src.pipeline.cli.run_pipeline_review", lambda **k: _Review())


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "task.dir=examples/circpack",
        f"paths.results_root={tmp_path / 'results'}",
        "run_id=20260101_000000",
        "container.enabled=false",
        "container.paper_enabled=true",  # the paper phase always needs Docker
        "execution.phases=investigation",
        *extra,
    ]


def _labels(calls: list[dict[str, Any]]) -> list[str]:
    return [str(call["label"]) for call in calls]


def _roots(calls: list[dict[str, Any]]) -> list[Path]:
    return [Path(call["run_root"]) for call in calls]


def _phase_checkpoint(cfg: PipelineConfig, phase: str, **kw: Any):
    return pipeline_cli._pipeline_checkpoint(
        cfg, derive_paths(cfg), label=f"{phase} phase", phase=phase, event="phase_end", **kw
    )


ALL_PHASES = "execution.phases=investigation,ablations,posthoc,paper,review"


# --------------------------------------------------------------------------- pipeline _main flow


def test_every_selected_phase_is_checkpointed_and_final_is_last_even_after_degraded_review(
    tmp_path: Path, monkeypatch
) -> None:
    """Covers the posthoc and review branches that short-circuit past the tail."""
    # every selected phase is checkpointed and final is last
    calls = _recorder(monkeypatch)
    _stub_phases(monkeypatch, tmp_path / "ok")

    rc = pipeline_cli._main(_argv(tmp_path / "ok", ALL_PHASES))

    assert rc == 0
    assert _labels(calls) == [
        "pipeline start",
        "investigation phase",
        "ablations phase",
        "posthoc phase",
        "paper phase",
        "review phase",
        "final",
    ]
    assert calls[-1]["final"] is True
    assert all(call.get("final") is False for call in calls[:-1])

    # a degraded review still checkpoints
    calls = _recorder(monkeypatch)
    _stub_phases(monkeypatch, tmp_path / "degraded", review_status="failed")

    pipeline_cli._main(_argv(tmp_path / "degraded", ALL_PHASES))

    assert "review phase" in _labels(calls)
    assert _labels(calls)[-1] == "final"


def test_phase_and_pipeline_end_events_precede_their_checkpoints(
    tmp_path: Path, monkeypatch
) -> None:
    lifecycle: list[tuple[str, str]] = []
    _stub_phases(monkeypatch, tmp_path)

    def record_checkpoint(**kwargs: Any) -> CheckpointResult:
        lifecycle.append(("checkpoint", str(kwargs["label"])))
        return CheckpointResult(status="unchanged", label=str(kwargs["label"]))

    monkeypatch.setattr("src.pipeline.cli.write_checkpoint", record_checkpoint)
    monkeypatch.setattr(
        "src.pipeline.cli.log_event", lambda event, **_fields: lifecycle.append(("event", event))
    )

    assert pipeline_cli._main(_argv(tmp_path, "execution.phases=investigation")) == 0

    assert lifecycle.index(("event", "phase_end")) < lifecycle.index(
        ("checkpoint", "investigation phase")
    )
    assert lifecycle.index(("event", "pipeline_end")) < lifecycle.index(("checkpoint", "final"))


def test_phase_failures_degraded_checkpoints_and_disabled_versioning_keep_return_codes(
    tmp_path: Path, monkeypatch
) -> None:
    # a failed phase still takes a final checkpoint
    calls = _recorder(monkeypatch)
    _stub_phases(monkeypatch, tmp_path / "failed")
    with monkeypatch.context() as m:
        m.setattr("src.pipeline.cli.run_cmd", lambda *a, **k: 3)
        rc = pipeline_cli._main(_argv(tmp_path / "failed", "execution.fail_fast=true"))

    assert rc == 3
    assert _labels(calls)[-1] == "final"

    # a degraded checkpoint does not change the return code
    _recorder(monkeypatch, status="degraded")
    _stub_phases(monkeypatch, tmp_path / "degraded")

    assert pipeline_cli._main(_argv(tmp_path / "degraded")) == 0

    # disabled versioning takes no checkpoints
    calls = _recorder(monkeypatch)
    _stub_phases(monkeypatch, tmp_path / "disabled")

    rc = pipeline_cli._main(_argv(tmp_path / "disabled", "versioning.enabled=false"))

    assert rc == 0
    assert calls == []


def test_final_checkpoint_skips_an_immediately_degraded_root_even_when_the_event_fails(
    tmp_path: Path, monkeypatch
) -> None:
    # _main: the final checkpoint does not repeat an immediately degraded phase root
    calls: list[dict[str, Any]] = []
    events = _event_log(monkeypatch)
    _stub_phases(monkeypatch, tmp_path)

    def record_checkpoint(**kwargs: Any) -> CheckpointResult:
        calls.append(kwargs)
        return CheckpointResult(
            status="degraded" if kwargs["label"] == "investigation phase" else "committed",
            label=str(kwargs["label"]),
            error="injected checkpoint failure",
        )

    monkeypatch.setattr(pipeline_cli, "write_checkpoint", record_checkpoint)

    rc = pipeline_cli._main(_argv(tmp_path))

    assert rc == 0
    assert _labels(calls) == ["pipeline start", "investigation phase"]
    assert any(
        event == "run_checkpoint"
        and fields.get("label") == "final"
        and fields.get("reason") == "previous_checkpoint_degraded"
        for event, fields in events
    )

    # _pipeline_checkpoint: a suppressed root is skipped with an event and no write
    cfg = _pipeline_cfg()
    paths = derive_paths(cfg)
    root = paths.investigation_results_dir.resolve()
    writes: list[Path] = []
    events = _event_log(monkeypatch)
    monkeypatch.setattr(pipeline_cli, "_checkpoint_roots", lambda *_args: (root,))
    monkeypatch.setattr(
        pipeline_cli, "write_checkpoint", lambda **kwargs: writes.append(Path(kwargs["run_root"]))
    )

    degraded = pipeline_cli._pipeline_checkpoint(
        cfg,
        paths,
        label="final",
        phase="pipeline",
        event="pipeline_end",
        final=True,
        suppress_roots=frozenset({root}),
    )

    assert degraded == frozenset()
    assert writes == []
    assert events == [
        (
            "run_checkpoint",
            {
                "label": "final",
                "run_root": str(root),
                "status": "skipped",
                "reason": "previous_checkpoint_degraded",
            },
        )
    ]

    # _finish_pipeline: a failing suppression event is non-gating
    monkeypatch.setattr(
        pipeline_cli,
        "write_checkpoint",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must stay suppressed")),
    )

    def log_or_fail(event: str, **_kwargs: Any) -> None:
        if event == "run_checkpoint":
            raise OSError("journal unavailable")

    monkeypatch.setattr(pipeline_cli, "log_event", log_or_fail)

    return_code = pipeline_cli._finish_pipeline(
        return_code=7,
        status="failed",
        pipeline_started_at=0.0,
        cfg=cfg,
        paths=paths,
        previous_checkpoint_degraded_roots=frozenset({root}),
    )

    assert return_code == 7


def test_dry_run_previews_checkpoints_and_degrades_gracefully_on_invalid_roots(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    # a dry run previews checkpoints without taking any
    calls = _recorder(monkeypatch)
    _stub_phases(monkeypatch, tmp_path)

    with caplog.at_level("INFO"):
        rc = pipeline_cli._main(_argv(tmp_path, "execution.dry_run=true"))

    assert rc == 0
    assert calls == []
    assert any("checkpoint 'final'" in record.message for record in caplog.records)

    # an invalid dry-run root degrades without failing
    caplog.clear()
    cfg = _pipeline_cfg()
    cfg.execution.dry_run = True
    cfg.versioning.run_root = "/"

    with caplog.at_level("WARNING"):
        pipeline_cli._pipeline_checkpoint(
            cfg, derive_paths(cfg), label="pipeline start", phase="pipeline", event="pipeline_start"
        )

    assert calls == []
    assert "Checkpoint 'pipeline start' degraded" in caplog.text


_TOPOLOGY_ERROR_CASES = [
    (
        "run-root-inside-output",
        {
            "execution.phases": "paper",
            "versioning.run_root": "{tmp}/bundle/paper",
            "paths.paper_dir": "{tmp}/bundle",
        },
        {"paper"},
        "versioning.run_root must not be inside a selected phase output",
    ),
    (
        "investigation-inside-another-root",
        {
            "execution.phases": "investigation,paper",
            "paths.run_dir": "{tmp}/bundle/investigation",
            "paths.paper_dir": "{tmp}/bundle",
        },
        {"investigation", "paper"},
        "selected checkpoint roots must not contain paths.run_dir",
    ),
]


def test_checkpoint_topology_errors_are_reported_and_reject_the_pipeline(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    # _checkpoint_topology_errors names each violation
    for name, overrides, selected, error in _TOPOLOGY_ERROR_CASES:
        cfg = _pipeline_cfg(tmp_path, **overrides)

        assert pipeline_cli._checkpoint_topology_errors(cfg, derive_paths(cfg), selected) == [
            error
        ], name

    # the pipeline rejects an output that contains the investigation sidecar
    calls = _recorder(monkeypatch)
    _stub_phases(monkeypatch, tmp_path)

    with caplog.at_level("ERROR"):
        rc = pipeline_cli._main(
            _argv(
                tmp_path,
                "execution.phases=investigation,ablations,paper",
                f"paths.paper_dir={tmp_path / 'results'}",
            )
        )

    assert rc == 1
    assert calls == []
    assert "must not contain paths.run_dir" in caplog.text


# --------------------------------------------------------------------------- checkpoint roots


_ROOT_TOPOLOGY_CASES: list[tuple[str, dict[str, str], str, bool, Callable[[Any, Path], list[Path]]]] = [
    (
        "configured-run-root",
        {
            "execution.phases": "investigation",
            "paths.run_dir": "{tmp}/checkpoint-root",
            "versioning.run_root": "{tmp}/checkpoint-root",
        },
        "investigation",
        False,
        lambda p, tmp: [(tmp / "checkpoint-root").resolve()],
    ),
    (
        "external-paper-gets-own-root",
        {
            "execution.phases": "investigation,paper",
            "paths.results_root": "{tmp}/results",
            "paths.paper_dir": "{tmp}/external-paper",
        },
        "paper",
        False,
        lambda p, tmp: [p.investigation_results_dir.resolve(), p.paper_output_dir.resolve()],
    ),
    (
        "read-only-external-ablations-skipped",
        {
            "execution.phases": "investigation,paper",
            "paths.results_root": "{tmp}/results",
            "paths.ablations_dir": "{tmp}/external-variants",
        },
        "paper",
        False,
        lambda p, tmp: [p.run_dir.resolve()],
    ),
    (
        "paper-only-uses-run-root",
        {"execution.phases": "paper", "paths.run_dir": "{tmp}/run"},
        "paper",
        False,
        lambda p, tmp: [p.run_dir.resolve()],
    ),
    (
        "review-only-includes-external-paper",
        {
            "execution.phases": "review",
            "paths.run_dir": "{tmp}/run",
            "paths.paper_dir": "{tmp}/published-paper",
        },
        "review",
        False,
        lambda p, tmp: [p.run_dir.resolve(), p.paper_dir.resolve()],
    ),
    (
        "configured-review-root-covers-bundle",
        {
            "execution.phases": "review",
            "paths.run_dir": "{tmp}/run",
            "paths.paper_dir": "{tmp}/run/paper",
            "versioning.run_root": "{tmp}/configured-root",
        },
        "review",
        False,
        lambda p, tmp: [(tmp / "configured-root").resolve(), p.run_dir.resolve()],
    ),
    (
        "final-covers-each-external-root",
        {
            "execution.phases": "investigation,ablations,posthoc,paper,review",
            "paths.results_root": "{tmp}/results",
            "paths.ablations_dir": "{tmp}/external-variants",
            "paths.posthoc_dir": "{tmp}/external-posthoc",
            "paths.paper_dir": "{tmp}/external-paper",
        },
        "pipeline",
        True,
        lambda p, tmp: [
            p.investigation_results_dir.resolve(),
            p.ablations_output_dir.resolve(),
            p.posthoc_output_dir.resolve(),
            p.paper_output_dir.resolve(),
        ],
    ),
]


def test_checkpoint_roots_follow_phase_topology_without_following_symlinks(
    tmp_path: Path, monkeypatch
) -> None:
    # roots follow the phase topology
    for name, overrides, phase, final, expected in _ROOT_TOPOLOGY_CASES:
        case_tmp = tmp_path / name
        calls = _recorder(monkeypatch)
        cfg = _pipeline_cfg(case_tmp, **overrides)
        paths = derive_paths(cfg)

        if final:
            pipeline_cli._pipeline_checkpoint(
                cfg, paths, label="final", phase="pipeline", event="pipeline_end", final=True
            )
        else:
            _phase_checkpoint(cfg, phase)

        assert _roots(calls) == expected(paths, case_tmp), name

    # a symlinked phase output is not followed
    calls = _recorder(monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    paper_link = tmp_path / "paper-link"
    paper_link.symlink_to(outside, target_is_directory=True)
    cfg = _pipeline_cfg(
        tmp_path,
        **{
            "execution.phases": "investigation,paper",
            "paths.results_root": "{tmp}/results",
            "paths.paper_dir": str(paper_link),
        },
    )
    paths = derive_paths(cfg)

    _phase_checkpoint(cfg, "paper")

    assert _roots(calls) == [paths.investigation_results_dir.resolve()]
    assert outside not in _roots(calls)


def test_checkpoint_writer_failures_do_not_skip_other_roots_and_only_degraded_roots_return(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    # a writer failure does not skip other roots
    cfg = _pipeline_cfg(
        tmp_path,
        **{
            "execution.phases": "investigation,paper",
            "paths.results_root": "{tmp}/results",
            "paths.paper_dir": "{tmp}/external-paper",
        },
    )
    attempted: list[Path] = []

    def write_or_fail(**kwargs: Any) -> CheckpointResult:
        attempted.append(Path(kwargs["run_root"]))
        if len(attempted) == 1:
            raise RuntimeError("unexpected writer failure")
        return CheckpointResult(status="unchanged", label=str(kwargs["label"]))

    monkeypatch.setattr(pipeline_cli, "write_checkpoint", write_or_fail)

    with caplog.at_level("WARNING"):
        _phase_checkpoint(cfg, "paper")

    assert len(attempted) == 2
    assert "degraded for one root: RuntimeError" in caplog.text

    # only degraded roots are returned
    cfg = _pipeline_cfg(
        tmp_path,
        **{
            "execution.phases": "investigation,paper",
            "paths.results_root": "{tmp}/results",
            "paths.paper_dir": "{tmp}/paper",
        },
    )
    paths = derive_paths(cfg)
    roots = (paths.investigation_results_dir.resolve(), paths.paper_output_dir.resolve())
    monkeypatch.setattr(pipeline_cli, "_checkpoint_roots", lambda *_args: roots)
    monkeypatch.setattr(
        pipeline_cli,
        "write_checkpoint",
        lambda **kwargs: CheckpointResult(
            status="degraded" if kwargs["run_root"] == roots[0] else "committed",
            label=str(kwargs["label"]),
        ),
    )

    degraded = _phase_checkpoint(cfg, "paper")

    assert degraded == frozenset({roots[0]})


# --------------------------------------------------------------------------- pipeline events


_EXCLUSION_EVENT_CASES = [
    ("sensitive", "skipped_sensitive", "agent.log"),
    ("oversize", "skipped_oversize", "large.bin"),
]


def _run_root_event(monkeypatch, run_root: Path) -> str:
    cfg = _pipeline_cfg(
        None,
        **{
            "execution.phases": "investigation",
            "paths.results_root": str(run_root / "results"),
            "versioning.run_root": str(run_root),
        }
    )
    events = _event_log(monkeypatch)
    monkeypatch.setattr(
        pipeline_cli,
        "write_checkpoint",
        lambda **kwargs: CheckpointResult(status="committed", label=str(kwargs["label"])),
    )
    _phase_checkpoint(cfg, "investigation")
    return str(events[0][1]["run_root"])


def test_checkpoint_events_report_exclusions_and_sanitize_the_run_root(
    tmp_path: Path, monkeypatch
) -> None:
    # an unchanged checkpoint with exclusions still emits a pipeline event
    cfg = _pipeline_cfg(tmp_path, **{"paths.results_root": "{tmp}"})
    paths = derive_paths(cfg)
    for case, field, name in _EXCLUSION_EVENT_CASES:
        events = _event_log(monkeypatch)
        exclusions: dict[str, Any] = {field: (name,)}
        excluded = CheckpointResult(status="unchanged", label="paper phase", **exclusions)
        monkeypatch.setattr(pipeline_cli, "write_checkpoint", lambda **_kwargs: excluded)

        pipeline_cli._pipeline_checkpoint(
            cfg, paths, label="paper phase", phase="paper", event="paper_completed"
        )

        assert events == [
            (
                "run_checkpoint",
                {
                    "label": "paper phase",
                    "run_root": str(paths.investigation_results_dir.resolve()),
                    "status": "unchanged",
                    "commit": None,
                    "tag": None,
                    "files": 0,
                    "skipped_oversize": [name] if field == "skipped_oversize" else [],
                    "skipped_sensitive": [name] if field == "skipped_sensitive" else [],
                    "error": None,
                },
            )
        ], case

    # the event redacts a sensitive run root
    reported = _run_root_event(monkeypatch, tmp_path / TOKEN)

    assert reported.startswith("<redacted-path:")
    assert "sk-proj" not in reported

    # the event escapes control characters
    reported = _run_root_event(monkeypatch, tmp_path / "line\nbreak")

    assert "\n" not in reported
    assert "\\x0a" in reported


def test_pipeline_end_records_human_and_numeric_total_runtime(monkeypatch) -> None:
    events = _event_log(monkeypatch)
    monkeypatch.setattr(pipeline_cli.time, "monotonic", lambda: 3823.6)

    return_code = pipeline_cli._finish_pipeline(
        return_code=0, status="completed", pipeline_started_at=100.0
    )

    assert return_code == 0
    assert events == [
        (
            "pipeline_end",
            {
                "status": "completed",
                "return_code": 0,
                "elapsed": "1h 2m 4s",
                "elapsed_seconds": 3723.6,
            },
        )
    ]


# --------------------------------------------------------------------------- investigation main_entry


def _cfg(**kw: object) -> InvestigationConfig:
    return _set_dotted(InvestigationConfig(), **kw)


def test_checkpoint_run_root_scopes_the_envelope_and_keeps_workers_out(tmp_path: Path) -> None:
    # a standalone investigation CLI run checkpoints its own results dir
    assert _checkpoint_run_root(_cfg(), tmp_path / "run") == (tmp_path / "run").resolve()

    # the gate that keeps N parallel workers off one shared git index
    run_root = tmp_path / "run"
    cfg = _cfg(**{"versioning.run_root": str(run_root)})

    assert _checkpoint_run_root(cfg, run_root / "worker_00" / "round_001") is None
    assert _checkpoint_run_root(cfg, run_root) == run_root.resolve()
    # The pipeline coordinator checkpoints its envelope, not only its workspace.
    assert _checkpoint_run_root(cfg, run_root / "investigation") == run_root.resolve()

    # disabled versioning has no run root
    assert _checkpoint_run_root(_cfg(**{"versioning.enabled": False}), tmp_path / "run") is None

    # the parallel worker command disables versioning
    command = build_parallel_worker_command(
        cfg=_cfg(),
        task_dir=tmp_path / "task",
        results_dir=tmp_path / "run" / "worker_00",
        global_memory_dir=tmp_path / "memory",
        instruct_path=tmp_path / "instruct.md",
        assigned_cycles=1,
        worker_index=0,
    )

    assert "versioning.enabled=false" in command


def test_checkpoint_fn_commits_the_run_envelope_but_is_a_noop_for_workers(tmp_path: Path) -> None:
    # the checkpoint fn commits the run envelope from the run root or the pipeline workspace
    for case, workspace in (("run-root", ""), ("pipeline-workspace", "investigation")):
        run_root = tmp_path / case / "run"
        workspace_dir = run_root / workspace
        workspace_dir.mkdir(parents=True)
        (workspace_dir / "outer_state.json").write_text("{}\n", encoding="utf-8")
        cfg = _cfg(**{"versioning.run_root": str(run_root)})
        events: list[tuple[str, dict[str, object]]] = []

        checkpoint_fn = _make_checkpoint_fn(
            cfg, workspace_dir, lambda event, **kw: events.append((event, kw))
        )
        checkpoint_fn(label="idea c001", phase="investigation", event="idea_drafted")

        assert (checkpoint_git_dir(run_root, ".agentic_history") / "HEAD").is_file(), case
        assert [event for event, _ in events] == ["run_checkpoint"], case
        assert events[0][1]["status"] == "committed", case

    # a worker subtree checkpoint fn is a noop
    run_root = tmp_path / "worker" / "run"
    run_root.mkdir(parents=True)
    cfg = _cfg(**{"versioning.run_root": str(run_root)})
    worker_events: list[str] = []

    checkpoint_fn = _make_checkpoint_fn(
        cfg, run_root / "worker_00", lambda event, **_kw: worker_events.append(event)
    )
    checkpoint_fn(label="idea", phase="investigation", event="idea_drafted")

    assert worker_events == []
    assert not checkpoint_git_dir(run_root, ".agentic_history").exists()


_LOG_CHECKPOINT_CASES: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    (
        "unchanged-sensitive",
        {"status": "unchanged", "skipped_sensitive": ("agent.log",)},
        {"status": "unchanged", "skipped_sensitive": ["agent.log"]},
    ),
    (
        "unchanged-oversize",
        {"status": "unchanged", "skipped_oversize": ("large.bin",)},
        {"status": "unchanged", "skipped_oversize": ["large.bin"]},
    ),
    (
        "degraded-reported-not-raised",
        {"status": "degraded", "error": "git exploded"},
        {"status": "degraded", "error": "git exploded"},
    ),
]


def test_checkpoint_settings_mirror_the_config_and_log_checkpoint_reports_only_notable_results() -> None:
    # settings mirror the config
    settings = _checkpoint_settings(
        _cfg(**{"versioning.max_file_bytes": 2048, "versioning.tag_checkpoints": False})
    )

    assert settings.max_file_bytes == 2048
    assert settings.tag_checkpoints is False
    assert settings.git_dir_name == ".agentic_history"

    # unchanged checkpoints emit no event
    silent: list[str] = []

    _log_checkpoint(
        CheckpointResult(status="unchanged", label="idea"), lambda event, **_kw: silent.append(event)
    )

    assert silent == []

    # exclusions and degradation are reported, not raised
    for case, result_kwargs, expected in _LOG_CHECKPOINT_CASES:
        events: list[dict[str, object]] = []

        _log_checkpoint(
            CheckpointResult(label="idea", **result_kwargs), lambda _event, **kw: events.append(kw)
        )

        assert all(events[0][key] == value for key, value in expected.items()), case

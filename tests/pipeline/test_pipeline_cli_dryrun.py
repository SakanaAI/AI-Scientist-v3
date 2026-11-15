"""Orchestration contracts of ``src.pipeline.cli.main`` driven with stubbed child commands."""

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.final_evaluation.runner import FinalEvaluationError
from src.investigation.program_spec import render_default_program_spec_toml
from src.investigation.runtime import PIPELINE_STARTED_AT_ENV
from src.pipeline import cli, config, downstream_selection
from src.pipeline.cli import (
    TASK_RUNTIME_CACHE_HANDOFF_ENV,
    _log_resume_hint,
    _prepare_posthoc_output,
    _record_posthoc_status,
    _verify_posthoc_output_identity,
    main,
)
from src.pipeline.config import (
    PipelineConfig,
    investigation_completion_payload,
    investigation_completion_receipt_path,
)
from src.pipeline.paths import derive_paths
from src.pipeline.phases import build_paper_cmd
from src.pipeline.resume_config import write_resume_policy
from src.pipeline.revision import build_revision_command
from src.pipeline.safe_fs import UnsafeGluePathError
from src.pipeline.task_snapshot import prepare_run_task, task_snapshot_ready_path

pytestmark = pytest.mark.usefixtures("stub_image_leases")

REPO_ROOT = Path(__file__).resolve().parents[2]
CIRCPACK = "task.dir=examples/circpack"
PAPER = "src/scripts/paper.sh"
FINAL = "src.final_evaluation.cli"
POSTHOC = "src.posthoc_pipeline.cli"
INVESTIGATION = "src.investigation.cli"


# --------------------------------------------------------------------------- helpers


def _no_glue(monkeypatch) -> None:
    monkeypatch.setattr("src.pipeline.cli.run_glue", lambda *args, **kwargs: None)


def _no_subprocess(monkeypatch) -> None:
    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("subprocess.run must not be called in dry-run")

    monkeypatch.setattr("src.pipeline.phases.subprocess.run", _boom)
    _no_glue(monkeypatch)


def _passthrough_images(monkeypatch) -> None:
    monkeypatch.setattr("src.pipeline.cli.resolve_immutable_image", lambda image: image)


def _capture_commands(monkeypatch, rc=0) -> list[list[str]]:
    """Stub ``run_cmd``; *rc* is an int or ``callable(command) -> int``."""
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> int:
        commands.append(command)
        return rc(command) if callable(rc) else rc

    monkeypatch.setattr("src.pipeline.cli.run_cmd", run)
    return commands


def _fail_on(marker: str, rc: int = 7):
    return lambda command: rc if marker in command else 0


def _capture_events(monkeypatch) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "src.pipeline.cli.log_event", lambda event, **fields: events.append((event, fields))
    )
    return events


def _phase_end(events, phase: str) -> dict[str, object]:
    return next(
        fields
        for event, fields in events
        if event == "phase_end" and fields.get("phase_name") == phase
    )


def _pipeline_end(events) -> dict[str, object]:
    return next(fields for event, fields in events if event == "pipeline_end")


def _journal(run_dir: Path) -> list[dict]:
    text = (run_dir / "pipeline_events.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _messages(caplog) -> list[str]:
    return [record.message for record in caplog.records]


def _mock_successful_investigation(monkeypatch) -> None:
    monkeypatch.setattr("src.pipeline.cli.validate_completed_investigation", lambda *_args: [])
    monkeypatch.setattr(
        "src.pipeline.cli.investigation_completion_payload",
        lambda *_args: {
            "schema_version": 1,
            "status": "completed",
            "outcome": "winner",
            "manifest_sha256": "0" * 64,
        },
    )


def _mock_full_run(monkeypatch, rc=0) -> list[list[str]]:
    """Successful investigation with every child command stubbed; returns the captured argv."""
    calls = _capture_commands(monkeypatch, rc)
    _no_glue(monkeypatch)
    _mock_successful_investigation(monkeypatch)
    _passthrough_images(monkeypatch)
    return calls


def _mock_contract_run(monkeypatch, *, contract: bool = True) -> None:
    """Run task carries a sealed contract (final evaluation required) and images are pinned."""

    def prepare_task(paths) -> None:
        paths.task_dir.mkdir(parents=True, exist_ok=True)
        if contract:
            (paths.task_dir / "task_contract.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr("src.pipeline.cli.prepare_run_task", prepare_task)
    monkeypatch.setattr("src.pipeline.cli._pin_evaluation_images", lambda *_args: None)
    _mock_successful_investigation(monkeypatch)


def _pipeline_cfg(run_dir: Path, *, task_dir="examples/circpack", run_id="test-run", **paths):
    cfg = PipelineConfig()
    cfg.task.dir = str(task_dir)
    if run_id is not None:
        cfg.run_id = run_id
    cfg.paths.run_dir = str(run_dir)
    for key, value in paths.items():
        setattr(cfg.paths, key, str(value))
    return cfg


def _source_task(tmp_path: Path, name: str = "source-task", text: str = "source") -> Path:
    source = tmp_path / name
    source.mkdir()
    (source / "task.md").write_text(text, encoding="utf-8")
    return source


def _write_investigation_checkpoint(results_dir: Path, *, include_task: bool = True) -> None:
    program_dir = results_dir / "best_program"
    entrypoint = program_dir / "src" / "program.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("", encoding="utf-8")
    if include_task:
        (results_dir / "task").mkdir()
    (results_dir / "outer_state.json").write_text('{"current_cycle": 1}', encoding="utf-8")
    (results_dir / "best_program_manifest.json").write_text(
        '{"program_dir": "'
        + str(program_dir)
        + '", "entrypoint_path": "src/program.py", '
        + '"combined_score": 1.0, "correct": true}',
        encoding="utf-8",
    )


def _write_completed_run(run_dir: Path, task_md: str | None = None) -> None:
    _write_investigation_checkpoint(run_dir / "investigation")
    (run_dir / "task").mkdir()
    if task_md is not None:
        (run_dir / "task" / "task.md").write_text(task_md, encoding="utf-8")


def _write_legacy_run(results: Path) -> None:
    _write_investigation_checkpoint(results / "investigation", include_task=False)


def _write_task_snapshot(results: Path, task_name: str) -> Path:
    task_snapshot = results / "task"
    task_snapshot.mkdir(parents=True)
    (task_snapshot / ".agentic_task_snapshot.json").write_text(
        json.dumps({"task_name": task_name, "version": 1}) + "\n", encoding="utf-8"
    )
    return task_snapshot


def _write_posthoc_analysis(output_dir: Path) -> None:
    attempt = output_dir / "analyses" / "analysis_a" / "attempt_1"
    attempt.mkdir(parents=True)
    (attempt / "analysis_summary.md").write_text("# Finding\n", encoding="utf-8")
    (attempt / "interestingness_rubric.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "ranking.json").write_text(
        '{"valid":[{"analysis_id":"analysis_a","selected_attempt":"attempt_1"}]}\n',
        encoding="utf-8",
    )


def _write_published_paper(paper_dir: Path) -> None:
    paper_dir.mkdir(parents=True)
    paper_pdf = paper_dir / "paper.pdf"
    paper_pdf.write_bytes(b"%PDF-1.4\n")
    (paper_dir / "latest_manifest.json").write_text(
        json.dumps({"paper_pdf": str(paper_pdf.resolve())}), encoding="utf-8"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def _program(path: Path, text: str = "baseline = True\n") -> Path:
    (path / "src").mkdir(parents=True)
    (path / "src/program.py").write_text(text)
    (path / "program_spec.toml").write_text(render_default_program_spec_toml())
    return path


def _write_search_with_candidate(run_dir: Path) -> Path:
    """Exhausted search: retained baseline winner plus one scored non-winner candidate."""
    root = run_dir / "investigation"
    baseline = _program(root / ".baseline/program_0")
    _write_json(root / "best_program_manifest.json", {
        "program_dir": str(baseline), "entrypoint_path": "src/program.py",
        "combined_score": 0.811173, "correct": True, "source_attempt_id": "bootstrap",
    })
    _write_json(investigation_completion_receipt_path(root), investigation_completion_payload(root))
    result = _program(root / "ideas" / "20" / "program_1", "candidate = 1\n")
    _write_json(result / "metrics.json", {"combined_score": 0.809734})
    _write_json(result / "correct.json", {"correct": True})
    _write_json(result / "evaluation_manifest.json", {
        "producer": "harness_eval", "attempt_id": "20", "idea_id": "20",
        "program_id": result.name, "command": ["evaluate"], "started_at": "start",
        "finished_at": "end", "exit_code": 0,
    })
    _write_json(result / "rubric_status.json", {"overall_pass": False})
    (root / "attempts_history.jsonl").write_text(json.dumps({
        "attempt_id": "20", "experiment_dir": str(result.parent),
        "metrics": {"combined_score": 0.809734, "correct": True},
    }) + "\n")
    return root


# --------------------------------------------------------------------------- posthoc output safety


def test_posthoc_output_safety_guards(tmp_path: Path) -> None:
    # status writer does not follow a predictable temp symlink
    paths = derive_paths(
        _pipeline_cfg(tmp_path / "investigation", posthoc_dir=tmp_path / "posthoc")
    )
    paths.posthoc_dir.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("unchanged", encoding="utf-8")
    (paths.posthoc_dir / ".integration_status.json.tmp").symlink_to(outside)
    _record_posthoc_status(paths, status="degraded", reason="test", return_code=1)
    assert outside.read_text(encoding="utf-8") == "unchanged"
    assert (paths.posthoc_dir / "integration_status.json").is_file()

    # posthoc output must be contained by the investigation run envelope
    cfg = _pipeline_cfg(tmp_path / "investigation", posthoc_dir=tmp_path / "outside-posthoc")
    with pytest.raises(UnsafeGluePathError, match="run-envelope posthoc directory"):
        _prepare_posthoc_output(derive_paths(cfg), dry_run=False)

    # posthoc output parent must not be a symlink
    investigation = tmp_path / "symlinked" / "investigation"
    investigation.mkdir(parents=True)
    investigation.chmod(0o700)
    outside_dir = tmp_path / "symlinked" / "outside"
    outside_dir.mkdir()
    (investigation / "posthoc").symlink_to(outside_dir, target_is_directory=True)
    with pytest.raises(UnsafeGluePathError, match="fresh posthoc output already exists"):
        _prepare_posthoc_output(derive_paths(_pipeline_cfg(investigation)), dry_run=False)

    # posthoc output replacement is rejected before launch
    investigation = tmp_path / "replaced" / "investigation"
    investigation.mkdir(parents=True)
    investigation.chmod(0o700)
    paths = derive_paths(_pipeline_cfg(investigation))
    identity = _prepare_posthoc_output(paths, dry_run=False)
    assert identity is not None
    paths.posthoc_dir.rename(paths.posthoc_dir.with_name("original"))
    paths.posthoc_dir.mkdir(mode=0o700)
    with pytest.raises(UnsafeGluePathError, match="changed after preparation"):
        _verify_posthoc_output_identity(paths, identity)


# --------------------------------------------------------------------------- dry runs


def test_dry_run_phase_selection_previews_commands(monkeypatch, caplog, capsys, tmp_path) -> None:
    _no_subprocess(monkeypatch)

    # full dry run prints all four commands in phase order
    with caplog.at_level("INFO"):
        rc = main([CIRCPACK, "run_id=20260101_000000", f"paths.run_dir={tmp_path / 'full'}", "--dry-run"])
    assert rc == 0
    logged = "\n".join(_messages(caplog))
    assert f"$ {sys.executable} -m {INVESTIGATION}" in logged
    assert f"$ {sys.executable} -m src.variants_pipeline.cli" in logged
    assert f"$ {sys.executable} -m {POSTHOC}" in logged
    assert f"$ bash {PAPER}" in logged
    assert logged.index(INVESTIGATION) < logged.index("variants_pipeline")
    assert logged.index("variants_pipeline") < logged.index("posthoc_pipeline")
    assert logged.index("posthoc_pipeline") < logged.index("paper.sh")
    assert "--posthoc-dir" in logged

    # task-only dry run prints one non-ideating investigation command
    caplog.clear()
    with caplog.at_level("INFO"):
        rc = main([
            CIRCPACK, "execution.phases=task",
            "investigation.execution.draft_task_if_missing=true",
            f"paths.run_dir={tmp_path / 'task-only'}", "--dry-run",
        ])
    assert rc == 0
    logged = "\n".join(_messages(caplog))
    assert logged.count("$ ") == 1
    assert INVESTIGATION in logged
    assert "execution.task_only=true" in logged

    # final phase dry run follows ablations
    caplog.clear()
    with caplog.at_level("INFO"):
        rc = main([
            CIRCPACK, "execution.phases=investigation,ablations,final_evaluation",
            f"paths.run_dir={tmp_path / 'final'}", "--dry-run",
        ])
    assert rc == 0
    logged = "\n".join(_messages(caplog))
    assert FINAL in logged
    assert logged.index("src.variants_pipeline.cli") < logged.index(FINAL)

    # required final dry run previews review without final artifacts
    with monkeypatch.context() as m:
        m.setattr("src.pipeline.cli.task_requires_final_evaluation", lambda *_args: True)
        rc = main([
            CIRCPACK, "execution.phases=investigation,final_evaluation,paper,review",
            f"paths.run_dir={tmp_path / 'required-final'}", "--dry-run",
        ])
    assert rc == 0

    # json dry run stdout is strict event jsonl
    capsys.readouterr()
    rc = main([
        CIRCPACK, "execution.phases=investigation", f"paths.run_dir={tmp_path / 'json'}",
        "--dry-run", "--log-format=json",
    ])
    assert rc == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["event"] for record in records] == [
        "pipeline_start", "phase_start", "phase_end", "pipeline_end",
    ]
    assert {record["source"] for record in records} == {"pipeline"}
    start_fields = records[0]["fields"]
    assert start_fields["investigation_selected"] is True
    assert start_fields["agent.name"] == "claude"
    assert start_fields["execution.max_idea_cycles"] == 3
    assert start_fields["paths.results_dir"] == str(tmp_path / "json")


def test_dry_runs_are_write_free(monkeypatch, tmp_path: Path) -> None:
    _no_subprocess(monkeypatch)

    # investigation dry run does not prepare the task snapshot
    source = _source_task(tmp_path)
    results = tmp_path / "results"
    rc = main([f"task.dir={source}", "execution.phases=investigation", f"paths.run_dir={results}", "--dry-run"])
    assert rc == 0
    assert not results.exists()

    # legacy task snapshot dry run remains write-free
    legacy_source = _source_task(tmp_path, "legacy-task", "legacy source")
    legacy = tmp_path / "investigation"
    _write_legacy_run(legacy)
    rc = main([f"task.dir={legacy_source}", "execution.phases=ablations", f"paths.run_dir={legacy}", "--dry-run"])
    assert rc == 0
    assert not (legacy / "task").exists()


def test_dry_run_entrypoints_mint_run_id_once(monkeypatch, caplog) -> None:
    _no_subprocess(monkeypatch)
    cases = [
        ("execution_flag", [CIRCPACK, "execution.dry_run=true"], None),
        (
            "positional_task_dir",
            ["examples/circpack", "--dry-run", "paper.final_outcome_policy=qualified_only"],
            "examples/circpack",
        ),
    ]
    for name, argv, logged in cases:
        caplog.clear()
        with caplog.at_level("INFO"):
            rc = main(argv)
        assert rc == 0, name
        if logged is not None:
            assert any(logged in message for message in _messages(caplog)), name

    # run_id is minted once per run
    caplog.clear()
    minted: list[str] = []

    def _mint() -> str:
        minted.append(f"generated-{len(minted)}")
        return minted[-1]

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.paths._generate_run_id", _mint)
        with caplog.at_level("INFO"):
            rc = main([CIRCPACK, "--dry-run"])
    assert rc == 0
    assert minted == ["generated-0"]
    assert any("run_id:      generated-0" in message for message in _messages(caplog))


# --------------------------------------------------------------------------- posthoc best effort


def test_posthoc_phase_is_best_effort_for_paper(monkeypatch, caplog, tmp_path) -> None:
    # posthoc failure is best-effort and paper continues
    with monkeypatch.context() as m:
        calls = _mock_full_run(m, _fail_on(POSTHOC, 9))
        with caplog.at_level("WARNING"):
            rc = main([CIRCPACK, f"paths.run_dir={tmp_path / 'failed'}", "run_id=test-run"])
    assert rc == 0
    assert any(POSTHOC in call for call in calls)
    paper = next(call for call in calls if PAPER in call)
    assert "--posthoc-dir" not in paper
    assert any("best-effort" in message for message in _messages(caplog))

    # valid posthoc artifacts are forwarded to paper
    investigation = tmp_path / "valid"
    posthoc = investigation / "posthoc"

    def write_valid(argv: list[str]) -> int:
        if POSTHOC in argv:
            _write_posthoc_analysis(posthoc)
        return 0

    with monkeypatch.context() as m:
        calls = _mock_full_run(m, write_valid)
        rc = main([CIRCPACK, f"paths.run_dir={investigation}", f"paths.posthoc_dir={posthoc}", "run_id=test-run"])
    assert rc == 0
    paper = next(call for call in calls if PAPER in call)
    assert paper[paper.index("--posthoc-dir") + 1] == str(posthoc)
    status = json.loads((posthoc / "integration_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"

    # empty posthoc success degrades and paper continues
    investigation = tmp_path / "empty"
    with monkeypatch.context() as m:
        calls = _mock_full_run(m)
        rc = main([CIRCPACK, f"paths.run_dir={investigation}", "run_id=test-run"])
    assert rc == 0
    paper = next(call for call in calls if PAPER in call)
    assert "--posthoc-dir" not in paper
    status = json.loads((investigation / "posthoc" / "integration_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "degraded"
    assert status["reason"] == "no_valid_analyses"

    # posthoc output replaced after launch is not consumed
    investigation = tmp_path / "replaced"
    output = investigation / "posthoc" / "fresh"

    def replace_output(argv: list[str]) -> int:
        if POSTHOC in argv:
            output.rename(output.with_name("replaced-original"))
            _write_posthoc_analysis(output)
        return 0

    with monkeypatch.context() as m:
        calls = _mock_full_run(m, replace_output)
        rc = main([CIRCPACK, f"paths.run_dir={investigation}", f"paths.posthoc_dir={output}", "run_id=test-run"])
    assert rc == 0
    paper = next(call for call in calls if PAPER in call)
    assert "--posthoc-dir" not in paper
    assert not (output / "integration_status.json").exists()


def test_posthoc_setup_failure_does_not_block_paper(monkeypatch, tmp_path) -> None:
    cases = [
        ("preflight", "paths.posthoc_dir={tmp}/outside-posthoc"),
        ("config", "posthoc.execution.max_selected_analyses=-1"),
    ]
    for name, override in cases:
        run_dir = tmp_path / name
        _write_completed_run(run_dir, task_md="test task")
        with monkeypatch.context() as m:
            calls = _capture_commands(m)
            _no_glue(m)
            rc = main(["execution.phases=posthoc,paper", f"paths.run_dir={run_dir}", override.format(tmp=tmp_path)])
        assert rc == 0, name
        assert not any(POSTHOC in call for call in calls), name
        assert any(PAPER in call for call in calls), name


# --------------------------------------------------------------------------- run-local task + investigation


def test_investigation_prepares_run_local_task_and_repairs_empty_one(
    monkeypatch, tmp_path: Path
) -> None:
    source = _source_task(tmp_path)
    commands = _capture_commands(monkeypatch)
    _mock_successful_investigation(monkeypatch)

    # investigation prepares a run-local task without mutating the source
    results = tmp_path / "results"
    rc = main([f"task.dir={source}", "execution.phases=investigation", f"paths.run_dir={results}", "run_id=test-run"])
    assert rc == 0
    assert (results / "task" / "task.md").read_text(encoding="utf-8") == "source"
    assert list(source.iterdir()) == [source / "task.md"]
    assert len(commands) == 1
    task_override = next(item for item in commands[0] if item.startswith("paths.task_dir="))
    results_override = next(item for item in commands[0] if item.startswith("paths.results_dir="))
    assert task_override == f"paths.task_dir='{results / 'task'}'"
    assert results_override == f"paths.results_dir='{results / 'investigation'}'"
    assert stat.S_IMODE(results.stat().st_mode) == 0o700
    assert stat.S_IMODE((results / "investigation").stat().st_mode) == 0o700

    # investigation repairs a pre-existing empty run task
    repaired = tmp_path / "repaired"
    (repaired / "task").mkdir(parents=True)
    rc = main([f"task.dir={source}", "execution.phases=investigation", f"paths.run_dir={repaired}", "run_id=test-run"])
    assert rc == 0
    assert (repaired / "task" / "task.md").read_text(encoding="utf-8") == "source"
    assert task_snapshot_ready_path(repaired / "task").is_file()


def test_completed_no_winner_outcome_is_terminal_and_survives_downstream_resume(
    monkeypatch, tmp_path: Path
) -> None:
    source = _source_task(tmp_path)

    # completed investigation without a winner is a terminal outcome
    results = tmp_path / "results"
    with monkeypatch.context() as m:
        commands = _capture_commands(m)
        events = _capture_events(m)
        m.setattr(
            "src.pipeline.cli.validate_completed_investigation",
            lambda *_args: ["investigation produced no scored correct program"],
        )
        m.setattr(
            "src.pipeline.cli.completed_investigation_outcome", lambda *_args: "completed_no_winner"
        )
        m.setattr(
            "src.pipeline.cli.investigation_completion_payload",
            lambda *_args: {
                "schema_version": 1, "status": "completed", "outcome": "completed_no_winner",
                "manifest_sha256": "0" * 64,
            },
        )
        m.setattr("src.pipeline.cli.task_requires_final_evaluation", lambda *_args: True)
        rc = main([
            f"task.dir={source}",
            "execution.phases=investigation,ablations,final_evaluation,posthoc,paper,review",
            f"paths.run_dir={results}", "run_id=test-run",
        ])
    assert rc == 0
    assert len(commands) == 1
    assert _phase_end(events, "investigation")["outcome"] == "completed_no_winner"
    ablations_end = _phase_end(events, "ablations")
    assert ablations_end["status"] == "skipped"
    assert "without a qualifying winner" in str(ablations_end["reason"])
    skipped = {
        str(fields["phase_name"]): fields
        for event, fields in events
        if event == "phase_end" and fields.get("phase_name") != "investigation"
    }
    assert set(skipped) == {"ablations", "final_evaluation", "posthoc", "paper", "review"}
    assert all(fields["status"] == "skipped" for fields in skipped.values())
    pipeline_end = _pipeline_end(events)
    assert pipeline_end["outcome"] == "completed_no_winner"
    assert pipeline_end["status"] == "completed"

    # downstream resume preserves the completed_no_winner outcome
    results = tmp_path / "resume"
    investigation = results / "investigation"
    _write_investigation_checkpoint(investigation)
    manifest_path = investigation / "best_program_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["combined_score"] = None
    manifest["correct"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = investigation_completion_payload(investigation)
    assert receipt is not None
    investigation_completion_receipt_path(investigation).write_text(json.dumps(receipt), encoding="utf-8")
    commands = _capture_commands(monkeypatch)
    events = _capture_events(monkeypatch)
    monkeypatch.setattr(
        "src.pipeline.cli._recover_pending_revision_publication",
        lambda *_args: pytest.fail("terminal downstream restart must not recover revision state"),
    )
    rc = main([
        f"task.dir={source}",
        "execution.phases=ablations,final_evaluation,posthoc,paper,review,revision",
        f"paths.run_dir={results}", "run_id=test-run",
    ])
    assert rc == 0
    assert commands == []
    skipped_phases = {
        str(fields["phase_name"])
        for event, fields in events
        if event == "phase_end" and fields.get("status") == "skipped"
    }
    assert skipped_phases == {"ablations", "final_evaluation", "posthoc", "paper", "review", "revision"}
    assert _pipeline_end(events)["outcome"] == "completed_no_winner"


def test_fresh_investigation_rejects_symlinked_or_existing_workspace(monkeypatch, tmp_path: Path) -> None:
    source = _source_task(tmp_path)
    commands = _capture_commands(monkeypatch)

    # investigation rejects a symlinked agent workspace
    results = tmp_path / "symlinked"
    results.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (results / "investigation").symlink_to(outside, target_is_directory=True)
    rc = main([f"task.dir={source}", "execution.phases=investigation", f"paths.run_dir={results}", "run_id=test-run"])
    assert rc == 1
    assert commands == []
    assert list(outside.iterdir()) == []

    # fresh investigation rejects any existing workspace
    results = tmp_path / "existing"
    prepare_run_task(derive_paths(_pipeline_cfg(results, task_dir=source)))
    workspace = results / "investigation"
    workspace.mkdir()
    (workspace / "memory.md").write_text("stale agent memory", encoding="utf-8")
    rc = main([f"task.dir={source}", "execution.phases=investigation", f"paths.run_dir={results}", "run_id=test-run"])
    assert rc == 1
    assert commands == []
    assert not (results / ".agentic_resume_policy.json").exists()


# --------------------------------------------------------------------------- task drafting


def test_task_drafting_leaves_investigation_workspace_unclaimed(monkeypatch, tmp_path: Path) -> None:
    source = _source_task(tmp_path)
    results = tmp_path / "results"
    commands: list[list[str]] = []
    command_kwargs: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs: object) -> int:
        commands.append(command)
        command_kwargs.append(kwargs)
        if "execution.task_only=true" in command:
            (results / "task" / "evaluate.py").write_text("drafted", encoding="utf-8")
        return 0

    monkeypatch.setattr("src.pipeline.cli.run_cmd", fake_run)
    _mock_successful_investigation(monkeypatch)
    common = [f"task.dir={source}", f"paths.run_dir={results}", "run_id=test-run"]

    task_rc = main([
        *common, "execution.phases=task", "investigation.execution.draft_task_if_missing=true",
    ])

    assert task_rc == 0
    assert len(commands) == 1
    drafting_environment = command_kwargs[0]["env_overrides"]
    assert isinstance(drafting_environment, dict)
    assert float(drafting_environment[PIPELINE_STARTED_AT_ENV]) >= 0
    assert drafting_environment[TASK_RUNTIME_CACHE_HANDOFF_ENV] == "1"
    assert not (results / "investigation").exists()
    assert not (results / ".agentic_resume_policy.json").exists()
    assert not (results / ".agentic_task_handoff").exists()

    investigation_rc = main([*common, "execution.phases=investigation"])

    assert investigation_rc == 0
    assert len(commands) == 2
    assert command_kwargs[1]["env_overrides"] is None
    assert (results / "investigation").is_dir()
    assert (results / ".agentic_resume_policy.json").is_file()

    repeated_rc = main([*common, "execution.phases=investigation"])

    assert repeated_rc == 1
    assert len(commands) == 2


def test_task_drafting_rejects_removed_task_and_unsafe_output_collisions(
    monkeypatch, tmp_path: Path
) -> None:
    source = _source_task(tmp_path, "source")

    # task drafting fails when the child removes the canonical task
    run_dir = tmp_path / "removed"

    def remove_task(_command: list[str], **_kwargs: object) -> int:
        shutil.rmtree(run_dir / "task")
        return 0

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.cli.run_cmd", remove_task)
        rc = main([f"task.dir={source}", "execution.phases=task_drafting", f"paths.run_dir={run_dir}", "run_id=test-run"])
    assert rc == 1

    # task drafting rejects an unsafe output collision (file / symlink)
    monkeypatch.setattr(
        "src.pipeline.cli.run_cmd", lambda *_args, **_kwargs: pytest.fail("child command must not run")
    )
    for collision in ("file", "symlink"):
        run_dir = tmp_path / f"collision-{collision}"
        run_dir.mkdir()
        output = run_dir / "task_drafting"
        if collision == "file":
            output.write_text("unsafe", encoding="utf-8")
        else:
            outside = tmp_path / f"outside-{collision}"
            outside.mkdir()
            output.symlink_to(outside, target_is_directory=True)
        rc = main([f"task.dir={source}", "execution.phases=task_drafting", f"paths.run_dir={run_dir}", "run_id=test-run"])
        assert rc == 1, collision


def test_task_drafting_child_failure_codes_propagate_without_resealing(
    monkeypatch, tmp_path: Path
) -> None:
    source = _source_task(tmp_path, "source")
    monkeypatch.setattr("src.pipeline.cli._validated_drafted_scaffold", lambda *_args: True)

    # a valid scaffold is rejected after a full runtime failure
    for child_rc in (1, 2, 3, 4, 75):
        resealed: list[Path] = []
        with monkeypatch.context() as m:
            _capture_commands(m, child_rc)
            m.setattr("src.pipeline.cli.discard_managed_task_runtime", lambda *_args: None)
            m.setattr("src.pipeline.cli.reseal_run_task", lambda paths: resealed.append(paths.task_dir))
            rc = main([
                f"task.dir={source}", "execution.phases=task_drafting",
                f"paths.run_dir={tmp_path / f'run-{child_rc}'}", "run_id=test-run",
            ])
        assert rc == child_rc, child_rc
        assert resealed == [], child_rc

    # cancellation is preserved with a valid scaffold
    downstream_commands = _capture_commands(monkeypatch, 130)
    rc = main([
        f"task.dir={source}", "execution.phases=task_drafting,investigation",
        f"paths.run_dir={tmp_path / 'cancelled'}", "run_id=test-run",
    ])
    assert rc == 130
    assert len(downstream_commands) == 1


def test_new_final_requirement_stops_before_investigation(monkeypatch, tmp_path: Path, caplog) -> None:
    source = _source_task(tmp_path, "source", "Draft a task")
    monkeypatch.setattr(
        "src.pipeline.cli.task_requires_final_evaluation", lambda task: (task / "evaluate.py").exists()
    )
    for fail_fast in (True, False):
        results = tmp_path / f"results-{fail_fast}"

        def run(command: list[str], results: Path = results) -> int:
            assert "execution.task_only=true" in command, "downstream work must not start"
            (results / "task" / "evaluate.py").write_text("drafted", encoding="utf-8")
            return 0

        caplog.clear()
        with monkeypatch.context() as m:
            commands = _capture_commands(m, run)
            rc = main([
                f"task.dir={source}", "execution.phases=task_drafting,investigation,ablations,posthoc,paper",
                f"execution.fail_fast={str(fail_fast).lower()}", f"paths.run_dir={results}", "run_id=test-run",
            ])
        assert rc == 2, fail_fast
        assert len(commands) == 1, fail_fast
        assert "drafted task requires final" in caplog.text, fail_fast
        assert (results / "task" / "evaluate.py").is_file(), fail_fast
        assert not (results / "investigation").exists(), fail_fast


def test_compatible_drafted_task_reaches_investigation(monkeypatch, tmp_path: Path) -> None:
    source = _source_task(tmp_path, "source", "Draft a task")
    cases = [
        (True, "task_drafting,investigation,final_evaluation,paper"),
        (False, "task_drafting,investigation,paper"),
        (True, "task_drafting,investigation"),
    ]
    for index, (requires_final, phases) in enumerate(cases):
        results = tmp_path / f"results-{index}"

        def run(command: list[str], results: Path = results) -> int:
            if "execution.task_only=true" in command:
                (results / "task" / "evaluate.py").write_text("drafted", encoding="utf-8")
                return 0
            return 42  # Stop at the downstream boundary; do not evaluate a candidate.

        with monkeypatch.context() as m:
            commands = _capture_commands(m, run)
            m.setattr("src.pipeline.cli.task_requires_final_evaluation", lambda task: requires_final)
            rc = main([f"task.dir={source}", f"execution.phases={phases}", f"paths.run_dir={results}", "run_id=test-run"])
        assert rc == 42, phases
        assert len(commands) == 2, phases


def test_task_contract_and_logger_failures_stop_before_downstream_phases(tmp_path, monkeypatch):
    source = _source_task(tmp_path, "source", "Draft a task")
    monkeypatch.setattr("src.pipeline.cli.task_requires_final_evaluation", lambda task: False)
    phases = "execution.phases=task_drafting,investigation,ablations,posthoc,paper"

    # a task contract failure after drafting stops before downstream phases
    cases = [
        (fail_fast, terminal_code, category)
        for terminal_code, category in [
            (86, "evaluator_output_contract_invalid"), (87, "agent_log_persistence_failed"),
        ]
        for fail_fast in (True, False)
    ]
    for fail_fast, terminal_code, category in cases:
        name = f"{terminal_code}-{fail_fast}"
        results = tmp_path / f"results-{name}"

        def run(command: list[str], results: Path = results, terminal_code: int = terminal_code) -> int:
            if "execution.task_only=true" in command:
                (results / "task/evaluate.py").write_text("drafted")
                return 0
            return terminal_code

        with monkeypatch.context() as m:
            commands = _capture_commands(m, run)
            rc = main([
                f"task.dir={source}", f"paths.run_dir={results}", "run_id=contract-failure",
                phases, f"execution.fail_fast={str(fail_fast).lower()}",
            ])
        assert rc == terminal_code, name
        assert len(commands) == 2, name
        events = _journal(results)
        phase_events = [event["fields"] for event in events if event["event"] == "phase_end"]
        assert [event["phase_name"] for event in phase_events] == ["task_drafting", "investigation"], name
        assert phase_events[-1]["failure_category"] == category, name
        ending = next(event for event in events if event["event"] == "pipeline_end")
        assert ending["fields"]["status"] == "failed", name
        assert ending["fields"].get("outcome") != "completed_no_winner", name

    # a drafting logger failure stops the pipeline even without fail-fast
    results = tmp_path / "log-failure"
    commands = _capture_commands(monkeypatch, 87)
    rc = main([
        f"task.dir={source}", f"paths.run_dir={results}", "run_id=log-failure",
        phases, "execution.fail_fast=false",
    ])
    assert rc == 87
    assert len(commands) == 1
    phase_events = [event["fields"] for event in _journal(results) if event["event"] == "phase_end"]
    assert len(phase_events) == 1
    assert phase_events[0]["failure_scope"] == "infrastructure"


# --------------------------------------------------------------------------- legacy task snapshots


def test_ablations_only_uses_stored_task_name_and_rejects_invalid_one(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    _passthrough_images(monkeypatch)

    # ablations-only uses the task name stored in the snapshot
    results = tmp_path / "stored"
    _write_task_snapshot(results, "stored-task")
    _write_legacy_run(results)
    commands = _capture_commands(monkeypatch)
    rc = main(["execution.phases=ablations", f"paths.run_dir={results}", "run_id=test-run"])
    assert rc == 0
    assert "paths.task_name='stored-task'" in commands[0]
    assert any(argument.endswith("/ablations'") for argument in commands[0])

    # ablations-only rejects an invalid stored task name
    results = tmp_path / "invalid"
    _write_task_snapshot(results, "../escape")
    _write_legacy_run(results)
    with monkeypatch.context() as m, caplog.at_level("WARNING"):
        m.setattr("src.pipeline.cli.run_cmd", lambda *_args, **_kwargs: pytest.fail("child command must not run"))
        rc = main(["execution.phases=ablations", f"paths.run_dir={results}"])
    assert rc == 1
    assert any("invalid task name" in message for message in _messages(caplog))


def test_legacy_task_snapshot_bootstrap_requires_explicit_source(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    source = _source_task(tmp_path, "legacy-task", "legacy source")
    _capture_commands(monkeypatch)
    _passthrough_images(monkeypatch)
    for phase in ("investigation", "ablations"):
        resume_override = ["investigation.execution.resume=true"] if phase == "investigation" else []

        # legacy task snapshot is bootstrapped from an explicit source
        results = tmp_path / f"bootstrap-{phase}"
        _write_legacy_run(results)
        rc = main([f"task.dir={source}", f"execution.phases={phase}", *resume_override, f"paths.run_dir={results}"])
        assert rc == 0, phase
        assert (results / "task" / "task.md").read_text(encoding="utf-8") == "legacy source", phase
        assert (results / "task" / ".agentic_task_snapshot.json").is_file(), phase

        # legacy task bootstrap requires an explicit source
        results = tmp_path / f"missing-{phase}"
        _write_legacy_run(results)
        caplog.clear()
        with caplog.at_level("ERROR"):
            rc = main([f"execution.phases={phase}", f"paths.run_dir={results}", *resume_override])
        assert rc == 1, phase
        assert not (results / "task").exists(), phase
        assert any("explicit task.dir" in message for message in _messages(caplog)), phase


def test_legacy_run_task_layouts_are_reused_or_rejected(monkeypatch, tmp_path: Path) -> None:
    _passthrough_images(monkeypatch)

    # investigation resume rejects a flat legacy workspace
    source = _source_task(tmp_path, "legacy-task", "legacy source")
    results = tmp_path / "legacy-run"
    _write_investigation_checkpoint(results, include_task=False)
    commands = _capture_commands(monkeypatch)
    rc = main([
        f"task.dir={source}", "execution.phases=investigation",
        "investigation.execution.resume=true", f"paths.run_dir={results}",
    ])
    assert rc == 1
    assert commands == []
    assert not (results / "investigation").exists()

    # ablations-only reuses a mutated versioned run task
    source = _source_task(tmp_path)
    results = tmp_path / "investigation"
    paths = derive_paths(_pipeline_cfg(results, task_dir=source))
    prepare_run_task(paths)
    (paths.task_dir / "task.md").write_text("drafted task", encoding="utf-8")
    _write_legacy_run(results)
    rc = main([f"task.dir={source}", "execution.phases=ablations", f"paths.run_dir={results}", "run_id=test-run"])
    assert rc == 0
    assert (paths.task_dir / "task.md").read_text(encoding="utf-8") == "drafted task"


# --------------------------------------------------------------------------- argv parsing / run_id


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        pytest.param("run_id=null", "generated-run", id="null"),
        pytest.param("run_id= null", "generated-run", id="null_space"),
        pytest.param("run_id=~", "generated-run", id="tilde"),
        pytest.param("run_id='abc'", "abc", id="quoted"),
        pytest.param("run_id=${oc.env:PIPELINE_RUN_ID}", "resolved-id", id="env_interpolation"),
        pytest.param("run_id=${agent.name}", "claude", id="config_reference"),
    ],
)
def test_run_id_override_is_resolved(monkeypatch, caplog, override: str, expected: str) -> None:
    _no_subprocess(monkeypatch)
    monkeypatch.setattr("src.pipeline.paths._generate_run_id", lambda: "generated-run")
    monkeypatch.setenv("PIPELINE_RUN_ID", "resolved-id")

    with caplog.at_level("INFO"):
        rc = main([CIRCPACK, override, "--dry-run"])

    assert rc == 0
    messages = _messages(caplog)
    assert any(f"run_id:      {expected}" in message for message in messages)
    assert not any(f"run_id:      '{expected}'" in message for message in messages)


INVALID_INVOCATIONS = [
    ("nested_run_id", [CIRCPACK, "run_id='/tmp/escape'"], "run_id must be one non-empty path component"),
    ("parent_task_dir", ["task.dir=examples/circpack/.."], "task.dir must identify one named directory"),
    ("invalid_phase", ["execution.phases=bogus"], "Invalid pipeline phases"),
    ("paper_only_without_anchor", [CIRCPACK, "execution.phases=paper"], "paths.run_dir"),
    ("two_positionals", ["examples/circpack", "examples/circpack"], "only one positional task directory"),
    ("positional_and_override", ["examples/circpack", CIRCPACK], "conflicts with an explicit task.dir="),
    ("interpolated_positional", ["${oc.env:HOME}"], "must not contain interpolations"),
    *[
        (f"hydra:{override}", [CIRCPACK, override], "Config error:")
        for override in [
            "unknown.key=1", "paper.paper_format=[broken",
            "task.dir=~missing_pipeline_user/task", "agent=foo", "paths=1", "paper=null",
            "container=x",
        ]
    ],
]


def test_help_output_and_invalid_invocations(monkeypatch, caplog, capsys) -> None:
    # --help prints schema keys and the positional task dir
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "task.dir" in out
    assert "execution.phases" in out
    assert "container.network" in out
    assert "Usage: agsci run [task_dir]" in out

    # invalid invocations return a config error
    _no_subprocess(monkeypatch)
    for name, argv, message in INVALID_INVOCATIONS:
        caplog.clear()
        with caplog.at_level("ERROR"):
            rc = main([*argv, "--dry-run"])
        assert rc == 1, name
        assert any(message in record for record in _messages(caplog)), name


def test_resume_phase_selections_preview_only_selected_commands(monkeypatch, tmp_path, caplog) -> None:
    _no_subprocess(monkeypatch)

    # paper-only with an existing anchor
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir)
    with caplog.at_level("INFO"):
        rc = main([CIRCPACK, "execution.phases=paper", f"paths.run_dir={run_dir}", "--dry-run"])
    assert rc == 0
    logged = "\n".join(_messages(caplog))
    assert f"$ bash {PAPER}" in logged
    assert INVESTIGATION not in logged

    # comma-separated resume phases bypass hydra sweep parsing
    results = tmp_path / "investigation"
    results.mkdir()
    _write_investigation_checkpoint(results / "investigation")
    caplog.clear()
    with caplog.at_level("INFO"):
        rc = main([CIRCPACK, "execution.phases=ablations,paper", f"paths.run_dir={results}", "--dry-run"])
    assert rc == 0
    logged = "\n".join(_messages(caplog))
    assert f"$ {sys.executable} -m src.variants_pipeline.cli" in logged
    assert f"$ bash {PAPER}" in logged
    assert f"$ {sys.executable} -m {INVESTIGATION}" not in logged

    # quoted and interpolated resume phases remain compatible
    cases = [
        ("single_quoted", "execution.phases='ablations,paper'", None),
        ("double_quoted", 'execution.phases="ablations,paper"', None),
        ("env", "execution.phases=${oc.env:PIPELINE_PHASES}", "ablations,paper"),
        ("env_default", "execution.phases=${oc.env:PIPELINE_PHASES,'ablations,paper'}", None),
    ]
    for name, phases_override, env_phases in cases:
        with monkeypatch.context() as m:
            if env_phases is None:
                m.delenv("PIPELINE_PHASES", raising=False)
            else:
                m.setenv("PIPELINE_PHASES", env_phases)
            rc = main([CIRCPACK, phases_override, f"paths.run_dir={results}", "--dry-run"])
        assert rc == 0, name


# --------------------------------------------------------------------------- required / optional final evaluation


def test_paper_rejects_contract_discovered_in_consumed_snapshot(monkeypatch, tmp_path, caplog) -> None:
    run_dir = tmp_path / "run"
    _write_completed_run(run_dir)

    def expose_contract(*_args: object, **_kwargs: object) -> None:
        (run_dir / "task" / "task_contract.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr("src.pipeline.cli.run_glue", expose_contract)
    monkeypatch.setattr(
        "src.pipeline.cli.run_cmd", lambda *_args, **_kwargs: pytest.fail("paper command must not run")
    )

    with caplog.at_level("ERROR"):
        rc = main([CIRCPACK, "execution.phases=paper", f"paths.run_dir={run_dir}", "container.enabled=false", "container.paper_enabled=true", "run_id=test-run"])

    assert rc == 2
    assert any("requires a successful final phase" in message for message in _messages(caplog))


def test_required_final_failure_blocks_paper_and_review_and_hints_resume(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    _mock_contract_run(monkeypatch)
    run_id = "run_id=20260829_173527"

    # required final failure stops downstream phases by default
    with monkeypatch.context() as m:
        calls = _capture_commands(m, _fail_on(FINAL))
        m.setattr("src.pipeline.cli.run_glue", lambda *_args, **_kwargs: pytest.fail("paper glue must not run"))
        with caplog.at_level("ERROR"):
            rc = main([
                CIRCPACK, "execution.phases=investigation,ablations,final_evaluation,posthoc,paper,review",
                f"paths.run_dir={tmp_path / 'default'}", "container.enabled=true", run_id,
            ])
    assert rc == 7
    assert any(FINAL in command for command in calls)
    assert not any(POSTHOC in command for command in calls)
    assert not any(PAPER in command for command in calls)
    messages = _messages(caplog)
    assert any("Final evaluation failed (rc=7)" in message for message in messages)
    assert any(
        message.strip() == "execution.phases=final_evaluation,posthoc,paper,review" for message in messages
    )
    assert any("run_id=20260829_173527" in message for message in messages)
    assert not any("paper will continue" in message for message in messages)

    # fail_fast=false runs posthoc but skips paper after a required final failure
    caplog.clear()
    with monkeypatch.context() as m:
        calls = _capture_commands(m, _fail_on(FINAL))
        m.setattr("src.pipeline.cli._current_posthoc_artifacts_available", lambda *_args: True)
        m.setattr("src.pipeline.cli.run_glue", lambda *_args, **_kwargs: pytest.fail("paper glue must not run"))
        with caplog.at_level("INFO"):
            rc = main([
                CIRCPACK, "execution.phases=investigation,final_evaluation,posthoc,paper,review",
                "execution.fail_fast=false", f"paths.run_dir={tmp_path / 'no-fail-fast'}",
                "container.enabled=true", run_id,
            ])
    assert rc == 7
    assert any(POSTHOC in command for command in calls)
    assert not any(PAPER in command for command in calls)
    messages = _messages(caplog)
    assert any("Skipping paper" in message and "final" in message for message in messages)
    assert any("Skipping review" in message and "final" in message for message in messages)

    # fail_fast=false skips review without a selected paper after a required final failure
    caplog.clear()
    run_dir = tmp_path / "review"
    paper_dir = run_dir / "paper"
    _write_published_paper(paper_dir)
    with monkeypatch.context() as m:
        calls = _capture_commands(m, _fail_on(FINAL))
        m.setattr("src.pipeline.cli._current_posthoc_artifacts_available", lambda *_args: True)
        m.setattr("src.pipeline.cli.run_pipeline_review", lambda **_kwargs: pytest.fail("review must not run"))
        with caplog.at_level("INFO"):
            rc = main([
                CIRCPACK, "execution.phases=investigation,final_evaluation,posthoc,review",
                "execution.fail_fast=false", f"paths.run_dir={run_dir}", f"paths.paper_dir={paper_dir}",
                "container.enabled=true", run_id,
            ])
    assert rc == 7
    assert any(POSTHOC in command for command in calls)
    assert any(
        "Skipping review" in message and "final evaluation failed" in message
        for message in _messages(caplog)
    )

    # the final resume hint rejects a retained once-only invocation
    caplog.clear()
    cfg = _pipeline_cfg(tmp_path / "hint", run_id="20260829_173527")
    cfg.execution.phases = "final_evaluation,posthoc,paper,review"
    paths = derive_paths(cfg)
    paths.final_dir.mkdir(parents=True)
    (paths.final_dir / "invocation.json").write_text("{}\n", encoding="utf-8")
    (paths.final_dir / "final_manifest.json").write_text("malformed\n", encoding="utf-8")
    with caplog.at_level("ERROR"):
        _log_resume_hint("final_evaluation", cfg, paths)
    messages = _messages(caplog)
    assert any("once-only invocation was retained" in message for message in messages)
    assert any("run_id=20260829_173527" in message for message in messages)
    assert not any("Rerun the original command" in message for message in messages)


def test_completed_final_outcome_is_verified_before_publication(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    _mock_contract_run(monkeypatch)
    calls = _capture_commands(monkeypatch)

    # a non-qualifying final completes without publication
    run_dir = tmp_path / "not-qualified"
    with monkeypatch.context() as m:
        m.setattr("src.pipeline.cli._verified_final_outcome", lambda *_args: "not_qualified")
        m.setattr("src.pipeline.cli._current_posthoc_artifacts_available", lambda *_args: True)
        m.setattr("src.pipeline.cli.run_glue", lambda *_args, **_kwargs: pytest.fail("paper glue must not run"))
        m.setattr("src.pipeline.cli.run_pipeline_review", lambda **_kwargs: pytest.fail("review must not run"))
        with caplog.at_level("INFO"):
            rc = main([
                CIRCPACK, "execution.phases=investigation,final_evaluation,posthoc,paper,review",
                f"paths.run_dir={run_dir}", "container.enabled=true", "run_id=20260829_173527",
            ])
    assert rc == 0
    assert any(FINAL in command for command in calls)
    assert any(POSTHOC in command for command in calls)
    assert not any(PAPER in command for command in calls)
    messages = _messages(caplog)
    assert any("Skipping paper" in message and "did not qualify" in message for message in messages)
    assert any("Skipping review" in message and "did not qualify" in message for message in messages)
    assert not any("Expected paper PDF" in message for message in messages)
    assert not any("Expected AI review" in message for message in messages)
    records = _journal(run_dir)
    final_end = next(
        record for record in records
        if record["event"] == "phase_end" and record["fields"].get("phase_name") == "final_evaluation"
    )
    pipeline_end = next(record for record in records if record["event"] == "pipeline_end")
    assert final_end["fields"]["status"] == "completed"
    assert final_end["fields"]["outcome"] == "not_qualified"
    assert pipeline_end["fields"]["status"] == "completed"
    assert pipeline_end["fields"]["outcome"] == "not_qualified"

    # an unverifiable completed final remains a pipeline failure
    caplog.clear()

    def reject_final(*_args: object) -> str:
        raise FinalEvaluationError("tampered result")

    monkeypatch.setattr("src.pipeline.cli._verified_final_outcome", reject_final)
    with caplog.at_level("ERROR"):
        rc = main([
            CIRCPACK, "execution.phases=investigation,final_evaluation",
            f"paths.run_dir={tmp_path / 'unverifiable'}", "container.enabled=true", "run_id=20260829_173527",
        ])
    assert rc == 1
    assert any("Could not verify completed final evaluation" in message for message in _messages(caplog))


def test_optional_final_outcome_never_blocks_paper(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setattr("src.pipeline.cli.task_requires_final_evaluation", lambda *_args: False)
    _no_glue(monkeypatch)

    # a failed or non-qualifying optional final lets paper continue without final input
    for outcome in ("failed", "not_qualified"):
        caplog.clear()
        with monkeypatch.context() as m:
            _mock_contract_run(m, contract=outcome == "failed")
            calls = _capture_commands(m, _fail_on(FINAL) if outcome == "failed" else 0)
            if outcome == "not_qualified":
                m.setattr("src.pipeline.cli._verified_final_outcome", lambda *_args: "not_qualified")
            with caplog.at_level("WARNING"):
                rc = main([
                    CIRCPACK, "execution.phases=investigation,final_evaluation,paper",
                    f"paths.run_dir={tmp_path / outcome}", "container.enabled=true", "run_id=20260829_173527",
                ])
        assert rc == 0, outcome
        paper_command = next(command for command in calls if PAPER in command)
        assert "--final-evaluation-dir" not in paper_command, outcome
        assert "--no-final-evaluation" in paper_command, outcome
        if outcome == "failed":
            assert any(
                "Final evaluation best-effort phase degraded" in message
                and "remaining selected phases will continue" in message
                for message in _messages(caplog)
            )

    # optional paper-only ignores an existing non-qualifying final
    run_dir = tmp_path / "paper-only"
    _write_completed_run(run_dir)
    final_dir = run_dir / "final_evaluation"
    final_dir.mkdir()
    (final_dir / "final_manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("src.pipeline.cli._verified_final_outcome", lambda *_args: "not_qualified")
    calls = _capture_commands(monkeypatch)
    rc = main([CIRCPACK, "execution.phases=paper", f"paths.run_dir={run_dir}", "container.enabled=false", "container.paper_enabled=true", "run_id=20260829_173527"])
    assert rc == 0
    paper_command = next(command for command in calls if PAPER in command)
    assert "--final-evaluation-dir" not in paper_command
    assert "--no-final-evaluation" in paper_command


# --------------------------------------------------------------------------- fresh outputs and journals


def test_late_phases_keep_fresh_output_and_journal_at_run_root(monkeypatch, tmp_path) -> None:
    # posthoc-only keeps the fresh output available for the child
    investigation = tmp_path / "investigation"
    investigation.mkdir()
    investigation.chmod(0o700)
    bundle = tmp_path / "bundle"
    (bundle / "task").mkdir(parents=True)
    with monkeypatch.context() as m:
        m.setattr("src.pipeline.cli.ensure_run_task_for_reuse", lambda *args, **kwargs: True)
        m.setattr("src.pipeline.cli.reuse_run_task", lambda *args, **kwargs: None)
        m.setattr("src.pipeline.cli._current_posthoc_artifacts_available", lambda *_args: True)
        commands = _capture_commands(m)
        rc = main([
            CIRCPACK, "execution.phases=posthoc", f"paths.run_dir={investigation}",
            f"paths.ablations_dir={bundle}", "container.enabled=false", "container.paper_enabled=true", "run_id=test-run",
        ])
    assert rc == 0
    assert len(commands) == 1
    assert POSTHOC in commands[0]
    assert (investigation / "pipeline_events.jsonl").is_file()
    assert not (investigation / "posthoc" / "pipeline_events.jsonl").exists()

    # paper-only keeps the fresh output available for the child and journals its events
    run_dir = tmp_path / "paper-run"
    _write_completed_run(run_dir)

    def assert_fresh_output(command: list[str], **kwargs: object) -> int:
        output_dir = Path(command[command.index("--output-dir") + 1])
        assert not output_dir.exists()
        assert "--resume-existing" not in command
        assert command[command.index("--event-transport") + 1] == "wrapper"
        output_line_fn = kwargs["output_line_fn"]
        assert callable(output_line_fn)
        record = {
            "timestamp": "2026-09-02T10:00:00.000Z", "source": "paper", "event": "paper_gate",
            "session_id": "paper-test", "gate": "draft", "state": "completed", "fields": {"round": 1},
        }
        assert output_line_fn(f"AGENTIC_SCIENTIST_PAPER_EVENT {json.dumps(record)}\n") == ""
        return 0

    with monkeypatch.context() as m:
        _no_glue(m)
        m.setattr("src.pipeline.cli.run_cmd", assert_fresh_output)
        rc = main([CIRCPACK, "execution.phases=paper", f"paths.run_dir={run_dir}", "container.enabled=false", "container.paper_enabled=true", "run_id=test-run"])
    assert rc == 0
    assert any(record["event"] == "paper_gate" for record in _journal(run_dir))

    # late phases use an existing journal root
    run_dir = tmp_path / "late-run"
    _write_completed_run(run_dir, task_md="task")
    with monkeypatch.context() as m:
        m.setattr("src.pipeline.cli.ensure_run_task_for_reuse", lambda *args, **kwargs: True)
        m.setattr("src.pipeline.cli.reuse_run_task", lambda *args, **kwargs: None)
        _capture_commands(m)
        rc = main([CIRCPACK, "execution.phases='posthoc,paper'", f"paths.run_dir={run_dir}", "container.enabled=false", "container.paper_enabled=true", "run_id=test-run"])
    assert rc == 0
    assert (run_dir / "pipeline_events.jsonl").is_file()


# --------------------------------------------------------------------------- investigation resume budgets and hints


def _resume_investigation_run(monkeypatch, run_dir: Path, overrides: list[str]) -> list[list[str]]:
    _write_investigation_checkpoint(run_dir / "investigation")
    monkeypatch.setattr("src.pipeline.cli.prepare_run_task", lambda *args, **kwargs: None)
    commands = _capture_commands(monkeypatch)
    rc = main([
        CIRCPACK, "execution.phases=investigation", f"paths.run_dir={run_dir}",
        "investigation.execution.resume=true", *overrides,
    ])
    assert rc == 0
    return commands


def test_investigation_resume_idea_cycle_budget_record(monkeypatch, caplog, tmp_path) -> None:
    cases = [
        ("uses_record", 7, [], 7, None, "recorded no idea-cycle budget"),
        (
            "explicit_overrides_record", 7, ["investigation.execution.max_idea_cycles=9"], 9,
            "recorded budget of 7", None,
        ),
        (
            "no_record", None, ["investigation.execution.max_idea_cycles=3"], 3,
            "recorded no idea-cycle budget", None,
        ),
    ]
    for name, recorded, overrides, budget, present, absent in cases:
        run_dir = tmp_path / name
        run_dir.mkdir()
        if recorded is not None:
            write_resume_policy(run_dir, max_idea_cycles=recorded)
        caplog.clear()
        with monkeypatch.context() as m, caplog.at_level("WARNING"):
            commands = _resume_investigation_run(m, run_dir, overrides)
        assert f"execution.max_idea_cycles={budget}" in commands[0], name
        messages = _messages(caplog)
        if present is not None:
            assert any(present in message for message in messages), name
        if absent is not None:
            assert not any(absent in message for message in messages), name

    # a corrupt budget record is rejected
    run_dir = tmp_path / "corrupt"
    run_dir.mkdir()
    _write_investigation_checkpoint(run_dir / "investigation")
    (run_dir / ".agentic_resume_policy.json").write_text("{", encoding="utf-8")
    _capture_commands(monkeypatch)
    caplog.clear()
    with caplog.at_level("ERROR"):
        rc = main([CIRCPACK, "execution.phases=investigation", f"paths.run_dir={run_dir}", "investigation.execution.resume=true"])
    assert rc == 1
    assert any("pass investigation.execution.max_idea_cycles explicitly" in message for message in _messages(caplog))


def test_investigation_failure_resume_hints(monkeypatch, caplog, tmp_path) -> None:
    monkeypatch.setattr("src.pipeline.cli.prepare_run_task", lambda *args, **kwargs: None)
    _capture_commands(monkeypatch, 7)

    # the failure hint preserves the resume contract
    run_dir = tmp_path / "resume"
    run_dir.mkdir()
    _write_investigation_checkpoint(run_dir / "investigation")
    with caplog.at_level("ERROR"):
        rc = main([
            CIRCPACK, "execution.phases=investigation", "run_id=20260101_000000",
            f"paths.run_dir={run_dir}", "investigation.execution.resume=true",
            "investigation.execution.max_idea_cycles=9",
            "investigation.execution.max_program_evals_per_attempt=4",
        ])
    assert rc == 7
    messages = _messages(caplog)
    assert any("Rerun the original command" in message for message in messages)
    assert any("execution.phases=investigation" == message.strip() for message in messages)
    assert any("investigation.execution.resume=true" in message for message in messages)
    assert any("run_id=20260101_000000" in message for message in messages)
    assert any("investigation.execution.max_idea_cycles=9" in message for message in messages)
    assert any("investigation.execution.max_program_evals_per_attempt=4" in message for message in messages)

    # the failure hint does not suggest an unsupported resume
    cases = [
        ("early_failure", [], ["No usable investigation checkpoint", "new paths.run_dir"]),
        *[
            (
                f"pruned_workers_{workers}",
                [
                    f"investigation.execution.workers={workers}",
                    "investigation.execution.max_idea_cycles=3",
                    "investigation.execution.prune_every_n_cycles=3",
                ],
                ["cannot be resumed"],
            )
            for workers in (1, 2)
        ],
    ]
    for name, overrides, expected in cases:
        run_dir = tmp_path / name
        run_dir.mkdir()
        caplog.clear()
        with caplog.at_level("ERROR"):
            rc = main([CIRCPACK, "execution.phases=investigation", f"paths.run_dir={run_dir}", *overrides])
        assert rc == 7, name
        messages = _messages(caplog)
        for fragment in expected:
            assert any(fragment in message for message in messages), name
        assert not any("investigation.execution.resume=true" in message for message in messages), name


# --------------------------------------------------------------------------- fail-fast chains


def test_failed_phases_skip_dependents_and_report_failure(monkeypatch, caplog, tmp_path: Path) -> None:
    # fail_fast=false skips the failed dependency chain and returns the failure
    with monkeypatch.context() as m:
        calls = _capture_commands(m, 7)
        _no_glue(m)
        with caplog.at_level("INFO"):
            rc = main([CIRCPACK, "execution.fail_fast=false", f"paths.run_dir={tmp_path / 'chain'}", "run_id=20260101_000000"])
    assert rc == 7
    assert len(calls) == 1
    messages = _messages(caplog)
    assert any("Skipping ablations" in message for message in messages)
    assert any("Skipping paper" in message for message in messages)
    assert any("Pipeline finished with failures" in message for message in messages)
    assert not any(message.startswith("Pipeline complete.") for message in messages)

    # failed selected ablations skip posthoc while paper continues
    caplog.clear()
    checkpoints: list[str] = []
    with monkeypatch.context() as m:
        calls = _capture_commands(m, _fail_on("src.variants_pipeline.cli"))
        m.setattr("src.pipeline.cli.prepare_run_task", lambda *args, **kwargs: None)
        _mock_successful_investigation(m)
        m.setattr(
            "src.pipeline.cli._pipeline_checkpoint", lambda *_args, **kwargs: checkpoints.append(kwargs["label"])
        )
        with caplog.at_level("WARNING"):
            rc = main([
                CIRCPACK, "execution.phases=investigation,ablations,posthoc,paper",
                f"paths.run_dir={tmp_path / 'ablations'}", "run_id=20260101_000000",
            ])
    assert rc == 0
    assert not any(POSTHOC in call for call in calls)
    assert any(PAPER in call for call in calls)
    messages = _messages(caplog)
    assert any("Ablations best-effort phase degraded" in message for message in messages)
    assert any("completed with degraded optional phases" in message for message in messages)
    assert any("Skipping posthoc because selected ablations did not succeed" in message for message in messages)
    assert "posthoc phase" in checkpoints

    # investigation failure still blocks paper
    caplog.clear()
    return_codes = iter([7, 0])
    glue_calls: list[object] = []
    with monkeypatch.context() as m:
        calls = _capture_commands(m, lambda _command: next(return_codes))
        m.setattr("src.pipeline.cli.prepare_run_task", lambda *args, **kwargs: None)
        m.setattr("src.pipeline.cli.run_glue", lambda *args, **kwargs: glue_calls.append(args))
        with caplog.at_level("INFO"):
            rc = main([
                CIRCPACK, "execution.phases=investigation,paper", "execution.fail_fast=false",
                f"paths.run_dir={tmp_path / 'investigation'}", "run_id=20260101_000000",
            ])
    assert rc == 7
    assert len(calls) == 1
    assert len(glue_calls) == 0
    assert any("Skipping paper" in message for message in _messages(caplog))


# --------------------------------------------------------------------------- no-winner studies / downstream selection


def test_policy_defaults_validation_and_forwarding(tmp_path) -> None:
    # policy defaults and validation
    for section, key, default, invalid in [
        ("execution", "no_winner_policy", "best_available", "promote"),
        ("paper", "final_outcome_policy", "qualified_only", "force"),
    ]:
        cfg = cli.load_pipeline_config(["execution.phases=investigation"])
        assert getattr(getattr(cfg, section), key) == default, key
        setattr(getattr(cfg, section), key, invalid)
        assert any(key in error for error in config.validate_pipeline_config(cfg)), key

    # paper.final_outcome_policy loads and is forwarded to both manuscript commands
    for value in ("qualified_only", "allow_not_qualified"):
        cfg = cli.load_pipeline_config([f"paper.final_outcome_policy={value}"])
        cfg.paths.run_dir = str(tmp_path / value)
        paths = derive_paths(cfg)
        commands = [build_paper_cmd(cfg, paths), build_revision_command(
            cfg=cfg, paths=paths, review_path=tmp_path / "review.json", output_dir=tmp_path / "edit",
        )]
        assert cfg.paper.final_outcome_policy == value
        for command in commands:
            assert command[command.index("--final-outcome-policy") + 1] == value, value


def _mock_study(monkeypatch, tmp_path: Path):
    """Exhausted search with a scored non-winner candidate; every downstream phase stubbed."""
    commands = _capture_commands(monkeypatch)
    events = _capture_events(monkeypatch)
    selection = {
        "selection_kind": "non_winner_candidate", "combined_score": 0.809734,
        "baseline_score": 0.811173, "source_attempt_id": "attempt_20",
        "program_tree_sha256": "a" * 64,
    }
    monkeypatch.setattr(cli, "_pin_evaluation_images", lambda *args: None)
    monkeypatch.setattr(cli, "_verify_image_leases", lambda *args: None)
    monkeypatch.setattr(cli, "run_glue", lambda *args, **kw: None)
    monkeypatch.setattr(cli, "validate_completed_investigation", lambda *args: ["no winner"])
    monkeypatch.setattr(cli, "investigation_completion_payload", lambda *args: {
        "schema_version": 1, "status": "completed", "outcome": "completed_no_winner",
        "manifest_sha256": "0" * 64,
    })
    for name in ("prepare_downstream_selection", "preview_downstream_selection", "load_downstream_selection"):
        monkeypatch.setattr(cli, name, lambda *args: selection, raising=False)
    monkeypatch.setattr(cli, "task_requires_final_evaluation", lambda *args: True)
    monkeypatch.setattr(cli, "_verified_final_outcome", lambda *args: "not_qualified")
    monkeypatch.setattr(cli, "_current_posthoc_artifacts_available", lambda *args: True)
    monkeypatch.setattr(cli, "validate_review_input", lambda *args: [])
    monkeypatch.setattr(cli, "run_pipeline_review", lambda **kw: SimpleNamespace(
        status="completed", review_path=tmp_path / "review.json", log_path=tmp_path / "review.log",
        error=None,
    ))
    return commands, events


def _winner_pipeline(monkeypatch, tmp_path: Path):
    commands, events = _mock_study(monkeypatch, tmp_path)
    _mock_successful_investigation(monkeypatch)
    monkeypatch.setattr(cli, "run_pipeline_revision", lambda **kw: SimpleNamespace(
        status="completed", revision_dir=tmp_path / "revision",
        before_review_path=tmp_path / "before.json", after_review_path=tmp_path / "after.json",
        error=None,
    ))
    return commands, events


def _break_final(monkeypatch, commands: list[list[str]], failure: str) -> None:
    if failure == "execution":
        def run(command, **kwargs):
            commands.append(command)
            return 7 if FINAL in command else 0
        monkeypatch.setattr(cli, "run_cmd", run)
    else:
        def invalid(*args):
            raise cli.FinalEvaluationError("mismatched program")
        monkeypatch.setattr(cli, "_verified_final_outcome", invalid)


def test_no_winner_study_reaches_paper_after_nonqualifying_final(monkeypatch, tmp_path: Path) -> None:
    for policy in ("best_available", "skip"):
        with monkeypatch.context() as m:
            commands, events = _mock_study(m, tmp_path)
            rc = cli.main([
                CIRCPACK, "execution.phases=investigation,ablations,final_evaluation,posthoc,paper,review",
                f"execution.no_winner_policy={policy}", f"paths.run_dir={tmp_path / policy}", "container.enabled=true",
            ])
        assert rc == 0, policy
        paper_commands = [command for command in commands if PAPER in command]
        assert bool(paper_commands) is (policy == "best_available"), policy
        if paper_commands:
            assert "--final-evaluation-dir" in paper_commands[0], policy
            assert "--no-final-evaluation" not in paper_commands[0], policy
            assert any(event == "downstream_selection" for event, _ in events), policy
        assert _pipeline_end(events)["outcome"] == "completed_no_winner", policy


def test_failed_unverified_or_invalid_selection_blocks_paper(monkeypatch, tmp_path: Path) -> None:
    # a failed or unverified required final blocks paper
    for study in ("no_winner", "winner_opt_in"):
        for failure in ("execution", "manifest"):
            name = f"{study}-{failure}"
            with monkeypatch.context() as m:
                if study == "no_winner":
                    commands, _ = _mock_study(m, tmp_path)
                    argv = [CIRCPACK, "execution.phases=investigation,final_evaluation,paper"]
                else:
                    commands, _ = _winner_pipeline(m, tmp_path)
                    argv = [
                        CIRCPACK, "execution.phases=investigation,final_evaluation,paper,review,revision",
                        "paper.final_outcome_policy=allow_not_qualified", "execution.fail_fast=false",
                    ]
                _break_final(m, commands, failure)
                rc = cli.main([*argv, f"paths.run_dir={tmp_path / name}", "container.enabled=true"])
            assert rc != 0, name
            assert not any(PAPER in command for command in commands), name

    # an invalid fallback selection stops before downstream phases
    commands, _ = _mock_study(monkeypatch, tmp_path)

    def invalid(*args):
        raise ValueError("stale downstream selection")

    monkeypatch.setattr(cli, "prepare_downstream_selection", invalid, raising=False)
    rc = cli.main([CIRCPACK, "execution.phases=investigation,ablations,paper", f"paths.run_dir={tmp_path / 'invalid'}"])
    assert rc != 0
    assert len(commands) == 1


def test_downstream_resume_selects_existing_candidate_without_restarting_search(
    monkeypatch, tmp_path: Path
) -> None:
    source = _source_task(tmp_path, "source", "Study this optimizer")
    for dry_run in (False, True):
        run_dir = tmp_path / f"run-{dry_run}"
        _write_search_with_candidate(run_dir)
        prepare_run_task(derive_paths(_pipeline_cfg(run_dir, task_dir=source, run_id=None)))
        with monkeypatch.context() as m:
            commands, events = _mock_study(m, tmp_path)
            for name in ("prepare_downstream_selection", "preview_downstream_selection", "load_downstream_selection"):
                m.setattr(cli, name, getattr(downstream_selection, name))
            rc = cli.main([
                f"task.dir={source}", "execution.phases=ablations,final_evaluation,paper",
                f"paths.run_dir={run_dir}", "container.enabled=true", f"execution.dry_run={str(dry_run).lower()}",
            ])
        assert rc == 0, dry_run
        assert not any(INVESTIGATION in command for command in commands), dry_run
        assert any(PAPER in command for command in commands), dry_run
        assert (run_dir / ".downstream_selection.json").exists() is (not dry_run), dry_run
        assert any(event == "downstream_selection" for event, _ in events), dry_run


# --------------------------------------------------------------------------- paper.final_outcome_policy


def test_winner_publication_opt_in_keeps_not_qualified_outcome(monkeypatch, tmp_path):
    for policy in ("qualified_only", "allow_not_qualified"):
        with monkeypatch.context() as m:
            commands, events = _winner_pipeline(m, tmp_path)
            rc = cli.main([
                CIRCPACK, "execution.phases=investigation,final_evaluation,paper,review,revision",
                f"paths.run_dir={tmp_path / policy}", "container.enabled=true", f"paper.final_outcome_policy={policy}",
            ])
        assert rc == 0, policy
        allowed = policy == "allow_not_qualified"
        assert any(PAPER in command for command in commands) is allowed, policy
        for event_name in ("review_result", "revision_result", "paper_final_outcome_override"):
            assert any(event == event_name for event, _ in events) is allowed, policy
        assert _pipeline_end(events)["outcome"] == "not_qualified", policy

    # the opt-in includes a verified optional non-qualifying final
    commands, _ = _winner_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "task_requires_final_evaluation", lambda *args: False)
    assert cli.main([
        CIRCPACK, "execution.phases=investigation,final_evaluation,paper",
        f"paths.run_dir={tmp_path / 'optional'}", "container.enabled=true",
        "paper.final_outcome_policy=allow_not_qualified",
    ]) == 0
    paper = next(command for command in commands if PAPER in command)
    assert "--final-evaluation-dir" in paper
    assert "--no-final-evaluation" not in paper


def test_manuscript_only_resume_verifies_saved_required_final(monkeypatch, tmp_path):
    for valid_final in (True, False):
        for dry_run in (False, True):
            name = f"{'valid' if valid_final else 'invalid'}-{'dry' if dry_run else 'live'}"
            run_dir = tmp_path / name
            _write_legacy_run(run_dir)
            prepare_run_task(derive_paths(_pipeline_cfg(run_dir, run_id=None)))
            with monkeypatch.context() as m:
                commands, events = _winner_pipeline(m, tmp_path)
                if not valid_final:
                    def invalid(*args):
                        raise cli.FinalEvaluationError("missing or invalid saved final")
                    m.setattr(cli, "_verified_final_outcome", invalid)
                rc = cli.main([
                    CIRCPACK, "execution.phases=paper,review,revision", f"paths.run_dir={run_dir}",
                    "container.enabled=true", "paper.final_outcome_policy=allow_not_qualified",
                    f"execution.dry_run={str(dry_run).lower()}",
                ])
            assert (rc == 0) is valid_final, name
            assert not any(FINAL in command for command in commands), name
            assert any(PAPER in command for command in commands) is valid_final, name
            if valid_final:
                assert _pipeline_end(events)["outcome"] == "not_qualified", name


def test_review_and_revision_only_recheck_explicit_policy(monkeypatch, tmp_path):
    for phase in ("review", "revision"):
        for policy in ("qualified_only", "allow_not_qualified"):
            name = f"{phase}-{policy}"
            bundle = tmp_path / name
            paper_dir = bundle / "paper"
            _write_published_paper(paper_dir)
            if phase == "revision":
                prepare_run_task(derive_paths(_pipeline_cfg(bundle, run_id=None)))
                review = paper_dir / "ai_review.before_revision.json"
                review.write_text("{}")
                manifest_path = paper_dir / "latest_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["before_revision"] = {"status": "completed", "review_json": str(review)}
                manifest_path.write_text(json.dumps(manifest))
            with monkeypatch.context() as m:
                _, events = _winner_pipeline(m, tmp_path)
                rc = cli.main([
                    CIRCPACK, f"execution.phases={phase}", f"paths.run_dir={bundle}", f"paths.paper_dir={paper_dir}",
                    "container.enabled=true", f"paper.final_outcome_policy={policy}",
                ])
            assert rc == 0, name
            assert any(event == f"{phase}_result" for event, _ in events) is (policy == "allow_not_qualified"), name


# --------------------------------------------------------------------------- event-logger lifecycle


def test_pipeline_reports_abnormal_end_and_always_closes_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [
        (
            "unexpected_failure", RuntimeError("unexpected"), (100.0, 3823.6),
            {"status": "failed", "error_type": "RuntimeError", "elapsed": "1h 2m 4s", "elapsed_seconds": 3723.6},
        ),
        (
            "interrupt", KeyboardInterrupt(), (100.0, 161.2),
            {"status": "interrupted", "return_code": 130, "elapsed": "1m 1s", "elapsed_seconds": 61.2},
        ),
    ]
    for name, exc, clock, expected in cases:
        shutdown_calls: list[bool] = []

        def fail(_argv: list[str] | None = None, exc: BaseException = exc) -> int:
            raise exc

        ticks = iter(clock)
        with monkeypatch.context() as m:
            events = _capture_events(m)
            m.setattr(cli, "_main", fail)
            m.setattr(cli, "event_logging_configured", lambda: True)
            m.setattr(cli, "shutdown_event_logging", lambda: shutdown_calls.append(True))
            m.setattr(cli.time, "monotonic", lambda: next(ticks))
            if isinstance(exc, KeyboardInterrupt):
                assert cli.main([]) == 130, name
            else:
                with pytest.raises(RuntimeError, match="unexpected"):
                    cli.main([])
        assert events == [("pipeline_end", expected)], name
        assert shutdown_calls == [True], name


# --------------------------------------------------------------------------- temporary storage entrypoints


def _storage_environment(root: Path) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("AGENTIC_TEMP")
        and key not in {"AGENTIC_CACHE_ROOT", "AGENTIC_CONTAINER_TEMP_ROOT"}
    }
    env["AGENTIC_TEMP_ROOT"] = str(root)
    return env


def _run_entrypoint(argv: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *argv], cwd=REPO_ROOT, env=_storage_environment(root),
        text=True, capture_output=True,
    )


def test_pipeline_entrypoint_previews_storage_and_rejects_invalid_storage(tmp_path: Path) -> None:
    # dry run previews storage without creating it
    root = tmp_path / "scratch"
    root.mkdir(mode=0o700)
    result = _run_entrypoint(["-m", "src.pipeline.cli", CIRCPACK, "--dry-run"], root)
    assert result.returncode == 0, result.stderr
    assert "Temporary storage" in result.stderr
    assert str(root) in result.stderr
    assert list(root.iterdir()) == []

    # invalid storage is rejected before a run starts
    bad_root = tmp_path / "scratch-file"
    bad_root.write_text("not a directory")
    run_dir = tmp_path / "run"
    program = (
        "from unittest.mock import patch; from src.pipeline.cli import main; "
        "patch('src.pipeline.cli._pin_evaluation_images', "
        "side_effect=AssertionError('unexpected phase startup')).start(); "
        "raise SystemExit(main())"
    )
    result = _run_entrypoint(["-c", program, CIRCPACK, f"paths.run_dir={run_dir}"], bad_root)
    assert result.returncode != 0
    assert "Temporary storage" in result.stderr
    assert not run_dir.exists()


def test_investigation_entrypoint_initializes_storage_before_main_flow(tmp_path: Path) -> None:
    # invalid storage is rejected before a run starts
    bad_root = tmp_path / "scratch-file"
    bad_root.write_text("not a directory")
    program = (
        "from unittest.mock import patch; from src.investigation import cli as investigation_cli; "
        "patch('src.investigation.cli.run_main_flow', "
        "side_effect=AssertionError('unexpected phase startup')).start(); "
        "raise SystemExit(investigation_cli.main())"
    )
    result = _run_entrypoint(["-c", program], bad_root)
    assert result.returncode != 0
    assert "Temporary storage" in result.stderr

    # the standalone investigation CLI initializes storage before the main flow
    root = tmp_path / "scratch"
    root.mkdir(mode=0o700)
    program = (
        "from unittest.mock import patch; import os; from src.investigation import cli as investigation_cli; "
        "patch('src.investigation.cli.run_main_flow', side_effect=lambda **kw: "
        "(print('FLOW_TMP=' + os.environ['TMPDIR']) or 0)).start(); "
        "raise SystemExit(investigation_cli.main())"
    )
    result = _run_entrypoint(["-c", program, f"paths.results_dir={tmp_path / 'run'}"], root)
    assert result.returncode == 0, result.stderr
    temporary = next(
        line.removeprefix("FLOW_TMP=") for line in result.stdout.splitlines() if line.startswith("FLOW_TMP=")
    )
    assert Path(temporary).is_relative_to(root)
    assert Path(temporary).is_dir()

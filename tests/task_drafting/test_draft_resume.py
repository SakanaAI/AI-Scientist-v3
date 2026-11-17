"""Resuming a task draft that stopped before publication (issue #2)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src.task_drafting import draft_resume
from src.task_drafting.draft_resume import DraftResumeError
from src.task_drafting.source_baseline import record_task_source_baseline


def _task_source(root: Path) -> Path:
    source = root / "my_task"
    source.mkdir()
    (source / "task.md").write_text("# Goal\nmaximize\n")
    return source


def _earlier_draft(drafting_root: Path, source: Path, name: str = "draft_20261101_000000_1") -> Path:
    draft = drafting_root / name
    staged = draft / "task"
    (staged / "program_0" / "src" / "__pycache__").mkdir(parents=True)
    (staged / ".venv" / "bin").mkdir(parents=True)
    record_task_source_baseline(source, draft / draft_resume.SOURCE_BASELINE_NAME)
    (staged / "task.md").write_text((source / "task.md").read_text())
    (staged / "evaluate.py").write_text("print('search')\n")
    (staged / "program_0" / "src" / "program.py").write_text("def handle(r):\n    return r\n")
    (staged / "program_0" / "src" / "__pycache__" / "program.pyc").write_bytes(b"stale")
    (staged / ".agentic_runtime.json").write_text("{}")
    return draft


def _accepted_design(draft: Path, *, verdict: str = "pass") -> None:
    design_root = draft / "_task_design"
    design_root.mkdir()
    (design_root / "task_design.json").write_text('{"design": 1}')
    (design_root / "accepted_review.json").write_text('{"review": 1}')
    digest = {
        name: hashlib.sha256((design_root / f"{name}.json").read_bytes()).hexdigest()
        for name in ("task_design", "accepted_review")
    }
    (design_root / "design_receipt.json").write_text(json.dumps({
        "schema_version": 1, "verdict": verdict,
        "design_sha256": digest["task_design"], "review_sha256": digest["accepted_review"],
    }))


@pytest.fixture
def layout(tmp_path: Path):
    source = _task_source(tmp_path)
    drafting_root = tmp_path / "run" / "_task_drafting"
    drafting_root.mkdir(parents=True)
    earlier = _earlier_draft(drafting_root, source)
    new = drafting_root / "draft_20261102_000000_2"
    new.mkdir()
    return source, earlier, new


def test_scaffold_is_copied_without_runtime_state(layout):
    source, earlier, new = layout
    previous = draft_resume.seed_resumed_scaffold(
        str(earlier), output_dir=new, task_source_dir=source,
    )
    assert previous == earlier
    staged = new / "task"
    assert (staged / "evaluate.py").read_text() == "print('search')\n"
    assert (staged / "program_0" / "src" / "program.py").is_file()
    assert not (staged / ".venv").exists()
    assert not (staged / ".agentic_runtime.json").exists()
    assert not (staged / "program_0" / "src" / "__pycache__").exists()


def test_latest_picks_the_most_recent_earlier_draft(layout):
    source, earlier, new = layout
    newer = _earlier_draft(earlier.parent, source, name="draft_20261101_120000_3")
    assert draft_resume.resolve_previous_draft("latest", new) == newer


def test_latest_without_earlier_draft_is_refused(tmp_path):
    new = tmp_path / "draft_20261102_000000_2"
    new.mkdir()
    with pytest.raises(DraftResumeError, match="no earlier draft"):
        draft_resume.resolve_previous_draft("latest", new)


def test_draft_outside_the_drafting_root_is_refused(layout, tmp_path):
    _, _, new = layout
    stray = tmp_path / "draft_elsewhere"
    stray.mkdir()
    with pytest.raises(DraftResumeError, match="must be a draft_"):
        draft_resume.resolve_previous_draft(str(stray), new)


def test_changed_task_source_is_refused(layout):
    source, earlier, new = layout
    (source / "task.md").write_text("# Goal\nminimize\n")
    with pytest.raises(DraftResumeError, match="task source changed"):
        draft_resume.seed_resumed_scaffold(str(earlier), output_dir=new, task_source_dir=source)
    assert not (new / "task").exists()


def test_drafting_output_inside_the_task_source_is_not_task_input(tmp_path):
    source = _task_source(tmp_path)
    drafting_root = source / "results" / "_task_drafting"
    drafting_root.mkdir(parents=True)
    earlier = _earlier_draft(drafting_root, source)
    new = drafting_root / "draft_20261102_000000_2"
    new.mkdir()
    draft_resume.seed_resumed_scaffold("latest", output_dir=new, task_source_dir=source)
    assert (new / "task" / "evaluate.py").is_file()
    assert earlier.is_dir()


def test_verified_accepted_design_is_reused(layout):
    _, earlier, new = layout
    _accepted_design(earlier)
    assert draft_resume.copy_accepted_design(earlier, new) is True
    assert (new / "_task_design" / "task_design.json").read_text() == '{"design": 1}'


def test_design_with_stale_receipt_is_designed_again(layout):
    _, earlier, new = layout
    _accepted_design(earlier)
    (earlier / "_task_design" / "task_design.json").write_text('{"design": 2}')
    assert draft_resume.copy_accepted_design(earlier, new) is False
    assert not (new / "_task_design").exists()


def test_rejected_design_is_designed_again(layout):
    _, earlier, new = layout
    _accepted_design(earlier, verdict="fail")
    assert draft_resume.copy_accepted_design(earlier, new) is False


def test_cli_reports_refusal_without_traceback(layout, capsys):
    source, _, new = layout
    (source / "task.md").write_text("changed\n")
    exit_code = draft_resume.main([
        "scaffold", "--resume-from", "latest", "--output-dir", str(new), "--task-dir", str(source),
    ])
    assert exit_code == 1
    assert "Cannot resume task draft: task source changed" in capsys.readouterr().err


def test_pipeline_override_reaches_the_drafting_stage():
    from tests.pipeline.test_pipeline_phases import CIRCPACK, effective_stage_configs

    stages = effective_stage_configs([CIRCPACK, "investigation.task_drafting.resume_from=latest"])
    assert stages["task_drafting"]["task_drafting.resume_from"] == "latest"


def test_edit_to_a_file_too_large_to_hash_is_refused(layout):
    source, earlier, new = layout
    large = source / "data.bin"
    large.write_bytes(b"\0" * (2 * 1024 * 1024))
    started = (earlier / draft_resume.SOURCE_BASELINE_NAME).stat().st_mtime
    os.utime(large, (started + 5, started + 5))
    with pytest.raises(DraftResumeError, match=r"task source changed .*data\.bin"):
        draft_resume.seed_resumed_scaffold(str(earlier), output_dir=new, task_source_dir=source)


def test_latest_skips_drafts_that_never_staged_a_scaffold(layout):
    _, earlier, new = layout
    (earlier.parent / "draft_20261101_120000_3").mkdir()
    assert draft_resume.resolve_previous_draft("latest", new) == earlier

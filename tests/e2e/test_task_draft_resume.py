"""A draft that fails after its scaffold resumes without rerunning design or scaffold (#2)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.e2e.test_agent_execution_pipeline import (
    _script_path,
    _write_fake_task_draft_agent,
    _write_task_md_only,
)


def _draft(root: Path, output_name: str, *extra: str, **environment: str):
    bin_dir = root / "bin"
    if not bin_dir.exists():
        bin_dir.mkdir()
        _write_fake_task_draft_agent(bin_dir, "headless")
    env = os.environ | {
        "HEADLESS_BIN": str(bin_dir / "headless"),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WRAPPER_PYTHON": sys.executable,
        "TASK_ALLOW_NATIVE_DRAFT_FOR_TESTS": "1",
        "TASK_ALLOW_NATIVE_REPAIR_FOR_TESTS": "1",
        **environment,
    }
    return subprocess.run(
        [
            "bash", str(_script_path("task.sh")), "--agent", "codex",
            "--task-dir", str(root / "task"),
            "--output-dir", str(root / "drafts" / output_name),
            "--workflow", "robust", "--smoke-max-repairs", "0", *extra,
        ],
        capture_output=True, text=True, timeout=180, env=env,
    )


def _agent_logs(draft: Path, prefix: str) -> list[Path]:
    return sorted(draft.glob(f"{prefix}_agent_*.log"))


def test_failed_draft_resumes_from_its_design_and_scaffold(tmp_path: Path) -> None:
    _write_task_md_only(tmp_path)
    (tmp_path / "drafts").mkdir()

    failed = _draft(tmp_path, "draft_1", FAKE_TASK_SEARCH_SMOKE_FAIL="1")
    assert failed.returncode != 0, failed.stdout + failed.stderr
    earlier = tmp_path / "drafts" / "draft_1"
    assert _agent_logs(earlier, "design") and _agent_logs(earlier, "draft")
    assert not (tmp_path / "task" / "evaluate.py").exists()

    resumed = _draft(tmp_path, "draft_2", "--resume-from", "latest")
    output = resumed.stdout + resumed.stderr
    assert resumed.returncode == 0, output
    assert f"resuming task draft from {earlier}" in output
    assert "(accepted design: reused)" in output
    assert "resumed scaffold is complete; skipping the scaffold agent" in output
    later = tmp_path / "drafts" / "draft_2"
    assert not _agent_logs(later, "draft")
    assert not _agent_logs(later, "design")
    assert (tmp_path / "task" / "evaluate.py").is_file()


def test_resume_is_refused_after_the_task_changes(tmp_path: Path) -> None:
    _write_task_md_only(tmp_path)
    (tmp_path / "drafts").mkdir()
    failed = _draft(tmp_path, "draft_1", FAKE_TASK_SEARCH_SMOKE_FAIL="1")
    assert failed.returncode != 0
    (tmp_path / "task" / "task.md").write_text("# Goal\n- something else\n", encoding="utf-8")

    resumed = _draft(tmp_path, "draft_2", "--resume-from", "latest")
    assert resumed.returncode == 1
    assert "Cannot resume task draft: task source changed" in resumed.stderr


def _fail_scaffold_agent(root: Path) -> None:
    headless = root / "bin" / "headless"
    headless.rename(root / "bin" / "headless-real")
    headless.write_text(
        '#!/bin/bash\n'
        'if [ "${TASK_AGENT_PHASE:-}" = "draft" ]; then echo "scaffold agent crashed" >&2; exit 1; fi\n'
        'exec "$(dirname "$0")/headless-real" "$@"\n',
        encoding="utf-8",
    )
    headless.chmod(0o755)


def test_failed_scaffold_revision_does_not_accept_the_stale_scaffold(tmp_path: Path) -> None:
    _write_task_md_only(tmp_path)
    (tmp_path / "drafts").mkdir()
    failed = _draft(tmp_path, "draft_1", FAKE_TASK_SEARCH_SMOKE_FAIL="1")
    assert failed.returncode != 0
    (tmp_path / "drafts" / "draft_1" / "_task_design" / "design_receipt.json").write_text("{}", encoding="utf-8")
    _fail_scaffold_agent(tmp_path)

    resumed = _draft(tmp_path, "draft_2", "--resume-from", "latest", "--draft-max-attempts", "1")
    output = resumed.stdout + resumed.stderr
    assert resumed.returncode == 3, output
    assert "(accepted design: rerun)" in output
    assert "using complete retained scaffold" not in output
    assert "Task scaffold remained incomplete after 1 draft attempt(s)" in output
    assert not (tmp_path / "task" / "evaluate.py").exists()

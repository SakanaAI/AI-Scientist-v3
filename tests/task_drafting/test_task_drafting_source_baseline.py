"""Publishing a repaired scaffold over a task that already shipped one (source baseline)."""
from __future__ import annotations

import errno
import json
import logging
import os
import shutil
from pathlib import Path

import pytest

from src.investigation.task_alignment import load_task_alignment_report
from src.task_drafting import publication, publication_fs, publication_resources, source_baseline
from tests.task_drafting.test_task_drafting_publication import _make_fx, _Publication, _sha256

REPAIRED_EVALUATOR = "SEARCH = 'repaired'\n"


def _make_predrafted_fx(root: Path) -> tuple[_Publication, Path]:
    """A user task that already ships a full scaffold, which drafting then repairs."""
    fx = _make_fx(root)
    shutil.rmtree(fx.task_dir)
    shutil.copytree(fx.staged_task_dir, fx.task_dir)
    baseline = fx.output_dir / "task_source_baseline.json"
    source_baseline.record_task_source_baseline(fx.task_dir, baseline)
    # Alignment repair edits only the staged copy.
    (fx.staged_task_dir / "evaluate.py").write_text(REPAIRED_EVALUATOR, encoding="utf-8")
    resource = fx.staged_task_dir / "resources/search/dev.json"
    resource.write_text('{"repaired": true}\n', encoding="utf-8")
    contract = json.loads(fx.contract_path.read_text(encoding="utf-8"))
    for artifact in contract["provenance"]["artifacts"]:
        if artifact["path"] == "resources/search/dev.json":
            artifact["sha256"] = _sha256(resource)
    fx.write_contract(contract)
    fx.refresh_report()
    return fx, baseline


def test_repaired_predrafted_task_without_baseline_is_still_rejected(tmp_path: Path) -> None:
    fx, _ = _make_predrafted_fx(tmp_path)

    with pytest.raises(publication.TaskPublicationError, match="differs from accepted scaffold"):
        fx.publish()

    assert not fx.marker.exists()


def test_repaired_scaffold_replaces_source_that_matches_baseline(tmp_path: Path) -> None:
    fx, baseline = _make_predrafted_fx(tmp_path)

    fx.publish(source_baseline_path=baseline)

    assert (fx.task_dir / "evaluate.py").read_text(encoding="utf-8") == REPAIRED_EVALUATOR
    assert (fx.task_dir / "resources/search/dev.json").read_bytes() == (
        fx.staged_task_dir / "resources/search/dev.json"
    ).read_bytes()
    assert len(publication.validate_published_task_plan(fx.task_dir)) == 64
    load_task_alignment_report(fx.marker, scaffold_dir=fx.task_dir)
    assert not any(path.name.startswith(".resources.") for path in fx.task_dir.iterdir())


def test_replaced_originals_are_saved_in_the_drafting_output(tmp_path: Path) -> None:
    fx, baseline = _make_predrafted_fx(tmp_path)
    original_evaluator = (fx.task_dir / "evaluate.py").read_bytes()
    original_resources = {
        path.relative_to(fx.task_dir): path.read_bytes()
        for path in (fx.task_dir / "resources").rglob("*")
        if path.is_file()
    }

    replaced = fx.publish(source_baseline_path=baseline)

    saved = fx.output_dir / source_baseline.REPLACED_SOURCE_DIRNAME
    assert replaced == ("evaluate.py", "task_contract.json", "resources/")
    assert (saved / "evaluate.py").read_bytes() == original_evaluator
    assert {
        relative: (saved / relative).read_bytes() for relative in original_resources
    } == original_resources


def test_publication_without_replacements_saves_nothing(tmp_path: Path) -> None:
    fx = _make_fx(tmp_path)
    baseline = fx.output_dir / "task_source_baseline.json"
    source_baseline.record_task_source_baseline(fx.task_dir, baseline)

    assert fx.publish(source_baseline_path=baseline) == ()
    assert not (fx.output_dir / source_baseline.REPLACED_SOURCE_DIRNAME).exists()


def test_republishing_a_replaced_scaffold_is_a_no_op(tmp_path: Path) -> None:
    fx, baseline = _make_predrafted_fx(tmp_path)
    fx.publish(source_baseline_path=baseline)
    published = (fx.task_dir / "evaluate.py").stat().st_mtime_ns

    fx.publish(source_baseline_path=baseline)

    assert (fx.task_dir / "evaluate.py").stat().st_mtime_ns == published


def test_source_file_edited_after_baseline_aborts_and_is_kept(tmp_path: Path) -> None:
    fx, baseline = _make_predrafted_fx(tmp_path)
    (fx.task_dir / "evaluate.py").write_text("SEARCH = 'user edit'\n", encoding="utf-8")

    with pytest.raises(publication.TaskPublicationError, match="differs from accepted scaffold"):
        fx.publish(source_baseline_path=baseline)

    assert (fx.task_dir / "evaluate.py").read_text(encoding="utf-8") == "SEARCH = 'user edit'\n"
    assert not fx.marker.exists()


def test_resource_added_after_baseline_aborts_and_is_kept(tmp_path: Path) -> None:
    fx, baseline = _make_predrafted_fx(tmp_path)
    (fx.task_dir / "resources/final/extra.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(publication.TaskPublicationError, match="managed task resources"):
        fx.publish(source_baseline_path=baseline)

    assert (fx.task_dir / "resources/final/extra.json").is_file()


def _resource_files(task_dir: Path) -> dict[str, bytes]:
    root = task_dir / "resources"
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _hidden_resource_trees(task_dir: Path) -> list[Path]:
    return [path for path in task_dir.iterdir() if path.name.startswith(".resources.")]


def test_failed_resource_install_restores_source_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx, baseline = _make_predrafted_fx(tmp_path)
    original = _resource_files(fx.task_dir)
    real_mkdir = os.mkdir

    def mkdir_without_space(path, *args, **kwargs):
        if str(path).startswith(".resources.publish-"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(publication_resources.os, "mkdir", mkdir_without_space)

    with pytest.raises(OSError, match="No space left"):
        fx.publish(source_baseline_path=baseline)

    assert _resource_files(fx.task_dir) == original
    assert _hidden_resource_trees(fx.task_dir) == []


def test_unremovable_replaced_resources_do_not_fail_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fx, baseline = _make_predrafted_fx(tmp_path)
    real_rmtree = shutil.rmtree

    def rmtree_refusing_replaced(path, *args, **kwargs):
        if Path(path).name.startswith(".resources.replaced-"):
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(publication_resources.shutil, "rmtree", rmtree_refusing_replaced)

    with caplog.at_level(logging.WARNING, logger=publication_resources.__name__):
        fx.publish(source_baseline_path=baseline)

    assert fx.marker.is_file()
    assert "could not remove replaced task resources" in caplog.text


def test_unrestorable_source_resources_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx, baseline = _make_predrafted_fx(tmp_path)
    original = _resource_files(fx.task_dir)

    def install_racing_with_user(source, target, parent_fd, *, expected_sha256):
        target.mkdir()
        (target / "user.json").write_text("{}\n", encoding="utf-8")
        raise publication.TaskPublicationError("install failed")

    monkeypatch.setattr(publication_resources, "_install_tree", install_racing_with_user)

    with pytest.raises(publication.TaskPublicationError, match="install failed") as caught:
        fx.publish(source_baseline_path=baseline)

    [kept] = _hidden_resource_trees(fx.task_dir)
    assert str(kept) in "\n".join(caught.value.__notes__)
    assert {
        path.relative_to(kept).as_posix(): path.read_bytes()
        for path in sorted(kept.rglob("*"))
        if path.is_file()
    } == original


def test_source_file_edited_during_publication_aborts_and_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx, baseline = _make_predrafted_fx(tmp_path)
    evaluator = fx.task_dir / "evaluate.py"
    real_move_aside = publication_fs.move_aside

    def move_aside_after_user_edit(parent_fd: int, name: str) -> str | None:
        if name == evaluator.name:
            evaluator.write_text("SEARCH = 'user edit'\n", encoding="utf-8")
        return real_move_aside(parent_fd, name)

    monkeypatch.setattr(publication_fs, "move_aside", move_aside_after_user_edit)

    with pytest.raises(publication.TaskPublicationError, match="changed during publication"):
        fx.publish(source_baseline_path=baseline)

    assert evaluator.read_text(encoding="utf-8") == "SEARCH = 'user edit'\n"
    assert not (fx.output_dir / source_baseline.REPLACED_SOURCE_DIRNAME / "evaluate.py").exists()
    assert not any(path.name.startswith(".evaluate.py.") for path in fx.task_dir.iterdir())
    assert not fx.marker.exists()


def test_resource_added_during_publication_aborts_and_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx, baseline = _make_predrafted_fx(tmp_path)
    original = _resource_files(fx.task_dir)
    real_move_aside = publication_resources.move_aside

    def move_aside_after_user_edit(parent_fd: int, name: str) -> str | None:
        (fx.task_dir / "resources/final/extra.json").write_text("{}\n", encoding="utf-8")
        return real_move_aside(parent_fd, name)

    monkeypatch.setattr(publication_resources, "move_aside", move_aside_after_user_edit)

    with pytest.raises(publication.TaskPublicationError, match="resources changed during publication"):
        fx.publish(source_baseline_path=baseline)

    assert _resource_files(fx.task_dir) == {**original, "final/extra.json": b"{}\n"}
    assert _hidden_resource_trees(fx.task_dir) == []
    assert not (fx.output_dir / source_baseline.REPLACED_SOURCE_DIRNAME / "resources").exists()
    assert not fx.marker.exists()


def test_invalid_baseline_is_rejected(tmp_path: Path) -> None:
    fx, baseline = _make_predrafted_fx(tmp_path)
    baseline.write_text('{"schema_version": 99}\n', encoding="utf-8")

    with pytest.raises(publication.TaskPublicationError, match="invalid task source baseline"):
        fx.publish(source_baseline_path=baseline)


def test_baseline_that_is_not_an_object_is_rejected(tmp_path: Path) -> None:
    fx, baseline = _make_predrafted_fx(tmp_path)
    baseline.write_text("[]\n", encoding="utf-8")

    with pytest.raises(publication.TaskPublicationError, match="invalid task source baseline"):
        fx.publish(source_baseline_path=baseline)


def test_baseline_is_write_once(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.md").write_text("# one\n", encoding="utf-8")
    record = tmp_path / "record.json"
    source_baseline.record_task_source_baseline(task_dir, record)
    first = record.read_bytes()
    (task_dir / "task.md").write_text("# two\n", encoding="utf-8")

    source_baseline.record_task_source_baseline(task_dir, record)

    assert record.read_bytes() == first


def test_baseline_skips_symlinks(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.md").write_text("# one\n", encoding="utf-8")
    (task_dir / "link.md").symlink_to(task_dir / "task.md")
    record = tmp_path / "record.json"

    source_baseline.record_task_source_baseline(task_dir, record)

    payload = json.loads(record.read_text(encoding="utf-8"))
    assert set(payload["files"]) == {"task.md"}
    assert payload["resources_sha256"] is None


def test_baseline_skips_drafting_output_and_hashes_resources_once(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    (task_dir / "resources").mkdir(parents=True)
    (task_dir / "resources" / "data.json").write_text("{}\n", encoding="utf-8")
    (task_dir / "task.md").write_text("# one\n", encoding="utf-8")
    earlier_draft = task_dir / ".task_drafting" / "draft_1" / "task"
    earlier_draft.mkdir(parents=True)
    (earlier_draft / "task.md").write_text("# copy\n", encoding="utf-8")
    custom_output = task_dir / "drafts" / "now"
    (custom_output / "task").mkdir(parents=True)
    (custom_output / "task" / "task.md").write_text("# copy\n", encoding="utf-8")
    record = custom_output / "baseline.json"

    source_baseline.record_task_source_baseline(task_dir, record)

    payload = json.loads(record.read_text(encoding="utf-8"))
    assert set(payload["files"]) == {"task.md"}
    assert payload["resources_sha256"] is not None


def test_baseline_skips_unreadable_files(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.md").write_text("# one\n", encoding="utf-8")
    secret = task_dir / "secret.txt"
    secret.write_text("x\n", encoding="utf-8")
    secret.chmod(0)
    if os.access(secret, os.R_OK):
        pytest.skip("running with privileges that ignore file permissions")
    record = tmp_path / "record.json"

    source_baseline.record_task_source_baseline(task_dir, record)

    assert set(json.loads(record.read_text(encoding="utf-8"))["files"]) == {"task.md"}


def test_baseline_skips_files_that_change_while_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.md").write_text("# one\n", encoding="utf-8")
    (task_dir / "train.log").write_text("step 1\n", encoding="utf-8")
    real_file_sha256 = source_baseline.file_sha256

    def file_sha256_racing_with_writer(path: Path) -> str:
        if path.name == "train.log":
            raise publication.TaskPublicationError(f"task publication source changed while reading: {path}")
        return real_file_sha256(path)

    monkeypatch.setattr(source_baseline, "file_sha256", file_sha256_racing_with_writer)
    record = tmp_path / "record.json"

    source_baseline.record_task_source_baseline(task_dir, record)

    assert set(json.loads(record.read_text(encoding="utf-8"))["files"]) == {"task.md"}

"""Seed a new task draft from an earlier draft that stopped before publication.

Only work products are carried over: the accepted task design (when its receipt still
verifies) and the staged scaffold. Contract, smoke, alignment and qualification gates run
again on the new draft. The design receipt is an unkeyed digest in agent-writable draft
output, so a reused design is trusted exactly as far as within a single draft, where
later agents can also rewrite the accepted design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from src.task_drafting.source_baseline import (
    load_source_baseline,
    record_task_source_baseline,
)
from src.task_drafting.publication_fs import TaskPublicationError

LATEST = "latest"
DRAFT_PREFIX = "draft_"
SOURCE_BASELINE_NAME = ".task_source_baseline.json"
STAGED_TASK_NAME = "task"
DESIGN_DIRNAME = "_task_design"
DESIGN_FILES = ("task_design.json", "accepted_review.json", "design_receipt.json")
# Harness-owned runtime state is rebuilt by the new draft, never inherited.
_STAGED_RUNTIME_ENTRIES = frozenset({".venv", ".agentic_runtime.json", ".agentic_repair_prompt.md"})


class DraftResumeError(RuntimeError):
    """The earlier draft cannot safely seed the new one."""


def resolve_previous_draft(requested: str, output_dir: Path) -> Path:
    """Return the earlier draft directory: ``latest`` or an explicit sibling of *output_dir*."""
    drafting_root = output_dir.parent.resolve(strict=True)
    if requested == LATEST:
        earlier = sorted(
            path for path in drafting_root.iterdir()
            if path.name.startswith(DRAFT_PREFIX) and path.name != output_dir.name
            and not path.is_symlink() and path.is_dir()
            and (path / STAGED_TASK_NAME).is_dir()
        )
        if not earlier:
            raise DraftResumeError(f"no earlier draft to resume under {drafting_root}")
        return earlier[-1]
    previous = Path(requested).expanduser()
    if previous.is_symlink() or not previous.is_dir():
        raise DraftResumeError(f"draft to resume is not a directory: {previous}")
    previous = previous.resolve(strict=True)
    if previous.parent != drafting_root or not previous.name.startswith(DRAFT_PREFIX):
        raise DraftResumeError(
            f"draft to resume must be a {DRAFT_PREFIX}* directory in {drafting_root}"
        )
    if previous.name == output_dir.name:
        raise DraftResumeError("a draft cannot resume itself")
    return previous


def require_unchanged_task_source(previous_dir: Path, task_source_dir: Path, output_dir: Path) -> None:
    """Refuse to resume when the user's task changed after the earlier draft started."""
    try:
        before = load_source_baseline(previous_dir / SOURCE_BASELINE_NAME)
    except TaskPublicationError as exc:
        raise DraftResumeError(f"earlier draft has no readable task source record: {exc}") from exc
    if before is None:
        raise DraftResumeError("earlier draft has no task source record")
    current_record = output_dir / ".resume_source_baseline.json"
    record_task_source_baseline(task_source_dir, current_record)
    try:
        now = load_source_baseline(current_record)
    finally:
        current_record.unlink(missing_ok=True)
    assert now is not None
    drafting_root = _relative_inside(output_dir.parent, task_source_dir)
    if (
        _outside(before.file_sha256, drafting_root) != _outside(now.file_sha256, drafting_root)
        or before.resources_sha256 != now.resources_sha256
    ):
        raise DraftResumeError(
            "task source changed since the earlier draft started; start a fresh draft instead"
        )
    # The digest record skips large files; their modification time still reveals an edit.
    started = (previous_dir / SOURCE_BASELINE_NAME).stat().st_mtime
    results_root = output_dir.parent.parent
    edited = _first_source_file_modified_after(task_source_dir, started, excluded=results_root)
    if edited is not None:
        raise DraftResumeError(
            f"task source changed since the earlier draft started ({edited}); "
            "start a fresh draft instead"
        )


_UNTRACKED_SOURCE_DIRS = frozenset({".git", ".venv", "__pycache__", ".task_drafting"})


def _first_source_file_modified_after(root: Path, since: float, *, excluded: Path) -> str | None:
    root = root.resolve()
    excluded = excluded.resolve()
    for directory, directory_names, file_names in os.walk(root):
        current = Path(directory)
        directory_names[:] = [
            name for name in directory_names
            if name not in _UNTRACKED_SOURCE_DIRS and current / name != excluded
        ]
        for name in file_names:
            path = current / name
            if not path.is_symlink() and path.is_file() and path.stat().st_mtime > since:
                return path.relative_to(root).as_posix()
    return None


def _relative_inside(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _outside(digests: dict[str, str], excluded_root: str | None) -> dict[str, str]:
    """Drop drafting output that lives inside the task source; it is not task input."""
    if excluded_root is None:
        return digests
    prefix = excluded_root + "/"
    return {name: digest for name, digest in digests.items() if not name.startswith(prefix)}


def copy_staged_scaffold(previous_dir: Path, output_dir: Path) -> None:
    source = previous_dir / STAGED_TASK_NAME
    if source.is_symlink() or not source.is_dir():
        raise DraftResumeError("earlier draft has no staged task scaffold")
    destination = output_dir / STAGED_TASK_NAME
    if destination.exists() or destination.is_symlink():
        raise DraftResumeError(f"new draft already has a staged task: {destination}")

    def ignored(directory: str, names: list[str]) -> set[str]:
        at_root = Path(directory) == source
        return {
            name for name in names
            if name == "__pycache__" or (at_root and name in _STAGED_RUNTIME_ENTRIES)
        }

    staging = output_dir / f".task.resume.{os.getpid()}"
    try:
        shutil.copytree(source, staging, symlinks=True, ignore=ignored)
        staging.replace(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def copy_accepted_design(previous_dir: Path, output_dir: Path) -> bool:
    """Carry over a design whose receipt still matches its files; otherwise design again."""
    source = previous_dir / DESIGN_DIRNAME
    paths = [source / name for name in DESIGN_FILES]
    if source.is_symlink() or any(path.is_symlink() or not path.is_file() for path in paths):
        return False
    design, review, receipt_path = paths
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not (
        isinstance(receipt, dict)
        and receipt.get("schema_version") == 1
        and receipt.get("verdict") == "pass"
        and receipt.get("design_sha256") == _sha256(design)
        and receipt.get("review_sha256") == _sha256(review)
    ):
        return False
    destination = output_dir / DESIGN_DIRNAME
    destination.mkdir(mode=0o700)
    for path in paths:
        shutil.copyfile(path, destination / path.name)
    return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_resumed_scaffold(requested: str, *, output_dir: Path, task_source_dir: Path) -> Path:
    """Copy the earlier draft's staged scaffold into *output_dir*; return the earlier draft."""
    output_dir = output_dir.resolve(strict=True)
    previous_dir = resolve_previous_draft(requested, output_dir)
    require_unchanged_task_source(previous_dir, task_source_dir, output_dir)
    copy_staged_scaffold(previous_dir, output_dir)
    return previous_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scaffold = commands.add_parser("scaffold", help="copy the staged scaffold; print the earlier draft")
    scaffold.add_argument("--resume-from", required=True, help=f"earlier draft path or '{LATEST}'")
    scaffold.add_argument("--output-dir", required=True, type=Path)
    scaffold.add_argument("--task-dir", required=True, type=Path)
    design = commands.add_parser("design", help="copy a verified accepted design (exit 1 if none)")
    design.add_argument("--previous-dir", required=True, type=Path)
    design.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "design":
            return 0 if copy_accepted_design(args.previous_dir, args.output_dir) else 1
        print(seed_resumed_scaffold(
            args.resume_from, output_dir=args.output_dir, task_source_dir=args.task_dir,
        ))
    except (DraftResumeError, OSError) as exc:
        print(f"Cannot resume task draft: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

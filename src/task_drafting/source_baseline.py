"""Record of the user-supplied task source, taken before drafting repairs the staged copy."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.task_drafting.publication_fs import (
    RESOURCE_MANIFEST,
    TaskPublicationError,
    atomic_copy,
    atomic_write,
    file_sha256,
    read_regular_bytes,
    real_path,
    regular_tree_files,
    resource_tree_digest,
    same_file_bytes,
)

_MAX_SOURCE_BASELINE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SourceBaseline:
    """Bytes of the user-supplied task directory before drafting staged and repaired it.

    A destination artifact that still matches this baseline was only changed by the
    harness's own staged repairs, so publication may replace it with the accepted
    scaffold. Anything else is a concurrent edit and still aborts publication.
    """

    file_sha256: dict[str, str]
    resources_sha256: str | None

    def matches_file(self, target: Path, relative: Path) -> bool:
        expected = self.file_sha256.get(relative.as_posix())
        if expected is None or target.is_symlink() or not target.is_file():
            return False
        return file_sha256(target) == expected

    def matches_resources(self, target: Path) -> bool:
        if self.resources_sha256 is None or target.is_symlink() or not target.is_dir():
            return False
        try:
            return resource_tree_digest(target, RESOURCE_MANIFEST) == self.resources_sha256
        except (OSError, TaskPublicationError):
            return False


_SOURCE_BASELINE_SCHEMA_VERSION = 1
_SOURCE_BASELINE_OMITTED_DIRS = frozenset({".git", ".venv", "__pycache__"})


def record_task_source_baseline(task_dir: Path, baseline_path: Path) -> None:
    """Record the user-supplied task bytes once, before drafting edits the staged copy.

    The record is write-once so a resumed run keeps the baseline of the original
    source. Symlinks and special files are not recorded, so they can never be
    replaced by publication.
    """
    if baseline_path.exists() or baseline_path.is_symlink():
        return
    root = real_path(task_dir, "task directory")
    file_digests: dict[str, str] = {}
    for directory, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(
            name for name in directory_names if name not in _SOURCE_BASELINE_OMITTED_DIRS
        )
        current = Path(directory)
        for name in sorted(file_names):
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            file_digests[path.relative_to(root).as_posix()] = file_sha256(path)
    resources_sha256: str | None = None
    resources = root / "resources"
    if resources.is_dir() and not resources.is_symlink():
        try:
            resources_sha256 = resource_tree_digest(resources, RESOURCE_MANIFEST)
        except TaskPublicationError:
            resources_sha256 = None
    payload = {
        "schema_version": _SOURCE_BASELINE_SCHEMA_VERSION,
        "files": file_digests,
        "resources_sha256": resources_sha256,
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(baseline_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def load_source_baseline(baseline_path: Path | None) -> SourceBaseline | None:
    if baseline_path is None:
        return None
    try:
        payload = json.loads(
            read_regular_bytes(baseline_path, maximum_bytes=_MAX_SOURCE_BASELINE_BYTES).decode(
                "utf-8"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TaskPublicationError) as exc:
        raise TaskPublicationError(f"invalid task source baseline: {baseline_path}") from exc
    files = payload.get("files") if isinstance(payload, dict) else None
    resources_sha256 = payload.get("resources_sha256") if isinstance(payload, dict) else None
    if (
        payload.get("schema_version") != _SOURCE_BASELINE_SCHEMA_VERSION
        or not isinstance(files, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())
        or not (resources_sha256 is None or isinstance(resources_sha256, str))
    ):
        raise TaskPublicationError(f"invalid task source baseline: {baseline_path}")
    return SourceBaseline(file_sha256=dict(files), resources_sha256=resources_sha256)


REPLACED_SOURCE_DIRNAME = "replaced_task_source"


def save_replaced_originals(
    baseline: SourceBaseline,
    *,
    task_dir: Path,
    staged_task_dir: Path,
    files: tuple[Path, ...],
    resources: Path | None,
    backup_dir: Path,
) -> tuple[str, ...]:
    """Copy every source artifact that publication is about to replace into ``backup_dir``.

    Only artifacts that still match the baseline and differ from the accepted scaffold
    are replaced, so only those are saved. Returns their task-relative paths.
    """
    saved: list[str] = []
    for relative in files:
        target = task_dir / relative
        if baseline.matches_file(target, relative) and not same_file_bytes(
            staged_task_dir / relative, target
        ):
            atomic_copy(target, backup_dir / relative, replace_existing=True)
            saved.append(relative.as_posix())
    if resources is not None:
        target = task_dir / resources
        accepted = resource_tree_digest(staged_task_dir / resources, RESOURCE_MANIFEST)
        if baseline.matches_resources(target) and baseline.resources_sha256 != accepted:
            for path in regular_tree_files(target):
                atomic_copy(path, backup_dir / resources / path.relative_to(target), replace_existing=True)
            saved.append(f"{resources.as_posix()}/")
    return tuple(saved)

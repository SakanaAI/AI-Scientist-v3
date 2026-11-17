"""Record of the user-supplied task source, taken before drafting repairs the staged copy."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.task_drafting.publication_fs import (
    RESOURCE_MANIFEST,
    TaskPublicationError,
    atomic_write,
    file_sha256,
    read_regular_bytes,
    real_path,
    resource_tree_digest,
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
# Drafting output lives here by default; resources/ is recorded as one tree digest instead.
_SOURCE_BASELINE_OMITTED_ROOT_ENTRIES = frozenset({".task_drafting", "resources"})


def record_task_source_baseline(task_dir: Path, baseline_path: Path) -> None:
    """Record the user-supplied task bytes once, before drafting edits the staged copy.

    The record is write-once so a resumed run keeps the baseline of the original
    source. Symlinks, special files and unreadable files are not recorded, so they
    can never be replaced by publication. The drafting output that holds the record
    is skipped wherever it lives inside the task.
    """
    if baseline_path.exists() or baseline_path.is_symlink():
        return
    root = real_path(task_dir, "task directory")
    payload = {
        "schema_version": _SOURCE_BASELINE_SCHEMA_VERSION,
        "files": _source_file_digests(root, output_dir=Path(os.path.abspath(baseline_path.parent))),
        "resources_sha256": _source_resources_digest(root / "resources"),
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(baseline_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _source_file_digests(root: Path, *, output_dir: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for directory, directory_names, file_names in os.walk(root):
        current = Path(directory)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in _SOURCE_BASELINE_OMITTED_DIRS
            and not (current == root and name in _SOURCE_BASELINE_OMITTED_ROOT_ENTRIES)
            and current / name != output_dir
        )
        for name in sorted(file_names):
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                digests[path.relative_to(root).as_posix()] = file_sha256(path)
            except (OSError, TaskPublicationError):
                # Unreadable files, and files still being written, stay unrecorded and
                # therefore never replaceable.
                continue
    return digests


def _source_resources_digest(resources: Path) -> str | None:
    if resources.is_symlink() or not resources.is_dir():
        return None
    try:
        return resource_tree_digest(resources, RESOURCE_MANIFEST)
    except (OSError, TaskPublicationError):
        return None


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
    if not isinstance(payload, dict):
        raise TaskPublicationError(f"invalid task source baseline: {baseline_path}")
    files = payload.get("files")
    resources_sha256 = payload.get("resources_sha256")
    if (
        payload.get("schema_version") != _SOURCE_BASELINE_SCHEMA_VERSION
        or not isinstance(files, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())
        or not (resources_sha256 is None or isinstance(resources_sha256, str))
    ):
        raise TaskPublicationError(f"invalid task source baseline: {baseline_path}")
    return SourceBaseline(file_sha256=dict(files), resources_sha256=resources_sha256)


REPLACED_SOURCE_DIRNAME = "replaced_task_source"

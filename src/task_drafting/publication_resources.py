"""Publication of the drafted ``resources/`` tree as one harness-managed directory."""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
from pathlib import Path

from src.task_drafting.publication_fs import (
    RESOURCE_MANIFEST,
    TaskPublicationError,
    atomic_write,
    fsync_tree,
    open_absolute_directory,
    read_regular_bytes,
    regular_tree_files,
    resource_tree_digest,
    validate_tree,
)

_MANIFEST_MAX_BYTES = 1024 * 1024
_logger = logging.getLogger(__name__)


def publish_resources(
    source: Path,
    target: Path,
    *,
    max_runtime_bytes: int,
    expected_sha256: str | None,
) -> None:
    """Install the accepted tree, or confirm an existing managed tree already matches it."""
    _validate_staged_resources(source, max_runtime_bytes=max_runtime_bytes)
    if target.exists():
        require_matching_managed_resources(source, target)
        return
    _replace_tree(source, target, expected_sha256=expected_sha256)


def replace_resources(
    source: Path,
    target: Path,
    *,
    max_runtime_bytes: int,
    expected_sha256: str | None,
) -> None:
    """Install the accepted tree in place of the unchanged pre-drafting source tree."""
    _validate_staged_resources(source, max_runtime_bytes=max_runtime_bytes)
    _replace_tree(source, target, expected_sha256=expected_sha256)


def _validate_staged_resources(source: Path, *, max_runtime_bytes: int) -> None:
    validate_tree(source)
    if (source / RESOURCE_MANIFEST).exists():
        raise TaskPublicationError("drafted resources contain a harness-owned manifest")
    total_bytes = sum(path.stat().st_size for path in regular_tree_files(source))
    if total_bytes > max_runtime_bytes:
        raise TaskPublicationError(
            f"drafted runtime resources exceed configured limit ({max_runtime_bytes} bytes): {source}"
        )


def _replace_tree(source: Path, target: Path, *, expected_sha256: str | None) -> None:
    parent_fd = open_absolute_directory(target.parent)
    try:
        # An existing tree still holds the pre-drafting source resources; move it
        # aside so the accepted tree can take its place, and restore it on failure.
        aside_name = _move_aside(parent_fd, target.name)
        try:
            _install_tree(source, target, parent_fd, expected_sha256=expected_sha256)
        except BaseException as exc:
            if aside_name is not None:
                _restore_aside(parent_fd, aside_name, target, exc)
            raise
        if aside_name is not None:
            _discard_aside(target.with_name(aside_name))
    finally:
        os.close(parent_fd)


def _move_aside(parent_fd: int, name: str) -> str | None:
    aside_name = f".{name}.replaced-{secrets.token_hex(12)}"
    try:
        os.rename(name, aside_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    return aside_name


def _restore_aside(parent_fd: int, aside_name: str, target: Path, failure: BaseException) -> None:
    try:
        os.rename(aside_name, target.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        failure.add_note(
            f"original task resources were kept at {target.with_name(aside_name)} "
            f"because {target} could not be restored: {exc}"
        )


def _discard_aside(aside: Path) -> None:
    try:
        shutil.rmtree(aside)
    except OSError as exc:
        _logger.warning("could not remove replaced task resources at %s: %s", aside, exc)


def _install_tree(
    source: Path,
    target: Path,
    parent_fd: int,
    *,
    expected_sha256: str | None,
) -> None:
    temporary_name = f".{target.name}.publish-{secrets.token_hex(12)}"
    os.mkdir(temporary_name, 0o700, dir_fd=parent_fd)
    temporary = (Path(f"/proc/self/fd/{parent_fd}") / temporary_name).resolve(strict=True)
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True, symlinks=True)
        validate_tree(temporary)
        resource_digest = resource_tree_digest(temporary, RESOURCE_MANIFEST)
        if resource_digest != expected_sha256:
            raise TaskPublicationError("drafted resources changed during publication")
        manifest = {
            "managed_by": "agentic-scientist-task-drafting",
            "schema_version": 1,
            "sha256": resource_digest,
        }
        atomic_write(
            temporary / RESOURCE_MANIFEST,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        fsync_tree(temporary)
        try:
            os.replace(
                temporary_name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        except OSError:
            if not target.exists():
                raise
            require_matching_managed_resources(source, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def require_matching_managed_resources(source: Path, target: Path) -> None:
    require_managed_resource_digest(
        target,
        resource_tree_digest(source, RESOURCE_MANIFEST),
    )


def require_managed_resource_digest(target: Path, expected_sha256: str | None) -> None:
    validate_tree(target)
    manifest_path = target / RESOURCE_MANIFEST
    try:
        manifest = json.loads(
            read_regular_bytes(
                manifest_path,
                maximum_bytes=_MANIFEST_MAX_BYTES,
            ).decode("utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskPublicationError(f"invalid managed task resources: {target}") from exc
    target_digest = resource_tree_digest(target, RESOURCE_MANIFEST)
    if manifest != {
        "managed_by": "agentic-scientist-task-drafting",
        "schema_version": 1,
        "sha256": target_digest,
    } or target_digest != expected_sha256:
        raise TaskPublicationError("drafted resources differ from existing managed task resources")

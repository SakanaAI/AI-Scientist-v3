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
    move_aside,
    open_absolute_directory,
    read_regular_bytes,
    regular_tree_files,
    resource_tree_digest,
    restore_aside,
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
    parent_fd = open_absolute_directory(target.parent)
    try:
        _install_tree(source, target, parent_fd, expected_sha256=expected_sha256)
    finally:
        os.close(parent_fd)


def replace_resources(
    source: Path,
    target: Path,
    *,
    max_runtime_bytes: int,
    expected_sha256: str | None,
    baseline_sha256: str,
    backup: Path,
) -> bool:
    """Install the accepted tree in place of the unchanged pre-drafting source tree.

    The source tree is moved aside first and only then compared with the baseline,
    so a concurrent edit aborts publication and the tree is restored. When its
    content differs from the accepted tree it is copied to ``backup`` before being
    discarded; returns whether that happened.
    """
    _validate_staged_resources(source, max_runtime_bytes=max_runtime_bytes)
    parent_fd = open_absolute_directory(target.parent)
    try:
        aside_name = move_aside(parent_fd, target.name)
        if aside_name is None:
            raise TaskPublicationError(f"task resources disappeared during publication: {target}")
        aside = target.with_name(aside_name)
        backed_up = False
        try:
            backed_up = _keep_verified_original(
                aside,
                baseline_sha256=baseline_sha256,
                backup=backup if baseline_sha256 != expected_sha256 else None,
            )
            _install_tree(source, target, parent_fd, expected_sha256=expected_sha256)
        except BaseException as exc:
            if backed_up:
                # The original goes back in place, so a retry must be able to back it up again.
                shutil.rmtree(backup, ignore_errors=True)
            restore_aside(parent_fd, aside_name, target, exc)
            raise
        _discard_aside(aside)
        return backed_up
    finally:
        os.close(parent_fd)


def _validate_staged_resources(source: Path, *, max_runtime_bytes: int) -> None:
    validate_tree(source)
    if (source / RESOURCE_MANIFEST).exists():
        raise TaskPublicationError("drafted resources contain a harness-owned manifest")
    total_bytes = sum(path.stat().st_size for path in regular_tree_files(source))
    if total_bytes > max_runtime_bytes:
        raise TaskPublicationError(
            f"drafted runtime resources exceed configured limit ({max_runtime_bytes} bytes): {source}"
        )


def _keep_verified_original(aside: Path, *, baseline_sha256: str, backup: Path | None) -> bool:
    """Require ``aside`` to hold the baseline tree, copying it to ``backup`` when given.

    The digest is taken from the backup itself, so the saved bytes are the verified ones.
    """
    verified = aside
    if backup is not None:
        shutil.copytree(aside, backup, symlinks=True)
        verified = backup
    if resource_tree_digest(verified, RESOURCE_MANIFEST) != baseline_sha256:
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        raise TaskPublicationError("task resources changed during publication")
    return backup is not None


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

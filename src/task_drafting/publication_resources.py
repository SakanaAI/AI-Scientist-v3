"""Publication of the drafted ``resources/`` tree as one harness-managed directory."""

from __future__ import annotations

import json
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


def publish_resources(
    source: Path,
    target: Path,
    *,
    max_runtime_bytes: int,
    expected_sha256: str | None,
    replaceable: bool = False,
) -> None:
    validate_tree(source)
    if (source / RESOURCE_MANIFEST).exists():
        raise TaskPublicationError("drafted resources contain a harness-owned manifest")
    total_bytes = sum(path.stat().st_size for path in regular_tree_files(source))
    if total_bytes > max_runtime_bytes:
        raise TaskPublicationError(
            f"drafted runtime resources exceed configured limit ({max_runtime_bytes} bytes): {source}"
        )
    if target.exists() and not replaceable:
        require_matching_managed_resources(source, target)
        return
    parent_fd = open_absolute_directory(target.parent)
    replaced_name: str | None = None
    if target.exists():
        # The existing tree still holds the pre-drafting source resources; move it
        # aside so the accepted tree can take its place, and restore it on failure.
        replaced_name = f".{target.name}.replaced-{secrets.token_hex(12)}"
        os.rename(target.name, replaced_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    temporary_name = f".{target.name}.publish-{secrets.token_hex(12)}"
    os.mkdir(temporary_name, 0o700, dir_fd=parent_fd)
    temporary = (Path(f"/proc/self/fd/{parent_fd}") / temporary_name).resolve(strict=True)
    published = False
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
        published = True
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if replaced_name is not None:
            if published:
                shutil.rmtree(Path(f"/proc/self/fd/{parent_fd}") / replaced_name)
            elif not target.exists():
                os.rename(replaced_name, target.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        os.close(parent_fd)


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

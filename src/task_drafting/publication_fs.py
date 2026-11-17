"""Descriptor-safe filesystem operations for task publication."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import secrets
import stat
import time
from collections.abc import Iterator
from pathlib import Path

TREE_MAX_DEPTH = 64
TREE_MAX_ENTRIES = 100_000
FileIdentity = tuple[int, int]
_logger = logging.getLogger(__name__)


class TaskPublicationError(ValueError):
    """Raised when a drafted task cannot be published safely."""


class TaskPublicationDeadlineExceeded(TaskPublicationError):
    """Raised when a bounded publication traversal exhausts its deadline."""


RESOURCE_MANIFEST = ".agentic_resource_manifest.json"


def real_path(path: Path, role: str) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise TaskPublicationError(f"unsafe {role}: {path}")
    try:
        return absolute.resolve(strict=True)
    except OSError as exc:
        raise TaskPublicationError(f"missing {role}: {path}") from exc


def validate_tree(
    root: Path,
    omitted_root_entries: set[str] | None = None,
    omitted_subtrees: set[Path] | None = None,
) -> None:
    for _ in regular_tree_files(
        root,
        omitted_root_entries=omitted_root_entries,
        omitted_subtrees=omitted_subtrees,
    ):
        pass


def regular_tree_files(
    root: Path,
    *,
    omitted_root_entries: set[str] | None = None,
    omitted_subtrees: set[Path] | None = None,
    deadline: float | None = None,
) -> Iterator[Path]:
    for path, is_directory in _tree_entries(
        root,
        omitted_root_entries=omitted_root_entries,
        omitted_subtrees=omitted_subtrees,
        deadline=deadline,
    ):
        if not is_directory:
            yield path


def _tree_entries(
    root: Path,
    *,
    omitted_root_entries: set[str] | None = None,
    omitted_subtrees: set[Path] | None = None,
    deadline: float | None = None,
) -> Iterator[tuple[Path, bool]]:
    omitted = omitted_root_entries or set()
    excluded = omitted_subtrees or set()
    entry_count = 0

    def visit(current: Path) -> Iterator[tuple[Path, bool]]:
        nonlocal entry_count
        records: list[tuple[str, Path, bool]] = []
        _require_traversal_time(deadline)
        with os.scandir(current) as scanner:
            for entry in scanner:
                _require_traversal_time(deadline)
                path = current / entry.name
                if (current == root and entry.name in omitted) or path in excluded:
                    continue
                entry_count += 1
                if entry_count > TREE_MAX_ENTRIES:
                    raise TaskPublicationError(
                        "task publication tree exceeds maximum entry count"
                    )
                relative = path.relative_to(root)
                if len(relative.parts) > TREE_MAX_DEPTH:
                    raise TaskPublicationError("task publication tree exceeds maximum depth")
                if entry.is_symlink():
                    raise TaskPublicationError(
                        f"unsafe task publication artifact: {relative.as_posix()}"
                    )
                is_directory = entry.is_dir(follow_symlinks=False)
                if not is_directory and not entry.is_file(follow_symlinks=False):
                    raise TaskPublicationError(
                        f"unsafe task publication artifact: {relative.as_posix()}"
                    )
                records.append((entry.name, path, is_directory))
        for _, path, is_directory in sorted(records):
            _require_traversal_time(deadline)
            yield path, is_directory
            if is_directory:
                yield from visit(path)

    yield from visit(root)


def _require_traversal_time(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TaskPublicationDeadlineExceeded("task publication traversal deadline exceeded")


def resource_tree_digest(root: Path, manifest_name: str) -> str:
    digest = hashlib.sha256()
    for path, is_directory in _tree_entries(root):
        relative = path.relative_to(root).as_posix()
        if relative == manifest_name:
            continue
        digest.update(relative.encode() + b"\0")
        if not is_directory:
            _update_digest_from_file(digest, path)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    _update_digest_from_file(digest, path)
    return digest.hexdigest()


def regular_file_identity(path: Path) -> FileIdentity:
    descriptor = _open_regular_file(path)
    try:
        metadata = os.fstat(descriptor)
        return metadata.st_dev, metadata.st_ino
    finally:
        os.close(descriptor)


def unlink_regular_file_if_identity(path: Path, expected: FileIdentity) -> None:
    parent_fd = open_absolute_directory(path.parent)
    try:
        try:
            current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_uid != os.getuid()
            or current.st_nlink != 1
            or (current.st_dev, current.st_ino) != expected
        ):
            raise TaskPublicationError(f"completion marker changed before cleanup: {path}")
        os.unlink(path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _update_digest_from_file(digest: hashlib._Hash, path: Path) -> None:
    descriptor = _open_regular_file(path)
    try:
        initial = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        _require_stable_file(initial, os.fstat(descriptor), path)
    finally:
        os.close(descriptor)


def same_file_bytes(left: Path, right: Path) -> bool:
    if right.is_symlink() or not right.is_file():
        return False
    left_fd = _open_regular_file(left)
    right_fd = _open_regular_file(right)
    try:
        left_stat = os.fstat(left_fd)
        right_stat = os.fstat(right_fd)
        if left_stat.st_size != right_stat.st_size:
            return False
        while left_chunk := os.read(left_fd, 1024 * 1024):
            if left_chunk != _read_exact(right_fd, len(left_chunk)):
                return False
        matches = not os.read(right_fd, 1)
        _require_stable_file(left_stat, os.fstat(left_fd), left)
        _require_stable_file(right_stat, os.fstat(right_fd), right)
        return matches
    finally:
        os.close(right_fd)
        os.close(left_fd)


def _read_exact(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_regular_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    descriptor = _open_regular_file(path)
    try:
        initial = os.fstat(descriptor)
        if initial.st_size > maximum_bytes:
            raise TaskPublicationError(f"drafted file exceeds {maximum_bytes} bytes: {path}")
        content = _read_exact(descriptor, maximum_bytes + 1)
        if len(content) > maximum_bytes:
            raise TaskPublicationError(f"drafted file exceeds {maximum_bytes} bytes: {path}")
        _require_stable_file(initial, os.fstat(descriptor), path)
        return content
    finally:
        os.close(descriptor)


def atomic_copy(source: Path, target: Path, *, replace_existing: bool = False) -> None:
    source_fd = _open_regular_file(source)
    source_stat = os.fstat(source_fd)
    parent_fd = open_absolute_directory(target.parent, create=True)
    temporary = f".{target.name}.publish-{secrets.token_hex(12)}"
    target_fd: int | None = None
    try:
        target_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            stat.S_IMODE(source_stat.st_mode) | stat.S_IWUSR,
            dir_fd=parent_fd,
        )
        while chunk := os.read(source_fd, 1024 * 1024):
            _write_all(target_fd, chunk)
        os.fchmod(target_fd, stat.S_IMODE(source_stat.st_mode))
        os.fsync(target_fd)
        _require_stable_file(source_stat, os.fstat(source_fd), source)
        _publish_temporary(
            parent_fd,
            temporary,
            target.name,
            replace_existing=replace_existing,
        )
        os.fsync(parent_fd)
    finally:
        if target_fd is not None:
            os.close(target_fd)
        _unlink_if_present(parent_fd, temporary)
        os.close(parent_fd)
        os.close(source_fd)


def atomic_write(path: Path, content: bytes, *, replace_existing: bool = False) -> None:
    parent_fd = open_absolute_directory(path.parent, create=True)
    temporary = f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        _write_all(descriptor, content)
        os.fsync(descriptor)
        _publish_temporary(
            parent_fd,
            temporary,
            path.name,
            replace_existing=replace_existing,
        )
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        _unlink_if_present(parent_fd, temporary)
        os.close(parent_fd)


def open_absolute_directory(path: Path, *, create: bool = False) -> int:
    absolute = Path(os.path.abspath(path))
    directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o755, dir_fd=directory_fd)
                os.fsync(directory_fd)
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            os.close(directory_fd)
            directory_fd = next_fd
    except BaseException:
        os.close(directory_fd)
        raise
    return directory_fd


def _open_regular_file(path: Path) -> int:
    parent_fd = open_absolute_directory(path.parent)
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise TaskPublicationError(f"unsafe drafted artifact: {path}")
    return descriptor


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        remaining = remaining[os.write(descriptor, remaining) :]


def _unlink_if_present(parent_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def move_aside(parent_fd: int, name: str) -> str | None:
    """Rename ``name`` to a hidden sibling and return that name, or None when absent.

    Once moved aside, the original can be verified without racing writes through
    its path: a concurrent writer recreates ``name`` instead of editing it.
    """
    aside_name = f".{name}.replaced-{secrets.token_hex(12)}"
    try:
        os.rename(name, aside_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    return aside_name


def restore_aside(parent_fd: int, aside_name: str, target: Path, failure: BaseException) -> None:
    """Move an original back to ``target`` without replacing anything that appeared there.

    When ``target`` was recreated meanwhile, the original stays aside and its location
    is attached to ``failure``.
    """
    try:
        if stat.S_ISDIR(os.stat(aside_name, dir_fd=parent_fd, follow_symlinks=False).st_mode):
            # rename(2) refuses to replace a non-empty directory.
            os.rename(aside_name, target.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        else:
            os.link(
                aside_name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            os.unlink(aside_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        failure.add_note(
            f"original task artifact was kept at {target.with_name(aside_name)} "
            f"because {target} could not be restored: {exc}"
        )


def replace_verified_file(source: Path, target: Path, *, baseline_sha256: str, backup: Path) -> bool:
    """Replace ``target`` with ``source`` only while it still holds the baseline bytes.

    ``target`` is moved aside, copied to ``backup`` and verified from that copy, so the
    saved original is exactly the verified one and a concurrent edit or recreation of
    ``target`` aborts the replacement. Returns whether ``target`` was replaced.
    """
    if same_file_bytes(source, target):
        return False
    parent_fd = open_absolute_directory(target.parent)
    try:
        aside_name = move_aside(parent_fd, target.name)
        if aside_name is None:
            raise TaskPublicationError(f"task artifact disappeared during publication: {target}")
        try:
            atomic_copy(target.with_name(aside_name), backup, replace_existing=True)
            if file_sha256(backup) != baseline_sha256:
                raise TaskPublicationError(f"task artifact changed during publication: {target}")
            atomic_copy(source, target)
        except BaseException as exc:
            # The original goes back in place, so its backup would be misleading.
            with contextlib.suppress(OSError):
                backup.unlink(missing_ok=True)
            restore_aside(parent_fd, aside_name, target, exc)
            raise
        _discard_aside_file(parent_fd, aside_name, target)
        return True
    finally:
        os.close(parent_fd)


def _discard_aside_file(parent_fd: int, aside_name: str, target: Path) -> None:
    try:
        os.unlink(aside_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        _logger.warning(
            "could not remove replaced task artifact at %s: %s", target.with_name(aside_name), exc
        )


def _publish_temporary(
    parent_fd: int,
    temporary: str,
    destination: str,
    *,
    replace_existing: bool,
) -> None:
    if replace_existing:
        os.replace(
            temporary,
            destination,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        return
    os.link(
        temporary,
        destination,
        src_dir_fd=parent_fd,
        dst_dir_fd=parent_fd,
        follow_symlinks=False,
    )
    os.unlink(temporary, dir_fd=parent_fd)


def _require_stable_file(expected: os.stat_result, actual: os.stat_result, path: Path) -> None:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if not stat.S_ISREG(actual.st_mode) or any(
        getattr(expected, field) != getattr(actual, field) for field in fields
    ):
        raise TaskPublicationError(f"task publication source changed while reading: {path}")


def fsync_directory(path: Path) -> None:
    descriptor = open_absolute_directory(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(root: Path) -> None:
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for path in regular_tree_files(root):
        descriptor = _open_regular_file(path)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(set(directories), key=lambda item: len(item.parts), reverse=True):
        fsync_directory(directory)

"""Inspect all editable candidate sources with fixed host-owned resource bounds."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import stat
from collections.abc import Iterator

MAX_SOURCE_FILE_BYTES = 1024 * 1024  # real training scripts exceed 128 KiB (noise track 3: 165 KB)
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_SOURCE_ENTRIES = 2048
MAX_SOURCE_DEPTH = 32
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_LIMIT_ERROR = 'program_1 source exceeds harness inspection limit'


def read_candidate_sources(program_dir: Path) -> tuple[str, ...]:
    sources: list[str] = []
    total_bytes = 0
    try:
        with _directory(program_dir) as program_descriptor:
            with _directory('src', program_descriptor) as source_descriptor:
                for count, (parent, name, metadata) in enumerate(_entries(source_descriptor), 1):
                    if count > MAX_SOURCE_ENTRIES:
                        raise SystemExit(_LIMIT_ERROR)
                    if stat.S_ISDIR(metadata.st_mode):
                        continue
                    if stat.S_ISLNK(metadata.st_mode):
                        raise SystemExit('program_1 source directory must not contain symlinks')
                    if not stat.S_ISREG(metadata.st_mode):
                        raise SystemExit('program_1 source must contain only regular files')
                    total_bytes += metadata.st_size
                    if metadata.st_size > MAX_SOURCE_FILE_BYTES or total_bytes > MAX_SOURCE_BYTES:
                        raise SystemExit(_LIMIT_ERROR)
                    if name.endswith('.py'):
                        sources.append(_read_source(parent, name, metadata))
    except OSError as error:
        raise SystemExit('program_1 source could not be safely inspected') from error
    return tuple(sources)


@contextmanager
def _directory(path: str | Path, parent: int | None = None) -> Iterator[int]:
    descriptor = os.open(path, _DIRECTORY_FLAGS, dir_fd=parent)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def _entries(
    descriptor: int, depth: int = 0,
) -> Iterator[tuple[int, str, os.stat_result]]:
    if depth > MAX_SOURCE_DEPTH:
        raise SystemExit(_LIMIT_ERROR)
    with os.scandir(descriptor) as children:
        for child in children:
            metadata = child.stat(follow_symlinks=False)
            yield descriptor, child.name, metadata
            if stat.S_ISDIR(metadata.st_mode):
                with _directory(child.name, descriptor) as nested:
                    _require_unchanged(metadata, os.fstat(nested))
                    yield from _entries(nested, depth + 1)


def _require_unchanged(before: os.stat_result, after: os.stat_result) -> None:
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise SystemExit('program_1 source changed during inspection')


def _read_source(parent: int, name: str, expected: os.stat_result) -> str:
    descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent)
    with os.fdopen(descriptor, 'rb') as stream:
        _require_unchanged(expected, os.fstat(stream.fileno()))
        content = stream.read(MAX_SOURCE_FILE_BYTES + 1)
        if len(content) > MAX_SOURCE_FILE_BYTES:
            raise SystemExit(_LIMIT_ERROR)
        _require_unchanged(expected, os.fstat(stream.fileno()))
        if len(content) != expected.st_size:
            raise SystemExit('program_1 source changed during inspection')
    return content.decode('utf-8', errors='replace')

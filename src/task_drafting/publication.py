"""Contract-driven publication for validated drafted task scaffolds."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from src.task_drafting.publication_fs import (
    TaskPublicationDeadlineExceeded as RuntimeResourceValidationDeadlineExceeded,
    TaskPublicationError,
    atomic_copy as _atomic_copy,
    atomic_write as _atomic_write,
    file_sha256 as _file_sha256,
    fsync_directory as _fsync_directory,
    fsync_tree as _fsync_tree,
    open_absolute_directory as _open_absolute_directory,
    read_regular_bytes as _read_regular_bytes,
    real_path as _real_path,
    regular_file_identity as _regular_file_identity,
    regular_tree_files as _regular_tree_files,
    resource_tree_digest,
    same_file_bytes as _same_file_bytes,
    unlink_regular_file_if_identity as _unlink_regular_file_if_identity,
    validate_tree as _validate_tree,
)
from src.task_drafting.qualification import (
    TaskQualificationError,
    load_task_qualification_receipt,
)
from src.investigation.files import TASK_COPY_IGNORE_FILENAME
from src.investigation.program_spec import ProgramSpecError, load_program_spec, resolve_entrypoint
from src.investigation.checkpoint_security import report_sensitive_path
from src.investigation.task_alignment import TaskAlignmentError, load_task_alignment_report
from src.investigation.task_contract import TaskContractError, load_task_contract

_CONTRACT_PATH = Path("task_contract.json")
_MARKER_PATH = Path(".agentic_task_validation.json")
_PUBLICATION_PATH = Path(".agentic_task_publication.json")
_QUALIFICATION_PATH = Path(".agentic_task_qualification.json")
_PROGRAM_SPEC_PATH = Path("program_0/program_spec.toml")
_RESOURCE_MANIFEST = ".agentic_resource_manifest.json"
_RUNTIME_FILE_MAX_BYTES = 1024 * 1024
_MAX_RUNTIME_PUBLICATION_BYTES = 64 * 1024 * 1024 * 1024
_MAX_SOURCE_BASELINE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _PublicationPlan:
    files: tuple[Path, ...]
    file_sha256: tuple[tuple[Path, str], ...]
    ignore_file: bytes
    resources: Path | None
    resource_sha256: str | None
    manifest: bytes


@dataclass(frozen=True, slots=True)
class _SourceBaseline:
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
        return _file_sha256(target) == expected

    def matches_resources(self, target: Path) -> bool:
        if self.resources_sha256 is None or target.is_symlink() or not target.is_dir():
            return False
        try:
            return resource_tree_digest(target, _RESOURCE_MANIFEST) == self.resources_sha256
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
    root = _real_path(task_dir, "task directory")
    file_sha256: dict[str, str] = {}
    for directory, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(
            name for name in directory_names if name not in _SOURCE_BASELINE_OMITTED_DIRS
        )
        current = Path(directory)
        for name in sorted(file_names):
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            file_sha256[path.relative_to(root).as_posix()] = _file_sha256(path)
    resources_sha256: str | None = None
    resources = root / "resources"
    if resources.is_dir() and not resources.is_symlink():
        try:
            resources_sha256 = resource_tree_digest(resources, _RESOURCE_MANIFEST)
        except TaskPublicationError:
            resources_sha256 = None
    payload = {
        "schema_version": _SOURCE_BASELINE_SCHEMA_VERSION,
        "files": file_sha256,
        "resources_sha256": resources_sha256,
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(baseline_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _load_source_baseline(baseline_path: Path | None) -> _SourceBaseline | None:
    if baseline_path is None:
        return None
    try:
        payload = json.loads(
            _read_regular_bytes(baseline_path, maximum_bytes=_MAX_SOURCE_BASELINE_BYTES).decode(
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
    return _SourceBaseline(file_sha256=dict(files), resources_sha256=resources_sha256)


def validate_staged_runtime_resources(
    staged_task_dir: Path,
    *,
    max_runtime_bytes: int,
    deadline: float | None = None,
) -> int:
    """Validate the drafted resource tree and return its bounded byte total."""
    if not 1 <= max_runtime_bytes <= _MAX_RUNTIME_PUBLICATION_BYTES:
        raise TaskPublicationError("max_runtime_bytes must be between 1 byte and 64 GiB")
    resource_root = staged_task_dir / "resources"
    if resource_root.is_symlink():
        raise TaskPublicationError(f"unsafe drafted runtime resources: {resource_root}")
    if not resource_root.exists():
        return 0
    if not resource_root.is_dir():
        raise TaskPublicationError(f"unsafe drafted runtime resources: {resource_root}")
    if deadline is not None and time.monotonic() >= deadline:
        raise RuntimeResourceValidationDeadlineExceeded(
            "runtime resource validation exhausted its deadline"
        )
    total_bytes = 0
    largest: list[tuple[int, str]] = []
    for path in _regular_tree_files(resource_root, deadline=deadline):
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeResourceValidationDeadlineExceeded(
                "runtime resource validation exhausted its deadline"
            )
        if path.name == _RESOURCE_MANIFEST:
            continue
        size = path.stat().st_size
        total_bytes += size
        relative = path.relative_to(resource_root).as_posix()
        largest.append((size, report_sensitive_path(relative)))
        largest = sorted(largest, reverse=True)[:5]
        if total_bytes > max_runtime_bytes:
            contributors = ", ".join(f"{name}:{size}" for size, name in largest)
            raise TaskPublicationError(
                "runtime_resources_exceed_limit: "
                f"observed={total_bytes} limit={max_runtime_bytes}; "
                f"largest_observed=[{contributors}]"
            )
    return total_bytes


def publish_task_scaffold(
    *,
    task_dir: Path,
    staged_task_dir: Path,
    output_dir: Path,
    alignment_report_path: Path,
    qualification_receipt_path: Path | None = None,
    max_runtime_bytes: int,
    source_baseline_path: Path | None = None,
) -> None:
    """Publish one validated scaffold without consuming its staged source.

    ``source_baseline_path`` points at the record written by
    :func:`record_task_source_baseline` when the task was staged. Destination
    artifacts that still match it are replaced by the accepted scaffold; without it,
    any differing destination artifact aborts publication.
    """
    task_dir, staged_task_dir, output_dir, alignment_report_path = _validated_roots(
        task_dir, staged_task_dir, output_dir, alignment_report_path
    )
    if max_runtime_bytes < 1:
        raise TaskPublicationError("max_runtime_bytes must be positive")

    plan = _build_plan(
        task_dir,
        staged_task_dir,
        max_runtime_bytes=max_runtime_bytes,
    )
    qualification_bytes = _validated_qualification_bytes(
        qualification_receipt_path,
        output_dir=output_dir,
        plan=plan,
    )
    if qualification_bytes is None and (
        (task_dir / _QUALIFICATION_PATH).exists()
        or (task_dir / _QUALIFICATION_PATH).is_symlink()
    ):
        raise TaskPublicationError("task destination contains a stale qualification receipt")
    _validate_staged_acceptance(staged_task_dir, alignment_report_path)
    baseline = _load_source_baseline(source_baseline_path)
    if _settle_existing_marker(
        task_dir,
        alignment_report_path,
        plan,
        qualification_bytes,
    ):
        return
    _preflight_destination(
        task_dir,
        staged_task_dir,
        alignment_report_path,
        plan,
        qualification_bytes,
        baseline,
    )
    _publish_bytes(plan.manifest, task_dir / _PUBLICATION_PATH)

    for relative in plan.files:
        source = staged_task_dir / relative
        target = task_dir / relative
        replaceable = baseline is not None and baseline.matches_file(target, relative)
        if relative == _PROGRAM_SPEC_PATH:
            _publish_program_spec(source, target, replaceable=replaceable)
        elif replaceable:
            _publish_regular_file(source, target, replaceable=True)
        else:
            _publish_regular_file(source, target)
    if plan.resources is not None:
        _publish_resources(
            staged_task_dir / plan.resources,
            task_dir / plan.resources,
            max_runtime_bytes=max_runtime_bytes,
            expected_sha256=plan.resource_sha256,
            replaceable=baseline is not None
            and baseline.matches_resources(task_dir / plan.resources),
        )
    _publish_replaceable_bytes(plan.ignore_file, task_dir / TASK_COPY_IGNORE_FILENAME)
    if qualification_bytes is not None:
        _publish_bytes(qualification_bytes, task_dir / _QUALIFICATION_PATH)

    _validate_published_plan(task_dir, plan)
    _validate_destination(task_dir, alignment_report_path)
    marker_path = task_dir / _MARKER_PATH
    _publish_regular_file(alignment_report_path, marker_path)
    marker_identity = _regular_file_identity(marker_path)
    try:
        _validate_published_plan(task_dir, plan)
        _validate_destination(task_dir, marker_path)
        if qualification_bytes is not None and not _bytes_match(
            task_dir / _QUALIFICATION_PATH, qualification_bytes
        ):
            raise TaskPublicationError("published task qualification receipt changed")
        if qualification_bytes is None and (task_dir / _QUALIFICATION_PATH).exists():
            raise TaskPublicationError("unexpected task qualification receipt appeared")
    except BaseException:
        _unlink_regular_file_if_identity(marker_path, marker_identity)
        raise


def task_publication_plan_sha256(
    *,
    task_dir: Path,
    staged_task_dir: Path,
    max_runtime_bytes: int,
) -> str:
    """Return the digest a qualification receipt must bind before publication."""
    task_dir = _real_path(task_dir, "task destination")
    staged_task_dir = _real_path(staged_task_dir, "staged task directory")
    if task_dir.is_symlink() or not task_dir.is_dir():
        raise TaskPublicationError(f"unsafe task destination: {task_dir}")
    _validate_tree(staged_task_dir, omitted_root_entries={".venv"})
    plan = _build_plan(task_dir, staged_task_dir, max_runtime_bytes=max_runtime_bytes)
    return hashlib.sha256(plan.manifest).hexdigest()


def validate_published_task_plan(task_dir: Path) -> str:
    """Validate every published file/resource against its sealed plan and return its digest."""
    task_dir = _real_path(task_dir, "published task directory")
    if task_dir.is_symlink() or not task_dir.is_dir():
        raise TaskPublicationError(f"unsafe published task directory: {task_dir}")
    try:
        sealed_manifest = json.loads(
            _read_regular_bytes(
                task_dir / _PUBLICATION_PATH,
                maximum_bytes=_RUNTIME_FILE_MAX_BYTES,
            )
        )
        max_runtime_bytes = sealed_manifest.get("max_runtime_bytes")
        if (
            isinstance(max_runtime_bytes, bool)
            or not isinstance(max_runtime_bytes, int)
            or not 1 <= max_runtime_bytes <= _MAX_RUNTIME_PUBLICATION_BYTES
        ):
            raise TaskPublicationError("published task has an invalid runtime byte cap")
        sealed_resources = sealed_manifest.get("resources")
        if isinstance(sealed_resources, dict):
            expected_bytes = sealed_resources.get("bytes")
            resource_root = task_dir / str(sealed_resources.get("path", ""))
            if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
                raise TaskPublicationError("published resource byte count is invalid")
            actual_bytes = sum(
                path.stat().st_size
                for path in _regular_tree_files(resource_root)
                if path.name != _RESOURCE_MANIFEST
            )
            if actual_bytes != expected_bytes:
                raise TaskPublicationError("published resources differ from publication plan")
        plan = _build_plan(task_dir, task_dir, max_runtime_bytes=max_runtime_bytes)
        _validate_published_plan(task_dir, plan)
    except TaskPublicationError:
        raise
    except (OSError, ProgramSpecError, TaskContractError, ValueError) as exc:
        raise TaskPublicationError(
            f"published task differs from publication plan: {exc}"
        ) from exc
    return hashlib.sha256(plan.manifest).hexdigest()


def _validated_qualification_bytes(
    path: Path | None,
    *,
    output_dir: Path,
    plan: _PublicationPlan,
) -> bytes | None:
    if path is None:
        return None
    path = _real_path(path, "qualification receipt")
    if not path.is_relative_to(output_dir):
        raise TaskPublicationError("qualification receipt escaped task drafting output")
    plan_sha256 = hashlib.sha256(plan.manifest).hexdigest()
    try:
        load_task_qualification_receipt(
            path,
            expected_publication_plan_sha256=plan_sha256,
        )
        return _read_regular_bytes(path, maximum_bytes=_RUNTIME_FILE_MAX_BYTES)
    except TaskQualificationError as exc:
        raise TaskPublicationError(str(exc)) from exc


def _validated_roots(
    task_dir: Path,
    staged_task_dir: Path,
    output_dir: Path,
    alignment_report_path: Path,
) -> tuple[Path, Path, Path, Path]:
    task_dir = _real_path(task_dir, "task destination")
    staged_task_dir = _real_path(staged_task_dir, "staged task directory")
    output_dir = _real_path(output_dir, "task drafting output")
    alignment_report_path = _real_path(alignment_report_path, "alignment report")
    if staged_task_dir != output_dir / "task" or not staged_task_dir.parent.samefile(output_dir):
        raise TaskPublicationError(f"unexpected staged task directory: {staged_task_dir}")
    if task_dir.is_symlink() or not task_dir.is_dir():
        raise TaskPublicationError(f"unsafe task destination: {task_dir}")
    if not alignment_report_path.is_relative_to(output_dir):
        raise TaskPublicationError("alignment report escaped task drafting output")
    _require_regular_file(alignment_report_path, output_dir)
    omitted_root_entries = {".venv"}
    omitted_subtrees: set[Path] = set()
    if output_dir.is_relative_to(task_dir):
        relative_output = output_dir.relative_to(task_dir)
        if relative_output.parts[0] == ".task_drafting":
            omitted_root_entries.add(".task_drafting")
        else:
            omitted_subtrees.add(output_dir)
    _validate_tree(
        task_dir,
        omitted_root_entries=omitted_root_entries,
        omitted_subtrees=omitted_subtrees,
    )
    _validate_tree(staged_task_dir, omitted_root_entries={".venv"})
    return task_dir, staged_task_dir, output_dir, alignment_report_path


def _build_plan(
    task_dir: Path,
    staged_task_dir: Path,
    *,
    max_runtime_bytes: int,
) -> _PublicationPlan:
    contract = load_task_contract(staged_task_dir / _CONTRACT_PATH)
    for artifact in contract.provenance.artifacts:
        if artifact.role == "search_helper":
            raise TaskPublicationError(
                "generated task-local evaluator helpers are unsupported; "
                f"inline helper {artifact.path} into {contract.evaluators.search}"
            )

    files = {
        Path("capture.py"),
        Path(contract.canonical_task.path),
        Path(contract.evaluators.search),
        Path(contract.evaluators.final),
        _CONTRACT_PATH,
        _PROGRAM_SPEC_PATH,
    }
    try:
        program_dir = staged_task_dir / "program_0"
        spec = load_program_spec(program_dir, required=True)
        assert spec is not None
        entrypoint = resolve_entrypoint(program_dir, spec)
        files.add(entrypoint.relative_to(staged_task_dir))
    except (ProgramSpecError, ValueError) as exc:
        raise TaskPublicationError(f"invalid drafted program specification: {exc}") from exc
    for editable in contract.candidate.editable_paths:
        files.update(_regular_files_below(staged_task_dir, Path(editable)))
    files.update(
        Path(artifact.path)
        for artifact in contract.provenance.artifacts
        if not _is_below(Path(artifact.path), Path("resources"))
    )
    for artifact in contract.provenance.artifacts:
        _reject_reserved_artifact(Path(artifact.path))
    requirements = staged_task_dir / "requirements.txt"
    if requirements.is_file():
        if requirements.stat().st_size > _RUNTIME_FILE_MAX_BYTES:
            raise TaskPublicationError(f"drafted runtime file exceeds 1 MiB: {requirements}")
        files.add(Path("requirements.txt"))

    ignore_file = _ignore_file_bytes(task_dir, contract)
    resources = Path("resources") if (staged_task_dir / "resources").is_dir() else None
    for relative in files:
        _require_regular_file(staged_task_dir / relative, staged_task_dir)
    file_sha256 = tuple(
        (relative, _file_sha256(staged_task_dir / relative))
        for relative in sorted(files, key=lambda item: item.as_posix())
    )
    resource_sha256 = None
    if resources is not None:
        resource_bytes = validate_staged_runtime_resources(
            staged_task_dir,
            max_runtime_bytes=max_runtime_bytes,
        )
        resource_sha256 = resource_tree_digest(staged_task_dir / resources, _RESOURCE_MANIFEST)
    manifest = _publication_manifest_bytes(
        file_sha256=file_sha256,
        ignore_file=ignore_file,
        resources=resources,
        resource_sha256=resource_sha256,
        resource_bytes=resource_bytes if resources is not None else None,
        max_runtime_bytes=max_runtime_bytes,
    )
    return _PublicationPlan(
        files=tuple(relative for relative, _ in file_sha256),
        file_sha256=file_sha256,
        ignore_file=ignore_file,
        resources=resources,
        resource_sha256=resource_sha256,
        manifest=manifest,
    )


def _publication_manifest_bytes(
    *,
    file_sha256: tuple[tuple[Path, str], ...],
    ignore_file: bytes,
    resources: Path | None,
    resource_sha256: str | None,
    resource_bytes: int | None,
    max_runtime_bytes: int,
) -> bytes:
    payload = {
        "schema_version": 1,
        "max_runtime_bytes": max_runtime_bytes,
        "files": {path.as_posix(): digest for path, digest in file_sha256},
        "generated": {
            TASK_COPY_IGNORE_FILENAME: hashlib.sha256(ignore_file).hexdigest(),
        },
        "resources": (
            None
            if resources is None
            else {
                "path": resources.as_posix(),
                "sha256": resource_sha256,
                "bytes": resource_bytes,
            }
        ),
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _regular_files_below(root: Path, relative: Path) -> set[Path]:
    path = root / relative
    if path.is_symlink() or not path.exists():
        raise TaskPublicationError(f"missing safe candidate editable path: {relative.as_posix()}")
    if path.is_file():
        return {relative}
    if not path.is_dir():
        raise TaskPublicationError(f"unsafe candidate editable path: {relative.as_posix()}")
    return {candidate.relative_to(root) for candidate in _regular_tree_files(path)}


def _reject_reserved_artifact(relative: Path) -> None:
    reserved = {
        _MARKER_PATH,
        _PUBLICATION_PATH,
        _QUALIFICATION_PATH,
        Path(TASK_COPY_IGNORE_FILENAME),
        Path(".agentic_runtime.json"),
    }
    if (
        any(_is_below(relative, path) for path in reserved)
        or relative.parts[0] == ".venv"
        or relative.name == _RESOURCE_MANIFEST
    ):
        raise TaskPublicationError(
            f"contract provenance artifact uses harness-owned path: {relative.as_posix()}"
        )


def _ignore_file_bytes(task_dir: Path, contract) -> bytes:
    path = task_dir / TASK_COPY_IGNORE_FILENAME
    patterns = (
        _read_regular_bytes(path, maximum_bytes=_RUNTIME_FILE_MAX_BYTES)
        .decode("utf-8")
        .splitlines()
        if path.is_file()
        else []
    )
    required = (
        ".venv/",
        "resources/",
        f"/{contract.evaluators.final}",
        f"/{contract.resources.final_root}/",
        "/task_contract.json",
        f"/{_PUBLICATION_PATH.as_posix()}",
        "/.agentic_task_validation.json",
        f"/{_QUALIFICATION_PATH.as_posix()}",
    )
    patterns.extend(pattern for pattern in required if pattern not in patterns)
    return ("\n".join(patterns) + "\n").encode()


def _validate_staged_acceptance(staged_task_dir: Path, report_path: Path) -> None:
    report = load_task_alignment_report(report_path, scaffold_dir=staged_task_dir)
    if report.verdict != "pass":
        raise TaskPublicationError("accepted task alignment report did not pass")


def _preflight_destination(
    task_dir: Path,
    staged_task_dir: Path,
    alignment_report_path: Path,
    plan: _PublicationPlan,
    qualification_bytes: bytes | None,
    baseline: _SourceBaseline | None = None,
) -> None:
    target_spec_is_valid = _program_spec_is_valid(task_dir / "program_0")
    for relative in plan.files:
        source = staged_task_dir / relative
        target = task_dir / relative
        _reject_symlink_components(target, task_dir)
        if not target.exists() or _same_file_bytes(source, target):
            continue
        if relative == _PROGRAM_SPEC_PATH and not target_spec_is_valid:
            continue
        if baseline is not None and baseline.matches_file(target, relative):
            continue
        raise TaskPublicationError(
            f"existing task artifact differs from accepted scaffold: {relative.as_posix()}"
        )
    publication_path = task_dir / _PUBLICATION_PATH
    if publication_path.is_file() and not _bytes_match(publication_path, plan.manifest):
        raise TaskPublicationError("task destination contains a different publication plan")
    _preflight_bytes(publication_path, plan.manifest)
    marker = task_dir / _MARKER_PATH
    if marker.exists() and not _same_file_bytes(alignment_report_path, marker):
        raise TaskPublicationError("existing task validation report differs from accepted report")
    if qualification_bytes is not None:
        _preflight_bytes(task_dir / _QUALIFICATION_PATH, qualification_bytes)
    elif (task_dir / _QUALIFICATION_PATH).exists() or (
        task_dir / _QUALIFICATION_PATH
    ).is_symlink():
        raise TaskPublicationError("task destination contains a stale qualification receipt")
    if plan.resources is not None and (task_dir / plan.resources).exists():
        if baseline is not None and baseline.matches_resources(task_dir / plan.resources):
            return
        _require_managed_resource_digest(task_dir / plan.resources, plan.resource_sha256)


def _program_spec_is_valid(program_dir: Path) -> bool:
    try:
        return load_program_spec(program_dir, required=True) is not None
    except ProgramSpecError:
        return False


def _settle_existing_marker(
    task_dir: Path,
    report_path: Path,
    plan: _PublicationPlan,
    qualification_bytes: bytes | None,
) -> bool:
    marker = task_dir / _MARKER_PATH
    if not marker.exists():
        return False
    try:
        if not _publication_is_complete(
            task_dir,
            report_path,
            plan,
            qualification_bytes,
        ):
            raise TaskPublicationError("published task closure is incomplete")
    except (OSError, ProgramSpecError, TaskAlignmentError, TaskContractError, TaskPublicationError):
        marker.unlink()
        _fsync_directory(task_dir)
        return False
    return True


def _publication_is_complete(
    task_dir: Path,
    report_path: Path,
    plan: _PublicationPlan,
    qualification_bytes: bytes | None,
) -> bool:
    marker = task_dir / _MARKER_PATH
    if not _same_file_bytes(report_path, marker):
        return False
    if not _bytes_match(task_dir / _PUBLICATION_PATH, plan.manifest):
        return False
    if not _bytes_match(task_dir / TASK_COPY_IGNORE_FILENAME, plan.ignore_file):
        return False
    if qualification_bytes is not None and not _bytes_match(
        task_dir / _QUALIFICATION_PATH, qualification_bytes
    ):
        return False
    if qualification_bytes is None and (
        (task_dir / _QUALIFICATION_PATH).exists()
        or (task_dir / _QUALIFICATION_PATH).is_symlink()
    ):
        return False
    _validate_published_plan(task_dir, plan)
    _validate_destination(task_dir, marker)
    return True


def _validate_published_plan(task_dir: Path, plan: _PublicationPlan) -> None:
    if not _bytes_match(task_dir / _PUBLICATION_PATH, plan.manifest):
        raise TaskPublicationError("published task differs from its publication plan")
    if not _bytes_match(task_dir / TASK_COPY_IGNORE_FILENAME, plan.ignore_file):
        raise TaskPublicationError("published ignore rules differ from publication plan")
    for relative, expected_sha256 in plan.file_sha256:
        path = task_dir / relative
        if not path.is_file() or _file_sha256(path) != expected_sha256:
            raise TaskPublicationError(
                f"published artifact differs from publication plan: {relative.as_posix()}"
            )
    if plan.resources is not None:
        _require_managed_resource_digest(task_dir / plan.resources, plan.resource_sha256)


def _bytes_match(path: Path, expected: bytes) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        return _read_regular_bytes(path, maximum_bytes=len(expected) + 1) == expected
    except TaskPublicationError:
        return False


def _preflight_bytes(target: Path, content: bytes) -> None:
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise TaskPublicationError(f"unsafe drafted target: {target}")
    if target.is_file() and _read_regular_bytes(target, maximum_bytes=len(content) + 1) != content:
        raise TaskPublicationError(f"existing task artifact differs from accepted scaffold: {target.name}")


def _publish_regular_file(source: Path, target: Path, *, replaceable: bool = False) -> None:
    if target.exists():
        if _same_file_bytes(source, target):
            return
        if replaceable:
            _atomic_copy(source, target, replace_existing=True)
            return
        raise TaskPublicationError(f"existing task artifact differs from accepted scaffold: {target}")
    _atomic_copy(source, target)


def _publish_program_spec(source: Path, target: Path, *, replaceable: bool = False) -> None:
    if not target.exists():
        _atomic_copy(source, target)
        return
    if not _program_spec_is_valid(target.parent):
        _atomic_copy(source, target, replace_existing=True)
        return
    _publish_regular_file(source, target, replaceable=replaceable)


def _publish_bytes(content: bytes, target: Path) -> None:
    if target.exists():
        _preflight_bytes(target, content)
        return
    _atomic_write(target, content)


def _publish_replaceable_bytes(content: bytes, target: Path) -> None:
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise TaskPublicationError(f"unsafe drafted target: {target}")
    if not target.is_file() or _read_regular_bytes(
        target, maximum_bytes=len(content) + 1
    ) != content:
        _atomic_write(target, content, replace_existing=True)


def _publish_resources(
    source: Path,
    target: Path,
    *,
    max_runtime_bytes: int,
    expected_sha256: str | None,
    replaceable: bool = False,
) -> None:
    _validate_tree(source)
    if (source / _RESOURCE_MANIFEST).exists():
        raise TaskPublicationError("drafted resources contain a harness-owned manifest")
    total_bytes = sum(path.stat().st_size for path in _regular_tree_files(source))
    if total_bytes > max_runtime_bytes:
        raise TaskPublicationError(
            f"drafted runtime resources exceed configured limit ({max_runtime_bytes} bytes): {source}"
        )
    if target.exists() and not replaceable:
        _require_matching_managed_resources(source, target)
        return
    parent_fd = _open_absolute_directory(target.parent)
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
        _validate_tree(temporary)
        resource_digest = resource_tree_digest(temporary, _RESOURCE_MANIFEST)
        if resource_digest != expected_sha256:
            raise TaskPublicationError("drafted resources changed during publication")
        manifest = {
            "managed_by": "agentic-scientist-task-drafting",
            "schema_version": 1,
            "sha256": resource_digest,
        }
        _atomic_write(
            temporary / _RESOURCE_MANIFEST,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        _fsync_tree(temporary)
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
            _require_matching_managed_resources(source, target)
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


def _require_matching_managed_resources(source: Path, target: Path) -> None:
    _require_managed_resource_digest(
        target,
        resource_tree_digest(source, _RESOURCE_MANIFEST),
    )


def _require_managed_resource_digest(target: Path, expected_sha256: str | None) -> None:
    _validate_tree(target)
    manifest_path = target / _RESOURCE_MANIFEST
    try:
        manifest = json.loads(
            _read_regular_bytes(
                manifest_path,
                maximum_bytes=_RUNTIME_FILE_MAX_BYTES,
            ).decode("utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskPublicationError(f"invalid managed task resources: {target}") from exc
    target_digest = resource_tree_digest(target, _RESOURCE_MANIFEST)
    if manifest != {
        "managed_by": "agentic-scientist-task-drafting",
        "schema_version": 1,
        "sha256": target_digest,
    } or target_digest != expected_sha256:
        raise TaskPublicationError("drafted resources differ from existing managed task resources")


def _validate_destination(task_dir: Path, report_path: Path) -> None:
    contract = load_task_contract(task_dir / _CONTRACT_PATH)
    report = load_task_alignment_report(report_path, scaffold_dir=task_dir)
    if report.verdict != "pass":
        raise TaskPublicationError("published task alignment report did not pass")
    program_dir = task_dir / "program_0"
    try:
        spec = load_program_spec(program_dir, required=True)
        assert spec is not None
        entrypoint = resolve_entrypoint(program_dir, spec)
    except ProgramSpecError as exc:
        raise TaskPublicationError(f"invalid published program specification: {exc}") from exc
    required = (
        task_dir / "capture.py",
        task_dir / contract.evaluators.search,
        task_dir / contract.evaluators.final,
        task_dir / _CONTRACT_PATH,
        program_dir / "program_spec.toml",
        entrypoint,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise TaskPublicationError("published task is missing required files: " + ", ".join(missing))


def _require_regular_file(path: Path, root: Path) -> None:
    _reject_symlink_components(path, root)
    if path.is_symlink() or not path.is_file():
        raise TaskPublicationError(f"unsafe drafted artifact: {path}")


def _reject_symlink_components(path: Path, root: Path) -> None:
    if path != root and root not in path.parents:
        raise TaskPublicationError(f"task publication path escaped its root: {path}")
    current = path
    while current != root:
        if current.is_symlink():
            raise TaskPublicationError(f"unsafe symlink in task publication path: {path}")
        current = current.parent


def _is_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents

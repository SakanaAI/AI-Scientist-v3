"""Run a task evaluator without placing bulky outputs in a candidate tree."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from .program_spec import load_program_spec, resolve_entrypoint

_OUTPUT_ROOT_NAME = ".evaluation_outputs"
_RUNTIME_ROOT_NAME = ".evaluation_runtime"
_SUMMARY_FILES = ("metrics.json", "correct.json")
_PENDING_RESULT_FILE = ".evaluation_pending"
_STAGED_RESULT_PREFIX = ".evaluation_stage_"
_MAX_SUMMARY_BYTES = 16 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 21_600.0
_PROCESS_TERMINATION_GRACE_SECONDS = 35.0
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37
_EVALUATION_PROCESS_LOCK = threading.Lock()
_SENSITIVE_ENV_TOKENS = (
    "AUTH",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "PROXY",
    "SECRET",
    "TOKEN",
)
_SAFE_ENV_NAMES = {
    "AGENTIC_DATA_DIR",
    "DATA_DIR",
    "FINEWEB_DATA_DIR",
    "GRPO_TEST_DATASET_PATH",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "PATH",
    "TASK_SOURCE_DIR",
    "TERM",
    "TZ",
    "USER",
    "VENV_DIR",
}
_SAFE_ENV_PREFIXES = (
    "CUDA_",
    "MKL_",
    "NCCL_",
    "NUMEXPR_",
    "NVIDIA_",
    "OMP_",
    "ROCR_",
    "TORCH_",
    "XLA_",
)
_PYTHON_ENV_BLOCKLIST = {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"}


class _ForwardedSignal(Exception):
    def __init__(self, signum: int) -> None:
        self.signum = signum


@dataclass(frozen=True)
class EvaluationResult:
    returncode: int
    output_dir: Path
    runtime_dir: Path


def _resolve_directory(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_dir():
        raise NotADirectoryError(f"{label} must be a real directory: {expanded}")
    return expanded.resolve()


def _open_child_directory(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(name, flags, dir_fd=parent_fd)


def _ensure_child_directory(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    return _open_child_directory(parent_fd, name)


def _remove_created_directory(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        raise RuntimeError(f"Created directory identity changed: {name}")
    if not stat.S_ISDIR(current.st_mode):
        raise RuntimeError(f"Created path is no longer a directory: {name}")
    os.rmdir(name, dir_fd=parent_fd)


def _create_attempt_directories(
    program_dir: Path,
    parent_fd: int,
) -> tuple[Path, int, Path, int]:
    output_root_fd = _ensure_child_directory(parent_fd, _OUTPUT_ROOT_NAME)
    try:
        program_output_fd = _ensure_child_directory(output_root_fd, program_dir.name)
    finally:
        os.close(output_root_fd)
    runtime_root_fd = _ensure_child_directory(parent_fd, _RUNTIME_ROOT_NAME)
    try:
        program_runtime_fd = _ensure_child_directory(runtime_root_fd, program_dir.name)
    finally:
        os.close(runtime_root_fd)
    try:
        for _ in range(100):
            attempt_name = f"attempt_{secrets.token_hex(8)}"
            output_fd: int | None = None
            runtime_fd: int | None = None
            output_created = False
            runtime_created = False
            output_stat: os.stat_result | None = None
            runtime_stat: os.stat_result | None = None
            try:
                os.mkdir(attempt_name, mode=0o700, dir_fd=program_output_fd)
                output_created = True
            except FileExistsError:
                continue
            try:
                output_stat = os.stat(
                    attempt_name,
                    dir_fd=program_output_fd,
                    follow_symlinks=False,
                )
                output_fd = _open_child_directory(program_output_fd, attempt_name)
                try:
                    os.mkdir(attempt_name, mode=0o700, dir_fd=program_runtime_fd)
                    runtime_created = True
                except FileExistsError:
                    os.close(output_fd)
                    output_fd = None
                    _remove_created_directory(
                        program_output_fd,
                        attempt_name,
                        output_stat,
                    )
                    continue
                runtime_stat = os.stat(
                    attempt_name,
                    dir_fd=program_runtime_fd,
                    follow_symlinks=False,
                )
                runtime_fd = _open_child_directory(program_runtime_fd, attempt_name)
                output_dir = (
                    program_dir.parent / _OUTPUT_ROOT_NAME / program_dir.name / attempt_name
                )
                runtime_dir = (
                    program_dir.parent / _RUNTIME_ROOT_NAME / program_dir.name / attempt_name
                )
                return output_dir, output_fd, runtime_dir, runtime_fd
            except BaseException as allocation_error:
                if runtime_fd is not None:
                    os.close(runtime_fd)
                if output_fd is not None:
                    os.close(output_fd)
                try:
                    unverified: list[str] = []
                    if runtime_created and runtime_stat is None:
                        unverified.append("runtime")
                    elif runtime_stat is not None:
                        _remove_created_directory(
                            program_runtime_fd,
                            attempt_name,
                            runtime_stat,
                        )
                    if output_created and output_stat is None:
                        unverified.append("output")
                    elif output_stat is not None:
                        _remove_created_directory(
                            program_output_fd,
                            attempt_name,
                            output_stat,
                        )
                    if unverified:
                        labels = ", ".join(unverified)
                        raise RuntimeError(
                            f"Could not verify partial evaluation allocation: {labels}"
                        )
                except BaseException as cleanup_error:
                    raise RuntimeError(
                        "Could not clean up partial evaluation allocation"
                    ) from cleanup_error
                raise allocation_error
        raise RuntimeError("Could not allocate unique evaluation directories")
    finally:
        os.close(program_output_fd)
        os.close(program_runtime_fd)


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags)


def _read_summary(dir_fd: int, filename: str) -> tuple[bytes, dict[str, object]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        fd = os.open(filename, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise RuntimeError(f"Evaluator did not produce a safe summary: {filename}") from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"Evaluator summary is not a regular file: {filename}")
        if file_stat.st_size > _MAX_SUMMARY_BYTES:
            raise RuntimeError(f"Evaluator summary exceeds {_MAX_SUMMARY_BYTES} bytes: {filename}")
        chunks: list[bytes] = []
        remaining = _MAX_SUMMARY_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > _MAX_SUMMARY_BYTES:
            raise RuntimeError(f"Evaluator summary exceeds {_MAX_SUMMARY_BYTES} bytes: {filename}")
    finally:
        os.close(fd)
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Evaluator produced invalid JSON: {filename}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Evaluator summary must contain a JSON object: {filename}")
    return payload, parsed


def _reserve_new_file(dir_fd: int, filename: str) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(filename, flags, 0o600, dir_fd=dir_fd)


def _write_reserved_file(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]
    os.fsync(fd)


def _close_reserved_files(*, dir_fd: int, reserved: dict[str, int], remove: bool) -> None:
    for filename, fd in reserved.items():
        try:
            os.close(fd)
        finally:
            if remove:
                try:
                    os.unlink(filename, dir_fd=dir_fd)
                except FileNotFoundError:
                    pass


def _reserve_result_files(dir_fd: int, filenames: tuple[str, ...]) -> dict[str, int]:
    reserved: dict[str, int] = {}
    try:
        for filename in filenames:
            reserved[filename] = _reserve_new_file(dir_fd, filename)
    except BaseException:
        _close_reserved_files(dir_fd=dir_fd, reserved=reserved, remove=True)
        raise
    return reserved


def _unlink_entry(dir_fd: int, filename: str) -> None:
    try:
        os.unlink(filename, dir_fd=dir_fd)
    except FileNotFoundError:
        pass


def _recover_interrupted_publication(program_fd: int) -> None:
    for filename in os.listdir(program_fd):
        if filename.startswith(_STAGED_RESULT_PREFIX):
            _unlink_entry(program_fd, filename)
    try:
        pending_stat = os.stat(
            _PENDING_RESULT_FILE,
            dir_fd=program_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        os.fsync(program_fd)
        return
    if not stat.S_ISREG(pending_stat.st_mode):
        raise RuntimeError("Unsafe interrupted evaluation marker")
    _, pending = _read_summary(program_fd, _PENDING_RESULT_FILE)
    owned_files = pending.get("owned_files")
    if pending.get("schema_version") != 1 or not isinstance(owned_files, dict):
        raise RuntimeError("Invalid interrupted evaluation marker")
    for filename, identity in owned_files.items():
        if filename not in (*_SUMMARY_FILES, "evaluation_manifest.json"):
            raise RuntimeError("Invalid file in interrupted evaluation marker")
        if not isinstance(identity, dict):
            raise RuntimeError("Invalid identity in interrupted evaluation marker")
        try:
            current = os.stat(filename, dir_fd=program_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if (current.st_dev, current.st_ino) == (
            identity.get("device"),
            identity.get("inode"),
        ):
            _unlink_entry(program_fd, filename)
    _unlink_entry(program_fd, _PENDING_RESULT_FILE)
    os.fsync(program_fd)


def _ensure_result_files_absent(program_fd: int) -> None:
    for filename in (*_SUMMARY_FILES, "evaluation_manifest.json"):
        try:
            os.stat(filename, dir_fd=program_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise FileExistsError(f"Evaluation result already exists: {filename}")


def _program_tree_fingerprint(program_dir: Path) -> dict[str, str]:
    fingerprint: dict[str, str] = {}
    excluded = {
        _PENDING_RESULT_FILE,
        "correct.json",
        "evaluation_manifest.json",
        "metrics.json",
    }
    for path in sorted(program_dir.rglob("*")):
        relative = path.relative_to(program_dir).as_posix()
        root_name = relative.split("/", 1)[0]
        if root_name in excluded or root_name.startswith(_STAGED_RESULT_PREFIX):
            continue
        path_stat = path.lstat()
        digest = hashlib.sha256()
        digest.update(path_stat.st_mode.to_bytes(8, "big"))
        if path.is_symlink():
            digest.update(os.readlink(path).encode("utf-8"))
        elif path.is_file():
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        fingerprint[relative] = digest.hexdigest()
    return fingerprint


def _validated_summaries(output_fd: int) -> dict[str, bytes]:
    metrics_payload, metrics = _read_summary(output_fd, "metrics.json")
    correct_payload, correct = _read_summary(output_fd, "correct.json")
    score = metrics.get("combined_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise RuntimeError("metrics.json combined_score must be numeric")
    if not math.isfinite(float(score)):
        raise RuntimeError("metrics.json combined_score must be finite")
    if not isinstance(correct.get("correct"), bool):
        raise RuntimeError("correct.json correct must be a boolean")
    return {"correct.json": correct_payload, "metrics.json": metrics_payload}


def _publish_result_bundle(
    *,
    output_fd: int,
    program_fd: int,
    program_dir: Path,
    output_dir: Path,
    returncode: int,
) -> None:
    payloads = _validated_summaries(output_fd)
    manifest = {
        "producer": "isolated_evaluator",
        "schema_version": 1,
        "program_id": program_dir.name,
        "output_dir": os.path.relpath(output_dir, program_dir),
        "returncode": returncode,
        "published_summaries": list(_SUMMARY_FILES),
    }
    payloads["evaluation_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    filenames = ("correct.json", "metrics.json", "evaluation_manifest.json")
    for filename in filenames:
        try:
            os.stat(filename, dir_fd=program_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise FileExistsError(f"Evaluation result already exists: {filename}")
    transaction_id = secrets.token_hex(8)
    staged_names = {
        filename: f"{_STAGED_RESULT_PREFIX}{transaction_id}_{filename}"
        for filename in filenames
    }
    staged: dict[str, int] = {}
    published: list[str] = []
    try:
        staged = _reserve_result_files(program_fd, tuple(staged_names.values()))
        for filename, staged_name in staged_names.items():
            _write_reserved_file(staged[staged_name], payloads[filename])
        owned_files = {}
        for filename, staged_name in staged_names.items():
            staged_stat = os.fstat(staged[staged_name])
            owned_files[filename] = {
                "device": staged_stat.st_dev,
                "inode": staged_stat.st_ino,
            }
        pending_fd = _reserve_new_file(program_fd, _PENDING_RESULT_FILE)
        try:
            pending_payload = (
                json.dumps(
                    {
                        "schema_version": 1,
                        "transaction_id": transaction_id,
                        "owned_files": owned_files,
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            _write_reserved_file(pending_fd, pending_payload)
        finally:
            os.close(pending_fd)
        os.fsync(program_fd)
        for filename in filenames:
            staged_name = staged_names[filename]
            os.link(
                staged_name,
                filename,
                src_dir_fd=program_fd,
                dst_dir_fd=program_fd,
                follow_symlinks=False,
            )
            published.append(filename)
            path_stat = os.stat(filename, dir_fd=program_fd, follow_symlinks=False)
            staged_stat = os.fstat(staged[staged_name])
            if (path_stat.st_dev, path_stat.st_ino) != (
                staged_stat.st_dev,
                staged_stat.st_ino,
            ):
                raise RuntimeError(f"Published result identity changed: {filename}")
            os.unlink(staged_name, dir_fd=program_fd)
        _close_reserved_files(dir_fd=program_fd, reserved=staged, remove=False)
        staged = {}
        os.fsync(program_fd)
        os.unlink(_PENDING_RESULT_FILE, dir_fd=program_fd)
        os.fsync(program_fd)
    except BaseException:
        _close_reserved_files(dir_fd=program_fd, reserved=staged, remove=True)
        for staged_name in staged_names.values():
            _unlink_entry(program_fd, staged_name)
        for filename in published:
            _unlink_entry(program_fd, filename)
        _unlink_entry(program_fd, _PENDING_RESULT_FILE)
        os.fsync(program_fd)
        raise


def credential_free_environment() -> dict[str, str]:
    """This process's environment without credential-like variables or Python path overrides."""
    return {
        key: value
        for key, value in os.environ.items()
        if key not in _PYTHON_ENV_BLOCKLIST
        and not any(token in key.upper() for token in _SENSITIVE_ENV_TOKENS)
    }


def _evaluation_environment(runtime_dir: Path, runtime_fd: int) -> dict[str, str]:
    environment = {
        key: value
        for key, value in credential_free_environment().items()
        if key in _SAFE_ENV_NAMES or key.startswith(_SAFE_ENV_PREFIXES)
    }
    home_dir = runtime_dir / "home"
    temp_dir = runtime_dir / "tmp"
    os.mkdir("home", mode=0o700, dir_fd=runtime_fd)
    os.mkdir("tmp", mode=0o700, dir_fd=runtime_fd)
    environment.update(
        HOME=str(home_dir),
        TMPDIR=str(temp_dir),
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
    )
    return environment


def _enable_child_subreaper() -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(previous), 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "Could not inspect evaluator descendant containment")
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "Could not enable evaluator descendant containment")
    return previous.value


def _restore_child_subreaper(previous: int | None) -> None:
    if previous is None:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, previous, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "Could not restore evaluator descendant containment")


def _descendant_pids(root_pid: int) -> set[int]:
    children_by_parent: dict[int, set[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            remainder = (entry / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
            parent_pid = int(remainder.split()[1])
        except (IndexError, OSError, ValueError):
            continue
        children_by_parent.setdefault(parent_pid, set()).add(int(entry.name))
    descendants: set[int] = set()
    pending = list(children_by_parent.get(root_pid, ()))
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(children_by_parent.get(pid, ()))
    return descendants


def _reap_adopted_children(pids: set[int]) -> None:
    for pid in pids:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass


def _terminate_descendants(*, excluded_pids: set[int]) -> None:
    if not sys.platform.startswith("linux"):
        return
    deadline = time.monotonic() + _PROCESS_TERMINATION_GRACE_SECONDS
    while True:
        descendants = _descendant_pids(os.getpid()) - excluded_pids
        _reap_adopted_children(descendants)
        descendants = _descendant_pids(os.getpid()) - excluded_pids
        if not descendants:
            return
        for pid in descendants:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        if time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    cleanup_deadline = time.monotonic() + 1.0
    while time.monotonic() < cleanup_deadline:
        descendants = _descendant_pids(os.getpid()) - excluded_pids
        for pid in descendants:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _reap_adopted_children(descendants)
        if not (_descendant_pids(os.getpid()) - excluded_pids):
            return
        time.sleep(0.02)
    raise RuntimeError("Could not prove evaluator descendant containment")


def _terminate_process_group(
    process: subprocess.Popen[bytes], *, excluded_pids: set[int] | None = None
) -> None:
    """SIGINT the group, wait for it to empty, then SIGKILL it.

    Group members orphaned by the leader are adopted by this subreaper and turn into zombies that
    keep the group alive until reaped, so the wait also reaps adopted descendants (except
    ``excluded_pids``, which other code may still wait on).
    """
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    except PermissionError:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + _PROCESS_TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        if excluded_pids is not None and sys.platform.startswith("linux"):
            _reap_adopted_children(_descendant_pids(os.getpid()) - excluded_pids)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        if process.poll() is None:
            process.kill()
    if process.poll() is None:
        process.wait()


@dataclass(frozen=True)
class ContainedRun:
    """How a contained evaluator ended: its exit code, a timeout, or a forwarded stop signal."""

    returncode: int | None = None
    timed_out: bool = False
    signum: int | None = None


def _run_contained_locked(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    stdout: Any = None,
) -> ContainedRun:
    previous_subreaper = _enable_child_subreaper()
    existing_descendants = _descendant_pids(os.getpid())
    previous_handlers: dict[int, Any] = {}
    forwarded_signum: int | None = None
    process: subprocess.Popen[bytes] | None = None
    handled_signals = (signal.SIGINT, signal.SIGTERM)
    can_install_handlers = threading.current_thread() is threading.main_thread()

    def forward_signal(signum: int, _frame: object) -> None:
        nonlocal forwarded_signum
        if forwarded_signum is not None:
            return
        forwarded_signum = signum
        if process is not None:
            raise _ForwardedSignal(signum)

    def contain() -> None:
        if process is not None:
            _terminate_process_group(process, excluded_pids=existing_descendants)
        _terminate_descendants(excluded_pids=existing_descendants)

    if can_install_handlers:
        for signum in handled_signals:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=subprocess.STDOUT if stdout is not None else None,
            start_new_session=True,
        )
        if forwarded_signum is not None:
            raise _ForwardedSignal(forwarded_signum)
        returncode = process.wait(timeout=timeout_seconds)
        contain()
        return ContainedRun(returncode=returncode)
    except subprocess.TimeoutExpired:
        contain()
        return ContainedRun(timed_out=True)
    except _ForwardedSignal as exc:
        contain()
        return ContainedRun(signum=exc.signum)
    except KeyboardInterrupt:
        contain()
        return ContainedRun(signum=signal.SIGINT)
    except BaseException:
        if process is not None:
            contain()
        raise
    finally:
        for restore_signum, previous_handler in previous_handlers.items():
            signal.signal(restore_signum, previous_handler)
        _restore_child_subreaper(previous_subreaper)


def run_contained(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    stdout: Any = None,
) -> ContainedRun:
    """Run an evaluator in its own session and leave no descendant behind.

    The caller becomes a child subreaper, so descendants that start their own sessions (torchrun
    workers do) are still found and stopped: SIGINT to the group, then every descendant, then
    SIGKILL. SIGINT/SIGTERM received meanwhile end the run as ``signum`` instead of raising.
    ``stdout`` (a binary file) receives stdout and stderr; ``None`` inherits them.
    """
    with _EVALUATION_PROCESS_LOCK:
        return _run_contained_locked(
            command, cwd=cwd, environment=environment, timeout_seconds=timeout_seconds, stdout=stdout
        )


def _run_evaluator(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float,
) -> int:
    run = run_contained(command, cwd=cwd, environment=environment, timeout_seconds=timeout_seconds)
    if run.timed_out:
        return 124
    if run.signum is not None:
        return 128 + run.signum
    assert run.returncode is not None
    return run.returncode


def _verify_pinned_directory(path: Path, dir_fd: int) -> None:
    pinned = os.fstat(dir_fd)
    current = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (
        pinned.st_dev,
        pinned.st_ino,
    ):
        raise RuntimeError(f"Directory identity changed during evaluation: {path}")


def run_evaluation(
    *,
    evaluator: Path,
    program_dir: Path,
    python_executable: str = sys.executable,
    evaluator_args: tuple[str, ...] = (),
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    report_paths: Callable[[Path, Path], None] | None = None,
) -> EvaluationResult:
    """Run one evaluator and publish only canonical summaries to the candidate."""
    resolved_program_dir = _resolve_directory(program_dir, label="program_dir")
    resolved_evaluator = evaluator.expanduser().resolve()
    if evaluator.is_symlink() or not resolved_evaluator.is_file():
        raise FileNotFoundError(f"Evaluator must be a real file: {evaluator}")

    spec = load_program_spec(resolved_program_dir, required=True)
    if spec is None:  # pragma: no cover - required=True raises first
        raise RuntimeError(f"Missing program specification: {resolved_program_dir}")
    entrypoint = resolve_entrypoint(resolved_program_dir, spec)
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise FileNotFoundError(f"Program entrypoint must be a real file: {entrypoint}")

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if (
        not resolved_program_dir.name.startswith("program_")
        or not resolved_program_dir.name[len("program_") :].isdigit()
    ):
        raise ValueError(f"program_dir must be named program_<index>: {resolved_program_dir}")
    reserved_args = {"--program_path", "--program-path", "--results_dir", "--results-dir"}
    if any(arg.split("=", 1)[0] in reserved_args for arg in evaluator_args):
        raise ValueError("evaluator_args must not override program or results paths")

    parent_fd = _open_directory(resolved_program_dir.parent)
    try:
        program_fd = _open_child_directory(parent_fd, resolved_program_dir.name)
        try:
            try:
                fcntl.flock(program_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"Evaluation already in progress: {resolved_program_dir}"
                ) from exc
            _verify_pinned_directory(resolved_program_dir, program_fd)
            _recover_interrupted_publication(program_fd)
            _ensure_result_files_absent(program_fd)
            program_fingerprint = _program_tree_fingerprint(resolved_program_dir)
            output_dir, output_fd, runtime_dir, runtime_fd = _create_attempt_directories(
                resolved_program_dir,
                parent_fd,
            )
            try:
                if report_paths is not None:
                    report_paths(output_dir, runtime_dir)
                environment = _evaluation_environment(runtime_dir, runtime_fd)
                returncode = _run_evaluator(
                    [
                        python_executable,
                        str(resolved_evaluator),
                        "--program_path",
                        str(entrypoint),
                        "--results_dir",
                        str(output_dir),
                        *evaluator_args,
                    ],
                    cwd=resolved_evaluator.parent,
                    environment=environment,
                    timeout_seconds=timeout_seconds,
                )
                _verify_pinned_directory(output_dir, output_fd)
                _verify_pinned_directory(runtime_dir, runtime_fd)
                if returncode == 0:
                    _verify_pinned_directory(resolved_program_dir, program_fd)
                    current_fingerprint = _program_tree_fingerprint(resolved_program_dir)
                    if any(
                        current_fingerprint.get(path) != digest
                        for path, digest in program_fingerprint.items()
                    ):
                        raise RuntimeError("Candidate program changed during evaluation")
                    _publish_result_bundle(
                        output_fd=output_fd,
                        program_dir=resolved_program_dir,
                        program_fd=program_fd,
                        output_dir=output_dir,
                        returncode=returncode,
                    )
            finally:
                os.close(output_fd)
                os.close(runtime_fd)
        finally:
            os.close(program_fd)
    finally:
        os.close(parent_fd)

    return EvaluationResult(
        returncode=returncode,
        output_dir=output_dir,
        runtime_dir=runtime_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--program-dir", required=True, type=Path)
    parser.add_argument("--num-replications", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=_DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    try:
        if args.num_replications <= 0:
            raise ValueError("--num-replications must be positive")
        evaluator_args = (
            ("--num_replications", str(args.num_replications)) if args.num_replications > 1 else ()
        )
        def report_paths(output_dir: Path, runtime_dir: Path) -> None:
            print(f"evaluation outputs: {output_dir}", flush=True)
            print(f"evaluation runtime: {runtime_dir}", flush=True)

        result = run_evaluation(
            evaluator=args.evaluator,
            program_dir=args.program_dir,
            evaluator_args=evaluator_args,
            timeout_seconds=args.timeout_seconds,
            report_paths=report_paths,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"evaluation wrapper error: {exc}", file=sys.stderr)
        return 1
    return result.returncode if 0 <= result.returncode <= 255 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

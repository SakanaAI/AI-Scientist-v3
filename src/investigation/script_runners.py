"""Script invocation helpers for idea, run, paper, and task steps."""

from __future__ import annotations

import io
import json
import os
import selectors
import secrets
import stat
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Callable, Optional, Sequence, Union, cast

from .constants import (
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_CONTAINER_NETWORK,
    DEFAULT_HARNESS_EVALUATION_TIMEOUT_SECONDS,
    DEFAULT_MAX_PROGRAM_EVALS_PER_ATTEMPT,
    DEFAULT_PAPER_MAX_REFS,
    DEFAULT_PAPER_MAX_REPAIR_ROUNDS,
    DEFAULT_PAPER_MIN_REFS,
    DEFAULT_PAPER_TIMEOUT_GRACE_SECONDS,
    DEFAULT_RUN_TIMEOUT_GRACE_SECONDS,
)
from .evaluation_artifact_cleanup import (
    ExperimentDirectoryGuard,
    cleanup_unmanaged_evaluation_outputs,
)
from .temp_storage import temporary_directory
from .files import parse_paper_dir
from .task_failure import (
    AGENT_LOG_FAILURE, INFRASTRUCTURE_EXIT_CODE, TASK_CONTRACT_FAILURE, TASK_FAILURE_FD_ENV,
    failure_details, read_task_failure,
)
from .models import (
    IdeaGenerationResult,
    PaperGenerationResult,
    RunAttemptResult,
    TaskDraftResult,
)
from src.orchestrator.paper_event_bridge import extract_transport_events
from src.task_drafting.progress import (
    TASK_PROGRESS_COST_FIELDS,
    extract_transport_events as extract_task_progress_events,
)

PopenFactory = Callable[..., subprocess.Popen[str]]
RegisterFn = Callable[[subprocess.Popen[str]], None]
UnregisterFn = Callable[[subprocess.Popen[str]], None]
TerminateFn = Callable[..., None]
ContainerEnvFn = Callable[..., dict[str, str]]
EnsureTextFn = Callable[[Optional[Union[str, bytes]]], str]
OutputLineFn = Callable[[str], bool]

SCRIPT_RESULT_TAIL_BYTES = 64 * 1024
SCRIPT_OUTPUT_LOG_MAX_BYTES = 16 * 1024 * 1024
_OUTPUT_READ_CHUNK_BYTES = 64 * 1024
_OUTPUT_DRAIN_BATCH_BYTES = 1024 * 1024


def _read_output_tail(output_file: BinaryIO) -> str:
    """Read at most the configured result tail from an open output file."""
    output_file.flush()
    output_file.seek(0, 2)
    output_size = output_file.tell()
    output_file.seek(max(0, output_size - SCRIPT_RESULT_TAIL_BYTES))
    return output_file.read(SCRIPT_RESULT_TAIL_BYTES).decode("utf-8", errors="replace")


class _BoundedOutputCapture:
    """Continuously drain output while bounding memory and temporary disk use."""

    def __init__(
        self,
        output_file: BinaryIO,
        *,
        output_line_fn: OutputLineFn | None = None,
    ) -> None:
        self.output_file = output_file
        self.output_line_fn = output_line_fn
        self.pending_line = bytearray()
        marker_text = (
            "\n[wrapper] output truncated after exceeding "
            f"{SCRIPT_OUTPUT_LOG_MAX_BYTES} bytes; retaining tail.\n"
        )
        self.marker = marker_text.encode("utf-8")
        self.tail_limit = max(
            0,
            min(
                SCRIPT_RESULT_TAIL_BYTES - len(self.marker),
                SCRIPT_OUTPUT_LOG_MAX_BYTES - len(self.marker),
            ),
        )
        self.prefix_limit = max(
            0,
            SCRIPT_OUTPUT_LOG_MAX_BYTES - len(self.marker) - self.tail_limit,
        )
        self.prefix_bytes = 0
        self.overflow_bytes = 0
        self.tail = bytearray()
        self.finalized = False

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        if self.output_line_fn is None:
            self._write_bytes(chunk)
            return
        self.pending_line.extend(chunk)
        while True:
            newline = self.pending_line.find(b"\n")
            if newline < 0:
                break
            line = bytes(self.pending_line[: newline + 1])
            del self.pending_line[: newline + 1]
            self._write_line(line)
        if len(self.pending_line) > _OUTPUT_READ_CHUNK_BYTES:
            self._write_bytes(bytes(self.pending_line))
            self.pending_line.clear()

    def _write_line(self, line: bytes) -> None:
        try:
            consumed = self.output_line_fn is not None and self.output_line_fn(
                line.decode("utf-8", errors="replace")
            )
        except Exception:
            consumed = False
        if not consumed:
            self._write_bytes(line)

    def _write_bytes(self, chunk: bytes) -> None:
        prefix_remaining = self.prefix_limit - self.prefix_bytes
        if prefix_remaining > 0:
            prefix = chunk[:prefix_remaining]
            self.output_file.write(prefix)
            self.prefix_bytes += len(prefix)
            chunk = chunk[len(prefix) :]
        if not chunk:
            return
        self.overflow_bytes += len(chunk)
        if self.tail_limit == 0:
            return
        self.tail.extend(chunk)
        if len(self.tail) > self.tail_limit:
            del self.tail[: len(self.tail) - self.tail_limit]

    def finalize(self) -> None:
        if self.finalized:
            return
        if self.pending_line:
            self._write_line(bytes(self.pending_line))
            self.pending_line.clear()
        if self.overflow_bytes > self.tail_limit:
            self.output_file.write(self.marker)
        self.output_file.write(self.tail)
        self.output_file.flush()
        self.finalized = True


def _open_directory_no_follow(path: Path, *, create: bool) -> int:
    """Open a direct child directory without following its final path component."""
    base_path = path.parent
    if create:
        base_path.mkdir(parents=True, exist_ok=True)
    base_fd = os.open(
        base_path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if create:
            try:
                os.mkdir(path.name, mode=0o755, dir_fd=base_fd)
            except FileExistsError:
                pass
        return os.open(
            path.name,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=base_fd,
        )
    finally:
        os.close(base_fd)


def _publish_output_log(output_file: BinaryIO, output_log_path: Path) -> None:
    """Atomically publish output through a no-follow parent directory descriptor."""
    directory_fd = _open_directory_no_follow(output_log_path.parent, create=True)
    temporary_name = f".wrapper-output-publish-{uuid.uuid4().hex}"
    temporary_fd: int | None = None
    try:
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(temporary_fd, "wb", closefd=False) as published_file:
            output_file.flush()
            output_file.seek(0)
            remaining_bytes = SCRIPT_OUTPUT_LOG_MAX_BYTES
            while remaining_bytes > 0:
                chunk = output_file.read(min(_OUTPUT_READ_CHUNK_BYTES, remaining_bytes + 1))
                if not chunk:
                    break
                if len(chunk) > remaining_bytes:
                    raise OSError("wrapper output exceeds publication quota")
                published_file.write(chunk)
                remaining_bytes -= len(chunk)
            if remaining_bytes == 0 and output_file.read(1):
                raise OSError("wrapper output exceeds publication quota")
        temporary_stat = os.fstat(temporary_fd)
        if not stat.S_ISREG(temporary_stat.st_mode):
            raise OSError("wrapper output publication target is not a regular file")
        os.close(temporary_fd)
        temporary_fd = None
        os.replace(
            temporary_name,
            output_log_path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        finally:
            os.close(directory_fd)


def copy_output_log_no_follow(source_path: Path, destination_path: Path) -> None:
    """Copy a regular output log without following its final path component."""
    source_directory_fd = _open_directory_no_follow(source_path.parent, create=False)
    source_fd: int | None = None
    try:
        source_fd = os.open(
            source_path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=source_directory_fd,
        )
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise OSError("wrapper output source is not a regular file")
        with os.fdopen(source_fd, "rb", closefd=False) as source_file:
            _publish_output_log(source_file, destination_path)
    finally:
        if source_fd is not None:
            os.close(source_fd)
        os.close(source_directory_fd)


def write_output_log_no_follow(output: str, destination_path: Path) -> None:
    """Publish wrapper-generated text without following the destination parent."""
    encoded_output = output.encode("utf-8")
    if len(encoded_output) > SCRIPT_OUTPUT_LOG_MAX_BYTES:
        raise OSError("wrapper output exceeds publication quota")
    with tempfile.TemporaryFile(mode="w+b", dir=temporary_directory()) as output_file:
        output_file.write(encoded_output)
        _publish_output_log(output_file, destination_path)


def _read_available_output(file_descriptor: int, capture: _BoundedOutputCapture) -> bool:
    """Drain currently available bytes; return whether the pipe reached EOF."""
    drained_bytes = 0
    while drained_bytes < _OUTPUT_DRAIN_BATCH_BYTES:
        try:
            chunk = os.read(file_descriptor, _OUTPUT_READ_CHUNK_BYTES)
        except BlockingIOError:
            return False
        if not chunk:
            return True
        capture.write(chunk)
        drained_bytes += len(chunk)
    return False


def _drain_process_output(
    proc: subprocess.Popen[str],
    stream: BinaryIO,
    capture: _BoundedOutputCapture,
    *,
    deadline: float,
    timeout_seconds: int,
) -> None:
    """Drain a process pipe until process exit or the command deadline."""
    file_descriptor = stream.fileno()
    os.set_blocking(file_descriptor, False)
    pipe_eof = False
    with selectors.DefaultSelector() as selector:
        selector.register(file_descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout_seconds)
            events = selector.select(timeout=min(0.1, remaining))
            if events and not pipe_eof:
                pipe_eof = _read_available_output(file_descriptor, capture)
                if pipe_eof:
                    selector.unregister(file_descriptor)
            if proc.poll() is not None:
                if not pipe_eof:
                    _read_available_output(file_descriptor, capture)
                return


def _execute_script_command(
    *,
    cmd: list[str],
    timeout_seconds: int,
    timeout_grace_seconds: int,
    script_label: str,
    missing_output: str,
    popen_factory: PopenFactory,
    register_subprocess: RegisterFn,
    unregister_subprocess: UnregisterFn,
    terminate_process_group: TerminateFn,
    ensure_text: EnsureTextFn,
    env: dict[str, str],
    output_log_path: Path | None = None,
    output_line_fn: OutputLineFn | None = None,
    pass_fds: tuple[int, ...] = (),
) -> tuple[str, int]:
    """Handle execute script command logic for this module."""
    started_at = time.monotonic()
    proc: subprocess.Popen[str] | None = None
    log_file: BinaryIO | None = None
    capture: _BoundedOutputCapture | None = None
    process_stream: BinaryIO | None = None
    if output_log_path is not None:
        output_log_path.parent.parent.mkdir(parents=True, exist_ok=True)
        log_file = tempfile.TemporaryFile(
            mode="w+b",
            prefix=f".{output_log_path.parent.name}-wrapper-output-",
            dir=temporary_directory(),
        )
    else:
        log_file = tempfile.TemporaryFile(mode="w+b", dir=temporary_directory())
    capture = _BoundedOutputCapture(log_file, output_line_fn=output_line_fn)

    def append_to_capture(text: str) -> None:
        if capture is None:
            return
        capture.write(text.encode("utf-8"))

    def output_tail() -> str:
        if log_file is None or capture is None:
            return ""
        capture.finalize()
        return _read_output_tail(log_file)

    def close_process_stream() -> None:
        nonlocal process_stream
        if process_stream is None:
            return
        try:
            process_stream.close()
        finally:
            process_stream = None

    try:
        proc = popen_factory(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if capture is not None else subprocess.PIPE,
            text=capture is None,
            env=env,
            start_new_session=True,
            **({"pass_fds": pass_fds} if pass_fds else {}),
        )
        register_subprocess(proc)
        if capture is not None and proc.stdout is not None:
            candidate_stream = proc.stdout
            try:
                candidate_file_descriptor = candidate_stream.fileno()
            except (AttributeError, io.UnsupportedOperation):
                pass
            else:
                if isinstance(candidate_file_descriptor, int):
                    process_stream = cast(BinaryIO, candidate_stream)
        if capture is not None and process_stream is not None:
            _drain_process_output(
                proc,
                process_stream,
                capture,
                deadline=started_at + timeout_seconds,
                timeout_seconds=timeout_seconds,
            )
            close_process_stream()
            output = output_tail()
        else:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            if capture is None:
                output = ensure_text(stdout) + ensure_text(stderr)
            else:
                append_to_capture(ensure_text(stdout) + ensure_text(stderr))
                output = output_tail()
        exit_code = proc.returncode if proc.returncode is not None else 1
        return output, exit_code
    except FileNotFoundError:
        append_to_capture(missing_output)
        if capture is not None:
            return output_tail(), 127
        return missing_output, 127
    except KeyboardInterrupt:
        if proc is not None:
            terminate_process_group(proc, grace_seconds=timeout_grace_seconds)
        close_process_stream()
        raise
    except subprocess.TimeoutExpired:
        elapsed_seconds = time.monotonic() - started_at
        stdout = ""
        stderr = ""
        if proc is not None:
            terminate_process_group(proc, grace_seconds=timeout_grace_seconds)
            if process_stream is not None and capture is not None:
                try:
                    _read_available_output(process_stream.fileno(), capture)
                except OSError:
                    pass
                close_process_stream()
            else:
                stdout, stderr = proc.communicate()
        timeout_text = (
            f"\n[wrapper] {script_label} timed out after {timeout_seconds} seconds; "
            f"grace={timeout_grace_seconds} seconds; elapsed={elapsed_seconds:.1f}s.\n"
        )
        if capture is not None:
            append_to_capture(ensure_text(stdout) + ensure_text(stderr))
            append_to_capture(timeout_text)
            return output_tail(), 124
        output = ensure_text(stdout) + ensure_text(stderr) + timeout_text
        return output, 124
    finally:
        close_process_stream()
        if proc is not None:
            unregister_subprocess(proc)
        if log_file is not None:
            try:
                if output_log_path is not None:
                    if capture is not None:
                        capture.finalize()
                    _publish_output_log(log_file, output_log_path)
            finally:
                log_file.close()


def execute_managed_command(
    *,
    cmd: list[str],
    timeout_seconds: int,
    timeout_grace_seconds: int,
    script_label: str,
    output_log_path: Path,
    popen_factory: PopenFactory,
    register_subprocess: RegisterFn,
    unregister_subprocess: UnregisterFn,
    terminate_process_group: TerminateFn,
    ensure_text: EnsureTextFn,
    env: dict[str, str],
) -> tuple[str, int]:
    """Execute an auxiliary wrapper command with bounded, registered process handling."""
    return _execute_script_command(
        cmd=cmd,
        timeout_seconds=timeout_seconds,
        timeout_grace_seconds=timeout_grace_seconds,
        script_label=script_label,
        missing_output=f"[wrapper] {script_label} could not start\n",
        popen_factory=popen_factory,
        register_subprocess=register_subprocess,
        unregister_subprocess=unregister_subprocess,
        terminate_process_group=terminate_process_group,
        ensure_text=ensure_text,
        env=env,
        output_log_path=output_log_path,
    )


def run_idea_generation(
    *,
    agent: str,
    agent_model: str | None = None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path | None,
    conclusions_dir: Path | None = None,
    worker_index: int = 0,
    cycle_index: int,
    output_dir: Path,
    instruct_path: Path,
    timeout_seconds: int,
    timeout_grace_seconds: int = DEFAULT_RUN_TIMEOUT_GRACE_SECONDS,
    harness_evaluated_attempts: bool = False,
    containerize: bool = False,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_network: str = DEFAULT_CONTAINER_NETWORK,
    container_auth_root: Path | None = None,
    container_env_file: Path | None = None,
    container_session_name: str | None = None,
    container_mount_root: Path | None = None,
    validation_feedback: str = "",
    script_path: Path,
    popen_factory: PopenFactory,
    register_subprocess: RegisterFn,
    unregister_subprocess: UnregisterFn,
    terminate_process_group: TerminateFn,
    container_execution_env: ContainerEnvFn,
    ensure_text: EnsureTextFn,
) -> IdeaGenerationResult:
    """Run idea generation."""
    from .idea_retry_feedback import sanitize_validation_feedback

    memory_dir = global_memory_dir if global_memory_dir is not None else results_dir
    conclusions_dir_resolved = conclusions_dir if conclusions_dir is not None else memory_dir
    model = agent_model.strip() if isinstance(agent_model, str) else ""
    profile = agent_profile.strip() if isinstance(agent_profile, str) else ""
    cmd = [
        "bash",
        str(script_path),
        "--agent",
        agent,
        "--task-dir",
        str(task_dir),
        "--results-dir",
        str(results_dir),
        "--global-memory-dir",
        str(memory_dir),
        "--conclusions-dir",
        str(conclusions_dir_resolved),
        "--worker-index",
        str(worker_index),
        "--cycle-index",
        str(cycle_index),
        "--output-dir",
        str(output_dir),
        "--instruct",
        str(instruct_path),
    ]
    if harness_evaluated_attempts:
        cmd.append("--harness-evaluated-attempt")
    if validation_feedback:
        cmd.extend(["--validation-feedback", sanitize_validation_feedback(validation_feedback)])
    if model:
        cmd.extend(["--agent-model", model])
    if profile:
        cmd.extend(["--agent-profile", profile])
    if reasoning_effort:
        cmd.extend(["--reasoning-effort", reasoning_effort])
    if containerize:
        cmd.append("--containerize")
        cmd.extend(["--container-image", container_image])
        cmd.extend(["--container-network", container_network])
        auth_root = container_auth_root if container_auth_root is not None else Path.home()
        cmd.extend(["--container-auth-root", str(auth_root)])
        if container_env_file is not None:
            cmd.extend(["--container-env-file", str(container_env_file)])

    env = container_execution_env(
        containerize=containerize,
        container_session_name=container_session_name,
        container_mount_root=container_mount_root,
    )
    identity = f"idea_generation_cycle_{cycle_index}_worker_{worker_index}"
    with tempfile.TemporaryFile(mode="w+b", dir=temporary_directory()) as control:
        env[TASK_FAILURE_FD_ENV] = str(control.fileno())
        try:
            output, exit_code = _execute_script_command(
                cmd=cmd,
                timeout_seconds=timeout_seconds,
                timeout_grace_seconds=timeout_grace_seconds,
                script_label="src/scripts/idea.sh",
                missing_output="[wrapper] src/scripts/idea.sh not found\n",
                popen_factory=popen_factory,
                register_subprocess=register_subprocess,
                unregister_subprocess=unregister_subprocess,
                terminate_process_group=terminate_process_group,
                ensure_text=ensure_text,
                env=env,
                output_log_path=output_dir / "wrapper_output.log",
                pass_fds=(control.fileno(),),
            )
        except Exception:
            if read_task_failure(control.fileno(), identity, "idea_generation") != AGENT_LOG_FAILURE:
                raise
            output = "[wrapper] idea logger infrastructure failure; secondary host error\n"
            exit_code = INFRASTRUCTURE_EXIT_CODE
        failure = read_task_failure(control.fileno(), identity, "idea_generation")
    return IdeaGenerationResult(
        output=output,
        exit_code=exit_code,
        idea_dir=output_dir,
        infrastructure_failure_category=AGENT_LOG_FAILURE if failure == AGENT_LOG_FAILURE else None,
    )


def _terminal_attempt_result(output: str, exit_code: int, category: str) -> RunAttemptResult:
    return RunAttemptResult(
        output=output, exit_code=exit_code, experiment_dir=None,
        task_failure_category=category if category == TASK_CONTRACT_FAILURE else None,
        infrastructure_failure_category=category if category == AGENT_LOG_FAILURE else None,
    )


def run_single_attempt(
    *,
    agent: str,
    agent_model: str | None = None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path | None,
    conclusions_dir: Path | None = None,
    worker_index: int = 0,
    seed_program_dir: Path,
    idea_file: Path,
    rubrics_file: Path,
    attempt_id: str,
    idea_dir_name: str,
    attempt_dir_name: str | None = None,
    instruct_path: Path | None,
    timeout_seconds: int,
    timeout_grace_seconds: int = DEFAULT_RUN_TIMEOUT_GRACE_SECONDS,
    max_program_evals_per_attempt: int = DEFAULT_MAX_PROGRAM_EVALS_PER_ATTEMPT,
    harness_evaluated_attempts: bool = False,
    harness_evaluation_timeout_seconds: int = DEFAULT_HARNESS_EVALUATION_TIMEOUT_SECONDS,
    gpu_evaluation: bool = False,
    containerize: bool = False,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_network: str = DEFAULT_CONTAINER_NETWORK,
    container_auth_root: Path | None = None,
    container_env_file: Path | None = None,
    container_session_name: str | None = None,
    container_mount_root: Path | None = None,
    script_path: Path,
    popen_factory: PopenFactory,
    register_subprocess: RegisterFn,
    unregister_subprocess: UnregisterFn,
    terminate_process_group: TerminateFn,
    container_execution_env: ContainerEnvFn,
    ensure_text: EnsureTextFn,
) -> RunAttemptResult:
    """Run single attempt."""
    if type(harness_evaluation_timeout_seconds) is not int or harness_evaluation_timeout_seconds < 1:
        raise ValueError("harness_evaluation_timeout_seconds must be a positive integer")
    memory_dir = global_memory_dir if global_memory_dir is not None else results_dir
    conclusions_dir_resolved = conclusions_dir if conclusions_dir is not None else memory_dir
    model = agent_model.strip() if isinstance(agent_model, str) else ""
    profile = agent_profile.strip() if isinstance(agent_profile, str) else ""
    cmd = [
        "bash",
        str(script_path),
        "--agent",
        agent,
        "--task-dir",
        str(task_dir),
        "--results-dir",
        str(results_dir),
        "--global-memory-dir",
        str(memory_dir),
        "--conclusions-dir",
        str(conclusions_dir_resolved),
        "--worker-index",
        str(worker_index),
        "--seed-program",
        str(seed_program_dir),
        "--idea-file",
        str(idea_file),
        "--rubrics-file",
        str(rubrics_file),
        "--attempt-id",
        attempt_id,
        "--idea-dir-name",
        idea_dir_name,
        "--max-program-evals-per-attempt",
        str(max_program_evals_per_attempt),
    ]
    if harness_evaluated_attempts:
        cmd.extend(["--harness-evaluated-attempt", "--agent-timeout-seconds", str(timeout_seconds)])
        cmd.extend(["--harness-evaluation-timeout-seconds", str(harness_evaluation_timeout_seconds)])
    if gpu_evaluation:
        cmd.append("--gpu-evaluation")
    if isinstance(attempt_dir_name, str) and attempt_dir_name.strip():
        cmd.extend(["--attempt-dir-name", attempt_dir_name.strip()])
    if model:
        cmd.extend(["--agent-model", model])
    if profile:
        cmd.extend(["--agent-profile", profile])
    if reasoning_effort:
        cmd.extend(["--reasoning-effort", reasoning_effort])
    if instruct_path is not None:
        cmd.extend(["--instruct", str(instruct_path)])
    if containerize:
        cmd.append("--containerize")
        cmd.extend(["--container-image", container_image])
        cmd.extend(["--container-network", container_network])
        auth_root = container_auth_root if container_auth_root is not None else Path.home()
        cmd.extend(["--container-auth-root", str(auth_root)])
        if container_env_file is not None:
            cmd.extend(["--container-env-file", str(container_env_file)])

    env = container_execution_env(
        containerize=containerize,
        container_session_name=container_session_name,
        container_mount_root=container_mount_root,
    )
    directory_guard = ExperimentDirectoryGuard.prepare(results_dir, idea_dir_name)
    experiment_fd: int | None = None
    control = None
    expected_idea_id = "idea_unknown"
    try:
        idea_payload = json.loads(idea_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    else:
        if isinstance(idea_payload, dict) and isinstance(idea_payload.get("idea_id"), str):
            expected_idea_id = idea_payload["idea_id"].strip() or expected_idea_id
    env.pop(TASK_FAILURE_FD_ENV, None)
    task_failure_category = None
    try:
        control = tempfile.TemporaryFile(mode="w+b", dir=temporary_directory())
        env[TASK_FAILURE_FD_ENV] = str(control.fileno())
        output, exit_code = _execute_script_command(
            cmd=cmd,
            timeout_seconds=(
                timeout_seconds + harness_evaluation_timeout_seconds
                if harness_evaluated_attempts
                else timeout_seconds
            ),
            timeout_grace_seconds=timeout_grace_seconds,
            script_label="src/scripts/run.sh",
            missing_output="[wrapper] src/scripts/run.sh not found\n",
            popen_factory=popen_factory,
            register_subprocess=register_subprocess,
            unregister_subprocess=unregister_subprocess,
            terminate_process_group=terminate_process_group,
            ensure_text=ensure_text,
            env=env,
            pass_fds=(control.fileno(),) if control is not None else (),
        )
        if control is not None:
            task_failure_category = read_task_failure(control.fileno(), attempt_id, expected_idea_id)
        if task_failure_category is not None:
            return _terminal_attempt_result(output, exit_code, task_failure_category)
        experiment_fd = directory_guard.open_experiment()
        experiment_dir = directory_guard.experiment_dir if experiment_fd is not None else None
        if experiment_dir is not None:
            try:
                cleanup_report = cleanup_unmanaged_evaluation_outputs(
                    experiment_dir,
                    experiment_fd=experiment_fd,
                )
            except OSError as exc:
                output += (
                    "\n[wrapper] unmanaged_evaluation_outputs_cleanup_failed "
                    f"error={type(exc).__name__}\n"
                )
            else:
                if cleanup_report.removed_roots:
                    output += (
                        "\n[wrapper] unmanaged_evaluation_outputs_removed "
                        f"roots={len(cleanup_report.removed_roots)} "
                        f"bytes={cleanup_report.removed_bytes}\n"
                    )
                if cleanup_report.failures:
                    output += (
                        "\n[wrapper] unmanaged_evaluation_outputs_cleanup_partial "
                        f"failures={len(cleanup_report.failures)}\n"
                    )
    except Exception:
        if control is not None:
            task_failure_category = read_task_failure(control.fileno(), attempt_id, expected_idea_id)
            if task_failure_category is not None:
                return _terminal_attempt_result(
                    "[wrapper] authenticated host failure; secondary host error\n",
                    failure_details(task_failure_category)[0], task_failure_category,
                )
        raise
    finally:
        try:
            if control is not None:
                control.close()
        finally:
            try:
                if experiment_fd is not None:
                    os.close(experiment_fd)
            finally:
                directory_guard.close()
    return RunAttemptResult(
        output=output,
        exit_code=exit_code,
        experiment_dir=experiment_dir,
        task_failure_category=task_failure_category,
    )


def run_paper_generation(
    *,
    agent: str,
    agent_model: str | None = None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    task_dir: Path,
    results_dir: Path,
    global_memory_dir: Path | None,
    output_dir: Path,
    instruct_path: Path,
    timeout_seconds: int,
    contributions_path: Path | None = None,
    timeout_grace_seconds: int = DEFAULT_PAPER_TIMEOUT_GRACE_SECONDS,
    paper_min_refs: int = DEFAULT_PAPER_MIN_REFS,
    paper_max_refs: int = DEFAULT_PAPER_MAX_REFS,
    paper_profile: str = "default",
    paper_skills: Sequence[str] | None = None,
    two_stage: bool = False,
    paper_review_rounds: int = 0,
    paper_max_repair_rounds: int = DEFAULT_PAPER_MAX_REPAIR_ROUNDS,
    containerize: bool = False,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_network: str = DEFAULT_CONTAINER_NETWORK,
    container_auth_root: Path | None = None,
    container_env_file: Path | None = None,
    container_session_name: str | None = None,
    container_mount_root: Path | None = None,
    script_path: Path,
    popen_factory: PopenFactory,
    register_subprocess: RegisterFn,
    unregister_subprocess: UnregisterFn,
    terminate_process_group: TerminateFn,
    container_execution_env: ContainerEnvFn,
    ensure_text: EnsureTextFn,
    paper_event_fn: Callable[..., None] | None = None,
) -> PaperGenerationResult:
    """Run paper generation."""
    memory_dir = global_memory_dir if global_memory_dir is not None else results_dir
    model = agent_model.strip() if isinstance(agent_model, str) else ""
    profile = agent_profile.strip() if isinstance(agent_profile, str) else ""
    cmd = [
        "bash",
        str(script_path),
        "--agent",
        agent,
        "--task-dir",
        str(task_dir),
        "--results-dir",
        str(results_dir),
        "--global-memory-dir",
        str(memory_dir),
        "--output-dir",
        str(output_dir),
        "--instruct",
        str(instruct_path),
        "--paper-min-refs",
        str(paper_min_refs),
        "--paper-max-refs",
        str(paper_max_refs),
        "--paper-profile",
        paper_profile,
        "--paper-review-rounds",
        str(paper_review_rounds),
        "--paper-max-repair-rounds",
        str(paper_max_repair_rounds),
        "--event-transport",
        "wrapper",
        "--heartbeat-seconds",
        "0",
    ]
    if model:
        cmd.extend(["--agent-model", model])
    if profile:
        cmd.extend(["--agent-profile", profile])
    if reasoning_effort:
        cmd.extend(["--reasoning-effort", reasoning_effort])
    if contributions_path is not None:
        cmd.extend(["--contributions-file", str(contributions_path)])
    if paper_skills:
        selected_skills = [skill.strip() for skill in paper_skills if skill.strip()]
        if selected_skills:
            cmd.extend(["--skills", ",".join(selected_skills)])
    if two_stage:
        cmd.append("--two-stage")
    if containerize:
        cmd.append("--containerize")
        cmd.extend(["--container-image", container_image])
        cmd.extend(["--container-network", container_network])
        auth_root = container_auth_root if container_auth_root is not None else Path.home()
        cmd.extend(["--container-auth-root", str(auth_root)])
        if container_env_file is not None:
            cmd.extend(["--container-env-file", str(container_env_file)])

    env = container_execution_env(
        containerize=containerize,
        container_session_name=container_session_name,
        container_mount_root=container_mount_root,
    )
    env["PAPER_WORKFLOW_TIMEOUT_SECONDS"] = str(timeout_seconds)

    def consume_paper_event_line(line: str) -> bool:
        retained, records = extract_transport_events(line)
        if retained:
            return False
        if paper_event_fn is not None:
            for record in records:
                paper_event_fn(
                    "paper_gate",
                    gate=record["gate"],
                    state=record["state"],
                    paper_session_id=record["session_id"],
                    paper_timestamp=record["timestamp"],
                    paper_fields=record["fields"],
                )
        return True

    output, exit_code = _execute_script_command(
        cmd=cmd,
        timeout_seconds=timeout_seconds,
        timeout_grace_seconds=timeout_grace_seconds,
        script_label="src/scripts/paper.sh",
        missing_output="[wrapper] src/scripts/paper.sh not found\n",
        popen_factory=popen_factory,
        register_subprocess=register_subprocess,
        unregister_subprocess=unregister_subprocess,
        terminate_process_group=terminate_process_group,
        ensure_text=ensure_text,
        env=env,
        output_line_fn=consume_paper_event_line,
    )
    paper_dir = parse_paper_dir(output)
    return PaperGenerationResult(
        output=output,
        exit_code=exit_code,
        paper_dir=paper_dir,
    )


def run_task_drafting(
    *,
    agent: str,
    agent_model: str | None = None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
    task_dir: Path,
    output_dir: Path,
    instruct_path: Path,
    timeout_seconds: int,
    timeout_grace_seconds: int = DEFAULT_RUN_TIMEOUT_GRACE_SECONDS,
    workflow: str = "robust",
    design_max_repairs: int = 8,
    contract_max_repairs: int = 3,
    smoke_max_repairs: int = 3,
    smoke_max_infrastructure_attempts: int = 3,
    alignment_max_repairs: int = 10,
    qualification_max_repairs: int = 4,
    agent_transient_max_attempts: int = 4,
    qualification_max_infrastructure_attempts: int = 3,
    production_entry_timeout_seconds: int = 300,
    max_gate_checks_per_repair: int = 12,
    draft_max_attempts: int = 3,
    evaluator_smoke_timeout_seconds: int = 1800,
    resume_from: str = "",
    max_runtime_bytes: int = 16 * 1024 * 1024 * 1024,
    max_idea_cycles: int = 20,
    max_program_evals_per_attempt: int = DEFAULT_MAX_PROGRAM_EVALS_PER_ATTEMPT,
    containerize: bool = False,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_network: str = DEFAULT_CONTAINER_NETWORK,
    container_auth_root: Path | None = None,
    container_env_file: Path | None = None,
    container_session_name: str | None = None,
    container_mount_root: Path | None = None,
    script_path: Path,
    popen_factory: PopenFactory,
    register_subprocess: RegisterFn,
    unregister_subprocess: UnregisterFn,
    terminate_process_group: TerminateFn,
    container_execution_env: ContainerEnvFn,
    ensure_text: EnsureTextFn,
    task_event_fn: Callable[..., None] | None = None,
) -> TaskDraftResult:
    """Run task drafting."""
    model = agent_model.strip() if isinstance(agent_model, str) else ""
    profile = agent_profile.strip() if isinstance(agent_profile, str) else ""
    cmd = [
        "bash",
        str(script_path),
        "--agent",
        agent,
        "--task-dir",
        str(task_dir),
        "--output-dir",
        str(output_dir),
        "--instruct",
        str(instruct_path),
    ]
    if model:
        cmd.extend(["--agent-model", model])
    if profile:
        cmd.extend(["--agent-profile", profile])
    if reasoning_effort:
        cmd.extend(["--reasoning-effort", reasoning_effort])
    cmd.extend(
        [
            "--workflow",
            workflow,
            "--design-max-repairs",
            str(design_max_repairs),
            "--contract-max-repairs",
            str(contract_max_repairs),
            "--smoke-max-repairs",
            str(smoke_max_repairs),
            "--smoke-max-infrastructure-attempts",
            str(smoke_max_infrastructure_attempts),
            "--alignment-max-repairs",
            str(alignment_max_repairs),
            "--qualification-max-repairs",
            str(qualification_max_repairs),
            "--agent-transient-max-attempts",
            str(agent_transient_max_attempts),
            "--qualification-max-infrastructure-attempts",
            str(qualification_max_infrastructure_attempts),
            "--production-entry-timeout-seconds",
            str(production_entry_timeout_seconds),
            "--qualification-timeout-seconds",
            str(timeout_seconds),
            "--max-gate-checks-per-repair",
            str(max_gate_checks_per_repair),
            "--max-runtime-bytes",
            str(max_runtime_bytes),
            "--draft-max-attempts",
            str(draft_max_attempts),
            "--evaluator-smoke-timeout-seconds",
            str(evaluator_smoke_timeout_seconds),
            "--max-idea-cycles",
            str(max_idea_cycles),
            "--max-program-evals-per-attempt",
            str(max_program_evals_per_attempt),
        ]
    )
    if resume_from:
        previous = resume_from if resume_from == "latest" else str(Path(resume_from).expanduser().absolute())
        cmd.extend(["--resume-from", previous])
    if containerize:
        cmd.append("--containerize")
        cmd.extend(["--container-image", container_image])
        cmd.extend(["--container-network", container_network])
        auth_root = container_auth_root if container_auth_root is not None else Path.home()
        cmd.extend(["--container-auth-root", str(auth_root)])
        if container_env_file is not None:
            cmd.extend(["--container-env-file", str(container_env_file)])

    env = container_execution_env(
        containerize=containerize,
        container_session_name=container_session_name,
        container_mount_root=container_mount_root,
    )
    progress_session_id = uuid.uuid4().hex
    progress_auth_token = secrets.token_hex(32)
    env["AGENTIC_TASK_PROGRESS_TRANSPORT"] = "wrapper"
    env["TASK_PROGRESS_SESSION_ID"] = progress_session_id
    env["TASK_PROGRESS_AUTH_TOKEN"] = progress_auth_token
    active_checkpoints: list[dict[str, Any]] = []

    def consume_task_event_line(line: str) -> bool:
        retained, records = extract_task_progress_events(
            line,
            expected_session_id=progress_session_id,
            auth_token=progress_auth_token,
        )
        if retained:
            return False
        for record in records:
            event_fields = {
                key: value
                for key, value in record["fields"].items()
                if key not in TASK_PROGRESS_COST_FIELDS
            }
            event_fields.update(
                {
                    "gate": record["gate"],
                    "state": record["state"],
                    "task_drafting_session_id": record["session_id"],
                    "task_drafting_timestamp": record["timestamp"],
                }
            )
            state = record["state"]
            if state == "started":
                if (
                    active_checkpoints
                    and active_checkpoints[-1].get("gate") == record["gate"]
                    and active_checkpoints[-1].get("state") == "retrying"
                ):
                    active_checkpoints.pop()
                active_checkpoints.append(dict(event_fields))
            elif (
                state == "retrying"
                and active_checkpoints
                and active_checkpoints[-1].get("gate") == record["gate"]
            ):
                active_checkpoints[-1].update(event_fields)
            elif state in {"completed", "failed", "skipped"}:
                for index in range(len(active_checkpoints) - 1, -1, -1):
                    if active_checkpoints[index].get("gate") == record["gate"]:
                        del active_checkpoints[index]
                        break
            if task_event_fn is not None:
                task_event_fn("task_drafting_checkpoint", **event_fields)
        return True

    with tempfile.TemporaryFile(mode="w+b", dir=temporary_directory()) as control:
        env[TASK_FAILURE_FD_ENV] = str(control.fileno())
        try:
            output, exit_code = _execute_script_command(
                cmd=cmd,
                timeout_seconds=timeout_seconds,
                timeout_grace_seconds=timeout_grace_seconds,
                script_label="src/scripts/task.sh",
                missing_output="[wrapper] src/scripts/task.sh not found\n",
                popen_factory=popen_factory,
                register_subprocess=register_subprocess,
                unregister_subprocess=unregister_subprocess,
                terminate_process_group=terminate_process_group,
                ensure_text=ensure_text,
                env=env,
                output_line_fn=consume_task_event_line,
                pass_fds=(control.fileno(),),
            )
        except Exception:
            if read_task_failure(control.fileno(), "task_drafting", "task_drafting") != AGENT_LOG_FAILURE:
                raise
            output = "[wrapper] task logger infrastructure failure; secondary host error\n"
            exit_code = INFRASTRUCTURE_EXIT_CODE
        failure = read_task_failure(control.fileno(), "task_drafting", "task_drafting")
    if active_checkpoints and task_event_fn is not None:
        for active_checkpoint in reversed(active_checkpoints):
            terminal_fields = dict(active_checkpoint)
            terminal_fields["state"] = "failed" if exit_code else "completed"
            terminal_fields["cost_checkpoint"] = True
            terminal_fields["synthetic"] = True
            try:
                task_event_fn("task_drafting_checkpoint", **terminal_fields)
            except Exception:
                pass
    return TaskDraftResult(
        output=output,
        exit_code=exit_code,
        draft_dir=output_dir,
        infrastructure_failure_category=AGENT_LOG_FAILURE if failure == AGENT_LOG_FAILURE else None,
    )


def build_agent_command(
    agent: str,
    prompt_file: str,
    *,
    agent_model: str | None = None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
) -> list[str]:
    """Build a pinned Headless command that references a staged prompt file."""
    model = agent_model.strip() if isinstance(agent_model, str) else ""
    profile = agent_profile.strip() if isinstance(agent_profile, str) else ""
    if profile and agent != "codex":
        raise ValueError("agent_profile is only supported for codex")
    if profile and not model:
        raise ValueError("agent_model is required when agent_profile is provided")
    if agent not in {"claude", "codex", "cursor", "opencode", "gemini", "pi"}:
        raise ValueError(f"unsupported agent: {agent}")
    if agent == "gemini" and reasoning_effort:
        raise ValueError("reasoning_effort is not supported for gemini")
    cmd = ["env", "HEADLESS_MODELS_DEV_CACHE="]
    cmd.extend(
        [
            "npx",
            "-y",
            "git+https://github.com/RobertTLange/headless-cli.git#"
            "93cd9b06b85f848af1308c41e018991b33907c5e",
            agent,
            "--debug",
            "--usage",
            "--allow",
            "yolo",
        ]
    )
    if model:
        cmd.extend(["--model", model])
    if profile:
        cmd.extend(["--profile", profile])
    if reasoning_effort:
        cmd.extend(["--reasoning-effort", reasoning_effort])
    if agent in {"claude", "codex", "gemini"}:
        cmd.extend(["--prompt-file", prompt_file])
    else:
        cmd.extend(
            [
                "--prompt",
                f"Read the complete task instructions from {prompt_file}, "
                "then follow them exactly.",
            ]
        )
    return cmd

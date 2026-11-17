"""Launch GPU containers only after durably registering their host processes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

# This entry point is invoked with Python -I from arbitrary evaluator directories.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.investigation.bounded_process import (
    BoundedProcessError, BoundedProcessResult, run_bounded_process,
)
from src.investigation.gpu_owner_process import (
    ProcessIdentityError, current_identity, process_is_alive,
)
from src.investigation.gpu_runtime_journal import (
    CONTAINER_ID_PATTERN, DAEMON_PATTERN, NAME_PATTERN, LaunchRecord,
    RuntimeJournalError, RuntimeRecord, open_runtime,
)

LOCAL_DOCKER_HOST = "unix:///var/run/docker.sock"
_OPERATION_TIMEOUT = 30.0
_BOOLEAN_OPTIONS = {"--read-only", "--init", "--privileged"}
_VALUE_OPTIONS = {
    "--cidfile", "--network", "--cap-drop", "--cap-add", "--security-opt",
    "--pids-limit", "--cpus", "--memory", "--memory-swap", "--user", "--tmpfs",
    "--mount", "--env", "--volume", "--gpus", "--entrypoint", "--workdir",
    "--ulimit", "--shm-size", "--hostname", "-e", "-v", "-w", "-u",
}


class RuntimeLaunchError(RuntimeError):
    """The owned launch could not establish its safety prerequisites."""


def prepare_create(
    command: list[str], token: str, launch_id: str,
) -> tuple[list[str], str, bool, bool]:
    """Parse the supported trusted run flags without interpreting image arguments."""
    if len(command) < 3 or command[:2] != ["docker", "run"]:
        raise RuntimeLaunchError("GPU ownership requires a direct docker run command")
    options: list[str] = []
    name: str | None = None
    interactive = remove = False
    index = 2
    while index < len(command) and command[index].startswith("-"):
        raw = command[index]
        option, separator, inline = raw.partition("=")
        if option in {"--rm", "-i", "--interactive"}:
            if separator:
                raise RuntimeLaunchError("GPU launch boolean options must use their plain form")
            if option == "--rm":
                remove = True
            else:
                interactive = True
                options.append(raw)
            index += 1
            continue
        if option in _BOOLEAN_OPTIONS and not separator:
            options.append(raw)
            index += 1
            continue
        if option not in _VALUE_OPTIONS | {"--name", "--label", "-l"}:
            raise RuntimeLaunchError("Unsupported GPU Docker launch option")
        if separator:
            value = inline
            consumed = 1
        else:
            if index + 1 >= len(command):
                raise RuntimeLaunchError("GPU Docker option is missing its value")
            value = command[index + 1]
            consumed = 2
        if option == "--name":
            if name is not None or NAME_PATTERN.fullmatch(value) is None:
                raise RuntimeLaunchError("GPU Docker container name is invalid or repeated")
            name = value
        if option in {"--label", "-l"} and value.partition("=")[0].startswith(
            "agentic-scientist."
        ):
            raise RuntimeLaunchError("GPU ownership labels are reserved")
        options.extend(command[index:index + consumed])
        index += consumed
    if index >= len(command):
        raise RuntimeLaunchError("GPU Docker launch is missing its image")
    if name is None:
        name = "agentic-" + launch_id
        options.extend(("--name", name))
    labels = ["--label", f"agentic-scientist.lease={token}",
              "--label", f"agentic-scientist.launch={launch_id}"]
    # No --quiet: it needs Docker >= 20.10, and pull progress already goes to stderr.
    create = ["docker", "create", *labels, *options, *command[index:]]
    return create, name, interactive, remove


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in ("DOCKER_CONTEXT", "DOCKER_HOST", "DOCKER_TLS", "DOCKER_TLS_VERIFY",
                "DOCKER_CERT_PATH"):
        environment.pop(key, None)
    return environment


def _digest(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, ensure_ascii=True).encode()).hexdigest()


def _captured(command: list[str]) -> BoundedProcessResult:
    return run_bounded_process(
        command, cwd=Path.cwd(), env=_environment(), timeout_seconds=_OPERATION_TIMEOUT,
        max_output_bytes=64 * 1024,
    )


def _daemon_id(docker: list[str]) -> str:
    result = _captured([*docker, "info", "--format", "{{.ID}}"])
    value = result.stdout.decode("ascii", errors="replace").strip()
    if result.returncode or DAEMON_PATTERN.fullmatch(value) is None:
        raise RuntimeLaunchError("Local Docker daemon identity could not be verified")
    return value


def _require_active(record: RuntimeRecord) -> None:
    if record.state != "active" or not process_is_alive(record.owner):
        raise RuntimeLaunchError("GPU lease owner stopped or launch permission was revoked")


def _helper(root: Path, token: str, launch_id: str, phase: str, command: list[str]) -> list[str]:
    return [sys.executable, "-I", str(Path(__file__).resolve()), "--lease-root", str(root),
            "--lease-id", token, "--launch-id", launch_id, "--exec-phase", phase,
            "--", *command]


def exec_phase(root: Path, token: str, launch_id: str, phase: str, command: list[str]) -> None:
    """Self-register under the revocation fence, then become the recorded Docker client."""
    with open_runtime(root, token, deadline=time.monotonic() + _OPERATION_TIMEOUT) as journal:
        record = journal.record
        _require_active(record)
        entry = record.launches.get(launch_id)
        if entry is None or not process_is_alive(entry.supervisor):
            raise RuntimeLaunchError("GPU launch supervisor is no longer registered and alive")
        if phase == "create":
            if entry.creator is not None or entry.create_finished:
                raise RuntimeLaunchError("GPU create phase has already been claimed")
            expected = entry.create_command_sha256
        elif phase == "start":
            if not entry.create_finished or entry.container_id is None or entry.starter is not None:
                raise RuntimeLaunchError("GPU start phase lacks a completed unclaimed container")
            expected = entry.start_command_sha256
        else:
            raise RuntimeLaunchError("Unknown GPU execution phase")
        if expected is None or _digest(command) != expected:
            raise RuntimeLaunchError("GPU execution command differs from its registered command")
        if phase == "create":
            entry.creator = current_identity()
        else:
            entry.starter = current_identity()
        journal.save()
        # Lock descriptors are close-on-exec. No unregistered child may issue a launch.
        os.execve(command[0], command, _environment())


def launch(root: Path, token: str, command: list[str]) -> int:
    launch_id = uuid.uuid4().hex
    create, name, interactive, remove = prepare_create(command, token, launch_id)
    executable = shutil.which("docker")
    if executable is None:
        raise RuntimeLaunchError("Docker executable is unavailable")
    docker = [str(Path(executable).resolve()), "--host", LOCAL_DOCKER_HOST]
    create = [*docker, *create[1:]]
    daemon_id = _daemon_id(docker)
    with open_runtime(root, token, deadline=time.monotonic() + _OPERATION_TIMEOUT) as journal:
        _require_active(journal.record)
        if journal.record.daemon_id not in {None, daemon_id}:
            raise RuntimeLaunchError("GPU lease Docker daemon identity changed")
        journal.record.daemon_id = daemon_id
        journal.record.launches[launch_id] = LaunchRecord(
            name=name, supervisor=current_identity(), create_command_sha256=_digest(create),
        )
        journal.save()
    created = _captured(_helper(root, token, launch_id, "create", create))
    sys.stderr.buffer.write(created.stderr)
    sys.stderr.buffer.flush()
    container_id = created.stdout.decode("ascii", errors="replace").strip()
    valid_id = CONTAINER_ID_PATTERN.fullmatch(container_id) is not None
    with open_runtime(root, token, deadline=time.monotonic() + _OPERATION_TIMEOUT) as journal:
        _require_active(journal.record)
        entry = journal.record.launches[launch_id]
        if created.returncode == 0 and valid_id:
            entry.create_finished = True
            entry.container_id = container_id
            start = [*docker, "start", "--attach"]
            if interactive:
                start.append("--interactive")
            start.append(container_id)
            entry.start_command_sha256 = _digest(start)
        journal.save()
    if created.returncode:
        return created.returncode if created.returncode > 0 else 128 - created.returncode
    if not valid_id:
        raise RuntimeLaunchError("Docker create returned an invalid container identity")
    # All inherited streams belong to the evaluator/RPC channel; never print container IDs.
    started = subprocess.run(_helper(root, token, launch_id, "start", start), env=_environment())
    if _daemon_id(docker) != daemon_id:
        raise RuntimeLaunchError("GPU lease Docker daemon identity changed after execution")
    inspected = _captured([*docker, "inspect", "--format", "{{json .State}}", container_id])
    try:
        state = json.loads(inspected.stdout)
        exit_code = state["ExitCode"]
        stopped = state["Running"] is False and state["Status"] in {"exited", "dead"}
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeLaunchError("GPU container exit could not be verified") from exc
    if inspected.returncode or not stopped or type(exit_code) is not int or not 0 <= exit_code <= 255:
        raise RuntimeLaunchError("GPU container exit could not be verified")
    if remove:
        removed = _captured([*docker, "rm", container_id])
        if removed.returncode:
            raise RuntimeLaunchError("Exited GPU container removal could not be verified")
    if started.returncode and exit_code == 0:
        return started.returncode if started.returncode > 0 else 128 - started.returncode
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease-root", required=True, type=Path)
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--launch-id")
    parser.add_argument("--exec-phase", choices=("create", "start"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        if args.exec_phase:
            if not args.launch_id:
                raise RuntimeLaunchError("GPU execution phase requires its launch identity")
            exec_phase(args.lease_root, args.lease_id, args.launch_id, args.exec_phase, command)
            raise RuntimeLaunchError("GPU execution did not replace its registered process")
        return launch(args.lease_root, args.lease_id, command)
    except (RuntimeLaunchError, RuntimeJournalError, ProcessIdentityError, BoundedProcessError,
            OSError) as exc:
        # Docker stderr is already bounded above. Internal exception text may contain paths.
        message = str(exc) if isinstance(exc, RuntimeLaunchError) else type(exc).__name__
        print(f"GPU owned launch failed: {message}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())

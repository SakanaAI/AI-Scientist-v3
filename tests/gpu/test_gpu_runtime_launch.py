"""Owned container launches: helper registration, fake-Docker supervision, and crash recovery."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

from src.task_drafting.resource_preflight import check_gpu_allocation
from src.investigation import gpu_runtime_launch as launch
from src.investigation.container_gpus import (
    GPU_ALLOCATION_BUSY_CATEGORY, GpuLeaseQuarantined, acquire_gpu_allocation_lease,
)
from src.investigation.gpu_owner_process import (
    ProcessIdentity, current_identity, process_is_alive, terminate_process,
)
from src.investigation.gpu_runtime_journal import LaunchRecord, create_runtime, open_runtime

PROJECT = Path(__file__).resolve().parents[2]
DOCKER_ENV = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")


def _open(root: Path, token: str):
    return open_runtime(root, token, deadline=time.monotonic() + 1)


def _first_launch(root: Path, token: str) -> LaunchRecord:
    with _open(root, token) as journal:
        return next(iter(journal.record.launches.values()))


def _write_fake_docker(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(0o700)


# --------------------------------------------------------------------------- prepare_create


def test_prepare_create_preserves_arguments_adds_labels_and_rejects_unsupported_commands():
    # container arguments are preserved and owned labels added
    command, name, interactive, remove = launch.prepare_create(
        ['docker', 'run', '--rm', '-i', '--env', 'KEY=value', 'image', '--name', 'inside'],
        'a' * 32, 'b' * 32,
    )
    assert command[:2] == ['docker', 'create']
    assert '--quiet' not in command  # unsupported before Docker 20.10 (issue #2)
    assert '--rm' not in command
    assert command[-3:] == ['image', '--name', 'inside']
    assert name == 'agentic-' + 'b' * 32
    assert interactive and remove
    assert 'agentic-scientist.lease=' + 'a' * 32 in command
    assert 'agentic-scientist.launch=' + 'b' * 32 in command
    # an explicit name is preserved
    command, name, interactive, remove = launch.prepare_create(
        ['docker', 'run', '--name', 'existing-worker', 'image'], 'a' * 32, 'b' * 32
    )
    assert name == 'existing-worker'
    assert command.count('--name') == 1
    assert not interactive and not remove
    # image arguments are never parsed as docker options
    command, _, interactive, _ = launch.prepare_create(
        ['docker', 'run', 'image', '-i', '--label', 'agentic-scientist.lease=literal'],
        'a' * 32, 'b' * 32,
    )
    assert not interactive
    assert command[-4:] == ['image', '-i', '--label', 'agentic-scientist.lease=literal']
    # malformed, unsupported, or ownership-overriding commands fail before launch
    rejected = [
        *(['docker', 'run', *option, 'image'] for option in [
            ['--label', 'agentic-scientist.lease=other'],
            ['--label=agentic-scientist.launch=other'],
            ['-l', 'agentic-scientist.lease=other'],
            ['--label-file', '/tmp/labels'],
            ['--detach'], ['-d'], ['-it'], ['--restart', 'always'],
        ]),
        [], ['docker', 'run'], ['docker', 'exec', 'image'],
        ['docker', 'run', '--name', 'one', '--name', 'two', 'image'],
        ['docker', 'run', '--name', '../unsafe', 'image'],
        ['docker', 'run', '--rm=false', 'image'],
        ['docker', 'run', '--name'], ['docker', 'run', '--read-only'],
    ]
    for command in rejected:
        with pytest.raises(launch.RuntimeLaunchError):
            launch.prepare_create(command, 'a' * 32, 'b' * 32)


# --------------------------------------------------------------------------- supervised launcher


FAKE_DOCKER = r'''
import json
import os
from pathlib import Path
import sys
args = sys.argv[1:]
assert args[:2] == ['--host', 'unix:///var/run/docker.sock'], args
assert 'DOCKER_CONTEXT' not in os.environ
args = args[2:]
record = json.loads(Path(os.environ['FAKE_JOURNAL']).read_text())
operation = args[0]
if operation == 'info':
    print('test-daemon')
elif operation == 'create':
    if '--quiet' in args:
        print('unknown flag: --quiet', file=sys.stderr)
        sys.exit(125)
    entry = next(iter(record['launches'].values()))
    assert entry['creator']['pid'] == os.getpid()
    assert entry['container_id'] is None
    assert '--rm' not in args
    if os.environ.get('FAKE_CREATE_FAIL'):
        print('create diagnostic', file=sys.stderr)
        sys.exit(17)
    print(os.environ.get('FAKE_CONTAINER_ID', 'c' * 64))
elif operation == 'start':
    entry = next(iter(record['launches'].values()))
    assert entry['starter']['pid'] == os.getpid()
    assert entry['create_finished'] and entry['container_id'] == 'c' * 64
    if '--interactive' in args:
        sys.stdout.buffer.write(sys.stdin.buffer.read())
    else:
        print('evaluator result')
    print('worker stderr', file=sys.stderr)
    sys.exit(int(os.environ.get('FAKE_ATTACH_EXIT', '0')))
elif operation == 'inspect':
    print(json.dumps({'Running': bool(os.environ.get('FAKE_RUNNING')), 'Status': 'exited',
                      'ExitCode': int(os.environ.get('FAKE_EXIT', '0'))}))
elif operation == 'rm':
    if os.environ.get('FAKE_REMOVE_FAIL'):
        sys.exit(1)
    Path(os.environ['FAKE_REMOVED']).write_text(args[-1])
else:
    raise AssertionError(args)
'''


def _fake_docker(directory: Path, monkeypatch: pytest.MonkeyPatch):
    """Fresh owned runtime plus a fake `docker` on PATH, both private to `directory`."""
    directory.mkdir(mode=0o700, exist_ok=True)
    root = directory / 'leases'
    root.mkdir(mode=0o700)
    token = create_runtime(root, ('GPU-aaaa',))
    _write_fake_docker(directory / 'docker', FAKE_DOCKER)
    monkeypatch.setenv('PATH', str(directory) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('DOCKER_CONTEXT', 'must-not-use-this-context')
    monkeypatch.setenv('FAKE_JOURNAL', str(root / f'runtime-{token}.json'))
    monkeypatch.setenv('FAKE_REMOVED', str(directory / 'removed'))
    return root, token, directory


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    return _fake_docker(tmp_path, monkeypatch)


@contextmanager
def _fresh_fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str):
    """Yield (root, token, directory, section_monkeypatch); patches end with the section."""
    with monkeypatch.context() as m:
        yield (*_fake_docker(tmp_path / name, m), m)


def run_launcher(root, token, arguments, input_bytes=b''):
    return subprocess.run(
        [sys.executable, '-I', str(Path(launch.__file__).resolve()),
         '--lease-root', str(root), '--lease-id', token, '--', 'docker', 'run', *arguments],
        input=input_bytes, capture_output=True, timeout=15,
    )


def test_real_helpers_register_before_create_and_start_and_exit_codes_are_authoritative(tmp_path, monkeypatch):
    # real helpers register before create and start, and the RPC stream is preserved
    with _fresh_fake_docker(tmp_path, monkeypatch, 'rpc') as (root, token, directory, m):
        result = run_launcher(root, token, ['--rm', '-i', 'image'], b'{"rpc":1}\n')
        assert result.returncode == 0, result.stderr
        assert result.stdout == b'{"rpc":1}\n'
        assert result.stderr == b'worker stderr\n'
        assert (directory / 'removed').read_text() == 'c' * 64
        with _open(root, token) as journal:
            entry = next(iter(journal.record.launches.values()))
            assert entry.create_finished and entry.container_id == 'c' * 64
            assert entry.creator is not None and entry.starter is not None
            assert journal.record.daemon_id == 'test-daemon'
    # container and attach exit codes are both authoritative
    for variable, code in [('FAKE_EXIT', 23), ('FAKE_ATTACH_EXIT', 19)]:
        with _fresh_fake_docker(tmp_path, monkeypatch, variable) as (root, token, _, m):
            m.setenv(variable, str(code))
            result = run_launcher(root, token, ['image'])
            assert result.returncode == code, result.stderr
            assert result.stdout == b'evaluator result\n'


def test_failed_create_or_invalid_create_identity_never_starts(tmp_path, monkeypatch):
    # a failed create preserves the diagnostic and never starts
    with _fresh_fake_docker(tmp_path, monkeypatch, 'create-fail') as (root, token, directory, m):
        m.setenv('FAKE_CREATE_FAIL', 'yes')
        result = run_launcher(root, token, ['image'])
        assert result.returncode == 17
        assert result.stdout == b''
        assert result.stderr == b'create diagnostic\n'
        assert not (directory / 'removed').exists()
        entry = _first_launch(root, token)
        assert not entry.create_finished and entry.container_id is None and entry.starter is None
    # an invalid create identity prevents start
    for index, container_id in enumerate(['short', 'c' * 64 + '\n' + 'd' * 64, 'Z' * 64]):
        with _fresh_fake_docker(tmp_path, monkeypatch, f'identity-{index}') as (root, token, _, m):
            m.setenv('FAKE_CONTAINER_ID', container_id)
            result = run_launcher(root, token, ['image'])
            assert result.returncode == 125, container_id
            assert b'invalid container identity' in result.stderr
            entry = _first_launch(root, token)
            assert entry.container_id is None and entry.starter is None


def test_revoked_or_redaemoned_journal_refuses_new_supervisor_launch(tmp_path, monkeypatch):
    for field, value, message in [
        ('state', 'revoked', b'revoked'), ('daemon_id', 'original-daemon', b'daemon identity changed'),
    ]:
        with _fresh_fake_docker(tmp_path, monkeypatch, field) as (root, token, _, m):
            with _open(root, token) as journal:
                setattr(journal.record, field, value)
                journal.save()
            result = run_launcher(root, token, ['image'])
            assert result.returncode == 125, field
            assert message in result.stderr
            with _open(root, token) as journal:
                assert not journal.record.launches


def test_unverified_exit_or_removal_is_reported(tmp_path, monkeypatch):
    for field, value in [('FAKE_RUNNING', 'yes'), ('FAKE_EXIT', '256'), ('FAKE_REMOVE_FAIL', 'yes')]:
        with _fresh_fake_docker(tmp_path, monkeypatch, field) as (root, token, _, m):
            m.setenv(field, value)
            result = run_launcher(root, token, ['--rm', 'image'])
            assert result.returncode == 125, field
            assert b'could not be verified' in result.stderr


# --------------------------------------------------------------------------- exec_phase


def register_phase(root, token, phase):
    command = ['/trusted/docker', '--host', launch.LOCAL_DOCKER_HOST, phase]
    launch_id = 'a' * 32
    with _open(root, token) as journal:
        entry = LaunchRecord(name='owned-worker', supervisor=current_identity())
        if phase == 'create':
            entry.create_command_sha256 = launch._digest(command)
        else:
            entry.creator = current_identity()
            entry.create_command_sha256 = launch._digest(['create'])
            entry.create_finished = True
            entry.container_id = 'c' * 64
            entry.start_command_sha256 = launch._digest(command)
        journal.record.launches[launch_id] = entry
        journal.save()
    return launch_id, command


def test_exec_phase_rejects_changed_command_revocation_missing_container_and_dead_owner(tmp_path, monkeypatch):
    # a changed command is rejected without registering the helper
    for phase in ['create', 'start']:
        with _fresh_fake_docker(tmp_path, monkeypatch, f'changed-{phase}') as (root, token, _, m):
            launch_id, command = register_phase(root, token, phase)
            with pytest.raises(launch.RuntimeLaunchError, match='differs'):
                launch.exec_phase(root, token, launch_id, phase, [*command, '--evil'])
            with _open(root, token) as journal:
                entry = journal.record.launches[launch_id]
                assert entry.starter is None
                if phase == 'create':
                    assert entry.creator is None
    # revocation is honoured before self registration
    for phase in ['create', 'start']:
        with _fresh_fake_docker(tmp_path, monkeypatch, f'revoked-{phase}') as (root, token, _, m):
            launch_id, command = register_phase(root, token, phase)
            with _open(root, token) as journal:
                journal.record.state = 'revoked'
                journal.save()
            with pytest.raises(launch.RuntimeLaunchError, match='revoked'):
                launch.exec_phase(root, token, launch_id, phase, command)
    # start requires a durable container; a dead owner refuses the helper
    with _fresh_fake_docker(tmp_path, monkeypatch, 'guards') as (root, token, _, m):
        launch_id, command = register_phase(root, token, 'create')
        with pytest.raises(launch.RuntimeLaunchError, match='lacks'):
            launch.exec_phase(root, token, launch_id, 'start', command)
        m.setattr(launch, 'process_is_alive', lambda identity: False)
        with pytest.raises(launch.RuntimeLaunchError, match='owner stopped'):
            launch.exec_phase(root, token, launch_id, 'create', command)


def test_exec_phase_registers_identity_before_exec(tmp_path, monkeypatch):
    for phase in ['create', 'start']:
        with _fresh_fake_docker(tmp_path, monkeypatch, phase) as (root, token, _, m):
            launch_id, command = register_phase(root, token, phase)
            observed = []

            def fake_exec(executable, argv, environment, phase=phase, root=root, token=token,
                          launch_id=launch_id, command=command, observed=observed):
                data = json.loads((root / f'runtime-{token}.json').read_text())
                key = 'creator' if phase == 'create' else 'starter'
                assert data['launches'][launch_id][key] == current_identity().fields()
                assert executable == command[0] and argv == command
                observed.append(True)

            m.setattr(launch.os, 'execve', fake_exec)
            launch.exec_phase(root, token, launch_id, phase, command)
            assert observed == [True], phase
            with pytest.raises(launch.RuntimeLaunchError):
                launch.exec_phase(root, token, launch_id, phase, command)


def test_late_exec_helper_cannot_cross_revocation_fence(fake_docker):
    root, token, directory = fake_docker
    launch_id, command = register_phase(root, token, 'create')
    command = [str(directory / 'docker'), '--host', launch.LOCAL_DOCKER_HOST, 'create']
    ready = directory / 'helper-entered-journal-lock'
    script = '''
import contextlib
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from src.investigation import gpu_runtime_launch as launch
original = launch.open_runtime
@contextlib.contextmanager
def observed_open(*args, **kwargs):
    Path(sys.argv[2]).write_text('waiting')
    with original(*args, **kwargs) as journal:
        yield journal
launch.open_runtime = observed_open
raise SystemExit(launch.main(sys.argv[3:]))
'''
    process = None
    try:
        with _open(root, token) as journal:
            journal.record.launches[launch_id].create_command_sha256 = launch._digest(command)
            journal.save()
            process = subprocess.Popen(
                [sys.executable, '-I', '-c', script, str(PROJECT), str(ready),
                 '--lease-root', str(root), '--lease-id', token,
                 '--launch-id', launch_id, '--exec-phase', 'create', '--', *command],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 5
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), 'Exec helper never reached the held journal fence'
            assert process.poll() is None
            journal.record.state = 'revoked'
            journal.save()
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 125
        assert stdout == b''
        assert b'launch permission was revoked' in stderr
        with _open(root, token) as journal:
            assert journal.record.launches[launch_id].creator is None
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)


# --------------------------------------------------------------------------- crash recovery (real pidfd)


SELECTOR = "device=GPU-aaaa,GPU-bbbb"
CRASH_DOCKER = r'''
import json, os, sys, time
from pathlib import Path
root = Path(os.environ["GPU_TEST_DOCKER_ROOT"])
args = sys.argv[1:]
assert args[:2] == ["--host", "unix:///var/run/docker.sock"]
assert not any(key in os.environ for key in (
    "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH",
))
args = args[2:]
operation = args[0]
inventory = root / "container.json"
event = {"operation": operation, "pid": os.getpid()}
if operation == "rm":
    event["marker_sizes"] = [p.stat().st_size for p in (root / "leases").glob("*.lock")]
with (root / "events.jsonl").open("a") as events:
    events.write(json.dumps(event) + "\n")
if operation == "info":
    print("test-local-daemon")
elif operation == "create":
    labels = {}
    for index, arg in enumerate(args):
        if arg == "--label":
            key, value = args[index + 1].split("=", 1)
            labels[key] = value
    name = args[args.index("--name") + 1]
    container = {"id": "c" * 64, "name": name, "labels": labels}
    inventory.write_text(json.dumps(container))
    print(container["id"])
elif operation == "start":
    (root / "started").write_text(str(os.getpid()))
    time.sleep(60)
elif operation == "inspect":
    if not inventory.exists():
        print("Error: No such container: " + args[-1], file=sys.stderr)
        raise SystemExit(1)
    container = json.loads(inventory.read_text())
    assert args[-1] in {container["id"], container["name"]}
    if "{{json .State}}" in args:
        print(json.dumps({"ExitCode": 0, "Running": False, "Status": "exited"}))
    else:
        print(json.dumps({"id": container["id"], "labels": container["labels"]}))
elif operation == "rm":
    if (root / "fail-removal").exists():
        print("injected Docker removal failure", file=sys.stderr)
        raise SystemExit(1)
    if inventory.exists():
        inventory.unlink()
else:
    raise AssertionError("unexpected Docker operation: " + operation)
'''

OWNER = r'''
import json, os, subprocess, sys, time
from pathlib import Path
from src.investigation.container_gpus import acquire_gpu_allocation_lease
root = Path(sys.argv[1])
lease = acquire_gpu_allocation_lease(
    "device=GPU-aaaa,GPU-bbbb", 2, lease_root=root / "leases", deadline=time.monotonic() + 5,
)
lease.prepare_runtime()
if sys.argv[2] == "launch":
    command = lease.wrap_command(["docker", "run", "--rm", "--gpus", "all", "fake-image"])
    subprocess.Popen(command, stdin=subprocess.DEVNULL)
    deadline = time.monotonic() + 15
    while not (root / "started").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("fake Docker start never arrived")
        time.sleep(0.01)
(root / "ready").write_text("ready")
while not (root / "crash").exists():
    time.sleep(0.01)
os._exit(73)
'''


def _launch_identities(record: dict) -> list[ProcessIdentity]:
    return [
        ProcessIdentity.from_fields(entry[key])
        for entry in record["launches"].values()
        for key in ("supervisor", "creator", "starter") if entry[key] is not None
    ]


@dataclass
class DisposableRuntime:
    root: Path
    owner: subprocess.Popen

    @property
    def lease_root(self) -> Path:
        return self.root / "leases"

    def record(self) -> dict:
        records = list(self.lease_root.glob("runtime-*.json"))
        assert len(records) == 1
        return json.loads(records[0].read_text())

    def launch_processes(self) -> list[ProcessIdentity]:
        return _launch_identities(self.record())

    def crash(self) -> None:
        (self.root / "crash").write_text("crash")
        assert self.owner.wait(timeout=5) == 73

    def assert_markers_cleared(self) -> None:
        markers = list(self.lease_root.glob("*.lock"))
        assert len(markers) == 2
        assert all(marker.read_bytes() == b"" for marker in markers)

    def preflight(self, **kwargs):
        return check_gpu_allocation(SELECTOR, lease_root=self.lease_root, **kwargs)


@pytest.fixture
def disposable_runtime(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(mode=0o700)
    _write_fake_docker(bin_dir / "docker", CRASH_DOCKER)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("GPU_TEST_DOCKER_ROOT", str(tmp_path))
    for key in DOCKER_ENV:
        monkeypatch.setenv(key, "must-not-reach-docker")

    @contextmanager
    def start(mode="launch"):
        with (tmp_path / "owner.log").open("w") as log:
            owner = subprocess.Popen(
                [sys.executable, "-c", textwrap.dedent(OWNER), str(tmp_path), mode],
                cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            )
        runtime = DisposableRuntime(tmp_path, owner)
        try:
            deadline = time.monotonic() + 20
            while not (tmp_path / "ready").exists():
                assert owner.poll() is None, (tmp_path / "owner.log").read_text()
                assert time.monotonic() < deadline, (tmp_path / "owner.log").read_text()
                time.sleep(0.02)
            yield runtime
        finally:
            if owner.poll() is None:
                owner.kill()
            owner.wait(timeout=5)
            # Only identities from this fixture's private journal are signaled; never bare PIDs.
            for record_path in runtime.lease_root.glob("runtime-*.json"):
                for identity in dict.fromkeys(_launch_identities(json.loads(record_path.read_text()))):
                    terminate_process(identity, deadline=time.monotonic() + 2)

    return start


@pytest.mark.parametrize("entrypoint", ["acquire", "preflight"])
def test_crashed_owner_is_reclaimed_through_normal_entrypoints(disposable_runtime, entrypoint):
    with disposable_runtime() as runtime:
        old_processes = runtime.launch_processes()
        assert any(process_is_alive(identity) for identity in old_processes)
        original_inodes = {path.name: path.stat().st_ino for path in runtime.lease_root.glob("*.lock")}
        runtime.crash()
        if entrypoint == "acquire":
            with acquire_gpu_allocation_lease(
                SELECTOR, 2, lease_root=runtime.lease_root, deadline=time.monotonic() + 10,
            ):
                runtime.assert_markers_cleared()
        else:
            report = runtime.preflight()
            assert report.status == "passed", report.fields()
        assert all(not process_is_alive(identity) for identity in old_processes)
        assert not (runtime.root / "container.json").exists()
        assert runtime.record()["state"] == "recovered"
        runtime.assert_markers_cleared()
        assert original_inodes == {path.name: path.stat().st_ino for path in runtime.lease_root.glob("*.lock")}
        events = [json.loads(line) for line in (runtime.root / "events.jsonl").read_text().splitlines()]
        removal = next(event for event in events if event["operation"] == "rm")
        assert len(removal["marker_sizes"]) == 2
        assert all(size > 0 for size in removal["marker_sizes"])


def test_failed_removal_keeps_markers_then_next_preflight_retries(disposable_runtime):
    with disposable_runtime() as runtime:
        old_processes = runtime.launch_processes()
        runtime.crash()
        failure = runtime.root / "fail-removal"
        failure.write_text("fail")
        with pytest.raises(GpuLeaseQuarantined):
            acquire_gpu_allocation_lease(
                SELECTOR, 2, lease_root=runtime.lease_root, deadline=time.monotonic() + 10,
            )
        assert runtime.record()["state"] == "revoked"
        assert all(not process_is_alive(identity) for identity in old_processes)
        assert (runtime.root / "container.json").exists()
        assert all(path.stat().st_size > 0 for path in runtime.lease_root.glob("*.lock"))
        failure.unlink()
        assert runtime.preflight().status == "passed"
        runtime.assert_markers_cleared()
        assert not (runtime.root / "container.json").exists()


def test_owner_crash_before_launch_recovers_without_docker(disposable_runtime):
    with disposable_runtime("no-launch") as runtime:
        runtime.crash()
        assert runtime.preflight().status == "passed"
        runtime.assert_markers_cleared()
        assert runtime.record()["state"] == "recovered"
        assert not (runtime.root / "events.jsonl").exists()


def test_live_owner_is_busy_and_its_workers_are_not_terminated(disposable_runtime):
    with disposable_runtime() as runtime:
        old_processes = runtime.launch_processes()
        report = runtime.preflight(timeout_seconds=0.05)
        assert report.category == GPU_ALLOCATION_BUSY_CATEGORY
        assert runtime.owner.poll() is None
        assert runtime.record()["state"] == "active"
        assert any(process_is_alive(identity) for identity in old_processes)

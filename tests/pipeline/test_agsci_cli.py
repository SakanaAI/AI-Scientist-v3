"""``agsci`` subcommand dispatch, ``agsci investigate``, ``agsci doctor``, ``agsci inspect``, and ``agsci review``."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.agsci import doctor, inspector, review
from src.agsci.main import main


# --- dispatch -----------------------------------------------------------------


def test_dispatch_forwards_run_and_investigate_arguments(monkeypatch) -> None:
    # run forwards arguments to the pipeline
    received: list[list[str]] = []

    def fake_pipeline(argv: list[str]) -> int:
        received.append(argv)
        return 7

    monkeypatch.setattr("src.pipeline.cli.main", fake_pipeline)
    assert main(["run", "examples/circpack", "--dry-run", "preset=circpack"]) == 7
    assert received == [["examples/circpack", "--dry-run", "preset=circpack"]]

    # investigate forwards overrides to the investigation cli
    received_cli: list[list[str] | None] = []

    def fake_run_cli(argv: list[str] | None = None) -> int:
        received_cli.append(argv)
        return 5

    monkeypatch.setattr("src.investigation.cli.run_cli", fake_run_cli)
    overrides = ["agent.name=claude", "paths.task_dir=examples/circpack", "--verbose"]
    assert main(["investigate", *overrides]) == 5
    assert received_cli == [overrides]

    # review forwards the PDF and output path to the standalone reviewer
    received_review: list[list[str] | None] = []

    def fake_review(argv: list[str] | None = None) -> int:
        received_review.append(argv)
        return 3

    monkeypatch.setattr("src.agsci.review.main", fake_review)
    assert main(["review", "paper.pdf", "--output", "r.json"]) == 3
    assert received_review == [["paper.pdf", "--output", "r.json"]]


def test_dispatch_usage_help_and_unknown_commands(capsys) -> None:
    # no arguments prints usage and fails
    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Usage: agsci <command>" in captured.err

    # help lists every command
    for flag in ("--help", "-h", "help"):
        assert main([flag]) == 0, flag
        out = capsys.readouterr().out
        for command in ("run", "investigate", "doctor", "inspect", "review"):
            assert f"  {command} " in out, flag

    # a bare task directory points at the run subcommand
    assert main(["examples/circpack", "preset=circpack"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'examples/circpack'" in err
    assert "agsci run examples/circpack preset=circpack" in err

    # an unknown word gets no run hint
    assert main(["frobnicate"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'frobnicate'" in err
    assert "agsci run" not in err.split("Usage:")[0]


# --- doctor -------------------------------------------------------------------


def _completed(argv: list[str], stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")


def _healthy_host(tmp_path: Path, *, missing: frozenset[str] = frozenset()) -> doctor.Host:
    repo = tmp_path / "repo"
    (repo / "configs" / "preset").mkdir(parents=True)
    for name in ("paper", "circpack"):
        (repo / "configs" / "preset" / f"{name}.yaml").write_text("", encoding="utf-8")
    (repo / "examples" / "circpack").mkdir(parents=True)
    (repo / "examples" / "circpack" / "task.md").write_text("# task\n", encoding="utf-8")
    (repo / ".env").write_text("", encoding="utf-8")
    (repo / "apps" / "run_inspector" / "frontend" / "dist").mkdir(parents=True)
    (repo / "apps" / "run_inspector" / "frontend" / "dist" / "index.html").write_text("")

    def which(name: str) -> str | None:
        return None if name in missing else f"/usr/bin/{name}"

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        if argv[0] == "nvidia-smi":
            return _completed(argv, "GPU-aaaa\nGPU-bbbb\n")
        if argv[:2] == ["docker", "run"] and "nvidia-smi" in argv:
            # A healthy container runtime shows exactly the GPUs the selector names.
            selector = argv[argv.index("--gpus") + 1].strip('"').removeprefix("device=")
            return _completed(argv, "".join(f"{uuid}\n" for uuid in selector.split(",")))
        return _completed(argv, f"{argv[0]} version 1.0\n")

    return doctor.Host(
        which=which,
        run=run,
        probe_docker=lambda: (True, ""),
        image_exists=lambda image: (True, ""),
        environ={"ANTHROPIC_API_KEY": "secret-value"},
        repo_root=repo,
        python_version=(3, 11, 9),
        python_executable="/venv/bin/python",
    )


def _stub_repo_checks(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor, "check_pipeline_config", lambda host: doctor.CheckResult("pipeline config", "ok", "")
    )
    monkeypatch.setattr(
        doctor, "check_temp_storage", lambda host: doctor.CheckResult("temporary storage", "ok", "")
    )


def test_doctor_reports_healthy_host_and_distinguishes_required_from_optional_tools(
    monkeypatch, tmp_path, capsys
) -> None:
    _stub_repo_checks(monkeypatch)

    # reports ok for a healthy host
    assert doctor.main(["--json"], host=_healthy_host(tmp_path / "healthy")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    statuses = {check["name"]: check["status"] for check in payload["checks"]}
    assert statuses["gpus"] == "ok"
    assert statuses["agent image"] == "ok"
    assert statuses["task presets"] == "ok"
    assert "resolved run" not in statuses, "run checks need run arguments"
    assert "secret-value" not in json.dumps(payload)

    # fails on a missing required tool but only warns on optional ones
    host = _healthy_host(tmp_path / "missing", missing=frozenset({"git", "codex", "nvidia-smi"}))
    assert doctor.main([], host=host) == 1
    out = capsys.readouterr().out
    assert "FAIL  git" in out
    assert "WARN  codex" in out
    assert "WARN  gpus" in out
    assert "1 failure" in out


def test_doctor_image_and_daemon_checks_and_hint_rendering(tmp_path) -> None:
    # warns on a missing agent image with a build hint
    host = _healthy_host(tmp_path)
    host.image_exists = lambda image: (False, "No such image")
    result = doctor.check_agent_image(host, doctor.DEFAULT_CONTAINER_IMAGE, required=False)
    assert result.status == "warn"
    assert result.hint == doctor.DEFAULT_CONTAINER_IMAGE_BUILD_HINT
    host_net = doctor.check_agent_image(
        host, doctor.DEFAULT_CONTAINER_IMAGE, network="host", required=True
    )
    assert host_net.status == "fail"
    assert "--network=host" in host_net.hint
    custom = doctor.check_agent_image(host, "ghcr.io/x/y:1", required=True)
    assert custom.hint == "build or pull ghcr.io/x/y:1"

    # a docker daemon failure uses the probe detail
    host.probe_docker = lambda: (False, "permission denied while trying to connect\nmore")
    result = doctor.check_docker_daemon(host)
    assert result.status == "fail"
    assert result.detail == "permission denied while trying to connect"
    assert "docker info" in result.hint

    # hints render on an indented second line
    results = [
        doctor.CheckResult("python", "ok", "3.11"),
        doctor.CheckResult("git", "fail", "not found on PATH", hint="install git"),
    ]
    text = doctor.render(results)
    lines = text.splitlines()
    assert lines[2].startswith("  FAIL  git")
    assert lines[3].strip() == "-> install git"
    assert "1 failure" in lines[-1]


# --- doctor: resolved-run checks ---------------------------------------------


def _run_config(**overrides: object):
    from src.pipeline.cli import load_pipeline_config

    return load_pipeline_config([f"{key}={value}" for key, value in overrides.items()])


def test_doctor_resolved_run_reads_overrides_like_run_and_fails_on_bad_override(monkeypatch, tmp_path, capsys):
    _stub_repo_checks(monkeypatch)

    # the resolved run reads task dir and overrides like `agsci run`
    host = _healthy_host(tmp_path / "ok")
    host.environ = {"AGENT_CONTAINER_GPUS": "device=GPU-aaaa,GPU-bbbb"}
    rc = doctor.main(
        ["--json", "examples/circpack", "preset=grpo", "container.env_file="],
        host=host,
    )
    payload = json.loads(capsys.readouterr().out)
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["resolved run"]["detail"].startswith("agent=claude mode=container")
    assert "gpu_count=2" in checks["resolved run"]["detail"]
    assert checks["required gpus"]["status"] == "ok"
    assert checks["agent cli"]["status"] == "ok"
    assert "agent image" in checks["agent cli"]["detail"]
    assert checks["agent image"]["status"] == "ok"
    assert checks["env file"]["detail"] == "none configured"
    assert checks["task"]["status"] == "ok"
    assert "scaffold" in checks["task"]["detail"]
    assert rc == 0, payload

    # a bad override is reported as a failed resolved run
    rc = doctor.main(["--json", "examples/grpo", "nope.key=1"], host=_healthy_host(tmp_path / "bad"))
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    resolved = next(check for check in payload["checks"] if check["name"] == "resolved run")
    assert resolved["status"] == "fail"
    assert "agsci run" in resolved["hint"]


def test_doctor_docker_daemon_failure_only_warns_for_native_runs(
    monkeypatch, tmp_path, capsys
) -> None:
    _stub_repo_checks(monkeypatch)

    calls = iter(range(10))

    def run_doctor(*overrides: str) -> tuple[int, dict]:
        host = _healthy_host(tmp_path / str(next(calls)))
        host.probe_docker = lambda: (False, "permission denied while trying to connect")
        rc = doctor.main(["--json", *overrides], host=host)
        payload = json.loads(capsys.readouterr().out)
        return rc, next(check for check in payload["checks"] if check["name"] == "docker daemon")

    rc, daemon = run_doctor(
        "examples/circpack",
        "container.enabled=false",
        'execution.phases="investigation,posthoc"',
    )
    assert daemon["status"] == "warn"
    assert "not needed: container.enabled=false" in daemon["detail"]
    assert rc == 0

    # a host without the docker binary at all passes a native run too
    host = _healthy_host(tmp_path / "no_docker", missing=frozenset({"docker"}))
    host.probe_docker = lambda: (False, "docker CLI is not installed")
    rc = doctor.main(
        ["--json", "examples/circpack", "container.enabled=false", "execution.phases=investigation"],
        host=host,
    )
    checks = {check["name"]: check for check in json.loads(capsys.readouterr().out)["checks"]}
    assert checks["docker"]["status"] == checks["docker daemon"]["status"] == "warn"
    assert rc == 0

    # native ablations evaluate on the host too
    rc, daemon = run_doctor(
        "examples/circpack",
        "container.enabled=false",
        'execution.phases="investigation,ablations,posthoc"',
    )
    assert daemon["status"] == "warn"
    assert rc == 0

    rc, daemon = run_doctor("examples/circpack")
    assert daemon["status"] == "fail"
    assert rc == 1

    rc, daemon = run_doctor()
    assert daemon["status"] == "fail"
    assert rc == 1


def test_doctor_requires_docker_for_a_native_runs_paper_phase(tmp_path, capsys) -> None:
    def run_doctor(*overrides: str) -> tuple[int, dict[str, dict]]:
        host = _healthy_host(tmp_path / str(len(overrides)))
        host.probe_docker = lambda: (False, "permission denied while trying to connect")
        rc = doctor.main(["--json", "examples/circpack", "container.enabled=false", *overrides], host=host)
        return rc, {check["name"]: check for check in json.loads(capsys.readouterr().out)["checks"]}

    rc, checks = run_doctor('execution.phases="investigation,paper"')
    assert checks["paper container"]["status"] == "fail"
    assert "container.paper_enabled=true" in checks["paper container"]["detail"]
    assert rc == 1

    rc, checks = run_doctor('execution.phases="investigation,paper"', "container.paper_enabled=true")
    assert checks["paper container"]["status"] == "ok"
    assert checks["docker daemon"]["status"] == "fail"  # the paper container needs a working daemon
    assert rc == 1


def test_doctor_rejects_a_gpu_selector_the_run_would_reject(tmp_path) -> None:
    cfg = doctor.load_run_config(["examples/circpack", "container.gpu_count=2"])
    assert not isinstance(cfg, doctor.CheckResult)
    for selector, status in (("GPU-aaaa,GPU-bbbb", "fail"), ("device=GPU-aaaa,GPU-bbbb", "ok")):
        host = _healthy_host(tmp_path / status)
        host.environ = {"AGENT_CONTAINER_GPUS": selector}
        result = doctor.check_required_gpus(host, cfg)
        assert result.status == status, (selector, result)
        assert ("device=<UUID" in result.hint) == (status == "fail")


def test_doctor_required_gpus_compares_preset_count_selector_host_and_workers(tmp_path) -> None:
    host = _healthy_host(tmp_path)  # nvidia-smi reports two GPUs

    # compares the preset count with the selector and the host
    cfg = _run_config(**{"container.gpu_count": 2})
    assert doctor.check_required_gpus(host, cfg).status == "ok"
    cfg4 = _run_config(**{"container.gpu_count": 4})
    short = doctor.check_required_gpus(host, cfg4)
    assert short.status == "fail" and "reports 2" in short.detail
    host.environ = {"AGENT_CONTAINER_GPUS": "device=GPU-aaaa"}
    mismatch = doctor.check_required_gpus(host, cfg)
    assert mismatch.status == "fail" and "selects 1" in mismatch.detail
    cpu = _run_config(**{"container.gpu_count": 0})
    assert doctor.check_required_gpus(host, cpu).status == "fail"
    host.environ = {}
    assert doctor.check_required_gpus(host, cpu).status == "ok"

    # multiplies by parallel investigation workers
    host.environ = {"ANTHROPIC_API_KEY": "secret-value"}
    two_workers = _run_config(**{"container.gpu_count": 1, "investigation.execution.workers": 2})
    ok = doctor.check_required_gpus(host, two_workers)
    assert ok.status == "ok" and "2 (1 x 2 workers)" in ok.detail
    three_workers = _run_config(**{"container.gpu_count": 1, "investigation.execution.workers": 3})
    short = doctor.check_required_gpus(host, three_workers)
    assert short.status == "fail" and "3 (1 x 3 workers)" in short.detail and "reports 2" in short.detail
    host.environ = {"AGENT_CONTAINER_GPUS": "device=GPU-aaaa"}
    mismatch = doctor.check_required_gpus(host, two_workers)
    assert mismatch.status == "fail" and "selects 1" in mismatch.detail
    host.environ = {}
    downstream = _run_config(**{
        "container.gpu_count": 1, "investigation.execution.workers": 3, "execution.phases": "paper",
    })
    assert doctor.check_required_gpus(host, downstream).status == "ok"


def _container_gpu_host(
    tmp_path: Path, container_stdout: str, returncode: int = 0
) -> tuple[doctor.Host, list[list[str]]]:
    """A healthy host whose agent image reports *container_stdout* from nvidia-smi."""
    host = _healthy_host(tmp_path)  # nvidia-smi reports GPU-aaaa and GPU-bbbb on the host
    docker_calls: list[list[str]] = []
    healthy_run = host.run

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if argv[:2] == ["docker", "run"]:
            docker_calls.append(argv)
            completed = _completed(argv, container_stdout, returncode)
            completed.stderr = "" if returncode == 0 else "could not select device driver"
            return completed
        return healthy_run(argv, **kwargs)

    host.run = run
    return host, docker_calls


def test_doctor_container_gpus_runs_nvidia_smi_with_selector_and_only_in_container_mode(tmp_path) -> None:
    # runs nvidia-smi in the image with the derived selector
    host, docker_calls = _container_gpu_host(tmp_path / "derived", "GPU-bbbb\nGPU-aaaa\n")
    cfg = _run_config(**{"container.gpu_count": 2, "container.image": "agent:test"})
    result = doctor.check_container_gpus(host, cfg)
    assert result.status == "ok" and "agent:test" in result.detail
    [argv] = docker_calls
    assert argv[:4] == ["docker", "run", "--rm", "--gpus"]
    assert argv[4] == '"device=GPU-aaaa,GPU-bbbb"'
    assert argv[5] == "agent:test" and argv[6] == "nvidia-smi"

    # honours an explicit selector
    host, docker_calls = _container_gpu_host(tmp_path / "explicit", "GPU-bbbb\n")
    host.environ = {"AGENT_CONTAINER_GPUS": "device=GPU-bbbb"}
    cfg = _run_config(**{"container.gpu_count": 1})
    assert doctor.check_container_gpus(host, cfg).status == "ok"
    [argv] = docker_calls
    assert argv[4] == "device=GPU-bbbb"

    # run checks include container gpus only in container mode
    host, _calls = _container_gpu_host(tmp_path / "mode", "GPU-aaaa\nGPU-bbbb\n")
    native = _run_config(**{"container.gpu_count": 2, "container.enabled": "false"})
    assert "container gpus" not in [r.name for r in doctor.run_checks_for(host, native)]
    container = _run_config(**{"container.gpu_count": 2, "container.enabled": "true"})
    names = [r.name for r in doctor.run_checks_for(host, container)]
    assert names.index("container gpus") == names.index("required gpus") + 1


def test_doctor_container_gpus_fails_on_mismatch_or_runtime_error_and_skips_without_prerequisites(tmp_path) -> None:
    # fails on a uuid mismatch or a runtime error
    cfg = _run_config(**{"container.gpu_count": 2})
    mismatch_host, _calls = _container_gpu_host(tmp_path / "mismatch", "GPU-aaaa\n")
    mismatch = doctor.check_container_gpus(mismatch_host, cfg)
    assert mismatch.status == "fail" and "sees 1 GPU(s), expected 2" in mismatch.detail
    broken_host, _calls = _container_gpu_host(tmp_path / "broken", "", returncode=125)
    broken = doctor.check_container_gpus(broken_host, cfg)
    assert broken.status == "fail" and "could not select device driver" in broken.detail
    assert "NVIDIA Container Toolkit" in broken.hint

    # skips when prerequisites are missing
    cfg = _run_config(**{"container.gpu_count": 4})
    host, docker_calls = _container_gpu_host(tmp_path / "skip", "")
    short = doctor.check_container_gpus(host, cfg)
    assert short.status == "warn" and "reports 2" in short.detail
    host.image_exists = lambda image: (False, "No such image")
    assert doctor.check_container_gpus(host, _run_config(**{"container.gpu_count": 2})).status == "warn"
    cpu = _run_config(**{"container.gpu_count": 0})
    assert doctor.check_container_gpus(host, cpu).status == "ok"
    assert docker_calls == []


def test_doctor_selected_agent_cli_and_credentials(tmp_path) -> None:
    # a native run needs the selected cli on PATH
    host = _healthy_host(tmp_path / "cli", missing=frozenset({"cursor-agent"}))
    native = _run_config(**{"agent.name": "cursor", "container.enabled": "false"})
    result = doctor.check_selected_agent_cli(host, native)
    assert result.status == "fail"
    assert "cursor-agent" in result.detail
    container = _run_config(**{"agent.name": "cursor"})
    assert doctor.check_selected_agent_cli(host, container).status == "ok"

    # credentials use the agent table and env file names only
    host = _healthy_host(tmp_path / "creds")
    env_file = host.repo_root / ".env"
    env_file.write_text(
        "# comment\nexport GEMINI_API_KEY='gem-secret'\nCURSOR_API_KEY=\n", encoding="utf-8"
    )
    host.environ = {}
    gemini = doctor.check_selected_credentials(host, _run_config(**{"agent.name": "gemini"}))
    assert gemini.status == "ok"
    assert "GEMINI_API_KEY" in gemini.detail and "gem-secret" not in gemini.detail
    cursor = doctor.check_selected_credentials(host, _run_config(**{"agent.name": "cursor"}))
    assert cursor.status == "warn"
    assert "CURSOR_API_KEY" in cursor.hint
    native_gemini = doctor.check_selected_credentials(
        host, _run_config(**{"agent.name": "gemini", "container.enabled": "false"})
    )
    assert native_gemini.status == "warn", "a native run never loads the env file"
    host.environ = {"CLAUDE_CODE_OAUTH_TOKEN": "x"}
    claude = doctor.check_selected_credentials(host, _run_config(**{"agent.name": "claude"}))
    assert claude.status == "ok" and "CLAUDE_CODE_OAUTH_TOKEN" in claude.detail
    pi = doctor.check_selected_credentials(host, _run_config(**{"agent.name": "pi"}))
    assert pi.status == "warn" and "provider-dependent" in pi.detail

    # codex accepts a subscription login directory
    host = _healthy_host(tmp_path / "codex")
    host.environ = {}
    auth_root = tmp_path / "codex"
    cfg = _run_config(
        **{"agent.name": "codex", "container.env_file": "", "container.auth_root": str(auth_root)}
    )
    missing = doctor.check_selected_credentials(host, cfg)
    assert missing.status == "warn" and "codex login" in missing.hint
    (auth_root / ".codex").mkdir()
    assert doctor.check_selected_credentials(host, cfg).status == "ok"
    assert doctor.check_auth_root(cfg).status == "ok"
    gone = _run_config(**{"container.auth_root": str(auth_root / "absent")})
    assert doctor.check_auth_root(gone).status == "fail"


def test_doctor_task_scaffold_and_preset_checks(tmp_path) -> None:
    # the task scaffold check distinguishes ready from drafted
    host = _healthy_host(tmp_path)
    drafted = doctor.check_task_scaffold(host, _run_config(**{"task.dir": "examples/circpack"}))
    assert drafted.status == "ok" and "will be drafted" in drafted.detail
    scaffold = host.repo_root / "examples" / "circpack"
    (scaffold / "capture.py").write_text("", encoding="utf-8")
    (scaffold / "evaluate.py").write_text("", encoding="utf-8")
    (scaffold / "program_0" / "src").mkdir(parents=True)
    shutil.copy(
        Path("examples/circpack/program_0/program_spec.toml"),
        scaffold / "program_0" / "program_spec.toml",
    )
    (scaffold / "program_0" / "src" / "program.py").write_text("", encoding="utf-8")
    ready = doctor.check_task_scaffold(host, _run_config(**{"task.dir": "examples/circpack"}))
    assert ready.status == "ok" and "scaffold ready" in ready.detail
    absent = doctor.check_task_scaffold(host, _run_config(**{"task.dir": "examples/nope"}))
    assert absent.status == "fail"

    # presets without an example task fail
    (host.repo_root / "configs" / "preset" / "orphan.yaml").write_text("", encoding="utf-8")
    result = doctor.check_presets(host)
    assert result.status == "fail"
    assert "orphan" in result.detail

    # the real pipeline config and presets pass
    real_host = doctor.Host()
    assert doctor.check_pipeline_config(real_host).status == "ok"
    presets = doctor.check_presets(real_host)
    assert presets.status == "ok", presets.detail
    assert "grpo" in presets.detail


# --- inspect ------------------------------------------------------------------


class _FakeBackend:
    def __init__(self) -> None:
        self.pid = 4242
        self.terminated = False

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        raise AssertionError("terminate should suffice")


class _Inspector:
    """An InspectorHost with fake npm/python plus the recorded (argv, cwd) calls and backends."""

    def __init__(self, tmp_path: Path) -> None:
        self.calls: list[tuple[list[str], Path | None]] = []
        self.backends: list[_FakeBackend] = []
        tmp_path.mkdir(parents=True, exist_ok=True)
        frontend = tmp_path / "frontend"
        frontend.mkdir()
        (frontend / "package-lock.json").write_text("{}", encoding="utf-8")
        launcher = tmp_path / "launcher.py"
        launcher.write_text("", encoding="utf-8")

        def run(argv: list[str], cwd: Path | None = None, **_kwargs: object) -> subprocess.CompletedProcess:
            self.calls.append((argv, cwd))
            return subprocess.CompletedProcess(argv, 0)

        def popen(argv: list[str], cwd: Path | None = None, **_kwargs: object) -> _FakeBackend:
            self.calls.append((["popen", *argv], cwd))
            self.backends.append(_FakeBackend())
            return self.backends[-1]

        self.host = inspector.InspectorHost(
            run=run,
            popen=popen,  # type: ignore[arg-type]
            which=lambda name: "/usr/bin/npm" if name == "npm" else None,
            python="/venv/bin/python",
            probe=lambda url: True,
            open_browser=lambda url: self.calls.append((["open", url], None)),
            sleep=lambda seconds: None,
            frontend_dir=frontend,
            launcher_path=launcher,
            repo_root=tmp_path,
        )


def test_inspect_build_installs_locked_dependencies_reuses_node_modules_and_requires_npm(tmp_path, capsys) -> None:
    # build installs locked dependencies then builds
    insp = _Inspector(tmp_path / "fresh")
    assert inspector.main(["build"], host=insp.host) == 0
    assert insp.calls == [
        (["npm", "ci"], insp.host.frontend_dir),
        (["npm", "run", "build"], insp.host.frontend_dir),
    ]

    # installed node_modules are reused
    insp = _Inspector(tmp_path / "installed")
    (insp.host.frontend_dir / "node_modules").mkdir()
    assert inspector.main(["build"], host=insp.host) == 0
    assert insp.calls == [(["npm", "run", "build"], insp.host.frontend_dir)]

    # npm is required
    insp = _Inspector(tmp_path / "no-npm")
    insp.host.which = lambda name: None
    assert inspector.main(["build"], host=insp.host) == 1
    assert "npm is required" in capsys.readouterr().err
    assert insp.calls == []


def test_inspect_start_builds_serves_opens_and_honours_no_build_and_no_open(tmp_path, capsys) -> None:
    # start builds, serves and opens the page
    root = tmp_path / "start"
    insp = _Inspector(root)
    host = insp.host
    assert inspector.main(["start", "--backend-port", "9000", "--results-root", "r"], host=host) == 0
    for thread in host.threads:
        thread.join(timeout=5)
    backend = next(call for call in insp.calls if call[0][0] == "/venv/bin/python")
    assert backend[1] == root
    assert backend[0] == [
        "/venv/bin/python",
        str(host.launcher_path),
        "--results-root",
        "r",
        "--host",
        "127.0.0.1",
        "--port",
        "9000",
        "--agentlens-base-url",
        "http://127.0.0.1:8787",
    ]
    assert (["open", "http://127.0.0.1:9000/"], None) in insp.calls

    # --no-build serves an existing build only
    insp = _Inspector(tmp_path / "no-build")
    host = insp.host
    assert inspector.main(["start", "--no-build", "--no-open"], host=host) == 1
    assert "agsci inspect build" in capsys.readouterr().err
    assert insp.calls == []
    (host.frontend_dir / "dist").mkdir()
    (host.frontend_dir / "dist" / "index.html").write_text("", encoding="utf-8")
    assert inspector.main(["start", "--no-build", "--no-open"], host=host) == 0
    assert [argv[1] for argv, _ in insp.calls] == [str(host.launcher_path)]

    # --no-open never opens
    insp = _Inspector(tmp_path / "no-open")
    assert inspector.main(["start", "--no-open"], host=insp.host) == 0
    assert insp.host.threads == []
    assert not any(argv[0] == "open" for argv, _ in insp.calls)


def test_inspect_dev_runs_reloading_backend_beside_vite_and_rejects_unknown_mode(tmp_path) -> None:
    # dev runs the reloading backend beside vite and stops it
    insp = _Inspector(tmp_path / "dev")
    assert inspector.main(["dev", "--host", "0.0.0.0", "--frontend-port", "5174"], host=insp.host) == 0
    popen_argv = insp.calls[1][0]
    assert popen_argv[0] == "popen"
    assert popen_argv[-2:] == ["--dev", "--reload"]
    assert "--host" in popen_argv and "0.0.0.0" in popen_argv
    assert insp.calls[2] == (
        ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5174", "--open"],
        insp.host.frontend_dir,
    )
    assert insp.backends[0].terminated

    # an unknown mode is rejected
    with pytest.raises(SystemExit) as excinfo:
        inspector.parse_args(["serve"])
    assert excinfo.value.code == 2


# --- review -------------------------------------------------------------------


def test_review_validates_inputs_and_runs_the_reviewer_in_a_restricted_environment(
    tmp_path, capsys
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    calls: list[tuple[list[str], dict[str, str]]] = []

    def runner(command: list[str], environment: dict[str, str]) -> int:
        calls.append((command, environment))
        return 0

    base_env = {"HOME": "/home/user", "ANTHROPIC_API_KEY": "anthropic-secret"}
    common = dict(environ=base_env, which=lambda name: "/usr/bin/uv", runner=runner, repo_root=repo_root)

    # a missing PDF is a usage error
    assert review.main([str(tmp_path / "missing.pdf")], **common) == 2
    assert "not a regular file" in capsys.readouterr().err
    assert calls == []

    # without OPENAI_API_KEY anywhere the command fails before touching uv
    assert review.main([str(pdf)], **common) == 1
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert "anthropic-secret" not in err
    assert calls == []

    # uv is required once the credential is available
    (repo_root / ".env").write_text(
        "OPENAI_API_KEY=file-openai-secret\nANTHROPIC_API_KEY=file-anthropic-secret\n",
        encoding="utf-8",
    )
    assert review.main([str(pdf)], **{**common, "which": lambda name: None}) == 1
    err = capsys.readouterr().err
    assert "uv" in err
    assert "file-openai-secret" not in err
    assert calls == []

    # happy path: the exact uv invocation with a restricted environment from the repo .env
    out = tmp_path / "review.json"
    assert review.main([str(pdf), "--output", str(out)], **common) == 0
    command, environment = calls[-1]
    assert command == [
        "/usr/bin/uv",
        "run",
        "--script",
        str(repo_root / "src" / "aireview.py"),
        str(pdf),
        "--output",
        str(out),
    ]
    assert environment["OPENAI_API_KEY"] == "file-openai-secret"
    assert environment["HOME"] == "/home/user"
    assert "ANTHROPIC_API_KEY" not in environment

    # --output is optional and --env-file overrides the repo .env
    other_env = tmp_path / "other.env"
    other_env.write_text("OPENAI_API_KEY=other-secret\n", encoding="utf-8")
    assert review.main([str(pdf), "--env-file", str(other_env)], **common) == 0
    command, environment = calls[-1]
    assert command == ["/usr/bin/uv", "run", "--script", str(repo_root / "src" / "aireview.py"), str(pdf)]
    assert environment["OPENAI_API_KEY"] == "other-secret"

    # a host OPENAI_API_KEY wins over the env file and the runner's exit code is returned
    host_env = {**base_env, "OPENAI_API_KEY": "host-secret"}
    assert review.main([str(pdf)], **{**common, "environ": host_env, "runner": lambda c, e: 4}) == 4
    assert review.main([str(pdf), "--env-file", str(tmp_path / "nope.env")], **common) == 2
    assert "env file not found" in capsys.readouterr().err

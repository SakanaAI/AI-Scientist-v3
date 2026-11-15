"""``agsci doctor``: report whether this host can run the pipeline.

Two layers of checks share one report:

* host inventory (interpreter, tools, Docker, GPUs, agent CLIs, presets,
  temporary storage), which never depends on configuration;
* resolved-run checks, computed from the same arguments ``agsci run`` takes
  (``agsci doctor examples/<task> preset=<name>``): the selected
  agent's CLI and credential, the configured image, auth root and env file,
  the GPU count the preset demands, whether the image sees those GPUs, and
  whether the task scaffold is ready or will be drafted.

Every probe reads the machine through an injectable :class:`Host`, so the
command is testable without Docker, GPUs, or the network. Checks never contact
remote services and never print secret values, only variable names.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.investigation.config import VALID_AGENTS
from src.investigation.container_gpus import (
    GpuSelectorError,
    bounded_docker_gpu_selector,
    gpu_pool_label,
    selector_devices,
    validate_gpu_allocation,
)
from src.investigation.constants import (
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_CONTAINER_IMAGE_BUILD_HINT,
    DEFAULT_CONTAINER_IMAGE_BUILD_HINT_HOST_NET,
)
from src.investigation.files import missing_task_scaffold_files
from src.investigation.runtime import docker_image_exists, probe_docker_available

if TYPE_CHECKING:
    from src.pipeline.config import PipelineConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
MINIMUM_PYTHON = (3, 11)
REQUIRED_TOOLS = ("git", "docker", "node", "npm", "npx")
# Agent name -> binary a native run expects on PATH.
AGENT_CLI_BINARIES: dict[str, str] = {
    "claude": "claude",
    "codex": "codex",
    "cursor": "cursor-agent",
    "opencode": "opencode",
    "gemini": "gemini",
    "pi": "pi",
}
# Agent name -> credential variables that satisfy it (any one is enough). An
# empty tuple means the requirement is provider-dependent and is not asserted.
AGENT_CREDENTIAL_ENV: dict[str, tuple[str, ...]] = {
    "claude": (
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
    ),
    "codex": ("CODEX_API_KEY", "OPENAI_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "cursor": ("CURSOR_API_KEY",),
    "opencode": (),
    "pi": (),
}
COMMAND_TIMEOUT_SECONDS = 30.0
# Starting a GPU container cold can take well over the plain command timeout.
CONTAINER_GPU_TIMEOUT_SECONDS = 180.0
NVIDIA_SMI_UUID_ARGS = ("nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits")
OK, WARN, FAIL = "ok", "warn", "fail"

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: str
    detail: str
    hint: str = ""


@dataclass(slots=True)
class Host:
    """Everything a check reads from the machine, replaceable in tests."""

    which: Callable[[str], str | None] = shutil.which
    run: CommandRunner = subprocess.run
    probe_docker: Callable[[], tuple[bool, str]] = probe_docker_available
    image_exists: Callable[[str], tuple[bool, str]] = docker_image_exists
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    repo_root: Path = REPO_ROOT
    python_version: tuple[int, int, int] = sys.version_info[:3]
    python_executable: str = sys.executable


# --- host inventory -----------------------------------------------------------


def _command(host: Host, argv: list[str]) -> tuple[int, str] | str:
    """Run a command; return ``(returncode, first output line)`` or an error string."""
    try:
        completed = host.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return f"{argv[0]} is not installed"
    except subprocess.TimeoutExpired:
        return f"{argv[0]} timed out after {COMMAND_TIMEOUT_SECONDS:.0f}s"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"could not run {argv[0]}: {exc}"
    output = completed.stdout.strip() or completed.stderr.strip()
    first_line = output.splitlines()[0].strip() if output else ""
    return completed.returncode, first_line[:200]


def _version_detail(host: Host, name: str, path: str) -> str:
    result = _command(host, [name, "--version"])
    if isinstance(result, str):
        return f"{path} ({result})"
    returncode, line = result
    if returncode != 0 or not line:
        return path
    return f"{line} ({path})"


def check_python(host: Host) -> CheckResult:
    version = ".".join(str(part) for part in host.python_version)
    detail = f"{version} ({host.python_executable})"
    if host.python_version[:2] < MINIMUM_PYTHON:
        minimum = ".".join(str(part) for part in MINIMUM_PYTHON)
        return CheckResult("python", FAIL, detail, hint=f"need Python >= {minimum}; run uv sync")
    return CheckResult("python", OK, detail)


def check_tool(host: Host, name: str, *, required: bool, hint: str = "") -> CheckResult:
    path = host.which(name)
    if path is None:
        return CheckResult(name, FAIL if required else WARN, "not found on PATH", hint=hint)
    return CheckResult(name, OK, _version_detail(host, name, path))


def docker_optional_for_native_run(result: CheckResult) -> CheckResult:
    """A native run (container.enabled=false) never talks to Docker: its daemon check only warns."""
    if result.name not in {"docker", "docker daemon"} or result.status != FAIL:
        return result
    return CheckResult(
        result.name,
        WARN,
        f"{result.detail} (not needed: container.enabled=false)",
        hint=result.hint,
    )


def check_docker_daemon(host: Host) -> CheckResult:
    if host.which("docker") is None:
        return CheckResult("docker daemon", FAIL, "docker CLI is not installed")
    available, detail = host.probe_docker()
    if not available:
        return CheckResult(
            "docker daemon",
            FAIL,
            detail.splitlines()[0][:200] if detail else "docker info failed",
            hint="start the Docker daemon; verify with: docker info",
        )
    return CheckResult("docker daemon", OK, "reachable")


def _image_build_hint(image: str, network: str = "") -> str:
    if image != DEFAULT_CONTAINER_IMAGE:
        return f"build or pull {image}"
    if network.strip() == "host":
        return DEFAULT_CONTAINER_IMAGE_BUILD_HINT_HOST_NET
    return DEFAULT_CONTAINER_IMAGE_BUILD_HINT


def check_agent_image(host: Host, image: str, *, network: str = "", required: bool) -> CheckResult:
    name = "agent image"
    if host.which("docker") is None:
        return CheckResult(name, WARN, f"{image}: docker CLI is not installed")
    exists, detail = host.image_exists(image)
    if not exists:
        return CheckResult(
            name,
            FAIL if required else WARN,
            f"{image} is not available locally",
            hint=_image_build_hint(image, network),
        )
    return CheckResult(name, OK, image)


def visible_gpus(host: Host) -> list[str] | str:
    """UUIDs ``nvidia-smi`` reports, or a diagnostic string when it cannot answer."""
    if host.which("nvidia-smi") is None:
        return "nvidia-smi not found"
    argv = list(NVIDIA_SMI_UUID_ARGS)
    try:
        completed = host.run(
            argv, capture_output=True, text=True, check=False, timeout=COMMAND_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        return f"nvidia-smi timed out after {COMMAND_TIMEOUT_SECONDS:.0f}s"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"could not run nvidia-smi: {exc}"
    if completed.returncode != 0:
        diagnostic = (completed.stderr.strip() or completed.stdout.strip()).splitlines()
        return f"nvidia-smi failed: {diagnostic[0][:200] if diagnostic else 'no output'}"
    return [entry.strip() for entry in completed.stdout.splitlines() if entry.strip()]


def check_gpus(host: Host) -> CheckResult:
    name = "gpus"
    selector = host.environ.get("AGENT_CONTAINER_GPUS")
    devices = visible_gpus(host)
    if isinstance(devices, str):
        return CheckResult(
            name, WARN, devices, hint="only a CPU-only preset can run without GPUs"
        )
    if not devices:
        return CheckResult(name, WARN, "nvidia-smi reports no GPUs")
    detail = f"{len(devices)} visible"
    if selector:
        detail = f"{detail}; AGENT_CONTAINER_GPUS selects {selector.count(',') + 1}"
    return CheckResult(name, OK, detail)


def check_pipeline_config(host: Host) -> CheckResult:
    try:
        from src.pipeline.cli import config_error_message, load_pipeline_config

        load_pipeline_config([])
    except Exception as exc:  # Hydra raises many types; any of them means "broken config".
        return CheckResult("pipeline config", FAIL, config_error_message(exc)[:300])
    return CheckResult("pipeline config", OK, "configs/pipeline_config.yaml composes")


def check_presets(host: Host) -> CheckResult:
    preset_dir = host.repo_root / "configs" / "preset"
    names = sorted(path.stem for path in preset_dir.glob("*.yaml") if path.stem != "paper")
    if not names:
        return CheckResult("task presets", FAIL, f"no task presets under {preset_dir}")
    missing = [
        name for name in names if not (host.repo_root / "examples" / name / "task.md").is_file()
    ]
    if missing:
        return CheckResult(
            "task presets", FAIL, "presets without examples/<task>/task.md: " + ", ".join(missing)
        )
    return CheckResult("task presets", OK, ", ".join(names))


def check_temp_storage(host: Host) -> CheckResult:
    try:
        from src.investigation.temp_storage import preview_temp_storage

        detail = preview_temp_storage()
    except (OSError, ValueError) as exc:
        return CheckResult(
            "temporary storage",
            FAIL,
            str(exc)[:300],
            hint="point AGENTIC_TEMP_ROOT at an existing writable directory",
        )
    return CheckResult("temporary storage", OK, detail.removeprefix("temporary storage "))


def check_inspector_frontend(host: Host) -> CheckResult:
    dist = host.repo_root / "apps" / "run_inspector" / "frontend" / "dist" / "index.html"
    if dist.is_file():
        return CheckResult("inspector frontend", OK, "built")
    return CheckResult("inspector frontend", WARN, "not built", hint="run: agsci inspect build")


def host_checks(host: Host) -> list[CheckResult]:
    results = [check_python(host)]
    results.extend(check_tool(host, name, required=True) for name in REQUIRED_TOOLS)
    results.append(check_docker_daemon(host))
    results.append(check_gpus(host))
    results.extend(
        check_tool(host, binary, required=False, hint="see README 'Install CLI agents'")
        for binary in AGENT_CLI_BINARIES.values()
    )
    results.append(
        check_tool(
            host,
            "autoillustrate",
            required=False,
            hint="optional on the host; the paper stage uses the agent image's copy",
        )
    )
    results.append(check_pipeline_config(host))
    results.append(check_presets(host))
    results.append(check_temp_storage(host))
    results.append(check_inspector_frontend(host))
    return results


# --- resolved-run checks ------------------------------------------------------


def _dotenv_names(path: Path) -> set[str]:
    """Variable names with a non-empty value in a ``.env`` file; never the values."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    names: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            names.add(name.removeprefix("export ").strip())
    return names


def _env_file_path(host: Host, cfg: PipelineConfig) -> Path | None:
    if not cfg.container.enabled:
        return None
    raw = str(cfg.container.env_file or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else host.repo_root / path


def check_run_summary(cfg: PipelineConfig) -> CheckResult:
    mode = "container" if cfg.container.enabled else "native"
    detail = (
        f"agent={cfg.agent.name} mode={mode} phases={cfg.execution.phases} "
        f"gpu_count={cfg.container.gpu_count}"
    )
    return CheckResult("resolved run", OK, detail)


def check_selected_agent_cli(host: Host, cfg: PipelineConfig) -> CheckResult:
    agent = cfg.agent.name
    if agent not in VALID_AGENTS:
        return CheckResult(
            "agent cli", FAIL, f"unknown agent {agent!r}", hint=f"valid: {', '.join(VALID_AGENTS)}"
        )
    binary = AGENT_CLI_BINARIES[agent]
    if cfg.container.enabled:
        return CheckResult("agent cli", OK, f"{agent} ({binary}) comes from the agent image")
    path = host.which(binary)
    if path is None:
        return CheckResult(
            "agent cli",
            FAIL,
            f"{agent} needs {binary!r} on PATH for a native run",
            hint=f"install the {agent} CLI or set container.enabled=true",
        )
    return CheckResult("agent cli", OK, f"{agent}: {path}")


def check_selected_credentials(host: Host, cfg: PipelineConfig) -> CheckResult:
    agent = cfg.agent.name
    names = AGENT_CREDENTIAL_ENV.get(agent)
    if names is None:
        return CheckResult("credentials", WARN, f"unknown agent {agent!r}; skipped")
    env_file = _env_file_path(host, cfg)
    available = {name for name in names if host.environ.get(name)}
    if env_file is not None and env_file.is_file():
        available |= _dotenv_names(env_file) & set(names)
    if available:
        found = ", ".join(name for name in names if name in available)
        return CheckResult("credentials", OK, f"{agent}: {found}")
    if agent == "codex":
        auth_dir = Path(str(cfg.container.auth_root)).expanduser() / ".codex"
        if auth_dir.is_dir():
            return CheckResult("credentials", OK, f"codex: subscription login under {auth_dir}")
        return CheckResult(
            "credentials",
            WARN,
            "codex: no API key and no ~/.codex login",
            hint="run `codex login`, or set CODEX_API_KEY or OPENAI_API_KEY",
        )
    if not names:
        return CheckResult(
            "credentials", WARN, f"{agent}: provider-dependent; verify its credentials manually"
        )
    source = "environment or env file" if env_file is not None else "environment"
    return CheckResult(
        "credentials",
        WARN,
        f"{agent}: nothing set in the {source}",
        hint="set one of: " + ", ".join(names),
    )


def check_auth_root(cfg: PipelineConfig) -> CheckResult:
    path = Path(str(cfg.container.auth_root)).expanduser()
    if path.is_dir():
        return CheckResult("auth root", OK, str(path))
    return CheckResult(
        "auth root",
        FAIL,
        f"{path} is not a directory",
        hint="set container.auth_root to an existing directory (default: ~)",
    )


def check_env_file(host: Host, cfg: PipelineConfig) -> CheckResult:
    path = _env_file_path(host, cfg)
    if path is None:
        return CheckResult("env file", OK, "none configured")
    if path.is_file():
        return CheckResult("env file", OK, str(path))
    return CheckResult(
        "env file",
        WARN,
        f"{path} does not exist",
        hint="create it with the provider keys, or set container.env_file=''",
    )


def _gpu_pool_shape(cfg: PipelineConfig) -> tuple[int, int]:
    """``(gpu_count, workers)``: workers multiply the need only when investigation runs."""
    from src.pipeline.config import parse_pipeline_phases

    try:
        selected = parse_pipeline_phases(cfg.execution.phases)
    except ValueError:
        selected = set()
    workers = cfg.investigation.workers if "investigation" in selected else 1
    return cfg.container.gpu_count, max(1, workers)


def _gpu_need(gpu_count: int, workers: int) -> str:
    if workers == 1:
        return str(gpu_count)
    return f"{gpu_count * workers} ({gpu_count} x {workers} workers)"


def check_required_gpus(host: Host, cfg: PipelineConfig) -> CheckResult:
    gpu_count, workers = _gpu_pool_shape(cfg)
    required = gpu_count * workers
    need = _gpu_need(gpu_count, workers)
    selector = host.environ.get("AGENT_CONTAINER_GPUS")
    if required <= 0:
        if selector:
            return CheckResult(
                "required gpus",
                FAIL,
                "task needs none but AGENT_CONTAINER_GPUS is set",
                hint="unset AGENT_CONTAINER_GPUS for a CPU task",
            )
        return CheckResult("required gpus", OK, "none")
    if selector:
        try:
            selected = len(selector_devices(selector))
        except GpuSelectorError as exc:
            return CheckResult(
                "required gpus",
                FAIL,
                f"AGENT_CONTAINER_GPUS is malformed: {exc}",
                hint="use AGENT_CONTAINER_GPUS=device=<UUID,...>, or unset it to discover via nvidia-smi",
            )
        if selected != required:
            return CheckResult(
                "required gpus",
                FAIL,
                f"task needs {need} but AGENT_CONTAINER_GPUS selects {selected}",
                hint="name exactly that many GPU UUIDs, or unset it to discover via nvidia-smi",
            )
        return CheckResult("required gpus", OK, f"{need} selected by AGENT_CONTAINER_GPUS")
    devices = visible_gpus(host)
    if isinstance(devices, str):
        return CheckResult("required gpus", FAIL, f"task needs {need}; {devices}")
    if len(devices) < required:
        return CheckResult(
            "required gpus", FAIL, f"task needs {need} but nvidia-smi reports {len(devices)}"
        )
    return CheckResult("required gpus", OK, f"{need} of {len(devices)} visible")


def check_container_gpus(host: Host, cfg: PipelineConfig) -> CheckResult:
    """Run ``nvidia-smi`` inside the agent image and compare UUIDs with the host.

    This is the only check that proves the NVIDIA container runtime hands the
    selected GPUs to Docker. It is skipped, with a warning, when an earlier row
    already explains why it cannot run (no Docker, no image, too few GPUs).
    """
    name = "container gpus"
    gpu_count, workers = _gpu_pool_shape(cfg)
    required = gpu_count * workers
    if required <= 0:
        return CheckResult(name, OK, "not needed")
    image = cfg.container.image
    if host.which("docker") is None:
        return CheckResult(name, WARN, "skipped: docker CLI is not installed")
    exists, _detail = host.image_exists(image)
    if not exists:
        return CheckResult(name, WARN, f"skipped: {image} is not available locally")
    selector = host.environ.get("AGENT_CONTAINER_GPUS")
    if not selector:
        devices = visible_gpus(host)
        if isinstance(devices, str):
            return CheckResult(name, WARN, f"skipped: {devices}")
        if len(devices) < required:
            return CheckResult(
                name,
                WARN,
                f"skipped: task needs {_gpu_need(gpu_count, workers)} but nvidia-smi reports "
                f"{len(devices)}",
            )
        selector = "device=" + ",".join(devices[:required])
    try:
        expected = validate_gpu_allocation(
            selector, required, label=gpu_pool_label(gpu_count, workers)
        )
        docker_selector = bounded_docker_gpu_selector(selector, required)
    except GpuSelectorError as exc:
        return CheckResult(name, FAIL, str(exc), hint="use AGENT_CONTAINER_GPUS=device=<UUID,...>")
    argv = ["docker", "run", "--rm", "--gpus", docker_selector, image, *NVIDIA_SMI_UUID_ARGS]
    try:
        completed = host.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=CONTAINER_GPU_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name, FAIL, f"docker run timed out after {CONTAINER_GPU_TIMEOUT_SECONDS:.0f}s"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(name, FAIL, f"could not run docker: {exc}")
    toolkit_hint = "install the NVIDIA Container Toolkit and restart the Docker daemon"
    if completed.returncode != 0:
        diagnostic = (completed.stderr.strip() or completed.stdout.strip()).splitlines()
        detail = diagnostic[0][:200] if diagnostic else "no output"
        return CheckResult(name, FAIL, f"docker run --gpus failed: {detail}", hint=toolkit_hint)
    reported = [entry.strip() for entry in completed.stdout.splitlines() if entry.strip()]
    if sorted(reported) != sorted(expected):
        return CheckResult(
            name,
            FAIL,
            f"container sees {len(reported)} GPU(s), expected {len(expected)} with matching UUIDs",
            hint=toolkit_hint,
        )
    return CheckResult(name, OK, f"{len(expected)} visible inside {image}")


def check_task_scaffold(host: Host, cfg: PipelineConfig) -> CheckResult:
    task_dir = Path(cfg.task.dir).expanduser()
    if not task_dir.is_absolute():
        task_dir = host.repo_root / task_dir
    if not task_dir.is_dir():
        return CheckResult("task", FAIL, f"{task_dir} is not a directory")
    if not (task_dir / "task.md").is_file():
        return CheckResult("task", FAIL, f"{task_dir} has no task.md")
    missing = missing_task_scaffold_files(task_dir)
    if not missing:
        return CheckResult("task", OK, f"{task_dir.name}: scaffold ready")
    names = ", ".join(str(path.relative_to(task_dir)) for path in missing)
    return CheckResult("task", OK, f"{task_dir.name}: scaffold will be drafted (missing {names})")


def _paper_selected(cfg: PipelineConfig) -> bool:
    from src.pipeline.config import parse_pipeline_phases

    try:
        return "paper" in parse_pipeline_phases(cfg.execution.phases)
    except ValueError:
        return False  # the run itself reports the malformed phase list


def uses_docker(cfg: PipelineConfig) -> bool:
    """Whether any selected phase runs in Docker (a native run may still containerize its paper)."""
    from src.pipeline.config import phase_uses_container

    return cfg.container.enabled or (_paper_selected(cfg) and phase_uses_container(cfg, "paper"))


def check_paper_container(cfg: PipelineConfig) -> CheckResult:
    from src.pipeline.config import PAPER_NEEDS_CONTAINER_ERROR, phase_uses_container

    if not _paper_selected(cfg):
        return CheckResult("paper container", OK, "paper phase not selected")
    if phase_uses_container(cfg, "paper"):
        return CheckResult("paper container", OK, "paper runs in Docker")
    return CheckResult(
        "paper container",
        FAIL,
        PAPER_NEEDS_CONTAINER_ERROR,
        hint="add container.paper_enabled=true (needs Docker and the agent image on this host)",
    )


def run_checks_for(host: Host, cfg: PipelineConfig) -> list[CheckResult]:
    results = [check_run_summary(cfg), check_paper_container(cfg)]
    if uses_docker(cfg):
        results.append(
            check_agent_image(host, cfg.container.image, network=cfg.container.network, required=True)
        )
        results.append(check_auth_root(cfg))
        results.append(check_env_file(host, cfg))
    results.append(check_selected_agent_cli(host, cfg))
    results.append(check_selected_credentials(host, cfg))
    results.append(check_required_gpus(host, cfg))
    if cfg.container.enabled:
        results.append(check_container_gpus(host, cfg))
    results.append(check_task_scaffold(host, cfg))
    return results


def load_run_config(arguments: list[str]) -> PipelineConfig | CheckResult:
    """Compose the config ``agsci run`` would use from the same arguments."""
    from src.pipeline.cli import (
        _normalize_logging_flags,
        _normalize_positional_task_dir,
        config_error_message,
        load_pipeline_config,
    )

    try:
        overrides = _normalize_positional_task_dir(_normalize_logging_flags(arguments))
        return load_pipeline_config(overrides)
    except Exception as exc:  # Hydra raises many types; report rather than crash.
        return CheckResult(
            "resolved run",
            FAIL,
            config_error_message(exc)[:300],
            hint="pass the task directory and key=value overrides exactly as to agsci run",
        )


# --- reporting ----------------------------------------------------------------


def render(results: list[CheckResult]) -> str:
    width = max(len(result.name) for result in results)
    labels = {OK: "ok  ", WARN: "WARN", FAIL: "FAIL"}
    lines = []
    for result in results:
        lines.append(f"  {labels[result.status]}  {result.name.ljust(width)}  {result.detail}")
        if result.hint:
            lines.append(f"  {' ' * (width + 6)}  -> {result.hint}")
    counts = {status: sum(1 for r in results if r.status == status) for status in (OK, WARN, FAIL)}
    summary = (
        f"{counts[OK]} ok, {counts[WARN]} warning{'s' if counts[WARN] != 1 else ''}, "
        f"{counts[FAIL]} failure{'s' if counts[FAIL] != 1 else ''}"
    )
    return "\n".join(["agsci doctor", *lines, "", summary])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agsci doctor",
        description=(
            "Check this host for the tools, images, and GPUs the pipeline needs. "
            "Pass a task directory and key=value overrides exactly as to `agsci run` "
            "to also check the resolved run (selected agent, image, credentials, GPUs, task)."
        ),
    )
    parser.add_argument("--json", action="store_true", help="print the checks as JSON")
    parser.add_argument(
        "run_arguments",
        nargs="*",
        metavar="run-argument",
        help="task directory and Hydra overrides, as for agsci run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, host: Host | None = None) -> int:
    """Print the report; exit 1 when any required check fails."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    host = host or Host()
    results = host_checks(host)
    if args.run_arguments:
        cfg = load_run_config(args.run_arguments)
        if isinstance(cfg, CheckResult):
            results.append(cfg)
        else:
            if not uses_docker(cfg):
                results = [docker_optional_for_native_run(result) for result in results]
            results.extend(run_checks_for(host, cfg))
    else:
        results.append(check_agent_image(host, DEFAULT_CONTAINER_IMAGE, required=False))
    failed = any(result.status == FAIL for result in results)
    if args.json:
        payload = {
            "ok": not failed,
            "checks": [
                {"name": r.name, "status": r.status, "detail": r.detail, "hint": r.hint}
                for r in results
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render(results))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

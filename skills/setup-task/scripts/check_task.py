"""Check an AI-Scientist-v3 task directory, and optionally run its evaluator like the harness does.

    python check_task.py <task_dir>                                   # static checks
    python check_task.py <task_dir> --run                             # + evaluate program_0 via the harness wrapper
    python check_task.py <task_dir> --run --profile smoke             # + evaluate.py directly with --profile
    python check_task.py <task_dir> --run --program <dir> --expect incorrect

Without `--profile`, `--run` evaluates a scratch copy of the program through the harness's own
wrapper (`python -m src.investigation.evaluate_program`), exactly as agents do. With `--profile`,
it calls `evaluate.py --profile ...` directly, with the wrapper's filtered environment and a fresh
HOME and TMPDIR. Either way the evaluator runs in its own process group, which is stopped (SIGINT,
then SIGKILL) on timeout or interruption. Exit code 0 means every check passed.
"""

import argparse
import filecmp
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.investigation import evaluate_program as harness_eval  # noqa: E402
from src.investigation import harness_sources  # noqa: E402
from src.investigation.files import (  # noqa: E402
    load_task_copy_ignore_patterns,
    missing_task_scaffold_files,
    read_task_sections,
    task_copy_path_is_ignored,
)
from src.investigation.program_spec import ProgramSpecError, load_program_spec, resolve_entrypoint  # noqa: E402
from src.investigation.task_runtime import _PINNED_REQUIREMENT  # noqa: E402

CANONICAL_CAPTURE = REPO_ROOT / "examples" / "circpack" / "capture.py"
CONSOLE_WARN_BYTES = 64 * 1024
RESULTS_WARN_BYTES = 64 * 1024 * 1024
SNAPSHOT_WARN_BYTES = 1 << 30
STOP_GRACE_SECONDS = 35


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def ok(self, message: str) -> None:
        print(f"  ok    {message}")

    def warn(self, message: str) -> None:
        print(f"  WARN  {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"  FAIL  {message}")


# --- static checks ---------------------------------------------------------------------


def check_required_files(task_dir: Path, report: Report) -> None:
    missing = missing_task_scaffold_files(task_dir)
    if missing:
        report.fail("missing or invalid: " + ", ".join(str(p.relative_to(task_dir)) for p in missing))
    else:
        report.ok("required files present; program_0 entrypoint resolves")
    if (task_dir / "task_contract.json").exists():
        report.warn("task_contract.json present: contract mode, which this checker does not validate")


def check_task_md(task_dir: Path, report: Report) -> None:
    goal, rubrics, _ = read_task_sections(task_dir)
    for name, body in (("Goal", goal), ("Rubrics", rubrics)):
        if body.startswith("(missing"):
            report.fail(f"task.md has no '# {name}' section")
        elif "TODO" in body:
            report.warn(f"task.md '# {name}' still contains TODO")
        else:
            report.ok(f"task.md '# {name}' section")


def check_capture(task_dir: Path, report: Report) -> None:
    capture = task_dir / "capture.py"
    if capture.is_file() and not filecmp.cmp(capture, CANONICAL_CAPTURE, shallow=False):
        report.warn("capture.py differs from the canonical examples/circpack/capture.py")


def check_evaluator_source(task_dir: Path, report: Report) -> None:
    evaluator = task_dir / "evaluate.py"
    if not evaluator.is_file():
        return
    source = evaluator.read_text(encoding="utf-8", errors="replace")
    for flag in ("--program_path", "--results_dir"):
        if flag not in source:
            report.fail(f"evaluate.py does not mention {flag}")
    if "--num_replications" not in source:
        report.warn("evaluate.py does not accept --num_replications (the harness passes it when > 1)")
    if "TODO" in source:
        report.warn("evaluate.py still contains TODO")


def check_requirements(task_dir: Path, report: Report) -> None:
    requirements = task_dir / "requirements.txt"
    if not requirements.is_file():
        return
    bad = [
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and not _PINNED_REQUIREMENT.fullmatch(line.strip())
    ]
    if bad:
        report.fail(f"requirements.txt allows only pinned name==version lines: {bad}")
    else:
        report.ok("requirements.txt pinned")


def snapshot_size(task_dir: Path) -> int:
    patterns = load_task_copy_ignore_patterns(task_dir)
    total = 0
    for root, dirs, files in os.walk(task_dir):
        rel_root = Path(root).relative_to(task_dir)
        dirs[:] = [d for d in dirs if not task_copy_path_is_ignored(rel_root / d, is_dir=True, patterns=patterns)]
        for name in files:
            if not task_copy_path_is_ignored(rel_root / name, is_dir=False, patterns=patterns):
                total += (Path(root) / name).lstat().st_size
    return total


def check_snapshot(task_dir: Path, report: Report) -> None:
    if (task_dir / ".venv").exists() and not (task_dir / ".agentic_task_ignore").is_file():
        report.warn(".venv exists but no .agentic_task_ignore excludes it from run snapshots")
    size = snapshot_size(task_dir)
    message = f"run snapshot size {size / 2**20:.1f} MiB"
    if size > SNAPSHOT_WARN_BYTES:
        report.warn(message + ": keep large data outside the task (see references/harness-contract.md)")
    else:
        report.ok(message)


def check_program_source_limits(task_dir: Path, report: Report) -> None:
    """Harness-evaluated attempts inspect every candidate source with these exact rules (size,
    entry count, depth, no symlinks); a program_0 that breaks them fails every such attempt."""
    try:
        harness_sources.read_candidate_sources(task_dir / "program_0")
    except SystemExit as exc:
        message = str(exc).replace("program_1", "program_0")
        report.warn(f"{message}: every harness-evaluated attempt (GPU tasks) would fail")
    else:
        report.ok("program_0 sources pass harness inspection")


def static_checks(task_dir: Path) -> Report:
    report = Report()
    print(f"static checks: {task_dir}")
    check_required_files(task_dir, report)
    check_task_md(task_dir, report)
    check_capture(task_dir, report)
    check_evaluator_source(task_dir, report)
    check_requirements(task_dir, report)
    check_snapshot(task_dir, report)
    check_program_source_limits(task_dir, report)
    return report


# --- evaluator run ---------------------------------------------------------------------


def harness_environment(home: Path, tmp: Path) -> dict[str, str]:
    """The variables `src.investigation.evaluate_program` lets through to evaluate.py."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in harness_eval._PYTHON_ENV_BLOCKLIST
        and not any(token in key.upper() for token in harness_eval._SENSITIVE_ENV_TOKENS)
        and (key in harness_eval._SAFE_ENV_NAMES or key.startswith(harness_eval._SAFE_ENV_PREFIXES))
    }
    env.update(
        HOME=str(home), TMPDIR=str(tmp), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
        PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
    )
    return env


def tree_state(task_dir: Path) -> dict[str, tuple[int, int]]:
    state = {}
    for root, dirs, files in os.walk(task_dir):
        dirs[:] = [d for d in dirs if d not in (".venv", "__pycache__")]
        for name in files:
            path = Path(root) / name
            stat = path.lstat()
            state[str(path.relative_to(task_dir))] = (stat.st_size, stat.st_mtime_ns)
    return state


def directory_bytes(path: Path) -> int:
    return sum(p.lstat().st_size for p in path.rglob("*") if p.is_file())


def stop_process_group(child: subprocess.Popen) -> None:
    """Ask like the harness does (SIGINT to the group), then force."""
    for signum, grace in ((signal.SIGINT, STOP_GRACE_SECONDS), (signal.SIGKILL, 10)):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def run_in_group(command: list[str], cwd: Path, env: dict[str, str] | None, console: Path, timeout: float) -> int | None:
    """Exit code, or None on timeout. The process group never outlives this call."""
    with open(console, "wb") as sink:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        finally:
            stop_process_group(child)


def validate_outputs(results: Path, expect: str, report: Report) -> None:
    try:
        metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
        correct = json.loads((results / "correct.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.fail(f"unreadable results: {exc}")
        return
    score = metrics.get("combined_score") if isinstance(metrics, dict) else None
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        report.fail(f"metrics.json combined_score must be a finite number, got {score!r}")
    else:
        report.ok(f"combined_score = {score}")
    verdict = correct.get("correct") if isinstance(correct, dict) else None
    if not isinstance(verdict, bool):
        report.fail(f"correct.json 'correct' must be a boolean, got {verdict!r}")
        return
    detail = f" ({correct['error']})" if correct.get("error") else ""
    if expect != "any" and verdict != (expect == "correct"):
        report.fail(f"correct = {verdict}{detail}, expected {expect}")
    else:
        report.ok(f"correct = {verdict}{detail}")


def evaluator_python(task_dir: Path, args: argparse.Namespace) -> str:
    task_python = task_dir / ".venv" / "bin" / "python"
    return str(args.python or (task_python if task_python.exists() else sys.executable))


def wrapper_command(task_dir: Path, program: Path, args: argparse.Namespace) -> tuple[list[str], Path]:
    """The agents' own evaluation command; results are published into the program dir."""
    command = [
        evaluator_python(task_dir, args), "-m", "src.investigation.evaluate_program",
        "--evaluator", str(task_dir / "evaluate.py"), "--program-dir", str(program),
        "--timeout-seconds", str(args.timeout), "--num-replications", str(args.num_replications),
    ]
    return command, program


def direct_command(task_dir: Path, program: Path, scratch: Path, args: argparse.Namespace) -> tuple[list[str], Path]:
    spec = load_program_spec(program, required=True)
    assert spec is not None  # required=True raises instead of returning None
    results = scratch / "results"
    results.mkdir()
    command = [
        evaluator_python(task_dir, args), str(task_dir / "evaluate.py"),
        "--program_path", str(resolve_entrypoint(program, spec)), "--results_dir", str(results),
        "--profile", args.profile,
    ]
    if args.num_replications > 1:
        command += ["--num_replications", str(args.num_replications)]
    return command, results


def run_evaluator(task_dir: Path, args: argparse.Namespace) -> Report:
    report = Report()
    source_program = (args.program or task_dir / "program_0").resolve()
    mode = f"evaluate.py --profile {args.profile}" if args.profile else "harness wrapper"
    print(f"evaluator run: {source_program} via {mode}")
    scratch = Path(tempfile.mkdtemp(prefix="check-task-"))
    try:
        program = scratch / "programs" / "program_0"
        shutil.copytree(source_program, program, symlinks=True)
        home, tmp, console = scratch / "home", scratch / "tmp", scratch / "console.log"
        home.mkdir()
        tmp.mkdir()
        try:
            if args.profile:
                command, results = direct_command(task_dir, program, scratch, args)
                cwd, env = task_dir, harness_environment(home, tmp)
            else:
                command, results = wrapper_command(task_dir, program, args)
                cwd, env = REPO_ROOT, None
        except ProgramSpecError as exc:
            report.fail(f"program_spec.toml: {exc}")
            return report
        before = tree_state(task_dir)
        started = time.monotonic()
        returncode = run_in_group(command, cwd, env, console, args.timeout + STOP_GRACE_SECONDS)
        report_run(task_dir, before, console, results, returncode, time.monotonic() - started, args, report)
        return report
    finally:
        if args.keep:
            print(f"  kept scratch dir: {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


def report_run(task_dir: Path, before: dict[str, Any], console: Path, results: Path, returncode: int | None,
               seconds: float, args: argparse.Namespace, report: Report) -> None:
    output = console.read_bytes()
    sys.stdout.write(output[-4000:].decode("utf-8", errors="replace"))
    if returncode is None:
        report.fail(f"evaluator exceeded --timeout {args.timeout} s and was stopped")
        return
    report.ok(f"evaluator finished in {seconds:.1f} s with exit code {returncode}")
    if len(output) > CONSOLE_WARN_BYTES:
        report.warn(f"console output {len(output)} bytes: drafting, final, and contract runs cap it at 64 KiB")
    changed = sorted({path for path, _ in set(before.items()) ^ set(tree_state(task_dir).items())})
    if changed:
        report.fail("evaluator wrote into the task dir (read-only under the harness): " + ", ".join(changed[:10]))
    if returncode != 0:
        report.fail("non-zero exit: the harness treats this as an evaluator failure, not a bad candidate")
        return
    size = directory_bytes(results)
    if size > RESULTS_WARN_BYTES:
        report.warn(f"results are {size / 2**20:.0f} MiB: containerized harness evaluation rejects more than 64 MiB")
    validate_outputs(results, args.expect, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--run", action="store_true", help="also run the evaluator on a program")
    parser.add_argument("--program", type=Path, help="program dir to evaluate (default: program_0)")
    parser.add_argument("--profile", help="call evaluate.py directly with --profile (default: harness wrapper, default profile)")
    parser.add_argument("--num-replications", type=int, default=1)
    parser.add_argument("--expect", choices=("correct", "incorrect", "any"), default="any")
    parser.add_argument("--python", type=Path, help="interpreter for the evaluator (default: task .venv, else this one)")
    parser.add_argument("--timeout", type=float, default=21600, help="seconds before the evaluator is stopped")
    parser.add_argument("--keep", action="store_true", help="keep the scratch dir with results and logs")
    args = parser.parse_args()
    task_dir = args.task_dir.expanduser().resolve()
    if not task_dir.is_dir():
        print(f"not a directory: {task_dir}", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))  # unwind so cleanup runs
    failures = static_checks(task_dir).failures
    if args.run:
        failures += run_evaluator(task_dir, args).failures
    print("PASS" if failures == 0 else f"FAIL ({failures} problem{'s' if failures != 1 else ''})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

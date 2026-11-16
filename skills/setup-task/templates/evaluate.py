"""Search evaluator for this task.

Called by the AI-Scientist-v3 harness as

    python evaluate.py --program_path <program_k>/src/<entrypoint> --results_dir <dir> [--num_replications N]

It copies the candidate into scratch space (on top of a pinned upstream tree for overlay
tasks), checks the static rules, runs the program with a timeout, scores the artifacts it
produced with evaluator-owned code, and writes metrics.json and correct.json. A candidate
failure exits 0 with correct=false; a non-zero exit means the evaluator itself failed.

Edit every TODO. Keep the CLI, the output files, and the failure semantics.
"""

import argparse
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# --- Task settings (TODO) --------------------------------------------------------------
# Overlay tasks: a pinned upstream tree that the candidate's src/ is copied onto.
# Leave as None for single-file/single-tree tasks.
UPSTREAM_SUBDIR: str | None = None  # e.g. "resources/upstream"
# Seconds the candidate may run, per profile. About 2x the measured baseline time.
PROGRAM_TIMEOUT_SECONDS = {"smoke": 120, "search": 600, "full": 600}
# Score written when the candidate fails. Must be finite and worse than any real score.
WORST_SCORE = -1e9
# Static rules: regexes that must not appear in candidate Python sources (bypassable; back
# important rules with runtime checks or a judgment rubric).
FORBIDDEN_PATTERNS: dict[str, str] = {
    # "no_autograd": r"\.backward\(|torch\.autograd",
}
# The program log tail kept in --results_dir. `stdout.log` is never copied into later programs.
PROGRAM_LOG_NAME = "stdout.log"
PROGRAM_LOG_TAIL_BYTES = 4 * 1024 * 1024
CONSOLE_ECHO_BUDGET_BYTES = 24 * 1024
CONSOLE_LINE_BYTES = 400
CONSOLE_ECHO_PATTERN = re.compile(r"step|score|loss|Error|Traceback|Exception", re.IGNORECASE)
# Variables the candidate must not inherit when the evaluator runs outside the harness.
SECRET_ENV_TOKENS = ("AUTH", "CREDENTIAL", "KEY", "PASSWORD", "PROXY", "SECRET", "TOKEN")
DROPPED_ENV = {"PYTHONPATH", "PYTHONHOME", "SSH_AUTH_SOCK", "TASK_SOURCE_DIR", "DATA_DIR", "AGENTIC_DATA_DIR"}


class CandidateFailure(Exception):
    """The candidate broke a rule, crashed, or timed out: correct=false, evaluation completed."""


def task_dir() -> Path:
    source = os.environ.get("TASK_SOURCE_DIR", "").strip()
    return Path(source).resolve() if source else Path(__file__).resolve().parent


def code_root_of(entrypoint: Path) -> Path:
    """The candidate's `src` dir: next to the closest program_spec.toml above the entrypoint."""
    for parent in entrypoint.parents:
        if (parent / "program_spec.toml").is_file():
            return parent / "src"
    raise FileNotFoundError(f"no program_spec.toml above {entrypoint}")


def assemble_work_tree(code_root: Path, work: Path) -> None:
    """Build the directory the candidate runs in; the candidate's files win over upstream."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv")
    if UPSTREAM_SUBDIR:
        shutil.copytree(task_dir() / UPSTREAM_SUBDIR, work, ignore=ignore, symlinks=True)
        shutil.copytree(code_root, work, ignore=ignore, dirs_exist_ok=True)
    else:
        shutil.copytree(code_root, work, ignore=ignore)


def check_static_rules(code_root: Path) -> None:
    for path in sorted(code_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        for rule, pattern in FORBIDDEN_PATTERNS.items():
            if re.search(pattern, source):
                raise CandidateFailure(f"rule {rule}: {path.relative_to(code_root)} matches {pattern!r}")


def program_command(work: Path, entrypoint_rel: Path, profile: str, replication: int) -> list[str]:
    """TODO: the command that runs the candidate (e.g. torchrun ... for multi-GPU training).

    `replication` counts from 0; map it to a seed if the task has one.
    """
    return [sys.executable, "-u", str(entrypoint_rel), "--profile", profile]


def program_environment(scratch: Path) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if key not in DROPPED_ENV and not any(token in key.upper() for token in SECRET_ENV_TOKENS)
    }
    caches = scratch / "cache"
    home = scratch / "home"
    home.mkdir(exist_ok=True)
    env.update(
        HOME=str(home),
        PYTHONUNBUFFERED="1",
        XDG_CACHE_HOME=str(caches),
        TORCHINDUCTOR_CACHE_DIR=str(caches / "inductor"),
        TRITON_CACHE_DIR=str(caches / "triton"),
        HF_HOME=str(caches / "hf"),
    )
    return env


class ProgramRun:
    """One candidate process group: started in its own session, killed as a group on every exit path."""

    active: "ProgramRun | None" = None

    def __init__(self, command: list[str], work: Path, env: dict[str, str], log_path: Path) -> None:
        self.log_path = log_path
        self.child = subprocess.Popen(
            command, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ProgramRun.active = self
        self.reader = threading.Thread(target=self._stream_output, daemon=True)
        self.reader.start()

    def _stream_output(self) -> None:
        """Copy all output to the scratch log; echo progress and error lines within the console budget."""
        echoed = 0
        assert self.child.stdout is not None
        with open(self.log_path, "wb") as log:
            try:
                for raw in self.child.stdout:
                    log.write(raw)
                    line = raw.decode("utf-8", errors="replace")
                    if echoed < CONSOLE_ECHO_BUDGET_BYTES and CONSOLE_ECHO_PATTERN.search(line):
                        line = line.rstrip("\n")[:CONSOLE_LINE_BYTES]
                        print(line, flush=True)
                        echoed += len(line) + 1
            except (OSError, ValueError):
                pass  # the pipe was closed under us after a detached process kept it open

    def kill_group(self) -> None:
        try:
            os.killpg(self.child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def wait(self, timeout: float) -> int | None:
        """Exit code, or None on timeout. Always kills leftovers and stops waiting on their pipe."""
        try:
            return self.child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        finally:
            self.kill_group()
            self.reader.join(timeout=10)
            if self.child.stdout is not None:
                self.child.stdout.close()
            self.child.wait()
            ProgramRun.active = None


def stop_on_signal(signum: int, frame: Any) -> None:
    """The harness stops evaluators with SIGINT or SIGTERM: kill the candidate, then unwind (cleanup runs)."""
    if ProgramRun.active is not None:
        ProgramRun.active.kill_group()
    raise SystemExit(128 + signum)


def keep_log_tail(log_path: Path, results_dir: Path) -> None:
    if not log_path.is_file():
        return
    with open(log_path, "rb") as source:
        source.seek(max(0, log_path.stat().st_size - PROGRAM_LOG_TAIL_BYTES))
        (results_dir / PROGRAM_LOG_NAME).write_bytes(source.read())


def run_program(command: list[str], work: Path, env: dict[str, str], log_path: Path, timeout: float) -> float:
    started = time.monotonic()
    returncode = ProgramRun(command, work, env, log_path).wait(timeout)
    if returncode is None:
        raise CandidateFailure(f"timed out after {timeout:.0f} s")
    if returncode != 0:
        raise CandidateFailure(f"program exited with code {returncode}; see {PROGRAM_LOG_NAME}")
    return time.monotonic() - started


def score(work: Path, profile: str) -> dict[str, float]:
    """TODO: compute metrics from artifacts the candidate left in `work`, with evaluator-owned code.

    Return a flat dict with a finite `combined_score` (higher is better). Never trust a score the
    candidate printed. Raise CandidateFailure when an artifact is missing or invalid.
    """
    try:
        payload = json.loads((work / "solution.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CandidateFailure(f"unreadable solution.json: {exc.__class__.__name__}") from exc
    value = payload.get("value") if isinstance(payload, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CandidateFailure("solution.json value must be a finite number")
    return {"combined_score": -abs(float(value) - 42.0), "value": float(value)}


def run_replication(code_root: Path, entrypoint_rel: Path, scratch: Path, results_dir: Path,
                    profile: str, replication: int) -> dict[str, float]:
    work, log_path = scratch / f"work_{replication}", scratch / f"program_{replication}.log"
    assemble_work_tree(code_root, work)
    try:
        seconds = run_program(
            program_command(work, entrypoint_rel, profile, replication), work, program_environment(scratch),
            log_path, PROGRAM_TIMEOUT_SECONDS[profile],
        )
    finally:
        keep_log_tail(log_path, results_dir)
    metrics = score(work, profile)
    metrics["wall_seconds"] = round(seconds, 2)
    return metrics


def aggregate(runs: list[dict[str, float]]) -> dict[str, float]:
    """Mean of every metric across replications, plus the spread of combined_score."""
    metrics = {key: statistics.fmean(run[key] for run in runs) for key in runs[0]}
    metrics["combined_score_std"] = statistics.pstdev(run["combined_score"] for run in runs)
    metrics["num_replications"] = len(runs)
    return metrics


def write_results(results_dir: Path, metrics: dict[str, Any], error: str | None) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    correct: dict[str, Any] = {"correct": error is None}
    if error:
        correct["error"] = error[:500]
    (results_dir / "correct.json").write_text(json.dumps(correct, indent=2, sort_keys=True), encoding="utf-8")


def evaluate(program_path: Path, results_dir: Path, profile: str, replications: int) -> None:
    code_root = code_root_of(program_path.resolve())
    entrypoint_rel = program_path.resolve().relative_to(code_root)
    scratch = Path(tempfile.mkdtemp(prefix="task-eval-"))
    try:
        check_static_rules(code_root)
        runs = [run_replication(code_root, entrypoint_rel, scratch, results_dir, profile, i) for i in range(replications)]
        metrics = aggregate(runs)
        write_results(results_dir, metrics, None)
        print(f"[evaluate] combined_score={metrics['combined_score']:.6g} over {replications} run(s)", flush=True)
    except CandidateFailure as failure:
        print(f"[evaluate] candidate failed: {failure}", file=sys.stderr, flush=True)
        write_results(results_dir, {"combined_score": WORST_SCORE}, str(failure))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--program_path", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--num_replications", type=int, default=1)  # the harness passes it when > 1
    parser.add_argument("--profile", choices=("smoke", "search", "full"), default="search")
    args = parser.parse_args()
    if args.num_replications < 1:
        parser.error("--num_replications must be >= 1")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop_on_signal)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    evaluate(args.program_path, args.results_dir, args.profile, args.num_replications)
    return 0


if __name__ == "__main__":
    sys.exit(main())

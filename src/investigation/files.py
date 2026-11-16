"""File and JSON helpers used across wrapper orchestration."""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

from .constants import (
    EXPERIMENT_DIR_RE,
    IDEA_DIR_RE,
    MARKDOWN_HEADING_RE,
    PAPER_DIR_RE,
    PROGRAM_DIR_RE,
)
from .program_spec import ProgramSpecError, load_program_spec, resolve_entrypoint
from .task_data_cache import require_no_legacy_data_cache

TASK_COPY_IGNORE_FILENAME = ".agentic_task_ignore"


def confirm_prune_warning() -> None:
    """Print a prominent prune-without-container warning and wait for Enter.

    When stdin is not a TTY (e.g. CI or piped input) the prompt is skipped
    and a log message is emitted instead.
    """
    banner = (
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  WARNING: Pruning may be ineffective without containers!   ║\n"
        "║                                                            ║\n"
        "║  The agent is NOT containerized, so it will still have     ║\n"
        "║  access to archived history and idea files from previous   ║\n"
        "║  rounds/chapters. Pruning only resets the orchestrator     ║\n"
        "║  state — the agent can still read old files on disk.       ║\n"
        "║                                                            ║\n"
        "║  Use container.enabled=true for full round isolation.      ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )
    print(banner, file=sys.stderr, flush=True)

    if sys.stdin.isatty():
        print("Press Enter to continue or Ctrl+C to abort... ", file=sys.stderr, end="", flush=True)
        input()
    else:
        logger.info("Non-interactive session detected — continuing without confirmation.")


def utc_now() -> str:
    """Return the current UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: str | Path) -> Path:
    """Resolve path."""
    return Path(path).expanduser().resolve()


def relative_display_path(path: str | Path, *, base_dir: str | Path) -> str:
    """Render `path` relative to `base_dir` for prompt/skill text."""
    base = resolve_path(base_dir)
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = (base / target).resolve()
    else:
        target = target.resolve()

    rel = Path(os.path.relpath(target, base)).as_posix()
    if rel == ".":
        return "."
    if rel.startswith("../"):
        return rel
    return f"./{rel}"


def staged_task_copy_dir(results_dir: str | Path) -> Path:
    """Return canonical staged task directory under a results root."""
    return resolve_path(results_dir) / "task"


def load_task_copy_ignore_patterns(source_task_dir: Path) -> tuple[str, ...]:
    """Load task-copy ignore patterns from the source task directory."""
    ignore_path = source_task_dir / TASK_COPY_IGNORE_FILENAME
    if not ignore_path.is_file():
        return ()
    patterns: list[str] = []
    for raw_line in ignore_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return tuple(patterns)


def task_copy_path_is_ignored(
    relative_path: str | Path,
    *,
    is_dir: bool,
    patterns: tuple[str, ...],
) -> bool:
    """Match a task-copy ignore pattern; a leading slash anchors it to the root."""
    rel_path = Path(relative_path).as_posix()
    if rel_path.startswith("./"):
        rel_path = rel_path[2:]
    name = Path(rel_path).name
    for pattern in patterns:
        anchored = pattern.startswith("/")
        directory_only = pattern.endswith("/")
        normalized = pattern.strip("/") if anchored else pattern.rstrip("/")
        if directory_only and not is_dir:
            continue
        if anchored:
            if _root_ignore_pattern_matches(rel_path, normalized):
                return True
            continue
        if (
            fnmatch.fnmatch(name, normalized)
            or fnmatch.fnmatch(rel_path, normalized)
            or rel_path.startswith(f"{normalized}/")
        ):
            return True
    return False


def _root_ignore_pattern_matches(relative_path: str, pattern: str) -> bool:
    path_parts = PurePosixPath(relative_path.lstrip("/")).parts
    raw_pattern_parts = PurePosixPath(pattern).parts
    if not raw_pattern_parts:
        return False
    pattern_parts = tuple(
        part
        for index, part in enumerate(raw_pattern_parts)
        if part != "**" or index == 0 or raw_pattern_parts[index - 1] != "**"
    )
    return _path_segments_match(path_parts, pattern_parts)


def _path_segments_match(
    path_parts: tuple[str, ...],
    pattern_parts: tuple[str, ...],
) -> bool:
    pending = [(0, 0)]
    visited: set[tuple[int, int]] = set()
    while pending:
        path_index, pattern_index = pending.pop()
        state = (path_index, pattern_index)
        if state in visited:
            continue
        visited.add(state)
        if pattern_index == len(pattern_parts):
            return True
        pattern_part = pattern_parts[pattern_index]
        if pattern_part == "**":
            pending.append((path_index, pattern_index + 1))
            if path_index < len(path_parts):
                pending.append((path_index + 1, pattern_index))
        elif path_index < len(path_parts) and fnmatch.fnmatchcase(
            path_parts[path_index], pattern_part
        ):
            pending.append((path_index + 1, pattern_index + 1))
    return False


def task_copy_ignore(source_task_dir: Path, patterns: tuple[str, ...]):
    """Build a ``copytree`` ignore callback for task-copy patterns."""

    def ignore(current_dir: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        if not patterns:
            return ignored
        current_path = Path(current_dir)
        for name in names:
            full_path = current_path / name
            try:
                rel_path = full_path.relative_to(source_task_dir).as_posix()
            except ValueError:
                rel_path = name
            if task_copy_path_is_ignored(
                rel_path,
                is_dir=stat.S_ISDIR(full_path.lstat().st_mode),
                patterns=patterns,
            ):
                ignored.add(name)
        return ignored

    return ignore


def task_copy_entry_is_ignored(
    *,
    name: str,
    relative_path: str,
    is_directory: bool,
    patterns: tuple[str, ...],
) -> bool:
    """Return whether one verified task-tree entry matches an ignore pattern."""
    return task_copy_path_is_ignored(
        relative_path or name,
        is_dir=is_directory,
        patterns=patterns,
    )


def ensure_task_copy_in_results(
    task_dir: str | Path,
    results_dir: str | Path,
    *,
    lock_timeout_seconds: float = 120.0,
) -> Path:
    """Copy task directory into results once and return staged path."""
    source_task_dir = resolve_path(task_dir)
    if not source_task_dir.is_dir():
        raise FileNotFoundError(f"Missing task directory: {source_task_dir}")

    require_no_legacy_data_cache(source_task_dir)
    results_root = resolve_path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    staged_dir = staged_task_copy_dir(results_root)

    def staged_copy_exists() -> bool:
        if staged_dir.is_symlink():
            raise FileExistsError(f"Unsafe staged task symlink: {staged_dir}")
        if staged_dir.exists() and not staged_dir.is_dir():
            raise FileExistsError(f"Unsafe staged task path: {staged_dir}")
        require_no_legacy_data_cache(staged_dir)
        return staged_dir.is_dir()

    if staged_copy_exists():
        return staged_dir

    lock_path = results_root / ".task_copy.lock"
    start = time.monotonic()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if staged_copy_exists():
                return staged_dir
            if (time.monotonic() - start) > lock_timeout_seconds:
                raise TimeoutError(f"Timed out waiting for task-copy lock: {lock_path}")
            time.sleep(0.05)
            continue
        os.close(fd)
        break

    tmp_dir: Path | None = None
    try:
        if staged_copy_exists():
            return staged_dir
        tmp_dir = results_root / f".task.tmp.{os.getpid()}.{time.time_ns()}"
        ignore_patterns = load_task_copy_ignore_patterns(source_task_dir)
        shutil.copytree(
            source_task_dir,
            tmp_dir,
            symlinks=True,
            ignore=task_copy_ignore(source_task_dir, ignore_patterns),
        )
        require_no_legacy_data_cache(tmp_dir)
        tmp_dir.replace(staged_dir)
        tmp_dir = None
        return staged_dir
    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def ensure_directory_alias(alias_path: str | Path, target_dir: str | Path) -> Path:
    """Create or refresh a relative directory symlink alias."""
    alias = Path(alias_path).expanduser()
    target = resolve_path(target_dir)
    if not target.is_dir():
        raise FileNotFoundError(f"Missing alias target directory: {target}")

    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() and not alias.is_symlink():
        if alias.resolve() == target:
            return alias
        raise FileExistsError(f"Alias path exists and is not a symlink: {alias}")

    if alias.is_symlink():
        existing_target = resolve_path(alias.parent / os.readlink(alias))
        if existing_target == target:
            return alias
        alias.unlink()

    relative_target = os.path.relpath(target, start=alias.parent.resolve())
    alias.symlink_to(relative_target, target_is_directory=True)
    return alias


def extract_markdown_section(markdown: str, heading: str) -> str:
    """Extract a markdown section body by heading title (case-insensitive)."""
    matches = list(MARKDOWN_HEADING_RE.finditer(markdown))
    target = heading.strip().lower()
    for idx, match in enumerate(matches):
        if match.group(1).strip().lower() != target:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        return markdown[start:end].strip()
    return ""


def read_task_sections(task_dir: Path) -> tuple[str, str, Path]:
    """Read `task.md` and return Goal + Rubrics section bodies."""
    task_md_path = task_dir / "task.md"
    if not task_md_path.exists():
        return "(missing `task.md`)", "(missing `task.md`)", task_md_path

    text = task_md_path.read_text(encoding="utf-8")
    goal = extract_markdown_section(text, "Goal") or "(missing Goal section)"
    rubric_section = extract_markdown_section(text, "Rubrics") or "(missing Rubrics section)"
    return goal, rubric_section, task_md_path


def load_instructions(
    path: Path,
    task_dir: Path,
    results_dir: Path,
    extra_replacements: dict[str, str] | None = None,
) -> str:
    """Load instructions from markdown file, resolving prompt placeholders."""
    from src.task_drafting.prompt_contract import expand_evidence_contract

    template = expand_evidence_contract(path.read_text(encoding="utf-8"))
    task_goal, task_rubrics, task_md_path = read_task_sections(task_dir)
    replacements = {
        "TASK_DIR": str(task_dir),
        "RESULTS_DIR": str(results_dir),
        "TASK_MD_PATH": str(task_md_path),
        "TASK_GOAL": task_goal,
        "TASK_RUBRICS": task_rubrics,
    }
    if extra_replacements:
        replacements.update(extra_replacements)

    for key, value in replacements.items():
        template = template.replace(f"{{{{{key}}}}}", value)
    return template


def read_json(path: Path) -> dict[str, Any] | None:
    """Read JSON data from disk when available."""
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def discover_program_indices(root_dir: Path) -> list[int]:
    """Discover program indices."""
    indices: list[int] = []
    if not root_dir.exists():
        return indices
    for path in root_dir.iterdir():
        if not path.is_dir():
            continue
        match = PROGRAM_DIR_RE.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


# Variant evaluations the host ran, and the boundary each producer's manifest must declare.
_HOST_EVALUATION_BOUNDARIES = {"sandbox_host_eval": "sandbox", "native_host_eval": "native"}


def evaluation_results_complete(program_dir: Path) -> bool:
    """Require a valid completion manifest when an evaluator created one."""
    pending_path = program_dir / ".evaluation_pending"
    if pending_path.exists() or pending_path.is_symlink():
        return False
    manifest_path = program_dir / "evaluation_manifest.json"
    try:
        manifest_stat = manifest_path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if not stat.S_ISREG(manifest_stat.st_mode):
        return False
    manifest = read_json(manifest_path)
    if manifest is None:
        return False
    if manifest.get("producer") == "isolated_evaluator" or "published_summaries" in manifest:
        summaries = manifest.get("published_summaries")
        return (
            manifest.get("schema_version") == 1
            and manifest.get("returncode") == 0
            and isinstance(manifest.get("output_dir"), str)
            and bool(manifest["output_dir"].strip())
            and isinstance(summaries, list)
            and set(summaries) == {"metrics.json", "correct.json"}
        )

    host_boundary = _HOST_EVALUATION_BOUNDARIES.get(str(manifest.get("producer")))
    if host_boundary is None and manifest.get("execution_boundary") == "sandbox" and "returncode" in manifest:
        host_boundary = "sandbox"  # legacy sandbox manifests carry no producer
    if host_boundary is not None:
        return (
            manifest.get("producer") in (None, *_HOST_EVALUATION_BOUNDARIES)
            and manifest.get("execution_boundary") == host_boundary
            and manifest.get("program_id") == program_dir.name
            and manifest.get("returncode") == 0
            and isinstance(manifest.get("command"), list)
            and bool(manifest["command"])
            and isinstance(manifest.get("started_at"), str)
            and isinstance(manifest.get("finished_at"), str)
        )

    required_harness_fields = {
        "attempt_id",
        "idea_id",
        "program_id",
        "command",
        "started_at",
        "finished_at",
        "exit_code",
    }
    return (
        manifest.get("producer") in (None, "harness_eval")
        and required_harness_fields <= manifest.keys()
        and manifest.get("program_id") == program_dir.name
        and manifest.get("exit_code") == 0
        and isinstance(manifest.get("command"), list)
        and bool(manifest["command"])
    )


def find_best_program_in_experiment(
    experiment_dir: Path,
) -> tuple[Path, str, float, bool] | None:
    """Handle find best program in experiment logic for this module."""
    best_path: Path | None = None
    best_entrypoint: str | None = None
    best_score: float | None = None

    for idx in discover_program_indices(experiment_dir):
        program_dir = experiment_dir / f"program_{idx}"
        if not evaluation_results_complete(program_dir):
            continue
        metrics = read_json(program_dir / "metrics.json")
        correctness = read_json(program_dir / "correct.json")
        if not metrics or not correctness:
            continue

        score = metrics.get("combined_score")
        correct = correctness.get("correct")
        if not isinstance(score, (int, float)):
            continue
        if correct is not True:
            continue

        try:
            spec = load_program_spec(program_dir, required=True)
            assert spec is not None
            entrypoint_abs = resolve_entrypoint(program_dir, spec)
            if not entrypoint_abs.is_file():
                continue
            entrypoint_rel = entrypoint_abs.relative_to(program_dir.resolve()).as_posix()
        except (ProgramSpecError, ValueError):
            continue

        score_f = float(score)
        if best_score is None or score_f > best_score:
            best_score = score_f
            best_path = program_dir
            best_entrypoint = entrypoint_rel

    if best_path is None or best_score is None or best_entrypoint is None:
        return None
    return best_path.resolve(), best_entrypoint, best_score, True


def _parse_directory_marker(run_output: str, pattern: Any) -> Path | None:
    """Parse directory marker."""
    matches = pattern.findall(run_output)
    if not matches:
        return None
    last = matches[-1].strip()
    if not last:
        return None
    return resolve_path(last)


def parse_experiment_dir(run_output: str) -> Path | None:
    """Parse experiment dir."""
    return _parse_directory_marker(run_output, EXPERIMENT_DIR_RE)


def parse_idea_dir(run_output: str) -> Path | None:
    """Parse idea dir."""
    return _parse_directory_marker(run_output, IDEA_DIR_RE)


def parse_paper_dir(run_output: str) -> Path | None:
    """Parse paper dir."""
    return _parse_directory_marker(run_output, PAPER_DIR_RE)


def missing_task_scaffold_files(task_dir: Path, *, require_contract_v1: bool = False) -> list[Path]:
    """Check scaffold files; explicit drafted validation also requires its receipt."""
    required = [
        task_dir / "capture.py",
        task_dir / "evaluate.py",
        task_dir / "program_0" / "program_spec.toml",
    ]
    contract_path = task_dir / "task_contract.json"
    if require_contract_v1 or contract_path.exists():
        required.extend(
            [
                contract_path,
                task_dir / "evaluate_final.py",
            ]
        )
    if require_contract_v1:
        required.append(task_dir / ".agentic_task_validation.json")
    return _missing_scaffold_sources(task_dir, required)


def _missing_scaffold_sources(task_dir: Path, required: list[Path]) -> list[Path]:
    missing = [path for path in required if not path.is_file()]
    if missing:
        return missing

    # Validate that the declared entrypoint exists.
    program_dir = task_dir / "program_0"
    try:
        spec = load_program_spec(program_dir, required=True)
        assert spec is not None
        entrypoint_abs = resolve_entrypoint(program_dir, spec)
    except ProgramSpecError:
        return [program_dir / "program_spec.toml"]
    if not entrypoint_abs.is_file():
        return [entrypoint_abs]
    return []

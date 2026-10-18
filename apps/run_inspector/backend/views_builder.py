"""Build payloads for the new inspector views from discovered run data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .discovery import (
    DiscoveredRun,
    Idea,
    Worker,
    _load_json_dict,
    _load_jsonl_dicts,
    idea_node_id,
    idea_rel_path,
)
from .models import (
    AncestorEntry,
    DiffLine,
    DiffStats,
    IdeaAncestors,
    IdeaAttemptSummary,
    IdeaTimeline,
    IdeaTimelineEntry,
    ProgramDirDiff,
    ProgramDirDiffFile,
    ProgramDirDiffStats,
    ProgramDiff,
    ProgramLocation,
    PruneEvent,
    SiblingProgramEntry,
)


def _load_per_worker_jsonl(
    run: DiscoveredRun, filename: str,
) -> list[tuple[Worker | None, list[dict[str, Any]]]]:
    """Load a JSONL file per worker.

    Tries each worker's own directory first.  When the worker-level file is
    missing (round isolation layout), falls back to collecting the file from
    ``round_NNN/`` sub-directories under the worker.  If *no* worker has its
    own file the single run-level file is used with ``worker=None`` so callers
    know the data is not attributable to a specific worker.
    """
    from .discovery import ROUND_DIR_RE

    results: list[tuple[Worker | None, list[dict[str, Any]]]] = []
    any_worker_has_file = False

    for worker in run.workers:
        rows = _load_jsonl_dicts(worker.path / filename)
        # Fallback: collect from round_NNN/ subdirs (round isolation layout)
        if not rows and worker.path.is_dir():
            for child in sorted(worker.path.iterdir(), key=lambda d: d.name):
                if child.is_dir() and ROUND_DIR_RE.match(child.name):
                    rows.extend(_load_jsonl_dicts(child / filename))
        if rows:
            any_worker_has_file = True
        results.append((worker, rows))

    if not any_worker_has_file:
        run_rows = _load_jsonl_dicts(run.path / filename)
        if run_rows:
            return [(None, run_rows)]
        return []

    return [(w, rows) for w, rows in results if rows]


def _worker_key(worker: Worker | None, record: dict[str, Any]) -> str | None:
    """Derive a collision-safe worker key from a history record.

    When per-worker files are available, ``worker`` is set and we use its
    name directly.  When data comes from the run-level aggregate
    (``worker is None``), the ``parallel_worker_index`` field embedded by
    aggregation is used to reconstruct a worker name.  For single-worker
    runs the field is absent, so ``None`` is returned — which is safe
    because there is only one worker and no collision is possible.
    """
    if worker is not None:
        return worker.worker_name
    pwi = record.get("parallel_worker_index")
    if isinstance(pwi, int) and pwi > 0:
        return f"worker_{pwi:02d}"
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _safe_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _best_rubric_status_for_idea(idea: Idea) -> dict[str, Any] | None:
    """Find the rubric_status with the most passes across all programs."""
    best_rs: dict[str, Any] | None = None
    best_pass_count = -1

    for program in idea.programs:
        if not program.rubric_status:
            continue
        rubrics = program.rubric_status.get("rubrics")
        if not isinstance(rubrics, list):
            continue
        pass_count = sum(1 for r in rubrics if r.get("pass"))
        if pass_count > best_pass_count:
            best_pass_count = pass_count
            best_rs = program.rubric_status

    # Also check attempt-level rubric_status files
    for attempt_dir in _attempt_dirs(idea.path):
        rs = _load_json_dict(attempt_dir / "rubric_status.json")
        if not rs:
            continue
        rubrics = rs.get("rubrics")
        if not isinstance(rubrics, list):
            continue
        pass_count = sum(1 for r in rubrics if r.get("pass"))
        if pass_count > best_pass_count:
            best_pass_count = pass_count
            best_rs = rs

    return best_rs


def _attempt_dirs(idea_path: Path) -> list[Path]:
    """Return sorted attempt directories under an idea path."""
    if not idea_path.is_dir():
        return []
    dirs = []
    for child in idea_path.iterdir():
        if child.is_dir() and child.name.endswith("_attempt"):
            dirs.append(child)
    return sorted(dirs, key=lambda d: d.name)


def _idea_best_score(idea: Idea) -> float | None:
    """Best combined_score across all programs (higher is better)."""
    best: float | None = None
    for program in idea.programs:
        if not program.metrics:
            continue
        score = program.metrics.get("combined_score")
        if isinstance(score, (int, float)):
            if best is None or float(score) > best:
                best = float(score)
    return best


def _idea_any_pass(idea: Idea) -> bool:
    """Check if any rubric_status has overall_pass.

    Only rubric pass counts — ``correct.json`` indicates base-task correctness
    which is a weaker signal (a program can be correct but fail hypothesis
    validation rubrics).
    """
    for program in idea.programs:
        if program.rubric_status and program.rubric_status.get("overall_pass"):
            return True
    for attempt_dir in _attempt_dirs(idea.path):
        rs = _load_json_dict(attempt_dir / "rubric_status.json")
        if rs and rs.get("overall_pass"):
            return True
    return False


def _compute_prune_chapter(
    cycle_index: int | None,
    prune_every_n: int,
    prune_chapters: list[int],
) -> int | None:
    """Determine which prune chapter a cycle belongs to.

    Chapter 0 is before the first prune, chapter 1 after the first prune, etc.
    Returns ``None`` when pruning is disabled or cycle_index is unknown.
    """
    if prune_every_n <= 0 or cycle_index is None:
        return None
    if not prune_chapters:
        return 0
    # Each archived chapter N means a prune happened after (ch+1)*prune_every_n
    # cycles.  The boundary for chapter ch is at cycle (ch+1)*prune_every_n + 1
    # — ideas at or above this cycle belong to the next chapter.
    chapter = 0
    for ch in prune_chapters:
        boundary_cycle = (ch + 1) * prune_every_n + 1
        if cycle_index >= boundary_cycle:
            chapter = ch + 1
        else:
            break
    return chapter


def _iter_prune_archive_dirs(run: DiscoveredRun) -> list[Path]:
    """Return all archived prune directories (chapter_NNN and round_NNN)."""
    pruned_root = run.path / "_pruned"
    if not pruned_root.is_dir():
        return []
    dirs: list[Path] = []
    for child in sorted(pruned_root.iterdir()):
        if child.is_dir() and (child.name.startswith("chapter_") or child.name.startswith("round_")):
            dirs.append(child)
    return dirs


def build_idea_timeline(run: DiscoveredRun) -> IdeaTimeline:
    entries: list[IdeaTimelineEntry] = []

    prune_every_n = run.prune_every_n_cycles
    prune_chapters = run.prune_chapters

    # Build lookup tables from history files, keyed by (worker_key, idea_id)
    # to avoid cross-worker collisions when workers emit the same idea_id.
    # _worker_key derives the key from the Worker object or, for run-level
    # aggregated data, from parallel_worker_index in each record.
    ideas_map: dict[tuple[str | None, str], dict[str, Any]] = {}
    for _w, ideas_history in _load_per_worker_jsonl(run, "ideas_history.jsonl"):
        for rec in ideas_history:
            iid = rec.get("idea_id", "")
            if iid:
                ideas_map[(_worker_key(_w, rec), iid)] = rec

    # Also load ideas from archived chapters/rounds so all cycles are represented
    for archive_dir in _iter_prune_archive_dirs(run):
        for rec in _load_jsonl_dicts(archive_dir / "ideas_history.jsonl"):
            iid = rec.get("idea_id", "")
            if iid:
                ideas_map[(None, iid)] = rec

    attempts_by_idea: dict[tuple[str | None, str], list[dict[str, Any]]] = {}
    for _w, attempts_history in _load_per_worker_jsonl(run, "attempts_history.jsonl"):
        for a in attempts_history:
            iid = a.get("idea_id", "")
            attempts_by_idea.setdefault((_worker_key(_w, a), iid), []).append(a)

    for archive_dir in _iter_prune_archive_dirs(run):
        for a in _load_jsonl_dicts(archive_dir / "attempts_history.jsonl"):
            iid = a.get("idea_id", "")
            attempts_by_idea.setdefault((None, iid), []).append(a)

    for worker in run.workers:
        for idea in worker.ideas:
            history_rec = (
                ideas_map.get((worker.worker_name, idea.idea_id))
                or ideas_map.get((None, idea.idea_id))
                or {}
            )
            idea_attempts = (
                attempts_by_idea.get((worker.worker_name, idea.idea_id))
                or attempts_by_idea.get((None, idea.idea_id))
                or []
            )

            # Try to get hypothesis from active_idea.json in first attempt dir
            hypothesis: str | None = None
            implementation_plan: list[str] | None = None
            notes_for_agent: str | None = None

            for attempt_dir in _attempt_dirs(idea.path):
                active_idea = _load_json_dict(attempt_dir / "active_idea.json")
                if active_idea:
                    hypothesis = active_idea.get("hypothesis")
                    raw_plan = active_idea.get("implementation_plan")
                    if isinstance(raw_plan, list):
                        implementation_plan = [str(s) for s in raw_plan]
                    notes_for_agent = active_idea.get("notes_for_agent")
                    break

            # Also try idea_bundle.json
            if not hypothesis:
                bundle = _load_json_dict(idea.path / "idea_bundle.json")
                if bundle:
                    hypothesis = hypothesis or bundle.get("hypothesis")

            best_score = _idea_best_score(idea)
            any_pass = _idea_any_pass(idea)

            rubric_summary: dict[str, int] | None = None
            best_rs = _best_rubric_status_for_idea(idea)
            if best_rs and isinstance(best_rs.get("rubrics"), list):
                rubrics = best_rs["rubrics"]
                rubric_summary = {
                    "passed": sum(1 for r in rubrics if r.get("pass")),
                    "total": len(rubrics),
                }

            attempt_summaries: list[IdeaAttemptSummary] = []
            for a in idea_attempts:
                attempt_summaries.append(IdeaAttemptSummary(
                    attempt_id=_safe_str(a.get("attempt_id")),
                    attempt_number=_safe_int(a.get("attempt_number")),
                    attempt_status=_safe_str(a.get("attempt_status")),
                    rubric_overall_pass=bool(a.get("rubric_overall_pass", False)),
                    best_updated=bool(a.get("best_updated", False)),
                    metrics=a.get("metrics") if isinstance(a.get("metrics"), dict) else None,
                    timestamp=_safe_str(a.get("timestamp")),
                ))

            cycle_idx = _safe_int(history_rec.get("cycle_index"))
            entries.append(IdeaTimelineEntry(
                node_id=idea_node_id(run, worker, idea),
                idea_id=idea.idea_id,
                idea_path=idea_rel_path(run, idea),
                cycle_index=cycle_idx,
                title=idea.title,
                timestamp=_safe_str(history_rec.get("timestamp")),
                hypothesis=hypothesis,
                implementation_plan=implementation_plan,
                notes_for_agent=notes_for_agent,
                target_rubrics=history_rec.get("target_rubrics", []) if isinstance(history_rec.get("target_rubrics"), list) else [],
                num_attempts=len(idea_attempts),
                best_score=best_score,
                any_pass=any_pass,
                rubric_summary=rubric_summary,
                attempts=attempt_summaries,
                worker_name=worker.worker_name,
                prune_chapter=_compute_prune_chapter(cycle_idx, prune_every_n, prune_chapters),
            ))

    # Infer cycle_index for entries that have no history record.  This happens
    # when the last round's history was never archived (run still in progress
    # or the run-level history was cleared after the last prune).
    needs_fix = [i for i, e in enumerate(entries) if e.cycle_index is None]
    if needs_fix and prune_every_n > 0:
        max_known_cycle = max(
            (e.cycle_index for e in entries if e.cycle_index is not None),
            default=0,
        )
        # Sort by (worker_name, idea_id) for deterministic ordering
        needs_fix.sort(
            key=lambda i: (entries[i].worker_name or "", entries[i].idea_id),
        )
        next_cycle = max_known_cycle + 1
        for idx in needs_fix:
            entry = entries[idx]
            inferred_chapter = _compute_prune_chapter(
                next_cycle, prune_every_n, prune_chapters,
            )
            entries[idx] = IdeaTimelineEntry(
                node_id=entry.node_id,
                idea_id=entry.idea_id,
                idea_path=entry.idea_path,
                cycle_index=next_cycle,
                title=entry.title,
                timestamp=entry.timestamp,
                hypothesis=entry.hypothesis,
                implementation_plan=entry.implementation_plan,
                notes_for_agent=entry.notes_for_agent,
                target_rubrics=entry.target_rubrics,
                num_attempts=entry.num_attempts,
                best_score=entry.best_score,
                any_pass=entry.any_pass,
                rubric_summary=entry.rubric_summary,
                attempts=entry.attempts,
                worker_name=entry.worker_name,
                prune_chapter=inferred_chapter,
            )
            next_cycle += 1

    # Build prune events list
    prune_events: list[PruneEvent] = []
    for ch in prune_chapters:
        boundary_cycle = (ch + 1) * prune_every_n + 1 if prune_every_n > 0 else 0
        prune_events.append(PruneEvent(
            chapter=ch,
            cycle_index=boundary_cycle,
            timestamp=None,
        ))

    return IdeaTimeline(run_id=run.run_id, ideas=entries, prune_events=prune_events)


# ---------------------------------------------------------------------------
# Tree ancestry & program diff
# ---------------------------------------------------------------------------


def _looks_like_program_dir(program_dir: Path) -> bool:
    # v1 layout
    if (program_dir / "program_spec.toml").is_file():
        return True
    # Legacy layout
    if (program_dir / "program.py").is_file():
        return True
    # Extremely defensive fallback: a src tree usually implies a project layout.
    if (program_dir / "src").is_dir():
        return True
    return False


def _program_entries(
    run: DiscoveredRun, idea_path: Path, *, exclude_path: str | None = None,
) -> list[SiblingProgramEntry]:
    """List an idea's program directories as run-relative comparison entries."""
    from .discovery import _discover_programs_flattened

    entries: list[SiblingProgramEntry] = []
    for prog in _discover_programs_flattened(idea_path):
        if not _looks_like_program_dir(prog.dir_path):
            continue
        try:
            rel = prog.dir_path.relative_to(run.path).as_posix()
        except ValueError:
            continue
        if rel == exclude_path:
            continue
        entries.append(SiblingProgramEntry(
            label=prog.display_label,
            attempt_name=prog.attempt_name,
            program_info=ProgramLocation(path=rel),
        ))
    return entries


def _find_best_program_dir(idea_path: Path) -> str | None:
    """Find the best program directory in the last non-empty attempt directory.

    Searches attempt directories in reverse order and returns the path to
    the highest-numbered ``program_N`` sub-directory of the first attempt
    that contains one. If there are no attempt sub-directories it falls
    back to ``program_N`` dirs directly under *idea_path*.
    """

    attempt_dirs = _attempt_dirs(idea_path)
    if not attempt_dirs:
        # No attempt sub-dirs – look for program_N dirs directly
        from .discovery import PROGRAM_DIR_RE

        prog_dirs = sorted(
            [d for d in idea_path.iterdir() if d.is_dir() and PROGRAM_DIR_RE.match(d.name)],
            key=lambda d: d.name,
        )
        for program_dir in reversed(prog_dirs):
            if _looks_like_program_dir(program_dir):
                return str(program_dir)
        return None

    # Use last attempt, last program
    for attempt_dir in reversed(attempt_dirs):
        from .discovery import PROGRAM_DIR_RE

        prog_dirs = sorted(
            [d for d in attempt_dir.iterdir() if d.is_dir() and PROGRAM_DIR_RE.match(d.name)],
            key=lambda d: d.name,
        )
        for program_dir in reversed(prog_dirs):
            if _looks_like_program_dir(program_dir):
                return str(program_dir)
    return None


def build_idea_ancestors(
    run: DiscoveredRun, idea_id: str, *, worker_name: str | None = None,
) -> IdeaAncestors:
    """Build the ancestor chain for an idea based on best_updated tracking.

    For parallel runs, each worker has an independent best-program chain.
    *worker_name* disambiguates when multiple workers share the same
    *idea_id*.
    """
    # Collect all ideas across all workers with their filesystem paths
    idea_path_by_id: dict[tuple[str | None, str], Path] = {}

    for worker in run.workers:
        for idea in worker.ideas:
            idea_path_by_id[(worker.worker_name, idea.idea_id)] = idea.path

    # Build attempts lookup from attempts_history.jsonl
    attempts_by_idea: dict[tuple[str | None, str], list[dict[str, Any]]] = {}
    for worker_or_none, attempts_history in _load_per_worker_jsonl(run, "attempts_history.jsonl"):
        for a in attempts_history:
            iid = a.get("idea_id", "")
            attempts_by_idea.setdefault((_worker_key(worker_or_none, a), iid), []).append(a)

    # Also load attempts from archived (pruned) chapters/rounds.
    # Archived records lack worker context; derive worker key from
    # parallel_worker_index when available, otherwise default to the
    # single worker's name so that parent-chain tracking stays connected.
    default_wk = run.workers[0].worker_name if len(run.workers) == 1 else None
    for archive_dir in _iter_prune_archive_dirs(run):
        for a in _load_jsonl_dicts(archive_dir / "attempts_history.jsonl"):
            iid = a.get("idea_id", "")
            wk = _worker_key(None, a) or default_wk
            attempts_by_idea.setdefault((wk, iid), []).append(a)

    # Load all ideas from history, sorted by cycle_index
    ideas_with_worker: list[tuple[str | None, dict[str, Any]]] = []
    for worker_or_none, ideas_history in _load_per_worker_jsonl(run, "ideas_history.jsonl"):
        for rec in ideas_history:
            ideas_with_worker.append((_worker_key(worker_or_none, rec), rec))

    # Also load ideas from archived (pruned) chapters/rounds
    for archive_dir in _iter_prune_archive_dirs(run):
        for rec in _load_jsonl_dicts(archive_dir / "ideas_history.jsonl"):
            iid = rec.get("idea_id", "")
            if iid:
                wk = _worker_key(None, rec) or default_wk
                ideas_with_worker.append((wk, rec))

    ideas_with_worker.sort(key=lambda x: x[1].get("cycle_index", 0))

    def _did_update_best(wk: str | None, iid: str) -> bool:
        attempts = (
            attempts_by_idea.get((wk, iid))
            or attempts_by_idea.get((None, iid))
            or []
        )
        return any(a.get("best_updated") for a in attempts)

    # Walk in cycle order tracking current_best_source per worker
    current_best_source: dict[str | None, str] = {}  # worker -> idea_id of last best
    parent_of: dict[tuple[str | None, str], tuple[str | None, str | None]] = {}  # (wk, idea_id) -> (wk, parent_idea_id or None)

    for wk, rec in ideas_with_worker:
        iid = rec.get("idea_id", "")
        if not iid:
            continue
        parent_id = current_best_source.get(wk)
        parent_of[(wk, iid)] = (wk, parent_id)

        if _did_update_best(wk, iid):
            current_best_source[wk] = iid

    # Find which worker this idea belongs to.  When *worker_name* is given
    # we match on both fields so colliding idea_ids are disambiguated.
    target_wk: str | None = None
    target_rec: dict[str, Any] | None = None
    for wk, rec in ideas_with_worker:
        if rec.get("idea_id") != idea_id:
            continue
        if worker_name is not None and wk != worker_name:
            continue
        target_wk = wk
        target_rec = rec
        break

    if target_rec is None:
        return IdeaAncestors(idea_id=idea_id)

    def _find_idea_path(wk: str | None, iid: str) -> Path | None:
        """Look up idea path, falling back across workers for archived ideas."""
        path = idea_path_by_id.get((wk, iid))
        if path is not None:
            return path
        # Archived records have worker_key=None; try all workers
        for key, p in idea_path_by_id.items():
            if key[1] == iid:
                return p
        return None

    # Find target program path
    target_path = _find_idea_path(target_wk, idea_id)
    target_program: ProgramLocation | None = None
    if target_path:
        prog_path = _find_best_program_dir(target_path)
        if prog_path:
            try:
                rel = Path(prog_path).relative_to(run.path).as_posix()
                target_program = ProgramLocation(path=rel)
            except ValueError:
                target_program = None

    # Walk ancestor chain backwards
    ancestors: list[AncestorEntry] = []
    visited: set[tuple[str | None, str | None]] = set()
    current = parent_of.get((target_wk, idea_id))

    while current is not None and current not in visited:
        cwk, cid = current
        if cid is None:
            break
        visited.add(current)

        # Find idea record for this ancestor
        anc_rec: dict[str, Any] | None = None
        for awk, arec in ideas_with_worker:
            if awk == cwk and arec.get("idea_id") == cid:
                anc_rec = arec
                break

        anc_program: ProgramLocation | None = None
        anc_path = _find_idea_path(cwk, cid)
        if anc_path:
            prog_path = _find_best_program_dir(anc_path)
            if prog_path:
                try:
                    rel = Path(prog_path).relative_to(run.path).as_posix()
                    anc_program = ProgramLocation(path=rel)
                except ValueError:
                    pass

        anc_programs = _program_entries(run, anc_path) if anc_path else []

        ancestors.append(AncestorEntry(
            idea_id=cid,
            title=anc_rec.get("title", "") if anc_rec else "",
            cycle_index=_safe_int(anc_rec.get("cycle_index")) if anc_rec else None,
            worker_name=cwk,
            program_info=anc_program,
            programs=anc_programs,
        ))

        current = parent_of.get((cwk, cid))

    # The target program itself is already shown as "current".
    target_rel = target_program.path if target_program else None
    sibling_programs = _program_entries(run, target_path, exclude_path=target_rel) if target_path else []

    return IdeaAncestors(
        idea_id=idea_id,
        target_program=target_program,
        ancestors=ancestors,
        sibling_programs=sibling_programs,
    )


def build_program_dir_diff(run: DiscoveredRun, from_dir: str, to_dir: str) -> ProgramDirDiff:
    """Compute a structured diff between two program directories within a run.

    The diff is scoped to:
    - program_spec.toml (program root)
    - src/** (all files under src/)
    """

    run_root = run.path.resolve()

    def _resolve_dir_safe(rel: str) -> Path:
        candidate = (run_root / rel).resolve()
        if candidate != run_root and run_root not in candidate.parents:
            raise ValueError(f"Path escapes run directory: {rel}")
        return candidate

    def _is_binary_file(path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                sample = handle.read(8192)
        except OSError:
            return False
        return b"\x00" in sample

    def _files_equal(left: Path, right: Path) -> bool:
        try:
            if left.stat().st_size != right.stat().st_size:
                return False
        except OSError:
            return False

        try:
            with left.open("rb") as lh, right.open("rb") as rh:
                while True:
                    lb = lh.read(65536)
                    rb = rh.read(65536)
                    if lb != rb:
                        return False
                    if not lb:
                        return True
        except OSError:
            return False

    def _collect_scoped_files(program_dir: Path) -> dict[str, Path]:
        files: dict[str, Path] = {}

        if not program_dir.is_dir():
            return files

        spec_path = program_dir / "program_spec.toml"
        if spec_path.is_file() and not spec_path.is_symlink():
            files["program_spec.toml"] = spec_path

        legacy_program = program_dir / "program.py"
        if legacy_program.is_file() and not legacy_program.is_symlink():
            files["program.py"] = legacy_program

        src_root = program_dir / "src"
        if not src_root.is_dir():
            return files

        skip_dirs: set[str] = {
            ".venv",
            "venv",
            ".env",
            "node_modules",
            "__pycache__",
            ".git",
            ".tox",
            ".nox",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "site-packages",
            "dist",
            "build",
        }

        for dirpath, dirnames, filenames in os.walk(src_root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for filename in filenames:
                if filename in {".DS_Store"}:
                    continue
                if filename.endswith((".pyc", ".pyo")):
                    continue
                abs_path = Path(dirpath) / filename
                if abs_path.is_symlink() or not abs_path.is_file():
                    continue
                relpath = abs_path.relative_to(program_dir).as_posix()
                files[relpath] = abs_path

        return files

    from_dir_abs = _resolve_dir_safe(from_dir)
    to_dir_abs = _resolve_dir_safe(to_dir)

    from_dir_rel = from_dir_abs.relative_to(run_root).as_posix()
    to_dir_rel = to_dir_abs.relative_to(run_root).as_posix()

    from_files = _collect_scoped_files(from_dir_abs)
    to_files = _collect_scoped_files(to_dir_abs)

    all_relpaths = sorted(set(from_files) | set(to_files))

    out_files: list[ProgramDirDiffFile] = []
    stats = ProgramDirDiffStats()

    for relpath in all_relpaths:
        from_abs = from_files.get(relpath)
        to_abs = to_files.get(relpath)
        from_exists = from_abs is not None and from_abs.is_file()
        to_exists = to_abs is not None and to_abs.is_file()

        if from_exists and not to_exists:
            stats.removed += 1
            out_files.append(
                ProgramDirDiffFile(
                    relpath=relpath,
                    status="removed",
                    from_path=(Path(from_dir_rel) / relpath).as_posix(),
                    to_path=(Path(to_dir_rel) / relpath).as_posix(),
                    from_exists=True,
                    to_exists=False,
                    is_binary=_is_binary_file(from_abs) if from_abs is not None else False,
                    from_size_bytes=from_abs.stat().st_size if from_abs is not None else None,
                    to_size_bytes=None,
                )
            )
            continue

        if to_exists and not from_exists:
            stats.added += 1
            out_files.append(
                ProgramDirDiffFile(
                    relpath=relpath,
                    status="added",
                    from_path=(Path(from_dir_rel) / relpath).as_posix(),
                    to_path=(Path(to_dir_rel) / relpath).as_posix(),
                    from_exists=False,
                    to_exists=True,
                    is_binary=_is_binary_file(to_abs) if to_abs is not None else False,
                    from_size_bytes=None,
                    to_size_bytes=to_abs.stat().st_size if to_abs is not None else None,
                )
            )
            continue

        if not from_exists or not to_exists:
            # Missing on both sides (e.g., non-existent program dirs).
            continue

        # Present on both sides
        if from_abs is None or to_abs is None:
            continue
        if _files_equal(from_abs, to_abs):
            stats.unchanged += 1
            continue

        stats.modified += 1
        out_files.append(
            ProgramDirDiffFile(
                relpath=relpath,
                status="modified",
                from_path=(Path(from_dir_rel) / relpath).as_posix(),
                to_path=(Path(to_dir_rel) / relpath).as_posix(),
                from_exists=True,
                to_exists=True,
                is_binary=_is_binary_file(from_abs) or _is_binary_file(to_abs),
                from_size_bytes=from_abs.stat().st_size,
                to_size_bytes=to_abs.stat().st_size,
            )
        )

    out_files.sort(key=lambda f: (f.status, f.relpath))
    return ProgramDirDiff(files=out_files, stats=stats)


def build_program_diff(run: DiscoveredRun, from_path: str, to_path: str) -> ProgramDiff:
    """Compute a unified diff between two program files within a run."""
    import difflib as _difflib

    run_root = run.path.resolve()

    def _resolve_safe(rel: str) -> Path:
        candidate = (run_root / rel).resolve()
        if candidate != run_root and run_root not in candidate.parents:
            raise ValueError(f"Path escapes run directory: {rel}")
        return candidate

    from_abs = _resolve_safe(from_path)
    to_abs = _resolve_safe(to_path)

    try:
        from_lines = from_abs.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except FileNotFoundError:
        from_lines = []

    try:
        to_lines = to_abs.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except FileNotFoundError:
        to_lines = []

    diff = list(_difflib.unified_diff(from_lines, to_lines, fromfile="before", tofile="after", lineterm=""))

    parsed: list[DiffLine] = []
    additions = 0
    deletions = 0
    hunks = 0
    fl = 0
    tl = 0

    for raw in diff:
        if raw.startswith("@@"):
            hunks += 1
            parts = raw.split()
            try:
                fl = abs(int(parts[1].split(",")[0]))
                tl = int(parts[2].split(",")[0])
            except (IndexError, ValueError):
                pass
            parsed.append(DiffLine(type="hunk_header", content=raw))
        elif raw.startswith("---") or raw.startswith("+++"):
            continue
        elif raw.startswith("-"):
            deletions += 1
            parsed.append(DiffLine(type="removal", content=raw[1:], from_line=fl))
            fl += 1
        elif raw.startswith("+"):
            additions += 1
            parsed.append(DiffLine(type="addition", content=raw[1:], to_line=tl))
            tl += 1
        else:
            content = raw[1:] if raw.startswith(" ") else raw
            parsed.append(DiffLine(type="context", content=content, from_line=fl, to_line=tl))
            fl += 1
            tl += 1

    return ProgramDiff(
        lines=parsed,
        stats=DiffStats(additions=additions, deletions=deletions, hunks=hunks),
    )

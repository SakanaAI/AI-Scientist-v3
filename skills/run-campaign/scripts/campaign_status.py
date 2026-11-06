"""One-screen status of an AI-Scientist-v3 run: state, phases, cycles, best score, top evaluations, cost.

    python campaign_status.py <run_dir>                  # one run
    python campaign_status.py <results_root>             # newest run under <root>/<task>/<agent>/<run_id>
    python campaign_status.py <run_dir> --job 12345      # add the Slurm job state
    python campaign_status.py <run_dir> --json           # machine-readable

Reads only files the harness writes; never modifies the run. A resumed run appends a new session
to the same journal: only the latest session counts.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PIPELINE_JOURNAL = "pipeline_events.jsonl"
PROGRAM_DIR = re.compile(r"program_\d+")
CONFIG_LABELS = {
    "agent.name": "agent",
    "agent.model": "model",
    "container.enabled": "container",
    "execution.workers": "workers",
    "execution.max_idea_cycles": "cycles",
    "execution.max_program_evals_per_attempt": "evals/attempt",
}


def read_object(path: Path) -> dict[str, Any]:
    """A JSON object from disk, or {} when missing, partial, or not an object."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # a line being written right now
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        pass
    return records


def parse_time(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def resolve_run_dir(path: Path) -> Path:
    """The run dir itself, or the most recently active run under a results root."""
    if (path / PIPELINE_JOURNAL).is_file():
        return path
    journals = [p for depth in ("*", "*/*", "*/*/*") for p in path.glob(f"{depth}/{PIPELINE_JOURNAL}")]
    if not journals:
        raise SystemExit(f"no {PIPELINE_JOURNAL} in {path} or up to three levels below")
    return max(journals, key=lambda p: p.stat().st_mtime).parent


def latest_session(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts = [index for index, event in enumerate(events) if event.get("event") == "pipeline_start"]
    return events[starts[-1]:] if starts else events


def phase_states(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    phases: dict[str, dict[str, Any]] = {}
    for event in events:
        fields = event.get("fields") or {}
        name = fields.get("phase_name")
        if event.get("event") == "phase_start" and name:
            phases[name] = {"status": "running", "started": event.get("timestamp")}
        elif event.get("event") == "phase_end" and name:
            phases.setdefault(name, {}).update(status=fields.get("status"), elapsed=fields.get("elapsed"))
    return phases


def investigation_journals(investigation: Path) -> list[Path]:
    patterns = ("events.jsonl", "worker_*/events.jsonl", "worker_*/round_*/events.jsonl")
    return [path for pattern in patterns for path in investigation.glob(pattern)]


def pipeline_summary(run_dir: Path, now: datetime, stale_minutes: float, last_evaluation: float | None) -> dict[str, Any]:
    events = latest_session(read_jsonl(run_dir / PIPELINE_JOURNAL))
    start = events[0] if events and events[0].get("event") == "pipeline_start" else {}
    end = next((e for e in reversed(events) if e.get("event") == "pipeline_end"), None)
    activity_times = [t.timestamp() for t in (parse_time(e.get("timestamp")) for e in events) if t]
    activity_times += [t for t in map(mtime, investigation_journals(run_dir / "investigation")) if t]
    if last_evaluation:
        activity_times.append(last_evaluation)
    idle_minutes = (now.timestamp() - max(activity_times)) / 60 if activity_times else None
    if end:
        state = f"FINISHED ({(end.get('fields') or {}).get('status')})"
    elif idle_minutes is None:
        state = "UNKNOWN"
    elif idle_minutes > stale_minutes:
        state = "STALLED"
    else:
        state = "RUNNING"
    heartbeat = next((e for e in reversed(events) if e.get("event") == "heartbeat"), None)
    config = start.get("fields") or {}
    started = parse_time(start.get("timestamp"))
    return {
        "state": state,
        "started": start.get("timestamp"),
        "sessions": sum(1 for e in read_jsonl(run_dir / PIPELINE_JOURNAL) if e.get("event") == "pipeline_start"),
        "elapsed_hours": round((now - started).total_seconds() / 3600, 2) if started and not end else None,
        "total_elapsed": (end.get("fields") or {}).get("elapsed") if end else None,
        "last_activity_minutes_ago": round(idle_minutes, 1) if idle_minutes is not None else None,
        "activity": None if end else ((heartbeat or {}).get("fields") or {}).get("label"),
        "config": {label: config[key] for key, label in CONFIG_LABELS.items() if key in config},
        "phases": phase_states(events),
    }


def state_dirs(investigation: Path) -> list[Path]:
    """The single-run state dir, plus the newest round of each parallel worker."""
    dirs = [investigation] if (investigation / "outer_state.json").is_file() else []
    for worker in sorted(investigation.glob("worker_*")):
        rounds = sorted(worker.glob("round_*/outer_state.json"))
        if rounds:
            dirs.append(rounds[-1].parent)
        elif (worker / "outer_state.json").is_file():
            dirs.append(worker)
    return dirs


def worker_summary(state_dir: Path, investigation: Path) -> dict[str, Any]:
    outer = read_object(state_dir / "outer_state.json")
    attempts = read_jsonl(state_dir / "attempts_history.jsonl")
    statuses = Counter(str(a.get("attempt_status")) for a in attempts)
    failures = Counter(str(a["failure_category"]) for a in attempts if a.get("failure_category"))
    return {
        "dir": str(state_dir.relative_to(investigation)) if state_dir != investigation else ".",
        "cycle": outer.get("current_cycle"),
        "best_score": outer.get("best_combined_score"),
        "best_correct": outer.get("best_correct"),
        "failure_streak": outer.get("failure_streak"),
        "attempts": dict(statuses),
        "failure_categories": dict(failures.most_common(3)),
    }


@dataclass(frozen=True)
class Evaluation:
    program: str
    score: float | None
    correct: bool
    mtime: float


def published_evaluations(investigation: Path) -> list[Evaluation]:
    """Program evaluations the harness published, without descending into hidden dirs or program sources."""
    found = []
    for root, dirs, _ in os.walk(investigation):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("src", "__pycache__")]
        if not PROGRAM_DIR.fullmatch(os.path.basename(root)):
            continue
        dirs[:] = []
        program = Path(root)
        modified = mtime(program / "metrics.json")
        if modified is None:
            continue
        metrics, correct = read_object(program / "metrics.json"), read_object(program / "correct.json")
        found.append(Evaluation(
            program=str(program.relative_to(investigation)),
            score=finite(metrics.get("combined_score")),
            correct=correct.get("correct") is True,
            mtime=modified,
        ))
    return found


def evaluation_summary(found: list[Evaluation], top: int) -> dict[str, Any]:
    ranked = sorted(((e.score, e.program) for e in found if e.correct and e.score is not None), reverse=True)
    latest = max((e.mtime for e in found), default=None)
    return {
        "count": len(found),
        "correct": sum(e.correct for e in found),
        "latest": datetime.fromtimestamp(latest, timezone.utc).isoformat() if latest else None,
        "top": [{"program": program, "score": score} for score, program in ranked[:top]],
    }


def investigation_summary(investigation: Path, found: list[Evaluation], top: int) -> dict[str, Any] | None:
    if not investigation.is_dir():
        return None
    best = read_object(investigation / "best_program_manifest.json")
    return {
        "best": {k: best.get(k) for k in ("combined_score", "correct", "program_dir", "source_attempt_id")},
        "workers": [worker_summary(d, investigation) for d in state_dirs(investigation)],
        "evaluations": evaluation_summary(found, top),
    }


def slurm_state(job: str) -> str:
    try:
        result = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T %M %N"], capture_output=True, text=True,
                                timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable ({exc.__class__.__name__})"
    return result.stdout.strip() or "not in queue (finished, failed, or cancelled; check sacct and the job log)"


def summarize(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    investigation = run_dir / "investigation"
    found = published_evaluations(investigation) if investigation.is_dir() else []
    last_evaluation = max((e.mtime for e in found), default=None)
    return {
        "run_dir": str(run_dir),
        "checked": now.isoformat(timespec="seconds"),
        "pipeline": pipeline_summary(run_dir, now, args.stale_minutes, last_evaluation),
        "investigation": investigation_summary(investigation, found, args.top),
        "cost_usd": read_object(run_dir / "metrics" / "cost_summary.json").get("total_cost_usd"),
        "papers": sorted(str(p.relative_to(run_dir)) for p in run_dir.glob("paper/*.pdf")),
        "slurm": slurm_state(args.job) if args.job else None,
    }


def fmt(value: Any) -> str:
    number = finite(value)
    return f"{number:.6g}" if number is not None else str(value)


def render(summary: dict[str, Any]) -> str:
    pipe = summary["pipeline"]
    idle = pipe["last_activity_minutes_ago"]
    lines = [f"run      {summary['run_dir']}",
             f"state    {pipe['state']}" + (f"  · last activity {idle} min ago" if idle is not None else "")]
    if summary["slurm"]:
        lines.append(f"slurm    {summary['slurm']}")
    elapsed = pipe["total_elapsed"] or (f"{pipe['elapsed_hours']} h" if pipe["elapsed_hours"] is not None else "?")
    sessions = f"  · session {pipe['sessions']}" if pipe["sessions"] > 1 else ""
    lines.append(f"started  {pipe['started']}  · elapsed {elapsed}{sessions}")
    if pipe["config"]:
        lines.append("config   " + "  ".join(f"{k}={v}" for k, v in pipe["config"].items()))
    if pipe["activity"]:
        lines.append(f"now      {pipe['activity']}")
    if pipe["phases"]:
        lines.append("phases   " + "  ".join(f"{name}:{info.get('status')}" for name, info in pipe["phases"].items()))
    inv = summary["investigation"]
    if inv:
        best = inv["best"]
        lines.append(f"best     {fmt(best['combined_score'])} (correct={best['correct']}) from {best['source_attempt_id']}")
        for worker in inv["workers"]:
            lines.append(f"  [{worker['dir']}] cycle {worker['cycle']}  attempts {worker['attempts']}  "
                         f"failure streak {worker['failure_streak']}"
                         + (f"  failures {worker['failure_categories']}" if worker["failure_categories"] else ""))
        ev = inv["evaluations"]
        lines.append(f"evals    {ev['count']} ({ev['correct']} correct), latest {ev['latest']}")
        lines += [f"  {fmt(item['score']):>12}  {item['program']}" for item in ev["top"]]
    lines.append(f"cost     ${fmt(summary['cost_usd'])} agent tokens (compute not included)")
    if summary["papers"]:
        lines.append("paper    " + ", ".join(summary["papers"]))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="run dir, or a results root to pick the newest run from")
    parser.add_argument("--job", help="Slurm job ID to include")
    parser.add_argument("--top", type=int, default=5, help="best correct evaluations to list")
    parser.add_argument("--stale-minutes", type=float, default=30.0,
                        help="minutes without journal or evaluation activity before STALLED")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = summarize(resolve_run_dir(args.path.expanduser().resolve()), args)
    print(json.dumps(summary, indent=2, default=str) if args.json else render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

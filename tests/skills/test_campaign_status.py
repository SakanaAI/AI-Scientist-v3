"""Checks for the run-campaign status summary."""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "skills" / "run-campaign" / "scripts"


def run_script(name: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *map(str, args)],
        capture_output=True, text=True, timeout=60, check=False,
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def event(name: str, minutes_ago: float, **fields: object) -> dict:
    stamp = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return {"event": name, "fields": fields, "timestamp": stamp.isoformat().replace("+00:00", "Z")}


def age(path: Path, minutes: float) -> None:
    stamp = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).timestamp()
    os.utime(path, (stamp, stamp))


def evaluation(investigation: Path, name: str, score: float, correct: bool) -> None:
    program = investigation / "ideas" / "01_idea" / "attempt" / name
    write_json(program / "metrics.json", {"combined_score": score})
    write_json(program / "correct.json", {"correct": correct})


def make_run(root: Path, *, idle_minutes: float, name: str = "run_1") -> Path:
    run = root / "task" / "claude" / name
    write_jsonl(run / "pipeline_events.jsonl", [
        event("pipeline_start", 120, **{"agent.name": "claude", "execution.workers": 1}),
        event("phase_start", 119, phase_name="investigation"),
        event("heartbeat", idle_minutes, label="Running Investigation…", phase_name="investigation"),
    ])
    investigation = run / "investigation"
    write_json(investigation / "outer_state.json", {"current_cycle": 3, "best_combined_score": -3.3, "failure_streak": 1})
    write_json(investigation / "best_program_manifest.json", {"combined_score": -3.3, "correct": True, "source_attempt_id": "a2"})
    write_jsonl(investigation / "attempts_history.jsonl", [
        {"attempt_status": "pass"}, {"attempt_status": "fail", "failure_category": "timeout"},
    ])
    for name_, score, correct in (("program_1", -3.4, True), ("program_2", -3.3, True), ("program_3", -1.0, False)):
        evaluation(investigation, name_, score, correct)
    write_json(investigation / ".evaluation_outputs" / "program_1" / "attempt_ab12" / "metrics.json", {"combined_score": 99.0})
    write_json(run / "metrics" / "cost_summary.json", {"total_cost_usd": "12.5"})
    for path in run.rglob("*.json*"):
        age(path, idle_minutes)
    return run


def status(*args: object) -> dict:
    result = run_script("campaign_status.py", *args, "--json")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_status_reports_best_correct_evaluations_and_skips_hidden_copies(tmp_path: Path) -> None:
    make_run(tmp_path, idle_minutes=1)

    summary = status(tmp_path)

    assert summary["pipeline"]["state"] == "RUNNING"
    evaluations = summary["investigation"]["evaluations"]
    assert evaluations["count"] == 3 and evaluations["correct"] == 2
    assert [item["score"] for item in evaluations["top"]] == [-3.3, -3.4]
    assert summary["investigation"]["workers"][0]["attempts"] == {"pass": 1, "fail": 1}


def test_status_picks_the_most_recently_active_run_under_a_root(tmp_path: Path) -> None:
    make_run(tmp_path, idle_minutes=600, name="old")
    newest = make_run(tmp_path, idle_minutes=1, name="new")

    assert status(tmp_path)["run_dir"] == str(newest)


def test_status_flags_silent_run_as_stalled(tmp_path: Path) -> None:
    run = make_run(tmp_path, idle_minutes=90)

    assert status(run)["pipeline"]["state"] == "STALLED"


def test_recent_investigation_journal_keeps_a_quiet_pipeline_running(tmp_path: Path) -> None:
    run = make_run(tmp_path, idle_minutes=90)
    write_jsonl(run / "investigation" / "worker_00" / "round_001" / "events.jsonl", [{"event": "cycle"}])

    assert status(run)["pipeline"]["state"] == "RUNNING"


def test_resumed_run_reports_the_latest_session(tmp_path: Path) -> None:
    run = make_run(tmp_path, idle_minutes=1)
    journal = run / "pipeline_events.jsonl"
    old_session = [event("pipeline_start", 600), event("pipeline_end", 500, status="failed", elapsed="1h")]
    journal.write_text("".join(json.dumps(e) + "\n" for e in old_session) + journal.read_text())

    pipeline = status(run)["pipeline"]

    assert pipeline["state"] == "RUNNING" and pipeline["sessions"] == 2


def test_finished_run_reports_final_status(tmp_path: Path) -> None:
    run = make_run(tmp_path, idle_minutes=1)
    with open(run / "pipeline_events.jsonl", "a") as journal:
        journal.write(json.dumps(event("pipeline_end", 0, status="completed", elapsed="2h")) + "\n")

    assert status(run)["pipeline"]["state"] == "FINISHED (completed)"


def test_parallel_workers_report_their_newest_round(tmp_path: Path) -> None:
    run = make_run(tmp_path, idle_minutes=1)
    for round_name, cycle in (("round_001", 2), ("round_002", 5)):
        write_json(run / "investigation" / "worker_00" / round_name / "outer_state.json", {"current_cycle": cycle})

    workers = status(run)["investigation"]["workers"]

    assert [(w["dir"], w["cycle"]) for w in workers] == [(".", 3), ("worker_00/round_002", 5)]


def test_non_object_state_files_do_not_crash_the_summary(tmp_path: Path) -> None:
    run = make_run(tmp_path, idle_minutes=1)
    write_json(run / "investigation" / "outer_state.json", [1, 2])
    write_json(run / "metrics" / "cost_summary.json", ["x"])

    summary = status(run)

    assert summary["investigation"]["workers"][0]["cycle"] is None and summary["cost_usd"] is None

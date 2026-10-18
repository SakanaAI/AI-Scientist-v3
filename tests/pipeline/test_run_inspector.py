"""apps.run_inspector.backend: discovery, views, graph, endpoints, lazy loading, node identity."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

if sys.version_info < (3, 11):
    pytest.skip("run inspector backend tests require Python 3.11+", allow_module_level=True)

from apps.run_inspector.backend.content_loader import load_text_file, resolve_path_under_run
from apps.run_inspector.backend.discovery import (
    DiscoveredRun,
    RunStub,
    Worker,
    _discover_ideas,
    _discover_workers,
    _extract_best_score,
    _extract_prune_info,
    _extract_session_hints,
    _infer_containerized,
    discover_run_stubs,
    discover_runs,
    discover_single_run,
)
from apps.run_inspector.backend.graph_builder import build_graph
from apps.run_inspector.backend.performance_builder import build_run_performance
from apps.run_inspector.backend.views_builder import build_idea_ancestors, build_idea_timeline

from tests.program_testutils import make_program_dir

AGENTLENS = "http://127.0.0.1:8787"


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(**fields: object) -> dict[str, object]:
    return {key: value for key, value in fields.items() if value is not None}


def _attempt(
    attempt_id: str, idea_id: str, cycle: int, *, number: int = 1, score: float | None = None,
    rubric_pass: bool | None = None, best_updated: bool | None = None, ts: str | None = None,
    worker: int | None = None, output_tail: str | None = None,
) -> dict[str, object]:
    return _row(
        attempt_id=attempt_id, idea_id=idea_id, cycle_index=cycle, attempt_number=number,
        metrics=None if score is None else {"combined_score": score}, rubric_overall_pass=rubric_pass,
        best_updated=best_updated, timestamp=ts, parallel_worker_index=worker, output_tail=output_tail,
    )


def _idea(idea_dir: Path, idea_id: str, title: str, **extra: object) -> None:
    _json(idea_dir / "idea_bundle.json", {"idea_id": idea_id, "title": title, **extra})


def _program(
    program_dir: Path, *, score: float | None = None, correct: bool | None = None,
    content: str = "print('ok')\n", spec: bool = True,
) -> Path:
    if spec:
        make_program_dir(program_dir, content=content)
    else:
        program_dir.mkdir(parents=True, exist_ok=True)
        (program_dir / "program.py").write_text(content, encoding="utf-8")
    if score is not None:
        _json(program_dir / "metrics.json", {"combined_score": score})
    if correct is not None:
        _json(program_dir / "correct.json", {"correct": correct})
    return program_dir


def _markers(run_dir: Path, *, outer: dict[str, object] | None = None, parallel_best: float | None = None,
             ideas: Sequence[dict[str, object]] | None = (), attempts: Sequence[dict[str, object]] | None = ()) -> None:
    """Write run-level marker files; pass ideas/attempts=None to omit that history file."""
    _json(run_dir / "outer_state.json", outer or {})
    if parallel_best is not None:
        _json(run_dir / "parallel_summary.json", {"best_combined_score": parallel_best})
    if ideas is not None:
        _jsonl(run_dir / "ideas_history.jsonl", list(ideas))
    if attempts is not None:
        _jsonl(run_dir / "attempts_history.jsonl", list(attempts))


def _single_run(root: Path) -> list[Any]:
    runs = discover_runs(root)
    assert len(runs) == 1
    return runs


# ---------------------------------------------------------------------------
# discovery + performance + graph
# ---------------------------------------------------------------------------


def test_discover_single_and_parallel_runs_with_performance_and_graph(tmp_path: Path) -> None:
    # single run layout
    single_root = tmp_path / "single"
    run_dir = single_root / "circpack" / "claude"
    idea_dir = run_dir / "ideas" / "01_smoke"
    _markers(run_dir, outer={"best_combined_score": 2.1})
    _idea(idea_dir, "idea_0001", "Smoke")
    _program(idea_dir / "program_0", score=2.1, correct=True)

    [run] = _single_run(single_root)
    assert run.run_id == "circpack/claude"
    assert not run.parallel
    assert len(run.workers) == 1
    assert run.workers[0].synthetic
    assert run.idea_count == 1
    assert run.program_eval_count == 1

    # parallel run, session hints, performance and graph
    parallel_root = tmp_path / "parallel"
    run_dir = parallel_root / "circpack" / "docker_parallel" / "claude"
    worker_dir = run_dir / "worker_01"
    idea_dir = worker_dir / "ideas" / "01_parallel_one"
    idea_dir_two = worker_dir / "ideas" / "02_parallel_two"
    program_dir = idea_dir / "program_1"
    log_name = "idea_agent_claude_20260222_130657.log"
    gen_log = worker_dir / "_idea_generation" / "cycle_0001_1771762017615369858" / log_name

    _markers(
        run_dir,
        parallel_best=2.6,
        attempts=[
            {"attempt_id": "idea_0001_attempt_01", "idea_id": "idea_0001",
             "output_tail": '..."session_id":"sess_abc123"...'}
        ],
    )
    _idea(idea_dir, "idea_0001", "Parallel One")
    _idea(idea_dir_two, "idea_0002", "Parallel Two")
    _program(program_dir, score=2.6, correct=True)
    (program_dir / "src" / "utils.py").write_text("ANSWER = 42\n", encoding="utf-8")
    (program_dir / "src" / "excluded.py").write_text("SHOULD_NOT_APPEAR = True\n", encoding="utf-8")
    (program_dir / "program_spec.toml").write_text(
        'schema_version = 1\ncode_root = "src"\n\n[entrypoint]\npath = "program.py"\n'
        'language = "python"\ntype = "script"\n\n[copy]\nexclude = ["src/excluded.py"]\ninclude = []\n\n'
        '[integration]\nkind = "generic"\nnotes = ""\n',
        encoding="utf-8",
    )
    _json(program_dir / "rubric_status.json", {"attempt_id": "idea_0001_attempt_01", "idea_id": "idea_0001"})
    _program(idea_dir_two / "program_2", score=2.2, correct=False, content="print('ok2')\n")
    gen_log.parent.mkdir(parents=True)
    gen_log.write_text("idea generation trace\n", encoding="utf-8")

    [run] = _single_run(parallel_root)
    assert run.parallel
    assert len(run.workers) == 1
    assert run.session_hints.by_attempt_id["idea_0001_attempt_01"] == "sess_abc123"
    assert run.best_combined_score == 2.6

    performance = build_run_performance(run)
    assert len(performance.ideas) == 2
    assert len(performance.programs) == 2
    assert performance.max_program_combined_score == 2.6
    assert performance.max_idea_combined_score == 2.6
    first, second = performance.ideas
    assert (first.idea_id, first.program_count, first.correct_program_count, first.incorrect_program_count) == (
        "idea_0001", 1, 1, 0)
    assert first.best_combined_score == 2.6
    assert first.best_correct
    assert (second.idea_id, second.correct_program_count, second.incorrect_program_count) == ("idea_0002", 0, 1)
    assert second.best_combined_score == 2.2
    assert not second.best_correct
    assert performance.programs[0].timestamp_unix is not None
    assert str(performance.programs[0].program_code_path).endswith("/src/program.py")

    graph = build_graph(run, agentlens_base_url=AGENTLENS)
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    [worker_node] = [node for node in graph.nodes if node.type == "worker"]
    idea_nodes = sorted(
        (node for node in graph.nodes if node.type == "idea"),
        key=lambda node: int(node.data.get("order_index", 0)),
    )
    assert (graph.edges[0].source, graph.edges[0].target) == (worker_node.id, idea_nodes[0].id)
    assert (graph.edges[1].source, graph.edges[1].target) == (idea_nodes[0].id, idea_nodes[1].id)

    detail = graph.details[idea_nodes[0].id]
    for node_detail in (graph.details[worker_node.id], detail):
        [gen_artifact] = [a for a in node_detail["artifacts"] if log_name in a.path]
        assert gen_artifact.exists
        assert "/trace-file/" in gen_artifact.agentlens_url
    assert "session_id=sess_abc123" in detail["agentlens_url"]
    [program] = detail["metadata"]["programs"]
    assert (program["label"], program["combined_score"], program["correct"]) == ("program_1", 2.6, True)
    assert any(artifact.kind == "code" for artifact in detail["artifacts"])
    artifact_paths = {artifact.path for artifact in detail["artifacts"]}
    assert "worker_01/ideas/01_parallel_one/program_1/src/utils.py" in artifact_paths
    assert "worker_01/ideas/01_parallel_one/program_1/src/excluded.py" not in artifact_paths


# ---------------------------------------------------------------------------
# views builder
# ---------------------------------------------------------------------------


def test_idea_timeline_entries_and_correct_without_rubric_is_not_any_pass(tmp_path: Path) -> None:
    # timeline entries
    root = tmp_path / "entries"
    run_dir = root / "task" / "agent"
    idea_dir = run_dir / "ideas" / "01_first_idea"
    attempt_dir = idea_dir / "01_attempt"
    _markers(
        run_dir,
        outer={"best_combined_score": 5.0},
        ideas=[_row(idea_id="idea_001", title="First Idea", cycle_index=0, timestamp="2026-02-20T10:00:00")],
        attempts=[
            _attempt("idea_001_att_01", "idea_001", 0, number=1, score=3.0, rubric_pass=False,
                     best_updated=False, ts="2026-02-20T10:01:00"),
            _attempt("idea_001_att_02", "idea_001", 0, number=2, score=5.0, rubric_pass=True,
                     best_updated=True, ts="2026-02-20T10:02:00"),
            _attempt("idea_001_att_03", "idea_001", 0, number=3, score=4.0, rubric_pass=True,
                     best_updated=False, ts="2026-02-20T10:03:00"),
        ],
    )
    _jsonl(run_dir / "global_memory.jsonl", [
        {"event": "baseline", "timestamp": "2026-02-20T10:00:00", "summary": "Baseline run", "tags": ["init"]},
        {"event": "evaluate", "timestamp": "2026-02-20T10:01:00", "idea_id": "idea_001", "cycle": 0,
         "metrics": {"combined_score": 5.0}},
    ])
    _idea(idea_dir, "idea_001", "First Idea", hypothesis="Test hypothesis")
    _json(attempt_dir / "active_idea.json",
          {"idea_id": "idea_001", "hypothesis": "Test hypothesis", "implementation_plan": ["step1", "step2"]})
    _json(attempt_dir / "rubric_status.json", {
        "overall_pass": True,
        "rubrics": [
            {"id": "rubric_speed", "pass": True, "evidence": "Fast enough"},
            {"id": "rubric_accuracy", "pass": False, "evidence": "Below threshold"},
        ],
    })
    _program(attempt_dir / "program_0", score=5.0, correct=True, content="print('hello')\n")

    [run] = _single_run(root)
    [entry] = build_idea_timeline(run).ideas
    assert (entry.idea_id, entry.title, entry.hypothesis) == ("idea_001", "First Idea", "Test hypothesis")
    assert entry.implementation_plan == ["step1", "step2"]
    assert entry.best_score == 5.0
    assert entry.any_pass
    assert entry.rubric_summary is not None
    assert (entry.rubric_summary["passed"], entry.rubric_summary["total"]) == (1, 2)

    # correct without rubric pass is not any_pass
    root = tmp_path / "no_rubric"
    run_dir = root / "task" / "agent"
    idea_dir = run_dir / "ideas" / "01_idea"
    _markers(run_dir, outer={"best_combined_score": 1.0},
             ideas=[{"idea_id": "idea_x", "title": "X", "cycle_index": 0}])
    _idea(idea_dir, "idea_x", "X")
    _program(idea_dir / "program_0", score=1.0, correct=True)  # correct=true but no rubric_status

    [run] = _single_run(root)
    [entry] = build_idea_timeline(run).ideas
    assert not entry.any_pass


def _two_worker_collision(run_dir: Path, titles: tuple[str, str], scores: tuple[float, float]) -> tuple[Path, Path]:
    """Two workers that share idea_id 'idea_0001'; returns the worker dirs."""
    workers = (run_dir / "worker_01", run_dir / "worker_02")
    for worker, title, score, content in zip(workers, titles, scores, ("x\n", "y\n")):
        idea_dir = worker / "ideas" / "01_idea_0001"
        _idea(idea_dir, "idea_0001", title)
        _program(idea_dir / "program_0", score=score, content=content)
    return workers


def _collision_history(worker: int, title: str, score: float, rubric_pass: bool, *, tagged: bool) -> tuple[
    dict[str, object], dict[str, object]]:
    tag = worker if tagged else None
    return (
        _row(idea_id="idea_0001", title=title, cycle_index=worker, parallel_worker_index=tag,
             timestamp=f"2026-01-01T0{worker}:00:00"),
        _attempt(f"w{worker}a1", "idea_0001", worker, score=score, rubric_pass=rubric_pass, worker=tag,
                 ts=f"2026-01-01T0{worker}:01:00"),
    )


def _assert_collision_timeline(run: Any) -> None:
    timeline = build_idea_timeline(run)
    assert len(timeline.ideas) == 2
    by_worker = {entry.worker_name: entry for entry in timeline.ideas}
    assert set(by_worker) == {"worker_01", "worker_02"}
    assert (by_worker["worker_01"].cycle_index, by_worker["worker_01"].num_attempts) == (1, 1)
    assert (by_worker["worker_02"].cycle_index, by_worker["worker_02"].num_attempts) == (2, 1)


def test_cross_worker_idea_id_collisions_stay_separate_in_timeline_and_ancestors(tmp_path: Path) -> None:
    # worker-level history placed after discovery must not mix metadata across workers
    root = tmp_path / "collision"
    run_dir = root / "task" / "par_collision"
    _markers(run_dir, parallel_best=9.0)
    w1, w2 = _two_worker_collision(run_dir, ("W1 Batch Idea", "W2 LR Idea"), (4.0, 9.0))
    [run] = _single_run(root)

    for worker, (idx, title, score, passed) in zip((w1, w2), ((1, "W1 Batch Idea", 4.0, False), (2, "W2 LR Idea", 9.0, True))):
        idea_row, attempt_row = _collision_history(idx, title, score, passed, tagged=False)
        _jsonl(worker / "ideas_history.jsonl", [idea_row])
        _jsonl(worker / "attempts_history.jsonl", [attempt_row])

    _assert_collision_timeline(run)

    # only run-level aggregated JSONL (with parallel_worker_index) keeps same-id ideas separate
    root = tmp_path / "aggregated"
    run_dir = root / "task" / "par_agg"
    rows = [_collision_history(1, "W1 Batch Idea", 4.0, False, tagged=True),
            _collision_history(2, "W2 LR Idea", 9.0, True, tagged=True)]
    _markers(run_dir, parallel_best=9.0, ideas=[r[0] for r in rows], attempts=[r[1] for r in rows])
    _two_worker_collision(run_dir, ("W1 Batch Idea", "W2 LR Idea"), (4.0, 9.0))

    [run] = _single_run(root)
    _assert_collision_timeline(run)

    # idea ancestors worker disambiguation
    root = tmp_path / "ancestors"
    run_dir = root / "task" / "par_wn"
    _markers(run_dir, parallel_best=9.0)
    workers = _two_worker_collision(run_dir, ("W1 Idea", "W2 Idea"), (1.0, 9.0))
    [run] = _single_run(root)
    for worker, title in zip(workers, ("W1 Idea", "W2 Idea")):
        _jsonl(worker / "ideas_history.jsonl", [{"idea_id": "idea_0001", "title": title, "cycle_index": 1}])
        _jsonl(worker / "attempts_history.jsonl", [])

    assert build_idea_ancestors(run, "idea_0001").idea_id == "idea_0001"
    anc_w2 = build_idea_ancestors(run, "idea_0001", worker_name="worker_02")
    assert anc_w2.idea_id == "idea_0001"
    if anc_w2.target_program is not None:
        assert "worker_02" in anc_w2.target_program.path


def test_parallel_run_views_and_ancestors_across_pruning_with_programs(tmp_path: Path) -> None:
    # parallel run views: worker dirs carry no marker files; history lives at the run level
    root = tmp_path / "parallel"
    run_dir = root / "task" / "parallel_run"
    _markers(
        run_dir,
        parallel_best=8.0,
        ideas=[{"idea_id": "w1_idea", "title": "Worker1 Idea", "cycle_index": 0},
               {"idea_id": "w2_idea", "title": "Worker2 Idea", "cycle_index": 0}],
        attempts=[_attempt("w1_att1", "w1_idea", 0, score=6.0, rubric_pass=True, ts="2026-02-20T11:00:00"),
                  _attempt("w2_att1", "w2_idea", 0, score=8.0, rubric_pass=True, ts="2026-02-20T11:01:00")],
    )
    for worker, idea_id, title, score in (("worker_01", "w1_idea", "Worker1 Idea", 6.0),
                                          ("worker_02", "w2_idea", "Worker2 Idea", 8.0)):
        idea_dir = run_dir / worker / "ideas" / f"01_{idea_id}"
        _idea(idea_dir, idea_id, title)
        program_dir = idea_dir / "program_0"
        program_dir.mkdir(parents=True)
        _json(program_dir / "metrics.json", {"combined_score": score})
        _json(program_dir / "correct.json", {"correct": True})

    [run] = _single_run(root)
    timeline = build_idea_timeline(run)
    assert len(timeline.ideas) == 2
    assert {entry.worker_name for entry in timeline.ideas} == {"worker_01", "worker_02"}

    # idea ancestors after pruning: ancestors are found from archived (pruned) chapters
    root = tmp_path / "pruned"
    run_dir = root / "task" / "agent"
    _markers(
        run_dir,
        outer={"best_combined_score": 7.0, "prune_every_n_cycles": 2},
        ideas=[{"idea_id": "idea_002", "title": "Post-prune Idea", "cycle_index": 2}],
        attempts=[_attempt("att_002", "idea_002", 2, score=7.0, best_updated=True, ts="2026-02-20T12:00:00")],
    )
    archive = run_dir / "_pruned" / "chapter_000"
    _jsonl(archive / "ideas_history.jsonl", [{"idea_id": "idea_001", "title": "Pre-prune Idea", "cycle_index": 0}])
    _jsonl(archive / "attempts_history.jsonl",
           [_attempt("att_001", "idea_001", 0, score=5.0, best_updated=True, ts="2026-02-20T10:00:00")])
    for name, idea_id, title, hyp, score in (("01_pre_prune_idea", "idea_001", "Pre-prune Idea", "H1", 5.0),
                                             ("02_post_prune_idea", "idea_002", "Post-prune Idea", "H2", 7.0)):
        idea_dir = run_dir / "ideas" / name
        _idea(idea_dir, idea_id, title)
        _json(idea_dir / "01_attempt" / "active_idea.json", {"idea_id": idea_id, "hypothesis": hyp})
        _program(idea_dir / "01_attempt" / "program_0", score=score, spec=False, content=f"print('{name}')\n")

    [run] = _single_run(root)
    anc_pre = build_idea_ancestors(run, "idea_001")
    assert anc_pre.idea_id == "idea_001"
    assert anc_pre.target_program is not None, "Pre-prune idea should have a target program"
    anc_post = build_idea_ancestors(run, "idea_002")
    assert anc_post.idea_id == "idea_002"
    assert anc_post.target_program is not None
    assert [ancestor.idea_id for ancestor in anc_post.ancestors] == ["idea_001"]

    # ancestor entry includes programs
    root = tmp_path / "programs"
    run_dir = root / "task" / "agent"
    _markers(
        run_dir,
        outer={"best_combined_score": 8.0},
        ideas=[{"idea_id": "idea_001", "title": "Idea 1", "cycle_index": 0},
               {"idea_id": "idea_002", "title": "Idea 2", "cycle_index": 1}],
        attempts=[_attempt("att_001", "idea_001", 0, best_updated=True, ts="2026-01-01T00:00:00"),
                  _attempt("att_002", "idea_002", 1, best_updated=True, ts="2026-01-01T01:00:00")],
    )
    idea1_dir = run_dir / "ideas" / "01_idea"
    idea2_dir = run_dir / "ideas" / "02_idea"
    _idea(idea1_dir, "idea_001", "Idea 1")
    _idea(idea2_dir, "idea_002", "Idea 2")
    for program_dir in (idea1_dir / "01_attempt/program_0", idea1_dir / "01_attempt/program_1",
                        idea2_dir / "01_attempt/program_0"):
        _program(program_dir, score=5.0, spec=False, content="# code\n")

    [run] = _single_run(root)
    anc = build_idea_ancestors(run, "idea_002")
    assert len(anc.ancestors) == 1
    assert anc.ancestors[0].idea_id == "idea_001"
    labels = {p.label for p in anc.ancestors[0].programs}
    assert len(anc.ancestors[0].programs) == 2, "Ancestor should list both programs from idea_001"
    assert any("program_0" in label for label in labels), labels
    assert any("program_1" in label for label in labels), labels


def test_ancestor_programs_list_spec_layout_program_dirs(tmp_path: Path) -> None:
    run_dir = tmp_path / "task" / "agent"
    _markers(
        run_dir,
        outer={"best_combined_score": 8.0},
        ideas=[{"idea_id": "idea_001", "title": "Idea 1", "cycle_index": 0},
               {"idea_id": "idea_002", "title": "Idea 2", "cycle_index": 1}],
        attempts=[_attempt("att_001", "idea_001", 0, best_updated=True, ts="2026-01-01T00:00:00"),
                  _attempt("att_002", "idea_002", 1, best_updated=True, ts="2026-01-01T01:00:00")],
    )
    for name, idea_id, title in (("01_idea", "idea_001", "Idea 1"), ("02_idea", "idea_002", "Idea 2")):
        _idea(run_dir / "ideas" / name, idea_id, title)
        _program(run_dir / "ideas" / name / "01_attempt" / "program_0", score=5.0)

    [run] = _single_run(tmp_path)
    [ancestor] = build_idea_ancestors(run, "idea_002").ancestors

    assert ancestor.program_info is not None
    assert [p.program_info for p in ancestor.programs] == [ancestor.program_info]


# ---------------------------------------------------------------------------
# endpoint routes (called directly; ASGI threadpool can deadlock in the sandbox)
# ---------------------------------------------------------------------------


def _endpoint_run(root: Path) -> Path:
    run_dir = root / "task" / "agent"
    idea_dir = run_dir / "ideas" / "01_first_idea"
    _markers(
        run_dir,
        outer={"best_combined_score": 1.0},
        ideas=[{"idea_id": "idea_001", "title": "Idea", "cycle_index": 0}],
        attempts=[_attempt("att_01", "idea_001", 0, score=1.0, rubric_pass=True, ts="2026-01-01T00:00:00")],
    )
    _jsonl(run_dir / "global_memory.jsonl", [{"event": "init", "timestamp": "2026-01-01T00:00:00"}])
    _idea(idea_dir, "idea_001", "Idea")
    _program(idea_dir / "program_0", score=1.0, correct=True)
    return run_dir


def _get_endpoint(app: Any, path_format: str) -> Any:
    for route in app.routes:
        if getattr(route, "path_format", None) == path_format and "GET" in getattr(route, "methods", set()):
            return route.endpoint
    pytest.fail(f"GET route not found: {path_format}")


def _app(root: Path) -> Any:
    from apps.run_inspector.backend.app import create_app

    return create_app(results_root=root, frontend_dist=Path("/nonexistent"))


def test_endpoints_list_ideas_diff_program_dirs_and_404_unknown_run(tmp_path: Path) -> None:
    from fastapi import HTTPException

    # ideas endpoint
    idea_dir = _endpoint_run(tmp_path) / "ideas" / "01_first_idea"
    app = _app(tmp_path)
    data = _get_endpoint(app, "/api/runs/{run_id}/ideas")("task/agent")
    assert data.run_id == "task/agent"
    assert isinstance(data.ideas, list)
    assert len(data.ideas) == 1

    # 404 for unknown run
    with pytest.raises(HTTPException) as ctx:
        _get_endpoint(app, "/api/runs/{run_id}/ideas")("nonexistent/run")
    assert ctx.value.status_code == 404

    # program dir diff endpoint
    (idea_dir / "program_0" / "src" / "old.py").write_text("old\n", encoding="utf-8")
    program1 = idea_dir / "program_1"
    make_program_dir(program1, content="print('new')\n")
    (program1 / "src" / "utils.py").write_text("utils\n", encoding="utf-8")
    (program1 / "src" / "blob.bin").write_bytes(b"abc\x00xyz")

    diff = _get_endpoint(_app(tmp_path), "/api/runs/{run_id}/program-dir-diff")(
        "task/agent", from_dir="ideas/01_first_idea/program_0", to_dir="ideas/01_first_idea/program_1"
    )

    by_rel = {f.relpath: f for f in diff.files}
    assert by_rel["src/program.py"].status == "modified"
    assert by_rel["src/utils.py"].status == "added"
    assert by_rel["src/blob.bin"].status == "added"
    assert by_rel["src/blob.bin"].is_binary
    assert by_rel["src/old.py"].status == "removed"
    assert (diff.stats.added, diff.stats.removed, diff.stats.modified) == (2, 1, 1)


# ---------------------------------------------------------------------------
# round-based isolation layout
# ---------------------------------------------------------------------------


def _round_idea(round_dir: Path, name: str, idea_id: str, title: str, score: float,
                correct: bool | None = None) -> None:
    idea_dir = round_dir / "ideas" / name
    _idea(idea_dir, idea_id, title)
    _program(idea_dir / "program_0", score=score, correct=correct, spec=False, content=f"# {idea_id}\n")


@pytest.fixture()
def round_run(tmp_path: Path) -> Path:
    """worker_01/round_000+round_001, worker_02/round_000, archived _pruned/round_000."""
    run_dir = tmp_path / "task" / "round_agent"
    for round_idx, idea_id in ((0, "idea_r0"), (1, "idea_r1")):
        _round_idea(run_dir / "worker_01" / f"round_{round_idx:03d}", f"01_{idea_id}", idea_id,
                    f"Idea {idea_id}", 1.0 + round_idx, correct=True)
    _round_idea(run_dir / "worker_02" / "round_000", "01_idea_r0_w2", "idea_r0_w2", "Worker2 Round0", 3.0)
    _markers(
        run_dir,
        outer={"best_combined_score": 3.0, "prune_every_n_cycles": 2},
        parallel_best=3.0,
        ideas=[{"idea_id": "idea_r0", "title": "Idea idea_r0", "cycle_index": 1},
               {"idea_id": "idea_r1", "title": "Idea idea_r1", "cycle_index": 2},
               {"idea_id": "idea_r0_w2", "title": "Worker2 Round0", "cycle_index": 1}],
    )
    archive = run_dir / "_pruned" / "round_000"
    _jsonl(archive / "ideas_history.jsonl", [{"idea_id": "idea_r0", "title": "Idea idea_r0", "cycle_index": 1}])
    _jsonl(archive / "attempts_history.jsonl", [])
    return run_dir


def test_round_isolation_layout_discovery_timeline_and_generation_logs(round_run: Path) -> None:
    # discover round isolation layout and ideas
    [run] = _single_run(round_run.parent.parent)
    assert run.parallel
    assert len(run.workers) == 2
    assert not any(worker.synthetic for worker in run.workers)
    by_name = {worker.worker_name: worker for worker in run.workers}
    assert {idea.idea_id for idea in by_name["worker_01"].ideas} == {"idea_r0", "idea_r1"}
    assert [idea.idea_id for idea in by_name["worker_02"].ideas] == ["idea_r0_w2"]
    assert run.idea_count == 3
    assert run.prune_every_n_cycles == 2
    assert run.prune_chapters == [0]

    # timeline with round isolation
    timeline = build_idea_timeline(run)
    assert {e.idea_id for e in timeline.ideas} == {"idea_r0", "idea_r1", "idea_r0_w2"}
    assert len(timeline.ideas) == 3
    assert len(timeline.prune_events) == 1
    assert timeline.prune_events[0].chapter == 0

    # idea generation logs in round dirs
    gen_dir = round_run / "worker_01" / "round_000" / "_idea_generation" / "cycle_0001"
    gen_dir.mkdir(parents=True)
    (gen_dir / "idea_agent_claude_20260301.log").write_text("trace data\n", encoding="utf-8")

    [run] = _single_run(round_run.parent.parent)
    graph = build_graph(run, agentlens_base_url=AGENTLENS)
    worker_nodes = [n for n in graph.nodes if n.type == "worker"]
    assert worker_nodes
    w1_node = [n for n in worker_nodes if "1" in n.label][0]
    [log_artifact] = [a for a in graph.details[w1_node.id]["artifacts"] if "idea_agent_claude_20260301.log" in a.path]
    assert log_artifact.exists


def test_round_candidate_promotion_and_history_from_round_subdirs(tmp_path: Path) -> None:
    # round candidate promotion: marker files inside round_NNN/ promote to the run dir
    root = tmp_path / "promo"
    run_dir = root / "task" / "round_promo"
    round_dir = run_dir / "worker_01" / "round_000"
    _json(round_dir / "outer_state.json", {})
    _jsonl(round_dir / "ideas_history.jsonl", [])
    _idea(round_dir / "ideas" / "01_smoke", "s1", "Smoke")
    _program(round_dir / "ideas" / "01_smoke" / "program_0", spec=False, content="ok\n")
    _json(run_dir / "parallel_summary.json", {})

    [run] = _single_run(root)
    assert run.run_id == "task/round_promo"

    # history only inside round_NNN/ (not at worker level) still yields cycle_index and attempts
    root = tmp_path / "history"
    run_dir = root / "task" / "round_history"
    for round_idx, idea_id, cycle in ((0, "idea_a", 0), (1, "idea_b", 2)):
        round_dir = run_dir / "worker_01" / f"round_{round_idx:03d}"
        _round_idea(round_dir, f"01_{idea_id}", idea_id, f"Idea {idea_id}", 1.0 + round_idx)
        _jsonl(round_dir / "ideas_history.jsonl", [{"idea_id": idea_id, "title": f"Idea {idea_id}", "cycle_index": cycle}])
        _jsonl(round_dir / "attempts_history.jsonl",
               [_attempt(f"att_{idea_id}", idea_id, cycle, best_updated=True, ts="2026-01-01T00:00:00")])
    _markers(run_dir, outer={"best_combined_score": 2.0, "prune_every_n_cycles": 2}, parallel_best=2.0,
             ideas=None, attempts=None)
    archive = run_dir / "_pruned" / "round_000"
    _jsonl(archive / "ideas_history.jsonl", [{"idea_id": "idea_a", "title": "Idea idea_a", "cycle_index": 0}])
    _jsonl(archive / "attempts_history.jsonl", [])

    [run] = _single_run(root)
    ideas_by_id = {e.idea_id: e for e in build_idea_timeline(run).ideas}
    assert ideas_by_id["idea_a"].cycle_index == 0
    assert ideas_by_id["idea_b"].cycle_index == 2
    assert any(a.best_updated for a in ideas_by_id["idea_b"].attempts), "idea_b should have best_updated attempt"


# ---------------------------------------------------------------------------
# content loader
# ---------------------------------------------------------------------------


def test_content_loader_blocks_escape_and_rejects_binary(tmp_path: Path) -> None:
    # resolve_path_under_run blocks escape
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_path_under_run(run_dir, str(outside))
    # binary rejected
    file_path = tmp_path / "binary.dat"
    file_path.write_bytes(b"abc\x00xyz")
    with pytest.raises(ValueError):
        load_text_file(file_path)


# ---------------------------------------------------------------------------
# node identity for duplicate idea ids across rounds
# ---------------------------------------------------------------------------


def _build_run_from_parts(run_dir: Path) -> DiscoveredRun:
    worker_descs, parallel = _discover_workers(run_dir)
    workers = [
        Worker(worker_index=w.worker_index, worker_name=w.worker_name, path=w.path, synthetic=w.synthetic,
               ideas=_discover_ideas(w.path))
        for w in worker_descs
    ]
    prune_every_n_cycles, prune_chapters = _extract_prune_info(run_dir)
    return DiscoveredRun(
        run_id="ed_conv/claude/20260331_094917", path=run_dir, task_name="ed_conv", agent="claude",
        containerized=_infer_containerized(run_dir), parallel=parallel, workers=workers,
        best_combined_score=_extract_best_score(run_dir), session_hints=_extract_session_hints(run_dir),
        prune_every_n_cycles=prune_every_n_cycles, prune_chapters=prune_chapters,
    )


def test_run_inspector_uses_unique_ids_for_duplicate_ideas(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "ed_conv" / "claude" / "20260331_094917"
    title = "Dual Stream Conv Error Diffusion for CIFAR-10"
    _json(run_dir / "outer_state.json", {"prune_every_n_cycles": 0})
    for round_name, cycle_index, best_updated in (("round_003", 1, True), ("round_004", 2, False)):
        round_dir = run_dir / "worker_01" / round_name
        _idea(round_dir / "ideas" / "01_dual_stream_conv_error_diffusion_for_cifar_10", "ed_conv_cifar10_v1", title)
        _jsonl(round_dir / "ideas_history.jsonl",
               [{"idea_id": "ed_conv_cifar10_v1", "cycle_index": cycle_index, "title": title}])
        _jsonl(round_dir / "attempts_history.jsonl",
               [{"idea_id": "ed_conv_cifar10_v1", "best_updated": best_updated, "attempt_id": f"attempt_{round_name}"}])
    run = _build_run_from_parts(run_dir)

    graph = build_graph(run, agentlens_base_url=AGENTLENS)
    idea_node_ids = [node.id for node in graph.nodes if node.type == "idea"]
    assert len(idea_node_ids) == 2
    assert len(set(idea_node_ids)) == 2
    assert len(graph.details) == len(graph.nodes)

    timeline = build_idea_timeline(run)
    timeline_node_ids = [entry.node_id for entry in timeline.ideas]
    assert len(timeline.ideas) == 2
    assert all(node_id is not None for node_id in timeline_node_ids)
    assert len(set(timeline_node_ids)) == 2
    assert timeline.ideas[0].idea_path != timeline.ideas[1].idea_path


# ---------------------------------------------------------------------------
# lazy loading (RunStore)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "task" / "agent_run"
    _json(run_dir / "outer_state.json", {"best_combined_score": 0.85})
    idea_dir = run_dir / "worker_01" / "ideas" / "1_my_idea"
    _idea(idea_dir, "idea_1", "My Idea")
    _json(idea_dir / "program_0" / "metrics.json", {"combined_score": 0.85})
    return tmp_path


def _store(root: Path) -> Any:
    from apps.run_inspector.backend.app import RunStore

    return RunStore(results_root=root, agentlens_base_url="http://localhost:8787")


def test_run_stubs_lazy_store_preparation_and_unknown_run(fake_run_dir: Path) -> None:
    # discover run stubs and single run
    [stub] = discover_run_stubs(fake_run_dir)
    assert isinstance(stub, RunStub)
    assert (stub.run_id, stub.task_name, stub.agent, stub.best_combined_score) == (
        "task/agent_run", "task", "agent_run", 0.85)

    run = discover_single_run(stub.path, stub.run_id, stub.task_name, stub.agent)
    assert run is not None
    assert run.run_id == "task/agent_run"
    assert (len(run.workers), run.idea_count, run.program_eval_count) == (1, 1, 1)

    # run store refresh is lightweight, then lazily prepares on access
    store = _store(fake_run_dir)
    with patch("apps.run_inspector.backend.app.discover_single_run") as mock_single:
        store.refresh()
        mock_single.assert_not_called()

    [summary] = store.list_runs()
    assert summary.run_id == "task/agent_run"
    assert (summary.worker_count, summary.idea_count, summary.program_eval_count) == (None, None, None)

    graph = store.run_graph("task/agent_run")
    assert graph is not None
    assert len(graph.nodes) > 0
    [summary] = store.list_runs()
    assert (summary.worker_count, summary.idea_count, summary.program_eval_count) == (1, 1, 1)

    # unknown run raises
    with pytest.raises(KeyError):
        store.run_graph("nonexistent/run")

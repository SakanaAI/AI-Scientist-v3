"""Draft smoke must reject evaluators that score one program differently twice (issue #5)."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from src.investigation import task_runtime as runtime
from src.investigation.evaluator_replay import EvaluationOutcome, replay_mismatch
from tests.final_evaluation.test_final_evaluation import _write_contract_task
from src.task_drafting.checks import run_gate
from tests.program_testutils import make_program_dir
from tests.task_drafting.test_baseline_artifacts import _Broker
from tests.task_drafting.test_task_drafting_checks import EVALUATOR, _scaffold, _settings

IMAGE = "sha256:" + "a" * 64


def test_identical_outcomes_agree():
    outcome = EvaluationOutcome(0.41, True)
    assert replay_mismatch(outcome, EvaluationOutcome(0.41, True)) is None


def test_summation_order_noise_is_tolerated():
    assert replay_mismatch(EvaluationOutcome(0.1 + 0.2, True), EvaluationOutcome(0.3, True)) is None


def test_score_spread_is_reported_with_both_values():
    diagnostic = replay_mismatch(EvaluationOutcome(0.41, True), EvaluationOutcome(0.62, True))
    assert diagnostic is not None
    assert diagnostic.startswith("evaluator is not deterministic")
    assert "0.41" in diagnostic and "0.62" in diagnostic
    assert "never drawn per invocation" in diagnostic


def test_correctness_flip_is_reported():
    assert replay_mismatch(EvaluationOutcome(1.0, True), EvaluationOutcome(1.0, False)) is not None


def _contracted_smoke_task(tmp_path: Path) -> Path:
    task = _write_contract_task(tmp_path)
    shutil.rmtree(task / ".venv")
    make_program_dir(task / "program_0")
    contract_path = task / "task_contract.json"
    payload = json.loads(contract_path.read_text())
    payload["metrics"]["search"] = [{"name": "combined_score", "data_role": "development"}]
    contract_path.write_text(json.dumps(payload))
    return task


def _docker_run(scores: Iterator[float]):
    def run(command, **kwargs):
        if command[:4] == ["docker", "image", "inspect", "--format={{.Id}}"]:
            return subprocess.CompletedProcess(command, 0, IMAGE + "\n", "")
        if "venv" in command:
            mount = next(value for value in command if value.endswith(":/runtime:rw"))
            environment = Path(mount.removesuffix(":/runtime:rw")) / ".venv"
            (environment / "bin").mkdir(parents=True)
            (environment / "bin/python").write_text("")
            (environment / "lib/python3.12/site-packages/src/investigation").mkdir(parents=True)
        if "/task/evaluate.py" in command:
            mount = next(value for value in command if value.endswith(",dst=/results"))
            results = Path(mount.removeprefix("type=bind,src=").removesuffix(",dst=/results"))
            (results / "metrics.json").write_text(json.dumps({"combined_score": next(scores)}))
            (results / "correct.json").write_text('{"correct":true}')
        return subprocess.CompletedProcess(command, 0, "", "")

    return run


def _prepare_smoke(task: Path, monkeypatch, scores: list[float]):
    monkeypatch.setattr(runtime, "CandidateBroker", _Broker)
    monkeypatch.setattr(runtime, "_force_remove_container", lambda *args, **kwargs: True)
    return runtime.prepare_task_runtime_result(
        task, containerize=True, container_image=IMAGE, run=_docker_run(iter(scores)),
        base_environment={"PATH": "/bin"}, draft_smoke=True,
        evaluation_requirement=runtime.TaskRuntimeEvaluationRequirement.VALID_RESULT,
    )


def test_draft_smoke_accepts_reproducible_evaluator(tmp_path, monkeypatch):
    task = _contracted_smoke_task(tmp_path)
    result = _prepare_smoke(task, monkeypatch, [0.41, 0.41])
    assert result is runtime.TaskRuntimePreparationResult.READY


def test_draft_smoke_rejects_per_invocation_entropy_as_authored(tmp_path, monkeypatch, caplog):
    task = _contracted_smoke_task(tmp_path)
    with caplog.at_level(logging.ERROR, logger=runtime.logger.name):
        result = _prepare_smoke(task, monkeypatch, [0.41, 0.62])
    assert result is runtime.TaskRuntimePreparationResult.AUTHORED_FAILURE
    assert "evaluator is not deterministic" in caplog.text


def test_production_evaluation_does_not_replay(tmp_path, monkeypatch):
    task = _contracted_smoke_task(tmp_path)
    monkeypatch.setattr(runtime, "CandidateBroker", _Broker)
    monkeypatch.setattr(runtime, "_force_remove_container", lambda *args, **kwargs: True)
    remaining = iter([0.41, 0.41])
    result = runtime.prepare_task_runtime_result(
        task, containerize=True, container_image=IMAGE, run=_docker_run(remaining),
        base_environment={"PATH": "/bin"},
        evaluation_requirement=runtime.TaskRuntimeEvaluationRequirement.VALID_RESULT,
    )
    assert result is runtime.TaskRuntimePreparationResult.READY
    assert next(remaining) == 0.41  # only one of the two scores was consumed


def test_native_search_smoke_rejects_per_invocation_secret(tmp_path):
    entropy = EVALUATOR.replace(
        "json.dumps({metric: 1.0})",
        "json.dumps({metric: __import__('secrets').randbelow(10**9) / 10**9})",
    )
    task = _scaffold(tmp_path / "entropy", search=entropy)
    report = run_gate("search-smoke", _settings(task))
    assert report.status == "authored_failure"
    assert report.category == "search_nondeterministic"
    assert "evaluator is not deterministic" in report.diagnostic


def test_native_search_smoke_accepts_task_owned_secret(tmp_path):
    fixed = EVALUATOR.replace(
        "json.dumps({metric: 1.0})",
        "json.dumps({metric: __import__('random').Random(b'task-owned').random()})",
    )
    task = _scaffold(tmp_path / "fixed", search=fixed)
    assert run_gate("search-smoke", _settings(task)).status == "pass"

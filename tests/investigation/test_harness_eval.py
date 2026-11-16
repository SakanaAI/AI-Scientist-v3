"""Harness evaluation: rubric status, sealed/direct execution, source inspection, feedback."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.investigation import harness_sources
from src.investigation.evaluator_tree import managed_evaluator_tree_digest
from src.investigation.files import evaluation_results_complete
from src.investigation.harness_incumbent import reconcile_harness_incumbent
from src.investigation.evaluate_program import ContainedRun
from src.investigation.harness_eval import (
    bootstrap_native_incumbent,
    SealedHarnessConfig,
    _ensure_agent_only_created_program_1,
    _read_candidate_sources,
    _rubric_status,
    main,
    run_harness_evaluation,
)
from src.investigation.harness_feedback import (
    evaluation_feedback,
    search_metric_feedback,
    summarize_recent_attempts,
)
from src.investigation.rubric_checks import (
    candidate_source_tokens,
    missing_source_patterns,
    rubric_metric_check,
)
from src.investigation.task_contract import MAX_TASK_CONTRACT_BYTES
from src.orchestrator.idea_bundle import IdeaBundleValidationError, normalize_target_rubrics
from src.orchestrator.rubrics import validate_rubric_status
from src.variants_pipeline.host_eval import (
    HostEvaluationResult,
    SandboxEvaluationError,
    SandboxEvaluationTimeout,
)
from tests.program_testutils import make_program_dir
from tests.task_drafting.test_public_task_contract import _contract_with_sentinels

IMAGE = "sha256:" + "a" * 64
GPUS = '"device=GPU-one,GPU-two"'
METRIC_RUBRIC: dict[str, Any] = {"id": "score", "description": "combined_score > 0.8", "type": "metric"}
DEFAULT_SOURCE = "def solve():\n    return 1\n"


def _impl_rubric(*patterns: str, id: str = "uses_solver") -> dict[str, object]:
    return {
        "id": id,
        "description": "The candidate implements the method.",
        "type": "implementation",
        "verification": {"type": "source_contains", "patterns": list(patterns)},
    }


def _write_evaluator(experiment_dir: Path, *, score: float, correct: bool = True) -> None:
    (experiment_dir / "evaluate.py").write_text(
        "\n".join(
            [
                "import argparse, json",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--program_path', required=True)",
                "parser.add_argument('--results_dir', required=True)",
                "args = parser.parse_args()",
                "results_dir = Path(args.results_dir)",
                "results_dir.mkdir(parents=True, exist_ok=True)",
                "Path(args.program_path).read_text(encoding='utf-8')",
                f"(results_dir / 'metrics.json').write_text(json.dumps({{'combined_score': {score!r}}}))",
                f"(results_dir / 'correct.json').write_text(json.dumps({{'correct': {correct!r}}}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_rubrics(tmp_path: Path, rubrics: list[dict[str, object]] | None) -> Path:
    rubrics_path = tmp_path / "rubrics.json"
    rubrics_path.write_text(json.dumps(rubrics or []), encoding="utf-8")
    return rubrics_path


def _experiment(tmp_path: Path, *, source: str = DEFAULT_SOURCE, score: float = 3.14) -> Path:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir(parents=True)
    make_program_dir(experiment_dir / "program_1", content=source)
    _write_evaluator(experiment_dir, score=score)
    return experiment_dir


def _evaluate(experiment_dir: Path, rubrics_path: Path, **overrides: Any) -> int:
    tmp_path = experiment_dir.parent
    kwargs: dict[str, Any] = dict(
        experiment_dir=experiment_dir,
        attempt_id="attempt_01",
        idea_id="idea_01",
        rubrics_path=rubrics_path,
        python_bin=sys.executable,
        global_memory_md=tmp_path / "memory.md",
        global_memory_jsonl=tmp_path / "memory.jsonl",
        timeout_seconds=10,
    )
    kwargs.update(overrides)
    return run_harness_evaluation(**kwargs)


def _status(path: Path) -> dict[str, Any]:
    return json.loads((path / "rubric_status.json").read_text(encoding="utf-8"))


def _run_harness(
    tmp_path: Path,
    rubrics: list[dict[str, object]],
    *,
    score: float = 3.14,
    source: str = DEFAULT_SOURCE,
) -> dict[str, Any]:
    experiment_dir = _experiment(tmp_path, source=source, score=score)
    assert _evaluate(experiment_dir, _write_rubrics(tmp_path, rubrics)) == 0
    return _status(experiment_dir)


def _sealed_experiment(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Experiment + task source dir whose python lives only inside the container."""
    experiment_dir = _experiment(tmp_path, source="print('ok')\n", score=0.9)
    task_source_dir = tmp_path / "task"
    task_source_dir.mkdir()
    _write_evaluator(task_source_dir, score=0.9)
    task_python = task_source_dir / ".venv/bin/python"
    task_python.parent.mkdir(parents=True)
    task_python.write_text("container-only runtime\n", encoding="utf-8")
    return experiment_dir, task_source_dir, _write_rubrics(tmp_path, [METRIC_RUBRIC]), task_python


def _sealed_evaluate(tmp_path: Path, side_effect: Any, **config: Any) -> tuple[int, Path]:
    experiment_dir, task_source_dir, rubrics_path, task_python = _sealed_experiment(tmp_path)
    sealed = SealedHarnessConfig(
        task_source_dir=task_source_dir, sandbox_image="sha256:test", sandbox_gpus=None, environment={}
    )
    with patch("src.variants_pipeline.host_eval.evaluate_variant_program", side_effect=side_effect):
        exit_code = _evaluate(
            experiment_dir, rubrics_path, python_bin=str(task_python), sealed_config=sealed, **config
        )
    return exit_code, experiment_dir / "program_1"


# --- rubric evaluation through the harness -----------------------------------------------------


@pytest.mark.parametrize(
    ("rubric", "source", "expected"),
    [
        pytest.param({"id": "t", "description": "combined_score > 2.6", "type": "metric"}, DEFAULT_SOURCE, True, id="metric_satisfied"),
        pytest.param({"id": "t", "description": "combined_score > 10", "type": "metric"}, DEFAULT_SOURCE, False, id="metric_unsatisfied"),
        pytest.param({"id": "t", "description": "Implementation uses restart diversity", "type": "implementation"}, DEFAULT_SOURCE, False, id="unverifiable_implementation"),
        pytest.param(_impl_rubric("def solve", "return 1"), DEFAULT_SOURCE, True, id="source_contains"),
        pytest.param(_impl_rubric('method="SLSQP"'), "def solve():\n    return optimize(method='SLSQP')\n", True, id="equivalent_string_quotes"),
        pytest.param(_impl_rubric('label=f"solver={solver}"'), "def solve():\n    return report(label=f'solver={solver}')\n", True, id="equivalent_fstring_quotes"),
        pytest.param(_impl_rubric("SAFETY_MARGIN = 1e-9"), "SAFETY_MARGIN = 1e-09\ndef solve():\n    return 1\n", True, id="equivalent_numeric_spelling"),
        pytest.param(
            _impl_rubric('linprog(objective, bounds=bounds, method="highs")'),
            "def solve():\n    return linprog(\n        objective,\n        bounds=bounds,\n        method='highs'\n    )\n",
            True,
            id="layout_changes",
        ),
        pytest.param(_impl_rubric("scipy.optimize.minimize"), DEFAULT_SOURCE, False, id="missing_pattern"),
        pytest.param(_impl_rubric('method="'), DEFAULT_SOURCE, False, id="malformed_pattern_fails_closed"),
        pytest.param(_impl_rubric('method="SLSQP"'), "def solve():\n    return optimize(method=SLSQP)\n", False, id="token_semantics_quotes"),
        pytest.param(_impl_rubric("limit = 1.0"), "limit = 1\ndef solve():\n    return limit\n", False, id="token_semantics_numeric"),
        pytest.param(_impl_rubric("scipy.optimize.minimize"), "# scipy.optimize.minimize\n" + DEFAULT_SOURCE, False, id="comment_only"),
        pytest.param(_impl_rubric("scipy.optimize.minimize"), '"""scipy.optimize.minimize"""\n' + DEFAULT_SOURCE, False, id="docstring_only"),
    ],
)
def test_harness_eval_rubric_outcomes(tmp_path: Path, rubric: dict[str, object], source: str, expected: bool) -> None:
    status = _run_harness(tmp_path, [rubric], source=source)
    assert status["overall_pass"] is expected
    assert status["rubrics"][0]["pass"] is expected


def test_harness_eval_batches_source_pattern_checks_and_ignores_invalid_auxiliary_source(tmp_path: Path) -> None:
    # all source patterns are checked in one batch
    rubrics = [_impl_rubric("def solve(", id="defines_solve"), _impl_rubric("return 1", id="returns_one")]
    with patch("src.investigation.harness_eval.missing_source_patterns", wraps=missing_source_patterns) as matcher:
        status = _run_harness(tmp_path / "batched", rubrics)
    assert status["overall_pass"] is True
    assert matcher.call_count == 1

    # an unparsable auxiliary source file is ignored
    root = tmp_path / "auxiliary"
    experiment_dir = _experiment(root)
    (experiment_dir / "program_1/src/invalid.py").write_text("def broken(:\n", encoding="utf-8")
    assert _evaluate(experiment_dir, _write_rubrics(root, [_impl_rubric("def solve(")])) == 0
    assert _status(experiment_dir)["overall_pass"] is True


# --- direct vs sealed execution ------------------------------------------------------------------


def test_harness_eval_sealed_execution_records_success_failure_and_timeout(tmp_path: Path) -> None:
    # success: the sealed task runtime is used inside the container, never the direct runner
    root = tmp_path / "success"
    root.mkdir()
    experiment_dir, task_source_dir, rubrics_path, task_python = _sealed_experiment(root)
    sealed_result = HostEvaluationResult(returncode=0, correct=True, metrics={"combined_score": 0.9}, command=("docker", "run"))
    environment = {"CUDA_VISIBLE_DEVICES": "0,1", "NCCL_DEBUG": "INFO"}

    def run_sealed_evaluation(**_kwargs: object) -> HostEvaluationResult:
        program_dir = experiment_dir / "program_1"
        (program_dir / "metrics.json").write_text(json.dumps(sealed_result.metrics), encoding="utf-8")
        (program_dir / "correct.json").write_text(json.dumps({"correct": True}), encoding="utf-8")
        (program_dir / "evaluation_manifest.json").write_text(
            json.dumps(
                {
                    "producer": "sandbox_host_eval",
                    "execution_boundary": "sandbox",
                    "program_id": "program_1",
                    "command": ["docker", "run"],
                    "started_at": "2026-08-21T00:00:00+00:00",
                    "finished_at": "2026-08-21T00:00:01+00:00",
                    "returncode": 0,
                }
            ),
            encoding="utf-8",
        )
        return sealed_result

    with (
        patch("src.investigation.harness_eval.run_contained") as direct_run,
        patch("src.variants_pipeline.host_eval.evaluate_variant_program", side_effect=run_sealed_evaluation) as sealed_run,
    ):
        exit_code = _evaluate(
            experiment_dir,
            rubrics_path,
            python_bin=str(task_python),
            sealed_config=SealedHarnessConfig(
                task_source_dir=task_source_dir,
                sandbox_image=IMAGE,
                sandbox_gpus=GPUS,
                environment=environment,
                gpu_evaluation=True,
            ),
        )
    assert exit_code == 0
    direct_run.assert_not_called()
    sealed_run.assert_called_once_with(
        task_bundle_dir=experiment_dir,
        evaluator_source_dir=task_source_dir,
        task_source_dir=task_source_dir,
        program_index=1,
        python_bin=task_python,
        sandbox_image=IMAGE,
        timeout_seconds=10,
        environment=environment,
        sandbox_gpus=GPUS,
    )
    assert evaluation_results_complete(experiment_dir / "program_1") is True

    # failure diagnostics are preserved
    root = tmp_path / "failure"
    root.mkdir()
    command = ["docker", "run", "sha256:test", "/task-venv/bin/python"]
    exit_code, program_dir = _sealed_evaluate(root, SandboxEvaluationError(7, command, "candidate process failed\n"))
    manifest = json.loads((program_dir / "evaluation_manifest.json").read_text())
    assert exit_code == 7
    assert manifest["exit_code"] == 7
    assert manifest["command"] == command
    assert (program_dir / "harness_evaluate_stderr.log").read_text() == "candidate process failed\n"
    assert _status(program_dir)["overall_pass"] is False

    # a sealed timeout is recorded
    root = tmp_path / "timeout"
    root.mkdir()
    exit_code, program_dir = _sealed_evaluate(root, SandboxEvaluationTimeout(10, command))
    manifest = json.loads((program_dir / "evaluation_manifest.json").read_text())
    assert exit_code == 124
    assert manifest["exit_code"] == 124
    assert manifest["returncode"] == 124
    assert manifest["timeout_seconds"] == 10
    assert manifest["command"] == command
    assert "timed out after 10s" in (program_dir / "harness_evaluate_stderr.log").read_text()
    assert _status(program_dir)["overall_pass"] is False


def test_direct_harness_controls_cuda_visibility(tmp_path: Path) -> None:
    for name, gpu_evaluation, expected_cuda in [("cpu_blanks", False, ""), ("gpu_preserves", True, "0,1")]:
        root = tmp_path / name
        experiment_dir = _experiment(root, source="print('ok')\n", score=0.9)

        def evaluate(command: list[str], **kwargs: Any) -> ContainedRun:
            assert kwargs["environment"]["CUDA_VISIBLE_DEVICES"] == expected_cuda, name
            assert "FAKE_API_KEY" not in kwargs["environment"], name
            program_dir = experiment_dir / "program_1"
            (program_dir / "metrics.json").write_text('{"combined_score": 0.9}')
            (program_dir / "correct.json").write_text('{"correct": true}')
            return ContainedRun(returncode=0)

        with patch("src.investigation.harness_eval.run_contained", side_effect=evaluate):
            exit_code = _evaluate(
                experiment_dir,
                _write_rubrics(root, []),
                environment={"CUDA_VISIBLE_DEVICES": "0,1", "FAKE_API_KEY": "secret"},
                gpu_evaluation=gpu_evaluation,
            )
        assert exit_code == 0, name


# --- CLI ------------------------------------------------------------------------------------------


def _cli_arguments(tmp_path: Path, task_source_dir: Path, *, python_bin: bool = True) -> list[str]:
    arguments = [
        "--experiment-dir", str(tmp_path / "experiment" if python_bin else tmp_path),
        "--attempt-id", "attempt_01",
        "--idea-id", "idea_01",
        "--rubrics-file", str(tmp_path / "rubrics.json"),
        "--global-memory-md", str(tmp_path / "memory.md"),
        "--global-memory-jsonl", str(tmp_path / "memory.jsonl"),
        "--task-source-dir", str(task_source_dir),
        "--sandbox-image", IMAGE,
    ]
    if python_bin:
        arguments += ["--python-bin", str(task_source_dir / ".venv/bin/python")]
    return arguments


def test_harness_cli_builds_sealed_configuration_and_rejects_incomplete_gpu_opt_in(tmp_path: Path) -> None:
    # a complete GPU opt-in builds the sealed configuration
    task_source_dir = tmp_path / "task"
    (tmp_path / "experiment").mkdir()
    task_source_dir.mkdir()
    _write_rubrics(tmp_path, [])
    with (
        patch.dict(os.environ, {"NCCL_DEBUG": "INFO"}),
        patch("src.investigation.harness_eval.run_harness_evaluation", return_value=0) as run,
    ):
        exit_code = main([*_cli_arguments(tmp_path, task_source_dir), "--sandbox-gpus", GPUS, "--gpu-evaluation"])
    assert exit_code == 0
    config = run.call_args.kwargs["sealed_config"]
    assert config.task_source_dir == task_source_dir
    assert config.sandbox_image == IMAGE
    assert config.sandbox_gpus == GPUS
    assert config.gpu_evaluation is True
    assert config.environment["NCCL_DEBUG"] == "INFO"

    # an incomplete GPU opt-in is rejected
    for gpu_arguments in (["--sandbox-gpus", '"device=GPU-one"'], ["--gpu-evaluation"]):
        with pytest.raises(SystemExit) as error:
            main([*_cli_arguments(tmp_path, tmp_path, python_bin=False), *gpu_arguments])
        assert error.value.code == 2, gpu_arguments


# --- agent-only program_1 and source inspection bounds -------------------------------------------


def test_harness_rejects_symlinked_program_1_entries(tmp_path: Path) -> None:
    # a dangling evaluator artifact symlink is rejected without being followed
    for artifact in ("metrics.json", "harness_evaluate_stdout.log"):
        root = tmp_path / artifact.replace(".", "-")
        experiment_dir = root / "experiment"
        make_program_dir(experiment_dir / "program_1", content="print('candidate')\n")
        outside = root / "outside"
        (experiment_dir / "program_1" / artifact).symlink_to(outside)
        with pytest.raises(SystemExit, match="agent-created evaluator artifact"):
            _ensure_agent_only_created_program_1(experiment_dir)
        assert not outside.exists(), artifact

    # a symlinked program root is rejected
    root = tmp_path / "program-root"
    experiment_dir = root / "experiment"
    outside_program = root / "outside_program"
    make_program_dir(outside_program, content="print('candidate')\n")
    experiment_dir.mkdir()
    (experiment_dir / "program_1").symlink_to(outside_program, target_is_directory=True)
    with pytest.raises(SystemExit, match="requires agent-created program_1"):
        _ensure_agent_only_created_program_1(experiment_dir)

    # a symlinked source file is rejected by the harness
    root = tmp_path / "source-file"
    experiment_dir = _experiment(root, source="print('candidate')\n")
    outside = root / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")
    (experiment_dir / "program_1" / "src" / "linked.py").symlink_to(outside)
    with pytest.raises(SystemExit, match="source directory must not contain symlinks"):
        _evaluate(experiment_dir, _write_rubrics(root, [_impl_rubric("candidate")]))


def _touch_many(program: Path) -> None:
    for index in range(2049):
        (program / "src" / str(index)).touch()


def _nest_deeply(program: Path) -> None:
    (program / "src" / Path(*(["nested"] * 33))).mkdir(parents=True)


def _fill_nonpython_bytes(program: Path) -> None:
    for index in range(33):
        (program / "src" / f"data-{index}.bin").write_bytes(b"x" * 128 * 1024)


def test_inspection_bounds_source_size_and_tree_shape(tmp_path: Path) -> None:
    # a large real-world training script (~165 KB) is inspected
    program_dir = tmp_path / "large" / "program_1"
    make_program_dir(program_dir, content="x = 1\n" * 28_000)
    assert len(_read_candidate_sources(program_dir)) == 1

    # an oversized candidate source file is rejected
    program_dir = tmp_path / "file" / "program_1"
    make_program_dir(program_dir, content="x" * (harness_sources.MAX_SOURCE_FILE_BYTES + 1))
    with pytest.raises(SystemExit, match="source exceeds harness inspection limit"):
        _read_candidate_sources(program_dir)

    # an oversized candidate source tree is rejected
    program_dir = tmp_path / "tree" / "program_1"
    make_program_dir(program_dir, content="x = 1\n" * 20_000)
    (program_dir / "src/helper.py").write_text("y = 2\n" * 20_000, encoding="utf-8")
    for index in range(34):
        (program_dir / f"src/other_{index}.py").write_text("z = 3\n" * 20_000, encoding="utf-8")
    with pytest.raises(SystemExit, match="source exceeds harness inspection limit"):
        _read_candidate_sources(program_dir)

    # too many entries, too deep nesting, or too many non-python bytes are rejected
    for name, populate in [("empty_entries", _touch_many), ("directory_depth", _nest_deeply), ("nonpython_bytes", _fill_nonpython_bytes)]:
        program = tmp_path / name / "program_1"
        make_program_dir(program)
        populate(program)
        with pytest.raises(SystemExit, match="inspection limit"):
            _read_candidate_sources(program)


def test_inspection_rejects_source_symlinks_and_nonregular_files(tmp_path: Path) -> None:
    # symlinked files, directories, source roots and program roots are rejected
    for kind in ("file", "directory", "source_root", "program_root"):
        root = tmp_path / kind
        root.mkdir()
        program = root / "program_1"
        source = program / "src"
        outside = root / "outside"
        if kind == "program_root":
            make_program_dir(outside)
            program.symlink_to(outside, target_is_directory=True)
        elif kind == "source_root":
            program.mkdir()
            source.symlink_to(outside, target_is_directory=True)
        else:
            make_program_dir(program)
            if kind == "directory":
                outside.mkdir()
            else:
                outside.write_text("private values")
            (source / "linked").symlink_to(outside, target_is_directory=kind == "directory")
        with pytest.raises(SystemExit):
            _read_candidate_sources(program)

    # non-regular files are rejected without being read
    for suffix in (".py", ".bin"):
        program = tmp_path / f"pipe{suffix.strip('.')}" / "program_1"
        make_program_dir(program)
        os.mkfifo(program / "src" / f"pipe{suffix}")
        with pytest.raises(SystemExit, match="regular files"):
            _read_candidate_sources(program)


def test_inspection_rejects_entry_replacement_and_growth_during_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # an entry replaced (by a file or a symlink) between listing and open is rejected
    for replacement in ("file", "symlink"):
        root = tmp_path / replacement
        root.mkdir()
        program = root / "program_1"
        make_program_dir(program, content="original = 1\n", entrypoint="main.py")
        target = program / "src/main.py"
        outside = root / "outside.py"
        outside.write_text("private = 1\n")
        open_file = os.open

        def replace_then_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            if path == "main.py":
                if replacement == "symlink":
                    target.unlink()
                    target.symlink_to(outside)
                else:
                    outside.replace(target)
            return open_file(path, flags, *args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(harness_sources.os, "open", replace_then_open)
            with pytest.raises(SystemExit):
                _read_candidate_sources(program)

    # reads stay bounded when the source grows while being read
    root = tmp_path / "growing"
    root.mkdir()
    program = root / "program_1"
    make_program_dir(program, content="original = 1\n", entrypoint="main.py")
    target = program / "src/main.py"
    fdopen = os.fdopen
    requested: list[int] = []

    class GrowingSource:
        def __init__(self, descriptor: int, mode: str) -> None:
            self.stream = fdopen(descriptor, mode)

        def __enter__(self) -> GrowingSource:
            return self

        def __exit__(self, *args: Any) -> None:
            self.stream.close()

        def fileno(self) -> int:
            return self.stream.fileno()

        def read(self, maximum: int) -> Any:
            requested.append(maximum)
            target.write_bytes(b"x" * (harness_sources.MAX_SOURCE_FILE_BYTES + 2))
            return self.stream.read(maximum)

    with monkeypatch.context() as context:
        context.setattr(harness_sources.os, "fdopen", GrowingSource)
        with pytest.raises(SystemExit, match="inspection limit"):
            _read_candidate_sources(program)
    assert requested == [harness_sources.MAX_SOURCE_FILE_BYTES + 1]


# --- rubric_checks: metric thresholds and source pattern limits ----------------------------------


def test_metric_check_parses_thresholds_and_rejects_malformed_or_nonfinite_values() -> None:
    # sentence punctuation and scientific notation are accepted
    for description, expected in [
        ("combined_score > 2.541.", ("combined_score", ">", 2.541)),
        ("combined_score > 2.6e-3.", ("combined_score", ">", 0.0026)),
        ("The score exceeds 2.541.", ("combined_score", ">", 2.541)),
        ("combined_score >= 2.", ("combined_score", ">=", 2.0)),
        ("combined_score < .5.", ("combined_score", "<", 0.5)),
    ]:
        assert rubric_metric_check({"description": description}) == expected, description

    # malformed or non-finite thresholds are rejected
    for description in (
        "combined_score > 2.6e",
        "combined_score > 2.6e+",
        "combined_score > 2.6.1",
        "combined_score > 2.6.e3",
        "combined_score > 2.6.foo",
        "combined_score > 2.6..",
        "combined_score > 1e999",
    ):
        assert rubric_metric_check({"description": description}) is None, description


def _normalize_patterns(patterns: Any) -> list[str]:
    rubrics = normalize_target_rubrics(
        [
            {
                "id": "method",
                "description": "Implement the proposed method",
                "type": "implementation",
                "verification": {"type": "source_contains", "patterns": patterns},
            }
        ]
    )
    return rubrics[0]["verification"]["patterns"]


_ACCEPTED_PATTERNS = [
    ("characters_not_bytes", ["é" * 1024], ["é" * 1024]),
    ("count_lower_bound", ["method_0"], ["method_0"]),
    ("count_upper_bound", [f"method_{i}" for i in range(8)], [f"method_{i}" for i in range(8)]),
    ("strips_outer_whitespace", ["  method(value) \n"], ["method(value)"]),
]
_INVALID_PATTERN_VALUES: list[object] = ["", " \t\n", None, 1, True, {}, []]
_INVALID_PATTERN_CONTAINERS: list[object] = [[], ["method"] * 9, "method", None]
_REJECTED_PATTERNS: list[tuple[str, Any, str]] = [
    ("oversized", ["x" * 1025], "up to 1024 characters"),
    ("outer_whitespace_counts", [" " + "x" * 1024], "up to 1024 characters"),
    *[(f"invalid_{index}", [value], "non-empty strings") for index, value in enumerate(_INVALID_PATTERN_VALUES)],
    *[(f"container_{index}", value, "1 to 8 strings") for index, value in enumerate(_INVALID_PATTERN_CONTAINERS)],
]


def test_pattern_normalization_bounds_and_matches_source_patterns() -> None:
    # long executable patterns are accepted and match
    for length in (232, 238, 1024):
        pattern = "result = " + "x" * (length - len("result = "))
        normalized = _normalize_patterns([pattern])
        assert normalized == [pattern], length
        assert missing_source_patterns(candidate_source_tokens([pattern]), normalized) == [], length

    # accepted shapes
    for name, patterns, expected in _ACCEPTED_PATTERNS:
        assert _normalize_patterns(patterns) == expected, name

    # rejected shapes
    for name, raw_patterns, message in _REJECTED_PATTERNS:
        with pytest.raises(IdeaBundleValidationError, match=message):
            _normalize_patterns(raw_patterns)


# --- harness feedback ---------------------------------------------------------------------------


_ASSESSMENTS = [
    (None, False, True),
    ("not_assessed", False, True),
    ("passed", True, True),
    ("failed", False, True),
    ("not_assessed", True, False),
    ("passed", False, False),
    ("unknown", False, False),
]


def test_rubric_status_and_feedback_label_outcomes_consistently() -> None:
    # a rejected evaluation does not claim the implementation was checked
    for exit_code, reason in [(0, "scientific acceptance"), (1, "evaluation failed")]:
        passed, evidence = _rubric_status(
            rubric={"id": "method", "type": "implementation"},
            metrics={"combined_score": 0.5},
            correct=False,
            exit_code=exit_code,
            missing_candidate_patterns=set(),
        )
        assert passed is False, exit_code
        assert "criterion not assessed" in evidence, exit_code
        assert reason in evidence, exit_code

    # feedback labels known and unknown outcomes
    for exit_code, correct, expected in [
        (124, False, "timeout"), (1, False, "cause unavailable"), (0, False, "does not establish"), (0, True, "acceptance passed"),
    ]:
        assert expected in evaluation_feedback({"correct": correct}, exit_code), (exit_code, correct)

    # rubric assessment is optional but must be consistent with pass
    for assessment, passed, valid in _ASSESSMENTS:
        rubric: dict[str, Any] = {"id": "method", "pass": passed}
        if assessment is not None:
            rubric["assessment"] = assessment
        status = {"attempt_id": "a", "idea_id": "i", "overall_pass": passed, "rubrics": [rubric]}
        assert validate_rubric_status(status, {"method"}, "a", "i").valid is valid, (assessment, passed)


def _task_with_metrics(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _contract_with_sentinels(tmp_path)
    task = tmp_path / "task"
    path = task / "task_contract.json"
    contract = json.loads(path.read_text())
    contract["metrics"]["search"].extend(
        {"name": name, "data_role": "validation"}
        for name in ("family_margin", "invalid_query_rate", "nested", "boolean", "nonfinite")
    )
    path.write_text(json.dumps(contract))
    return task


def test_search_metric_feedback_projects_declared_metrics_and_ignores_unsafe_input(tmp_path: Path) -> None:
    # only finite declared search metrics are projected
    metrics = {
        "combined_score": -0.08,
        "family_margin": -0.01,
        "invalid_query_rate": 0,
        "nested": {"secret": 1},
        "boolean": True,
        "nonfinite": float("inf"),
        "SEALED_FINAL_METRIC": 999,
        "undeclared": 321,
    }
    assert search_metric_feedback(_task_with_metrics(tmp_path / "declared"), metrics) == {
        "combined_score": -0.08,
        "family_margin": -0.01,
        "invalid_query_rate": 0,
    }

    # a missing or invalid contract yields no extra feedback
    for index, contents in enumerate((None, "{}", "not json")):
        root = tmp_path / f"contract-{index}"
        root.mkdir()
        if contents is not None:
            (root / "task_contract.json").write_text(contents)
        assert search_metric_feedback(root, {"secret": 42}) == {}, contents

    # unsafe contract entries are rejected
    for entry in ("symlink", "directory", "oversized"):
        root = tmp_path / f"unsafe-{entry}"
        root.mkdir()
        path = root / "task_contract.json"
        if entry == "symlink":
            path.symlink_to(_task_with_metrics(root / "linked") / "task_contract.json")
        elif entry == "directory":
            path.mkdir()
        else:
            path.write_bytes(b" " * (MAX_TASK_CONTRACT_BYTES + 1))
        assert search_metric_feedback(root, {"family_margin": 1}) == {}, entry

    # unusable metric values are omitted
    task = _task_with_metrics(tmp_path / "unusable")
    unusable: tuple[object, ...] = (float("nan"), 10**400, True, {}, None)
    for value in unusable:
        assert search_metric_feedback(task, {"family_margin": value}) == {}, repr(value)


def test_harness_memory_and_recent_attempts_keep_rejection_and_bound_declared_metrics(tmp_path: Path) -> None:
    # the harness keeps the rejection and adds diagnostics to memory
    task = _task_with_metrics(tmp_path / "memory")
    (task / "evaluate.py").write_text("# mocked evaluator\n")
    make_program_dir(task / "program_1", content="def handle(request): return {}\n")
    rubrics = task / "rubrics.json"
    rubrics.write_text(json.dumps([{"id": "method", "type": "implementation"}]))

    def evaluate(**kwargs: Any) -> int:
        program = task / "program_1"
        (program / "metrics.json").write_text(
            json.dumps({"combined_score": -0.08, "family_margin": -0.01, "invalid_query_rate": 0, "SEALED_FINAL_METRIC": 999})
        )
        (program / "correct.json").write_text('{"correct": false}')
        return 0

    with patch("src.investigation.harness_eval._run_direct_evaluation", side_effect=evaluate):
        exit_code = run_harness_evaluation(
            experiment_dir=task,
            attempt_id="a",
            idea_id="i",
            rubrics_path=rubrics,
            python_bin="python",
            global_memory_md=task / "memory.md",
            global_memory_jsonl=task / "memory.jsonl",
            timeout_seconds=10,
        )
    assert exit_code == 0
    status = _status(task)
    assert status["overall_pass"] is False
    assert status["rubrics"][0]["pass"] is False
    assert status["rubrics"][0]["assessment"] == "not_assessed"
    assert status["metrics"]["family_margin"] == -0.01
    memory = (task / "memory.md").read_text()
    assert "family_margin" in memory
    assert "cause unavailable" in memory
    assert "SEALED_FINAL_METRIC" not in memory

    # recent attempt summaries include bounded declared scientific metrics
    summary = summarize_recent_attempts(
        _task_with_metrics(tmp_path / "recent"),
        [
            {
                "attempt_id": "a",
                "attempt_status": "fail",
                "metrics": {
                    "correct": False,
                    "combined_score": 0.9,
                    "family_margin": -0.01,
                    "SEALED_FINAL_METRIC": 99,
                    "payload": "secret" * 10000,
                },
            }
        ],
    )
    assert '"correct": false' in summary
    assert '"combined_score": 0.9' in summary
    assert '"family_margin": -0.01' in summary
    assert "SEALED_FINAL_METRIC" not in summary
    assert "secret" not in summary


# --- evaluator tree digest -----------------------------------------------------------------------


def _add_pycache(task: Path) -> None:
    nested = task / "helpers"
    for directory, name in ((task, "evaluate"), (nested, "geometry")):
        cache = directory / "__pycache__"
        cache.mkdir()
        (cache / f"{name}.cpython-312.pyc").write_bytes(b"compiled")


def _rewrite_snapshot(task: Path) -> None:
    (task / ".agentic_task_snapshot.json").write_text('{"source_digest":"after"}\n', encoding="utf-8")


def _edit_source(task: Path) -> None:
    (task / "evaluate.py").write_text("VALUE = 2\n", encoding="utf-8")


def test_managed_digest_tracks_only_evaluator_sources_and_rejects_cache_symlinks(tmp_path: Path) -> None:
    # caches and snapshot metadata are ignored; source edits change the digest
    for name, mutate, unchanged in [
        ("ignores_pycache", _add_pycache, True),
        ("ignores_snapshot_metadata", _rewrite_snapshot, True),
        ("covers_source_changes", _edit_source, False),
    ]:
        task = tmp_path / name / "task"
        (task / "helpers").mkdir(parents=True)
        (task / "evaluate.py").write_text("VALUE = 1\n", encoding="utf-8")
        (task / "helpers" / "geometry.py").write_text("RADIUS = 1.0\n", encoding="utf-8")
        (task / ".agentic_task_snapshot.json").write_text('{"source_digest":"before"}\n', encoding="utf-8")
        expected = managed_evaluator_tree_digest(task)
        mutate(task)
        assert (managed_evaluator_tree_digest(task) == expected) is unchanged, name

    # cache symlinks are rejected
    task = tmp_path / "symlinked" / "task"
    task.mkdir(parents=True)
    outside = tmp_path / "symlinked" / "outside"
    outside.mkdir()
    (task / "__pycache__").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="contains a symlink"):
        managed_evaluator_tree_digest(task)


# --- native incumbent -----------------------------------------------------------------------------


def _reconcile(root: Path, seed: Path, *, trust_native: bool) -> tuple[bool, dict[str, Any]]:
    best_manifest: dict[str, Any] = {"program_dir": str(seed)}
    ok = reconcile_harness_incumbent(
        task_dir=root / "task", results_dir=root, seed_program_dir=seed, baseline_score=None,
        baseline_correct=None, best_manifest=best_manifest, outer_state={},
        trust_native=trust_native,
    )
    return ok, best_manifest


def test_native_bootstrap_earns_an_incumbent_only_native_runs_trust(tmp_path: Path) -> None:
    root = tmp_path / "run"
    seed = root / ".baseline-seed-x" / "program_0"
    make_program_dir(seed, content=DEFAULT_SOURCE)
    _write_evaluator(root, score=2.5)

    # without an evaluation nothing vouches for the seed
    assert _reconcile(root, seed, trust_native=True)[0] is False

    returncode = bootstrap_native_incumbent(
        program_dir=seed, task_dir=root, python_bin=sys.executable,
        timeout_seconds=30, environment={"CUDA_VISIBLE_DEVICES": "0"}, gpu_evaluation=False,
    )
    manifest = json.loads((seed / "evaluation_manifest.json").read_text())
    assert returncode == 0
    assert (manifest["producer"], manifest["program_id"], manifest["exit_code"]) == (
        "harness_eval", "program_0", 0,
    )

    ok, best = _reconcile(root, seed, trust_native=True)
    assert ok and best["combined_score"] == 2.5 and best["correct"] is True
    assert best["program_dir"] == str(seed.resolve())
    # a containerized run never trusts a host-run evaluation
    assert _reconcile(root, seed, trust_native=False)[0] is False


def test_native_bootstrap_of_an_incorrect_seed_earns_no_incumbent(tmp_path: Path) -> None:
    root = tmp_path / "run"
    seed = root / ".baseline-seed-x" / "program_0"
    make_program_dir(seed, content=DEFAULT_SOURCE)
    _write_evaluator(root, score=2.5, correct=False)
    assert bootstrap_native_incumbent(
        program_dir=seed, task_dir=root, python_bin=sys.executable,
        timeout_seconds=30, environment={}, gpu_evaluation=False,
    ) == 0
    assert _reconcile(root, seed, trust_native=True)[0] is False


def test_native_bootstrap_runs_like_an_attempt_and_logs_to_disk(tmp_path: Path) -> None:
    root = tmp_path / "run"
    seed = root / ".baseline-seed-x" / "program_0"
    make_program_dir(seed, content=DEFAULT_SOURCE)
    (root / "evaluate.py").write_text(
        "import argparse, json, os, pathlib\n"
        "p = argparse.ArgumentParser(); p.add_argument('--program_path'); p.add_argument('--results_dir')\n"
        "a = p.parse_args(); out = pathlib.Path(a.results_dir)\n"
        "print('cwd', os.getcwd(), 'task', os.environ.get('TASK_SOURCE_DIR'), 'key', os.environ.get('FAKE_API_KEY'))\n"
        "(out / 'metrics.json').write_text(json.dumps({'combined_score': 1.0}))\n"
        "(out / 'correct.json').write_text(json.dumps({'correct': True}))\n"
    )

    assert bootstrap_native_incumbent(
        program_dir=seed, task_dir=root, python_bin=sys.executable, timeout_seconds=30,
        environment={"FAKE_API_KEY": "secret", "PATH": os.environ["PATH"]}, gpu_evaluation=False,
    ) == 0

    log = (seed / "harness_evaluate_stdout.log").read_text()
    assert f"cwd {seed.parent}" in log and f"task {root.resolve()}" in log and "key None" in log
    assert (seed.parent / "evaluate.py").read_bytes() == (root / "evaluate.py").read_bytes()


def test_native_bootstrap_timeout_stops_detached_descendants(tmp_path: Path) -> None:
    root = tmp_path / "run"
    seed = root / ".baseline-seed-x" / "program_0"
    make_program_dir(seed, content=DEFAULT_SOURCE)
    pid_file = tmp_path / "worker.pid"
    (root / "evaluate.py").write_text(
        "import subprocess, sys, time\n"
        "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'], start_new_session=True)\n"
        f"open({str(pid_file)!r}, 'w').write(str(worker.pid))\n"
        "print('training', flush=True)\n"
        "time.sleep(300)\n"
    )

    with pytest.raises(SystemExit, match="timed out after 3s"):
        bootstrap_native_incumbent(
            program_dir=seed, task_dir=root, python_bin=sys.executable, timeout_seconds=3,
            environment={"PATH": os.environ["PATH"]}, gpu_evaluation=False,
        )

    worker = int(pid_file.read_text())
    assert not Path(f"/proc/{worker}").exists() or Path(f"/proc/{worker}/stat").read_text().split()[2] == "Z"
    assert "training" in (seed / "harness_evaluate_stdout.log").read_text()
    manifest = json.loads((seed / "evaluation_manifest.json").read_text())
    assert manifest["exit_code"] == 124

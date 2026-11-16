"""Native (container.enabled=false) variant evaluation: same output contract as the sandbox."""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from src.investigation.files import evaluation_results_complete
from src.investigation.program_clone import clone_program_dir
from src.investigation.task_contract import TASK_CONTRACT_FILENAME
from src.variants_pipeline.host_eval import SandboxEvaluationError, SandboxEvaluationTimeout
from src.variants_pipeline.native_eval import (
    evaluate_variant_program_natively,
    native_evaluation_python,
)

SPEC = textwrap.dedent(
    """\
    schema_version = 1
    code_root = "src"
    [entrypoint]
    path = "program.py"
    language = "python"
    type = "script"
    [copy]
    exclude = []
    include = []
    [integration]
    kind = "generic"
    notes = ""
    """
)


def _bundle(tmp_path: Path, evaluator_body: str) -> tuple[Path, Path, Path]:
    """A task bundle with program_1 and a sealed evaluator running ``evaluator_body``."""
    bundle, evaluator, task = tmp_path / "bundle", tmp_path / "evaluator", tmp_path / "task"
    program = bundle / "program_1"
    (program / "src").mkdir(parents=True)
    (program / "program_spec.toml").write_text(SPEC)
    (program / "src" / "program.py").write_text("print('candidate')\n")
    evaluator.mkdir()
    task.mkdir()
    header = textwrap.dedent(
        """\
        import argparse, json, os, subprocess, sys, time
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--program_path", required=True)
        parser.add_argument("--results_dir", required=True)
        args = parser.parse_args()  # strict, like the shipped examples: no --num_replications
        results = Path(args.results_dir)
        """
    )
    (evaluator / "evaluate.py").write_text(header + textwrap.dedent(evaluator_body))
    return bundle, evaluator, task


def _evaluate(bundle: Path, evaluator: Path, task: Path, *, timeout: int = 60, env=None):
    return evaluate_variant_program_natively(
        task_bundle_dir=bundle,
        evaluator_source_dir=evaluator,
        task_source_dir=task,
        program_index=1,
        python_bin=Path(sys.executable),
        timeout_seconds=timeout,
        environment=env or {"VARIANT_ID": "ablation_x"},
    )


WRITE_RESULTS = """\
(results / "metrics.json").write_text(json.dumps({
    "combined_score": 0.5, "replications": int(os.environ.get("NUM_REPLICATIONS", "1")),
    "home": os.environ.get("HOME", ""),
    "tmpdir": os.environ.get("TMPDIR", ""),
    "hf_home": os.environ.get("HF_HOME", ""),
    "leaked_key": os.environ.get("FAKE_API_KEY", ""),
    "variant": os.environ.get("VARIANT_ID", ""),
    "task_source": os.environ.get("TASK_SOURCE_DIR", ""),
}))
(results / "correct.json").write_text(json.dumps({"correct": True}))
(results / "stdout.log").write_text("not on the promotion allowlist")
print("evaluator says hello")
"""


def test_native_evaluation_promotes_only_the_allowlist_and_records_a_manifest(
    tmp_path, monkeypatch
):
    bundle, evaluator, task = _bundle(tmp_path, WRITE_RESULTS)
    monkeypatch.setenv("HF_HOME", "/data/hf")  # the native environment, as investigation sees it
    monkeypatch.setenv("FAKE_API_KEY", "secret")  # minus credentials

    result = _evaluate(
        bundle, evaluator, task, env={"VARIANT_ID": "ablation_x", "NUM_REPLICATIONS": "3"}
    )

    program = bundle / "program_1"
    assert result.returncode == 0 and result.correct
    assert result.metrics["combined_score"] == 0.5
    assert result.metrics["replications"] == 3  # NUM_REPLICATIONS only, as in the sandbox
    assert "--num_replications" not in result.command
    assert result.metrics["variant"] == "ablation_x"
    assert result.metrics["home"] == os.environ["HOME"] and result.metrics["hf_home"] == "/data/hf"
    assert result.metrics["leaked_key"] == ""
    assert "variant-native-eval-" in result.metrics["tmpdir"]  # private scratch, removed afterwards
    assert not Path(result.metrics["tmpdir"]).exists()
    assert result.metrics["task_source"] == str(task.resolve())
    assert "evaluator says hello" in result.diagnostic
    assert json.loads((program / "correct.json").read_text()) == {"correct": True}
    assert "evaluator says hello" in (program / "harness_evaluate_stdout.log").read_text()
    assert not (program / "stdout.log").exists()
    assert not (program / ".harness_results").exists()
    manifest = json.loads((program / "evaluation_manifest.json").read_text())
    assert manifest["producer"] == "native_host_eval" and manifest["returncode"] == 0
    assert manifest["execution_boundary"] == "native"


def test_native_results_count_as_complete_and_can_be_cloned(tmp_path):
    bundle, evaluator, task = _bundle(tmp_path, WRITE_RESULTS)
    _evaluate(bundle, evaluator, task)
    program = bundle / "program_1"
    (program / "rubric_status.json").write_text(
        json.dumps({"attempt_id": "ablation_x_1", "idea_id": "ablation_x", "overall_pass": True, "rubrics": []})
    )

    assert evaluation_results_complete(program)
    clone_program_dir(src_dir=program, dst_dir=bundle / "program_2")
    assert (bundle / "program_2" / "src" / "program.py").is_file()


def test_evaluator_cannot_hang_the_harness_by_replacing_its_log(tmp_path):
    bundle, evaluator, task = _bundle(
        tmp_path,
        """\
        log = Path(os.environ["TMPDIR"]).parent / "evaluator.log"
        log.unlink()
        os.mkfifo(log)  # opening this by path would block forever
        (results / "metrics.json").write_text(json.dumps({"combined_score": 1.0}))
        (results / "correct.json").write_text(json.dumps({"correct": True}))
        print("still logged", flush=True)
        """,
    )

    result = _evaluate(bundle, evaluator, task, timeout=30)

    assert result.correct and "still logged" in result.diagnostic


def test_native_evaluation_failure_discards_artifacts_and_reports_the_output(tmp_path):
    bundle, evaluator, task = _bundle(
        tmp_path,
        """\
        (results / "metrics.json").write_text("{}")
        print("evaluator broke", flush=True)
        sys.exit(3)
        """,
    )

    with pytest.raises(SandboxEvaluationError) as raised:
        _evaluate(bundle, evaluator, task)

    assert raised.value.returncode == 3
    assert "evaluator broke" in raised.value.diagnostic
    program = bundle / "program_1"
    assert not (program / "metrics.json").exists()
    assert not (program / "evaluation_manifest.json").exists()


def _wait_gone(pid: int, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _spawn_children(pid_file: Path) -> str:
    """Evaluator code that starts one child in its group and one in its own session, like a
    torchrun worker, and records both PIDs."""
    return f"""\
in_group = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
own_session = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True
)
Path({str(pid_file)!r}).write_text(f"{{in_group.pid}} {{own_session.pid}}")
"""


def _children_gone(pid_file: Path) -> bool:
    return all(_wait_gone(int(pid)) for pid in pid_file.read_text().split())


def test_native_evaluation_timeout_stops_every_descendant(tmp_path):
    pid_file = tmp_path / "children.pid"
    bundle, evaluator, task = _bundle(tmp_path, _spawn_children(pid_file) + "time.sleep(600)\n")

    with pytest.raises(SandboxEvaluationTimeout):
        _evaluate(bundle, evaluator, task, timeout=3)

    assert _children_gone(pid_file)
    assert not (bundle / "program_1" / "evaluation_manifest.json").exists()


def test_native_evaluation_stops_descendants_left_behind_by_a_finished_evaluator(tmp_path):
    pid_file = tmp_path / "children.pid"
    bundle, evaluator, task = _bundle(tmp_path, WRITE_RESULTS + _spawn_children(pid_file))

    assert _evaluate(bundle, evaluator, task).correct
    assert _children_gone(pid_file)


def test_native_evaluation_stop_signal_exits_after_containing_the_evaluator(tmp_path):
    # The harness receives SIGTERM while the evaluator runs (here the evaluator sends it).
    pid_file = tmp_path / "children.pid"
    bundle, evaluator, task = _bundle(
        tmp_path,
        _spawn_children(pid_file)
        + "import signal\nos.kill(os.getppid(), signal.SIGTERM)\ntime.sleep(600)\n",
    )

    with pytest.raises(SystemExit) as raised:
        _evaluate(bundle, evaluator, task, timeout=120)

    assert raised.value.code == 128 + 15
    assert _children_gone(pid_file)
    assert not (bundle / "program_1" / "evaluation_manifest.json").exists()


def test_native_evaluation_rejects_contract_tasks_and_missing_inputs(tmp_path):
    bundle, evaluator, task = _bundle(tmp_path, WRITE_RESULTS)
    (task / TASK_CONTRACT_FILENAME).write_text("{}")
    with pytest.raises(ValueError, match="contract tasks"):
        _evaluate(bundle, evaluator, task)

    (task / TASK_CONTRACT_FILENAME).unlink()
    (evaluator / "evaluate.py").unlink()
    with pytest.raises(ValueError, match="missing sealed evaluator"):
        _evaluate(bundle, evaluator, task)


def test_native_python_prefers_the_task_venv_and_falls_back_to_this_interpreter(tmp_path):
    assert (
        native_evaluation_python(task_source_dir=tmp_path, venv_dir=None)
        == Path(sys.executable).absolute()
    )
    task_python = tmp_path / ".venv" / "bin" / "python"
    task_python.parent.mkdir(parents=True)
    task_python.write_text("")
    assert (
        native_evaluation_python(task_source_dir=tmp_path, venv_dir=None) == task_python.absolute()
    )

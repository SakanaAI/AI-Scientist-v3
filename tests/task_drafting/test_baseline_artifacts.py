"""Baseline artifact archives, digests, and native runtime result retention."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from src.investigation import baseline_artifacts
from src.investigation import task_runtime as runtime
from src.investigation.baseline_artifacts import archive_baseline_artifacts, baseline_archive_digest
from src.investigation.task_contract_types import TaskContract
from tests.program_testutils import make_program_dir
from tests.final_evaluation.test_final_evaluation import _write_contract_task


def write_outputs(root: Path) -> None:
    root.mkdir()
    (root / "metrics.json").write_text('{"combined_score": 0.25}')
    (root / "correct.json").write_text('{"correct": true}')
    (root / "qualification.json").write_text('{"schema_version": 1, "feasible": true}')
    (root / "evaluation_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [{"seed": 1, "score": 0.25, "correct": True, "duration_seconds": 1.0}],
            }
        )
    )


def archive(source: Path, destination: Path) -> None:
    archive_baseline_artifacts(
        source,
        destination,
        metric_names=("combined_score",),
        combined_score=0.25,
        correct=True,
        feasible=True,
    )


def _fresh(root: Path, name: str) -> tuple[Path, Path]:
    """Create an isolated (source, destination) pair under root/name."""
    base = root / name
    base.mkdir()
    source = base / "source"
    write_outputs(source)
    return source, base / "baseline"


def _fresh_archive(root: Path, name: str) -> Path:
    source, destination = _fresh(root, name)
    archive(source, destination)
    return destination


class _Broker:
    calls = 2
    calls_before_first_reset = 0
    reset_seeds: tuple[int, ...] = (1,)
    peer_disconnects = 0

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "source"
    write_outputs(path)
    return path


@pytest.fixture
def destination(tmp_path: Path) -> Path:
    return tmp_path / "baseline"


@pytest.fixture
def archived(source: Path, destination: Path) -> Path:
    archive(source, destination)
    return destination


# ---------------------------------------------------------------- archive_baseline_artifacts


def test_archive_retains_validated_bytes_and_accepts_absent_optional_evidence(
    tmp_path, source, destination
):
    # validated bytes are retained without console output or extra files
    (source / "stdout.log").write_text("untrusted self report")
    archive(source, destination)
    assert {p.name for p in destination.iterdir()} == {
        "metrics.json",
        "correct.json",
        "qualification.json",
        "evaluation_evidence.json",
        "archive_manifest.json",
    }
    assert (destination / "metrics.json").read_bytes() == (source / "metrics.json").read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in destination.iterdir())
    archive(source, destination)
    assert len(list(tmp_path.glob("baseline.previous-*"))) == 1

    # absent optional evidence and qualification are accepted
    minimal = tmp_path / "minimal"
    minimal.mkdir()
    (minimal / "metrics.json").write_text('{"combined_score": 0.25}')
    (minimal / "correct.json").write_text('{"correct": false}')
    minimal_destination = tmp_path / "minimal-baseline"
    archive_baseline_artifacts(
        minimal,
        minimal_destination,
        metric_names=("combined_score",),
        combined_score=0.25,
        correct=False,
        feasible=None,
    )
    assert {p.name for p in minimal_destination.iterdir()} == {
        "metrics.json",
        "correct.json",
        "archive_manifest.json",
    }


def test_archive_rejects_unsafe_sources_destinations_and_invalid_evidence(tmp_path):
    # symlinked source files are refused
    for name in ("metrics.json", "evaluation_evidence.json"):
        source, destination = _fresh(tmp_path, f"symlink-{name}")
        original = source / name
        target = source.parent / "target"
        original.rename(target)
        original.symlink_to(target)
        with pytest.raises((OSError, ValueError)):
            archive(source, destination)
        assert not destination.exists(), name

    # oversized evidence is rejected before publication
    source, destination = _fresh(tmp_path, "oversized")
    with (source / "evaluation_evidence.json").open("wb") as handle:
        handle.truncate(16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="limit"):
        archive(source, destination)
    assert not destination.exists()

    # the evidence schema is revalidated before publication
    source, destination = _fresh(tmp_path, "schema")
    (source / "evaluation_evidence.json").write_text('{"self_report": "success"}')
    with pytest.raises(ValueError, match="fields"):
        archive(source, destination)
    assert not destination.exists()

    # a symlinked destination is refused
    source, destination = _fresh(tmp_path, "symlink-destination")
    destination.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe"):
        archive(source, destination)

    # a destination containing the source is rejected
    destination = tmp_path / "archive-parent"
    destination.mkdir()
    source = destination / "source"
    write_outputs(source)
    with pytest.raises(ValueError, match="overlap"):
        archive(source, destination)
    assert (source / "metrics.json").is_file()


def test_archive_protects_existing_archives_and_unrelated_directories(tmp_path, source, archived):
    # an unrelated existing directory is never moved aside
    unrelated_root = tmp_path / "unrelated"
    unrelated_root.mkdir()
    unrelated = unrelated_root / "baseline"
    unrelated.mkdir()
    (unrelated / "unrelated.txt").write_text("keep this")
    with pytest.raises(ValueError):
        archive(source, unrelated)
    assert (unrelated / "unrelated.txt").read_text() == "keep this"
    assert not list(unrelated_root.glob("baseline.previous-*"))

    # a tampered existing archive is rejected without moving it aside
    (archived / "metrics.json").write_text('{"combined_score": 99}')
    with pytest.raises(ValueError, match="changed"):
        archive(source, archived)
    assert not list(tmp_path.glob("baseline.previous-*"))
    (archived / "metrics.json").write_text('{"combined_score": 0.25}')

    # a changed result is detected and the previous archive preserved
    (source / "metrics.json").write_text('{"combined_score": 0.9}')
    with pytest.raises(ValueError, match="changed"):
        archive(source, archived)
    assert json.loads((archived / "metrics.json").read_text())["combined_score"] == 0.25


# ---------------------------------------------------------------- baseline_archive_digest


def test_archive_digest_hashes_validated_manifest_and_rejects_post_read_changes(
    tmp_path, archived, monkeypatch
):
    # the digest hashes the manifest it validated, reading every file once
    manifest = (archived / "archive_manifest.json").read_bytes()
    read = baseline_artifacts._read_regular
    calls = []

    def observed(path, maximum):
        calls.append(path.name)
        return read(path, maximum)

    with monkeypatch.context() as patch:
        patch.setattr(baseline_artifacts, "_read_regular", observed)
        assert baseline_archive_digest(archived) == hashlib.sha256(manifest).hexdigest()
    assert sorted(calls) == sorted(path.name for path in archived.iterdir())

    # files changed after their read are rejected
    for name in ("archive_manifest.json", "metrics.json"):
        target_archive = _fresh_archive(tmp_path, f"changed-{name}")

        def changed(path, maximum, _archive=target_archive, _name=name):
            payload = read(path, maximum)
            if path.name == "qualification.json":
                target = _archive / _name
                target.write_bytes(target.read_bytes() + b" ")
            return payload

        with monkeypatch.context() as patch:
            patch.setattr(baseline_artifacts, "_read_regular", changed)
            with pytest.raises(ValueError, match="changed"):
                baseline_archive_digest(target_archive)


DIGEST_BOUNDS = [
    ("archive_manifest.json", 1024 * 1024),
    ("metrics.json", 1024 * 1024),
    ("evaluation_evidence.json", 16 * 1024 * 1024),
]


def test_archive_digest_rejects_invalid_manifests_oversized_files_and_unsafe_parents(
    tmp_path, source, archived
):
    # invalid manifests are rejected
    for payload in (b"[]", b"{", b'{"schema_version":1,"schema_version":1,"files":{}}'):
        (archived / "archive_manifest.json").write_bytes(payload)
        with pytest.raises(ValueError, match="manifest"):
            baseline_archive_digest(archived)

    # every file is bounded before it is read
    for name, maximum in DIGEST_BOUNDS:
        bounded = _fresh_archive(tmp_path, f"bounded-{name}")
        with (bounded / name).open("wb") as stream:
            stream.truncate(maximum + 1)
        with pytest.raises(ValueError, match="limit"):
            baseline_archive_digest(bounded)

    # an unsafe (symlinked) parent directory is rejected
    parent = tmp_path / "parent"
    parent.mkdir()
    archive(source, parent / "baseline")
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe"):
        baseline_archive_digest(alias / "baseline")


# ---------------------------------------------------------------- runtime archive integration


def test_runtime_archives_baseline_only_after_verified_shutdown_into_safe_destination(
    tmp_path, monkeypatch
):
    # the contracted baseline archives only after verified shutdown
    for cleanup in (True, False):
        root = tmp_path / f"cleanup-{cleanup}"
        root.mkdir()
        task = _write_contract_task(root / "task")
        contract = runtime.load_task_contract(task / "task_contract.json")
        state = {}

        class Broker(_Broker):
            reset_seeds = contract.protocol.search_reset_sequence
            shutdown_verified = False

            def __exit__(self, *args, _cleanup=cleanup):
                self.shutdown_verified = _cleanup

        def command(_state=state, **kwargs):
            _state["results"] = kwargs["results_dir"]
            return ["evaluate"]

        def evaluate(*args, _state=state, _cleanup=cleanup, **kwargs):
            results = _state["results"]
            results.rmdir()
            write_outputs(results)
            return runtime._CheckedCommandResult(
                runtime._CheckedCommandStatus.SUCCEEDED,
                subprocess.CompletedProcess(["evaluate"], 0, "", ""),
                cleanup_succeeded=_cleanup,
            )

        destination = root / "baseline"
        with monkeypatch.context() as patch:
            patch.setattr(runtime, "CandidateBroker", Broker)
            patch.setattr(runtime, "_evaluation_command", command)
            patch.setattr(runtime, "_candidate_command", lambda **kwargs: ["candidate"])
            patch.setattr(runtime, "_run_checked", evaluate)
            result = runtime._run_contracted_baseline(
                task_dir=task,
                runtime_venv=task / ".venv",
                image_id="sha256:" + "a" * 64,
                contract=contract,
                entrypoint="main.py",
                deadline=time.monotonic() + 60,
                environment={"PATH": "/bin"},
                run=subprocess.run,
                baseline_artifacts_dir=destination,
                evaluation_requirement=runtime.TaskRuntimeEvaluationRequirement.FEASIBLE_RESULT,
            )
        assert (destination / "metrics.json").exists() is cleanup
        assert result.status is (
            runtime._CheckedCommandStatus.SUCCEEDED
            if cleanup
            else runtime._CheckedCommandStatus.INFRASTRUCTURE_FAILED
        ), cleanup
        assert not state["results"].exists(), cleanup

    # the runtime report forwards the optional archive without overriding the runner
    calls = []

    def result_only(*args, **kwargs):
        calls.append(kwargs)
        return runtime.TaskRuntimePreparationResult.READY

    with monkeypatch.context() as patch:
        patch.setattr(runtime, "prepare_task_runtime_result", result_only)
        runtime.prepare_task_runtime_report(tmp_path, baseline_artifacts_dir=tmp_path / "baseline")
    assert calls[0]["baseline_artifacts_dir"] == tmp_path / "baseline"
    assert calls[0]["run"] is subprocess.run

    # an archive destination containing the task is refused
    task = tmp_path / "task"
    task.mkdir()
    result = runtime._prepare_task_runtime(
        task, deadline=time.monotonic() + 30, baseline_artifacts_dir=tmp_path, containerize=True
    )
    assert result is runtime.TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
    assert task.is_dir()


# ---------------------------------------------------------------- retained native results


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("entrypoint", ["report", "result", "bool"])
@pytest.mark.parametrize("contracted", [False, True])
def test_preparation_retains_incorrect_smoke_evidence(
    tmp_path, monkeypatch, retain, entrypoint, contracted
):
    task = _write_contract_task(tmp_path)
    shutil.rmtree(task / ".venv")
    make_program_dir(task / "program_0")
    contract_path = task / "task_contract.json"
    payload = json.loads(contract_path.read_text())
    payload["metrics"]["search"] = [{"name": "combined_score", "data_role": "development"}]
    contract_path.write_text(json.dumps(payload))
    if not contracted:
        contract_path.unlink()
    image = "sha256:" + "a" * 64
    observed = []
    evidence = {"schema_version": 1, "episodes": [{
        "seed": 1, "score": -2.0, "correct": False, "duration_seconds": 1.0,
        "solution": {"control_failure": "short smoke"},
    }]}

    def run(command, **kwargs):
        if command[:4] == ["docker", "image", "inspect", "--format={{.Id}}"]:
            return subprocess.CompletedProcess(command, 0, image + "\n", "")
        if "venv" in command:
            mount = next(value for value in command if value.endswith(":/runtime:rw"))
            environment = Path(mount.removesuffix(":/runtime:rw")) / ".venv"
            python = environment / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("")
            (environment / "lib/python3.12/site-packages/src/investigation").mkdir(parents=True)
        if "/task/evaluate.py" in command:
            if contracted:
                mount = next(value for value in command if value.endswith(",dst=/results"))
                results = Path(mount.removeprefix("type=bind,src=").removesuffix(",dst=/results"))
            else:
                mount = next(value for value in command if value.endswith(":/results:rw"))
                results = Path(mount.removesuffix(":/results:rw"))
            observed.append(results)
            (results / "metrics.json").write_text('{"combined_score":-2.0}')
            (results / "correct.json").write_text('{"correct":false}')
            (results / "evaluation_evidence.json").write_text(json.dumps(evidence))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime, "CandidateBroker", _Broker)
    monkeypatch.setattr(runtime, "_force_remove_container", lambda *args, **kwargs: True)
    retained = tmp_path / "retained"
    kwargs = {"retained_results_dir": retained} if retain else {}
    prepare = {"report": runtime.prepare_task_runtime_report,
               "result": runtime.prepare_task_runtime_result,
               "bool": runtime.prepare_task_runtime}[entrypoint]
    result = prepare(
        task, containerize=True, container_image=image, run=run,
        base_environment={"PATH": "/bin"}, draft_smoke=True,
        evaluation_requirement=runtime.TaskRuntimeEvaluationRequirement.VALID_RESULT,
        **kwargs,
    )
    if entrypoint == "report":
        assert result.result is runtime.TaskRuntimePreparationResult.READY
    elif entrypoint == "result":
        assert result is runtime.TaskRuntimePreparationResult.READY
    else:
        assert result is True
    assert len(observed) == (2 if contracted else 1)  # contracted smoke replays once
    assert observed[0].exists() is retain
    manifest = json.loads((task / ".agentic_runtime.json").read_text())
    assert manifest["baseline_correct"] is False
    if retain:
        assert observed[0] == retained
        assert json.loads((retained / "evaluation_evidence.json").read_text()) == evidence


def test_baseline_results_retention_keeps_explicit_outputs_and_refuses_unsafe_paths(
    tmp_path: Path, monkeypatch
):
    # a command setup failure retains only explicitly requested results
    for retain in (False, True):
        root = tmp_path / f"setup-{retain}"
        root.mkdir()
        observed: list[Path] = []

        def fail_command(_observed=observed, **kwargs):
            results = kwargs["results_dir"]
            _observed.append(results)
            (results / "evaluation_evidence.json").write_text('{"partial": true}')
            raise ValueError("command setup failed")

        contract = SimpleNamespace(
            compute=SimpleNamespace(
                search=SimpleNamespace(
                    wall_clock_seconds=600, evaluator_gpu_count=0, cpu_cores=1, memory_gb=1
                )
            ),
            evaluators=SimpleNamespace(search="evaluate.py"),
            candidate=SimpleNamespace(editable_paths=("program_0/src",)),
            resources=SimpleNamespace(public_artifact_bundle=None),
        )
        retained = root / "retained" if retain else None
        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_evaluation_command", fail_command)
            patch.setattr(runtime, "_prepare_task_broker_directory", lambda *a, _r=root, **kw: _r)
            result = runtime._run_contracted_baseline(
                task_dir=root,
                runtime_venv=root / ".venv",
                image_id="unused",
                contract=cast(TaskContract, contract),  # SimpleNamespace stub
                entrypoint="src/main.py",
                deadline=time.monotonic() + 600,
                environment={},
                run=subprocess.run,
                retained_results_dir=retained,
            )
        assert result.status.value == "infrastructure_failed", retain
        assert len(observed) == 1, retain
        assert observed[0].exists() is retain, retain
        if retain:
            assert observed[0] == retained
            assert (retained / "evaluation_evidence.json").read_text() == '{"partial": true}'
            assert stat.S_IMODE(retained.stat().st_mode) == 0o700

    # partial outputs are preserved on interrupt
    destination = tmp_path / "results"
    with pytest.raises(KeyboardInterrupt):
        with runtime._baseline_results_directory(destination) as results:
            (results / "partial.json").write_text("{}")
            raise KeyboardInterrupt
    assert (destination / "partial.json").read_text() == "{}"

    # an existing output directory is refused
    with pytest.raises(FileExistsError):
        with runtime._baseline_results_directory(tmp_path):
            pytest.fail("existing output directory was accepted")

    # a symlinked parent is refused
    parent = tmp_path / "real"
    parent.mkdir()
    link = tmp_path / "link"
    link.symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|canonical"):
        with runtime._baseline_results_directory(link / "results"):
            pytest.fail("symlink parent was accepted")
    assert not (parent / "results").exists()

    # a relative path is refused
    with pytest.raises(ValueError, match="absolute"):
        with runtime._baseline_results_directory(Path("results")):
            pytest.fail("relative output path was accepted")

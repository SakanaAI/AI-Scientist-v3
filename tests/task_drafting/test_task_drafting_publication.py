"""Scaffold publication plus runtime-adjacent helpers: sealed shadows, setup.py, agent failure
classification, and host storage failures."""
from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from src.investigation import task_runtime as runtime
from src.investigation import task_shadow
from src.investigation.storage_failures import storage_failure_category
from src.investigation.task_alignment import (
    TaskAlignmentError,
    load_task_alignment_report,
    task_alignment_fingerprints,
)
from src.investigation.task_contract import load_task_contract
from src.investigation.task_setup import run_task_setup
from src.investigation.task_shadow import create_sealed_task_shadow
from src.task_drafting import publication, publication_fs
from src.task_drafting.agent_failure import (
    AgentFailureKind,
    classify_agent_failure_log,
    classify_agent_failure_text,
)
from src.task_drafting.qualification import (
    TaskQualificationReceipt,
    load_task_qualification_receipt,
    write_task_qualification_receipt,
)
from tests.task_drafting.test_task_alignment import _report, _write_scaffold

MAX_RUNTIME_BYTES = 1024 * 1024


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_contents(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


@dataclass
class _Publication:
    task_dir: Path
    staged_task_dir: Path
    output_dir: Path
    alignment_report_path: Path

    @property
    def marker(self) -> Path:
        return self.task_dir / ".agentic_task_validation.json"

    @property
    def contract_path(self) -> Path:
        return self.staged_task_dir / "task_contract.json"

    def arguments(self, **overrides) -> dict:
        return {
            "task_dir": self.task_dir,
            "staged_task_dir": self.staged_task_dir,
            "output_dir": self.output_dir,
            "alignment_report_path": self.alignment_report_path,
            "max_runtime_bytes": MAX_RUNTIME_BYTES,
            **overrides,
        }

    def publish(self, **overrides) -> None:
        publication.publish_task_scaffold(**self.arguments(**overrides))

    def plan_sha256(self) -> str:
        return publication.task_publication_plan_sha256(
            task_dir=self.task_dir,
            staged_task_dir=self.staged_task_dir,
            max_runtime_bytes=MAX_RUNTIME_BYTES,
        )

    def write_contract(self, contract: dict) -> None:
        self.contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    def refresh_report(self) -> None:
        self.alignment_report_path.write_text(
            json.dumps(_report(task_alignment_fingerprints(self.staged_task_dir)), indent=2) + "\n",
            encoding="utf-8",
        )

    def add_helper(self, editable_paths: list[str] | None = None) -> Path:
        helper = self.staged_task_dir / "program_0/src/helper.py"
        helper.write_text("VALUE = 1\n", encoding="utf-8")
        if editable_paths is not None:
            contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
            contract["candidate"]["editable_paths"] = editable_paths
            self.write_contract(contract)
        self.refresh_report()
        return helper

    def write_receipt(self, plan_sha256: str) -> Path:
        qualification_path = self.output_dir / "qualification.json"
        write_task_qualification_receipt(
            qualification_path,
            TaskQualificationReceipt(
                publication_plan_sha256=plan_sha256,
                image_id="sha256:" + "2" * 64,
                evaluator_tree_sha256="3" * 64,
                requirements_sha256="4" * 64,
                tooling_sha256="5" * 64,
                runtime_cache_key="6" * 64,
                baseline_combined_score=0.75,
                accelerator_model="none",
                evaluator_gpu_count=0,
                candidate_gpu_count=0,
            ),
        )
        return qualification_path


def _make_fx(root: Path) -> _Publication:
    output_dir = root / "draft"
    staged_task_dir = output_dir / "task"
    staged_task_dir.mkdir(parents=True)
    _write_scaffold(staged_task_dir)

    root_artifact = staged_task_dir / "dependency_manifest.json"
    root_artifact.write_text('{"schema_version": 1}\n', encoding="utf-8")
    nested_artifact = staged_task_dir / "manifests" / "requirements.lock"
    nested_artifact.parent.mkdir()
    nested_artifact.write_text("example-package==1.2.3\n", encoding="utf-8")

    fixture = _Publication(root / "task", staged_task_dir, output_dir, output_dir / "alignment.json")
    contract = json.loads(fixture.contract_path.read_text(encoding="utf-8"))
    contract["provenance"]["artifacts"].extend(
        [
            {
                "role": "dependency_manifest",
                "path": root_artifact.relative_to(staged_task_dir).as_posix(),
                "sha256": _sha256(root_artifact),
                "source": "generated fixture",
            },
            {
                "role": "dependency_lock",
                "path": nested_artifact.relative_to(staged_task_dir).as_posix(),
                "sha256": _sha256(nested_artifact),
                "source": "generated fixture",
            },
        ]
    )
    fixture.write_contract(contract)
    fixture.refresh_report()
    fixture.task_dir.mkdir()
    (fixture.task_dir / "task.md").write_bytes((staged_task_dir / "task.md").read_bytes())
    return fixture


# --- publish_task_scaffold ------------------------------------------------------------------


def test_publish_task_scaffold_copies_declared_artifacts_resources_and_candidate_tree(
    tmp_path: Path,
) -> None:
    # test_publish_task_scaffold_copies_every_declared_provenance_artifact
    fx = _make_fx(tmp_path / "artifacts")
    staged_resources_before = _tree_contents(fx.staged_task_dir / "resources")
    fx.publish()
    contract = load_task_contract(fx.task_dir / "task_contract.json")
    for artifact in contract.provenance.artifacts:
        published = fx.task_dir / artifact.path
        assert published.is_file(), artifact.path
        assert _sha256(published) == artifact.sha256
    assert _tree_contents(fx.staged_task_dir / "resources") == staged_resources_before
    load_task_alignment_report(fx.marker, scaffold_dir=fx.task_dir)

    # test_publish_task_scaffold_copies_resources_without_hard_links
    source = fx.staged_task_dir / "resources/search/dev.json"
    published = fx.task_dir / "resources/search/dev.json"
    assert source.stat().st_ino != published.stat().st_ino

    # test_publish_task_scaffold_copies_complete_candidate_editable_tree
    fx = _make_fx(tmp_path / "editable")
    helper = fx.add_helper()
    fx.publish()
    assert (fx.task_dir / "program_0/src/helper.py").read_bytes() == helper.read_bytes()

    # test_publish_task_scaffold_always_copies_declared_entrypoint
    fx = _make_fx(tmp_path / "entrypoint")
    fx.add_helper(editable_paths=["program_0/src/helper.py"])
    fx.publish()
    assert (fx.task_dir / "program_0/src/program.py").is_file()


def test_publish_task_scaffold_qualification_receipt_must_match_plan(tmp_path: Path) -> None:
    # test_publish_task_scaffold_publishes_matching_qualification_receipt
    fx = _make_fx(tmp_path / "matching")
    plan_sha256 = fx.plan_sha256()
    qualification_path = fx.write_receipt(plan_sha256)
    fx.publish(qualification_receipt_path=qualification_path)
    receipt = load_task_qualification_receipt(
        fx.task_dir / ".agentic_task_qualification.json",
        expected_publication_plan_sha256=plan_sha256,
    )
    assert receipt.baseline_combined_score == pytest.approx(0.75)

    # test_qualified_publication_rejects_later_unqualified_retry_without_unpublishing
    fx = _make_fx(tmp_path / "later-unqualified")
    fx.publish(qualification_receipt_path=fx.write_receipt(fx.plan_sha256()))
    marker_before = fx.marker.read_bytes()
    with pytest.raises(publication.TaskPublicationError, match="stale qualification"):
        fx.publish()
    assert fx.marker.read_bytes() == marker_before
    assert (fx.task_dir / ".agentic_task_qualification.json").is_file()

    # test_publish_task_scaffold_rejects_stale_qualification_receipt
    fx = _make_fx(tmp_path / "stale")
    qualification_path = fx.write_receipt("9" * 64)
    with pytest.raises(Exception, match="publication plan"):
        fx.publish(qualification_receipt_path=qualification_path)
    assert not (fx.task_dir / ".agentic_task_qualification.json").exists()

    # test_unqualified_publication_rejects_existing_qualification_receipt
    fx = _make_fx(tmp_path / "existing")
    (fx.task_dir / ".agentic_task_qualification.json").write_text("stale\n", encoding="utf-8")
    with pytest.raises(publication.TaskPublicationError, match="stale qualification"):
        fx.publish()


def test_publish_task_scaffold_rejects_drift_and_removes_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # test_published_plan_verifier_rejects_resource_drift
    fx = _make_fx(tmp_path / "resource-drift")
    fx.publish()
    assert len(publication.validate_published_task_plan(fx.task_dir)) == 64
    resource = next(
        path
        for path in (fx.task_dir / "resources").rglob("*")
        if path.is_file() and path.name != ".agentic_resource_manifest.json"
    )
    resource.write_bytes(resource.read_bytes() + b"drift")
    with pytest.raises(publication.TaskPublicationError, match="publication plan"):
        publication.validate_published_task_plan(fx.task_dir)

    # test_publish_task_scaffold_removes_marker_before_reporting_live_conflict
    fx = _make_fx(tmp_path / "live-conflict")
    fx.add_helper(editable_paths=["program_0/src"])
    fx.publish()
    (fx.task_dir / "program_0/src/helper.py").write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(publication.TaskPublicationError, match="differs from accepted scaffold"):
        fx.publish()
    assert not fx.marker.exists()

    original_publish = publication._publish_regular_file

    # test_publish_task_scaffold_rejects_source_drift_after_plan
    fx = _make_fx(tmp_path / "source-drift")
    helper = fx.add_helper(editable_paths=["program_0/src"])

    def mutate_helper_before_copy(source: Path, target: Path) -> None:
        if source == helper:
            source.write_text("VALUE = 2\n", encoding="utf-8")
        original_publish(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(publication, "_publish_regular_file", mutate_helper_before_copy)
        with pytest.raises(publication.TaskPublicationError, match="publication plan"):
            fx.publish()
    assert not fx.marker.exists()

    # test_publish_task_scaffold_removes_marker_after_post_marker_drift
    fx = _make_fx(tmp_path / "post-marker")
    fx.add_helper(editable_paths=["program_0/src"])

    def mutate_after_marker(source: Path, target: Path) -> None:
        original_publish(source, target)
        if target.name == ".agentic_task_validation.json":
            (fx.task_dir / "program_0/src/helper.py").write_text("VALUE = 4\n", encoding="utf-8")

    with monkeypatch.context() as patch:
        patch.setattr(publication, "_publish_regular_file", mutate_after_marker)
        with pytest.raises(publication.TaskPublicationError, match="publication plan"):
            fx.publish()
    assert not fx.marker.exists()


def test_publish_task_scaffold_interruptions_retry_exact_plan_and_quarantine_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_publish = publication._publish_regular_file

    # test_publish_task_scaffold_retries_exact_partial_publication
    fx = _make_fx(tmp_path / "partial")
    published = 0

    def interrupt_after_first_file(source: Path, target: Path) -> None:
        nonlocal published
        original_publish(source, target)
        published += 1
        if published == 1:
            raise RuntimeError("injected publication interruption")

    with monkeypatch.context() as patch:
        patch.setattr(publication, "_publish_regular_file", interrupt_after_first_file)
        with pytest.raises(RuntimeError, match="injected publication interruption"):
            fx.publish()
    assert not fx.marker.exists()

    fx.publish()
    load_task_contract(fx.task_dir / "task_contract.json")
    load_task_alignment_report(fx.marker, scaffold_dir=fx.task_dir)

    # test_publish_task_scaffold_quarantines_false_legacy_completion_marker
    fx = _make_fx(tmp_path / "legacy-marker")
    fx.marker.write_bytes(fx.alignment_report_path.read_bytes())

    def interrupt_publication(source: Path, target: Path) -> None:
        original_publish(source, target)
        raise RuntimeError("injected publication interruption")

    with monkeypatch.context() as patch:
        patch.setattr(publication, "_publish_regular_file", interrupt_publication)
        with pytest.raises(RuntimeError, match="injected publication interruption"):
            fx.publish()
    assert not fx.marker.exists()

    # test_publish_task_scaffold_rejects_different_plan_after_interruption
    fx = _make_fx(tmp_path / "different-plan")

    def interrupt_after_root_artifact(source: Path, target: Path) -> None:
        original_publish(source, target)
        if target.name == "dependency_manifest.json":
            raise RuntimeError("injected publication interruption")

    with monkeypatch.context() as patch:
        patch.setattr(publication, "_publish_regular_file", interrupt_after_root_artifact)
        with pytest.raises(RuntimeError, match="injected publication interruption"):
            fx.publish()

    contract = json.loads(fx.contract_path.read_text(encoding="utf-8"))
    contract["provenance"]["artifacts"] = [
        item for item in contract["provenance"]["artifacts"] if item["path"] != "dependency_manifest.json"
    ]
    fx.write_contract(contract)
    (fx.staged_task_dir / "dependency_manifest.json").unlink()
    fx.refresh_report()
    with pytest.raises(publication.TaskPublicationError, match="different publication plan"):
        fx.publish()
    assert not fx.marker.exists()

    # test_publish_task_scaffold_publishes_marker_after_live_validation
    fx = _make_fx(tmp_path / "live-validation")
    original_validate = publication._validate_destination
    marker_states: list[bool] = []

    def observe_marker(scaffold: Path, report: Path) -> None:
        marker_states.append((scaffold / ".agentic_task_validation.json").exists())
        original_validate(scaffold, report)

    with monkeypatch.context() as patch:
        patch.setattr(publication, "_validate_destination", observe_marker)
        fx.publish()
    assert marker_states == [False, True]


def test_publish_task_scaffold_validation_failure_repair_nested_output_and_empty_directory(
    tmp_path: Path,
) -> None:
    # test_publish_task_scaffold_validation_failure_leaves_both_trees_unchanged
    fx = _make_fx(tmp_path / "validation-failure")
    report = json.loads(fx.alignment_report_path.read_text(encoding="utf-8"))
    report["fingerprints"]["task_contract.json"] = "0" * 64
    fx.alignment_report_path.write_text(json.dumps(report), encoding="utf-8")
    task_before = _tree_contents(fx.task_dir)
    staged_before = _tree_contents(fx.staged_task_dir)
    with pytest.raises(TaskAlignmentError, match="fingerprints do not match"):
        fx.publish()
    assert _tree_contents(fx.task_dir) == task_before
    assert _tree_contents(fx.staged_task_dir) == staged_before
    assert not fx.marker.exists()

    # test_publish_task_scaffold_repairs_missing_file_behind_valid_marker
    fx = _make_fx(tmp_path / "missing-file")
    helper = fx.add_helper(editable_paths=["program_0/src"])
    fx.publish()
    published_helper = fx.task_dir / "program_0/src/helper.py"
    published_helper.unlink()
    fx.publish()
    assert published_helper.read_bytes() == helper.read_bytes()

    # test_publish_task_scaffold_allows_output_below_task_directory
    fx = _make_fx(tmp_path / "nested-output")
    nested_output = fx.task_dir / ".task_drafting/draft"
    nested_output.parent.mkdir()
    shutil.move(fx.output_dir, nested_output)
    fx.publish(
        staged_task_dir=nested_output / fx.staged_task_dir.name,
        output_dir=nested_output,
        alignment_report_path=nested_output / fx.alignment_report_path.name,
    )
    assert fx.marker.is_file()

    # test_publish_task_scaffold_accepts_existing_v1_resource_manifest_with_empty_directory
    fx = _make_fx(tmp_path / "empty-directory")
    (fx.staged_task_dir / "resources/search/empty").mkdir()
    fx.refresh_report()
    fx.publish()
    fx.publish()
    assert (fx.task_dir / "resources/.agentic_resource_manifest.json").is_file()


def test_publication_tree_limit_and_runtime_resource_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # test_publication_tree_entry_limit_stops_before_yielding_extra_files
    tree = tmp_path / "tree"
    tree.mkdir()
    for index in range(3):
        (tree / f"file_{index}").touch()
    with monkeypatch.context() as patch:
        patch.setattr(publication_fs, "TREE_MAX_ENTRIES", 2)
        with pytest.raises(publication_fs.TaskPublicationError, match="maximum entry count"):
            tuple(publication_fs.regular_tree_files(tree))

    # test_runtime_resource_preflight_accepts_exact_byte_limit
    root = tmp_path / "exact"
    resources = root / "resources" / "search"
    resources.mkdir(parents=True)
    (resources / "payload.bin").write_bytes(b"12345678")
    assert publication.validate_staged_runtime_resources(root, max_runtime_bytes=8) == 8

    # test_runtime_resource_preflight_stops_over_limit_with_contributors
    root = tmp_path / "over"
    resources = root / "resources" / "search"
    resources.mkdir(parents=True)
    (resources / "model.bin").write_bytes(b"x" * 9)
    with pytest.raises(publication.TaskPublicationError) as error:
        publication.validate_staged_runtime_resources(root, max_runtime_bytes=8)
    for fragment in ("runtime_resources_exceed_limit", "observed=9", "limit=8", "search/model.bin:9"):
        assert fragment in str(error.value)

    # test_runtime_resource_preflight_rejects_symlink
    root = tmp_path / "symlink"
    resources = root / "resources"
    resources.mkdir(parents=True)
    (resources / "escape").symlink_to(root.parent)
    with pytest.raises(publication.TaskPublicationError, match="unsafe task publication"):
        publication.validate_staged_runtime_resources(root, max_runtime_bytes=8)


# --- sealed task shadows --------------------------------------------------------------------


def _write_shadow_source(root: Path) -> Path:
    task = root / "task"
    (task / "resources" / "search").mkdir(parents=True)
    (task / "resources" / "final").mkdir()
    (task / "task.md").write_text("public task\n", encoding="utf-8")
    (task / "evaluate.py").write_text("public evaluator\n", encoding="utf-8")
    (task / "evaluate_final.py").write_text("sealed evaluator\n", encoding="utf-8")
    (task / "resources/search/visible.json").write_text("public search resource\n", encoding="utf-8")
    (task / "resources/final/hidden.json").write_text("sealed final resource\n", encoding="utf-8")
    contract = {
        "canonical_task": {"path": "task.md", "sha256": _sha256(task / "task.md")},
        "provenance": {
            "artifacts": [
                {"path": relative, "sha256": _sha256(task / relative)}
                for relative in ("resources/search/visible.json", "resources/final/hidden.json")
            ]
        },
    }
    (task / "task_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    return task


def _create_shadow(source: Path, shadow: Path) -> None:
    shadow.mkdir(exist_ok=True)
    create_sealed_task_shadow(
        source,
        shadow,
        final_evaluator="evaluate_final.py",
        final_resources="resources/final",
        contract_sha256=_sha256(source / "task_contract.json"),
    )


def test_shadow_masks_sealed_paths_supports_read_only_sources_and_avoids_prehash(
    tmp_path: Path,
) -> None:
    # test_shadow_keeps_search_resources_and_masks_sealed_paths
    root = tmp_path / "masks"
    root.mkdir()
    source = _write_shadow_source(root)
    shadow = root / "shadow"
    _create_shadow(source, shadow)
    assert (shadow / "task.md").read_text(encoding="utf-8") == "public task\n"
    assert (shadow / "resources/search/visible.json").read_text(encoding="utf-8") == (
        "public search resource\n"
    )
    assert (shadow / "evaluate_final.py").read_bytes() == b""
    assert (shadow / "task_contract.json").read_bytes() == b""
    assert list((shadow / "resources/final").iterdir()) == []

    # test_shadow_supports_read_only_source_directories
    root = tmp_path / "read-only"
    root.mkdir()
    source = _write_shadow_source(root)
    (source / "resources").chmod(0o555)
    source.chmod(0o555)
    shadow = root / "shadow"
    try:
        _create_shadow(source, shadow)
    finally:
        source.chmod(0o755)
        (source / "resources").chmod(0o755)
    assert (shadow / "task.md").is_file()
    assert (shadow / "resources/final").is_dir()

    # test_shadow_masks_manual_validation_report_and_preserves_host_evidence
    root = tmp_path / "manual-report"
    root.mkdir()
    source = _write_shadow_source(root)
    report = source / "manual_validation.json"
    report.write_text('{"final": {"score": 0.75}}\n')
    shadow = root / "shadow"
    _create_shadow(source, shadow)
    assert (shadow / "manual_validation.json").read_bytes() == b""
    assert report.read_text() == '{"final": {"score": 0.75}}\n'

    # test_shadow_does_not_prehash_contract_verified_resources
    root = tmp_path / "prehash"
    root.mkdir()
    source = _write_shadow_source(root)
    with patch(
        "src.investigation.task_shadow._hash_descriptor", wraps=task_shadow._hash_descriptor
    ) as hash_descriptor:
        _create_shadow(source, root / "shadow")
    assert hash_descriptor.call_count == 1


def test_shadow_rejects_sealed_hard_link_and_same_size_source_rewrite(tmp_path: Path) -> None:
    # test_shadow_rejects_public_hard_link_to_sealed_content
    root = tmp_path / "hard-link"
    root.mkdir()
    source = _write_shadow_source(root)
    os.link(source / "evaluate_final.py", source / "leaked_final.py")
    with pytest.raises(ValueError, match="aliases sealed content"):
        _create_shadow(source, root / "shadow")

    # test_shadow_rejects_same_size_source_rewrite_with_restored_mtime
    root = tmp_path / "rewrite"
    root.mkdir()
    source = _write_shadow_source(root)
    original_copy = task_shadow._copy_regular
    mutated = False

    def mutate_and_copy(*args, **kwargs) -> None:
        nonlocal mutated
        expected = args[2]
        if not mutated and expected.relative.as_posix() == "task.md":
            target = source / "task.md"
            metadata = target.stat()
            target.write_text("sealed task\n", encoding="utf-8")
            os.utime(target, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
            mutated = True
        original_copy(*args, **kwargs)

    with patch("src.investigation.task_shadow._copy_regular", mutate_and_copy):
        with pytest.raises(RuntimeError, match="task source (content )?changed"):
            _create_shadow(source, root / "shadow")


# --- run_task_setup -------------------------------------------------------------------------


_SETUP_OUTCOMES = [
    ("no-setup", None, None, True),
    ("exit-0", "import sys; sys.exit(0)\n", None, True),
    ("exit-1", "import sys; sys.exit(1)\n", None, False),
    ("timeout", "import time; time.sleep(60)\n", 1, False),
]


def test_run_task_setup_reports_outcome_and_passes_args_as_json(tmp_path: Path) -> None:
    # test_run_task_setup_reports_setup_script_outcome
    for name, script, timeout, expected in _SETUP_OUTCOMES:
        root = tmp_path / name
        root.mkdir()
        if script is not None:
            (root / "setup.py").write_text(script, encoding="utf-8")
        kwargs = {} if timeout is None else {"timeout_seconds": timeout}
        assert bool(run_task_setup(root, {"task_id": "abc"}, **kwargs)) is expected, name

    # test_run_task_setup_passes_args_as_json
    root = tmp_path / "args"
    root.mkdir()
    (root / "setup.py").write_text(
        "import argparse, os, pathlib, sys\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--args')\n"
        "a = p.parse_args()\n"
        "out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'received_args.json')\n"
        "pathlib.Path(out).write_text(a.args)\n",
        encoding="utf-8",
    )
    extra = {"task_id": "da515329", "problem_set": "training"}
    assert run_task_setup(root, extra)
    assert json.loads((root / "received_args.json").read_text(encoding="utf-8")) == extra


# --- agent failure classification -----------------------------------------------------------


_TRANSIENT_MESSAGES = [
    "HTTP 429: rate limit exceeded",
    "HTTP 429 Too Many Requests",
    "RESOURCE_EXHAUSTED: quota temporarily unavailable",
    "The upstream service is temporarily overloaded",
    "provider returned 503 service unavailable",
    "connection reset by peer",
    "request timed out while contacting the model provider",
    "the model provider request timed out",
]

_NON_TRANSIENT_MESSAGES = [
    ("invalid-api-key", "invalid API key"),
    ("permission-denied", "permission denied"),
    ("no-edit", "agent completed without editing the task"),
    ("incidental-429", "the report discusses HTTP 429 handling"),
    ("empty", ""),
    ("permanent-wins-over-stale-transient", "rate limit exceeded\ninvalid API key"),
    (
        "structured-error-wins-over-incidental-output",
        '{"type":"error","message":"Selected model is at capacity"}\n'
        '{"type":"turn.failed","error":{"message":"invalid API key"}}',
    ),
    ("codex-capacity", "Selected model is at capacity. Please try a different model."),
    (
        "codex-capacity-structured",
        '{"type":"turn.failed","error":{"message":"Selected model is at capacity. Please try a different model."}}',
    ),
    ("codex-capacity-after-429", "HTTP 429: rate limit exceeded\nSelected model is at capacity"),
]


def test_agent_failure_classification_of_text_and_bounded_owned_logs(tmp_path: Path) -> None:
    # test_transient_provider_failures_are_retryable
    for message in _TRANSIENT_MESSAGES:
        assert classify_agent_failure_text(message) is AgentFailureKind.TRANSIENT_PROVIDER, message
    # test_non_transient_failures_are_not_retried
    for name, message in _NON_TRANSIENT_MESSAGES:
        assert classify_agent_failure_text(message) is AgentFailureKind.NON_RETRYABLE, name

    # test_log_classifier_reads_only_bounded_tail
    log = tmp_path / "agent.log"
    log.write_text(
        "Selected model is at capacity\n" + "x" * (70 * 1024) + "\ninvalid API key\n", encoding="utf-8"
    )
    assert classify_agent_failure_log(log) is AgentFailureKind.NON_RETRYABLE

    # test_log_classifier_rejects_aliased_file
    for alias in ("symlink", "hardlink"):
        root = tmp_path / alias
        root.mkdir()
        target = root / "target.log"
        target.write_text("Selected model is at capacity\n", encoding="utf-8")
        log = root / "agent.log"
        if alias == "symlink":
            log.symlink_to(target)
        else:
            os.link(target, log)
        with pytest.raises(ValueError, match="owned regular file"):
            classify_agent_failure_log(log)


# --- host storage failures are infrastructure, not authored defects -------------------------


def test_storage_errors_classify_errno_through_wrapping_and_stderr_fallback(monkeypatch):
    # test_storage_errors_classify_errno_not_untrusted_text
    for number, category in (
        (errno.EDQUOT, "storage_quota_exceeded"),
        (errno.ENOSPC, "storage_space_exhausted"),
        (errno.EACCES, None),
    ):
        assert storage_failure_category(OSError(number, "private filename")) == category, number
        assert storage_failure_category(ValueError("[Errno 122] Disk quota exceeded")) is None

    # test_storage_error_survives_wrapped_and_cleanup_exceptions
    original = OSError(errno.EDQUOT, "quota")
    wrapped = ValueError("staging failed")
    wrapped.__cause__ = original
    cleanup = OSError(errno.EIO, "cleanup failed")
    cleanup.__context__ = wrapped
    original.__context__ = cleanup  # Defensive handling of cycles.
    assert storage_failure_category(cleanup) == "storage_quota_exceeded"

    # test_stderr_fallback_failure_does_not_replace_storage_error
    from src.investigation import storage_failures

    class FullStream:
        def write(self, _text):
            raise OSError(errno.ENOSPC, "stderr full")

        def flush(self):
            pass

    monkeypatch.setattr(storage_failures.sys, "stderr", FullStream())
    storage_failures.warn_storage_failure(OSError(errno.EDQUOT, "quota"), operation="checkpoint")


def _fail_cid_unlink(monkeypatch, cid_path: Path, number: int, message: str) -> None:
    original_unlink = Path.unlink

    def fail_cleanup(path, *args, **kwargs):
        if path == cid_path:
            raise OSError(number, message)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    monkeypatch.setattr(runtime, "_remove_container_before_deadline", lambda *_args: True)


def test_runtime_storage_failures_are_infrastructure_and_cleanup_cannot_hide_primary_failure(
    tmp_path, monkeypatch
):
    # test_runtime_staging_reports_storage_infrastructure_failure
    for number in (errno.EDQUOT, errno.ENOSPC):
        root = tmp_path / f"staging-{number}"
        root.mkdir()

        def fail_staging(*_args, _number=number, **_kwargs):
            raise OSError(_number, "private-token-must-not-leak")

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_prepare_task_runtime", fail_staging)
            report = runtime.prepare_task_runtime_report(root)
        assert report.result is runtime.TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE, number
        assert report.failure_category == storage_failure_category(OSError(number, ""))
        assert "private-token" not in report.diagnostic

    # test_checked_command_cleanup_cannot_hide_primary_storage_error
    root = tmp_path / "checked"
    root.mkdir()
    cid_path = root / "eval.cid"
    cid_path.write_text("container")

    def fail(*_args, **_kwargs):
        raise OSError(errno.EDQUOT, "quota")

    with monkeypatch.context() as patch:
        _fail_cid_unlink(patch, cid_path, errno.EIO, "cleanup failed")
        handler = runtime._RuntimeReportDiagnosticHandler()
        runtime.logger.addHandler(handler)
        try:
            result = runtime._run_checked(
                ["eval"], task_dir=root, deadline=time.monotonic() + 30,
                environment={}, run=fail, cid_path=cid_path,
            )
        finally:
            runtime.logger.removeHandler(handler)
    assert result.status is runtime._CheckedCommandStatus.INFRASTRUCTURE_FAILED
    assert "storage_quota_exceeded" in handler.diagnostic()

    # test_cleanup_storage_does_not_replace_primary_authored_failure
    from src.investigation.bounded_process import ProcessOutputLimitError

    for failure in ("output_limit", "nonzero"):
        root = tmp_path / f"authored-{failure}"
        root.mkdir()
        cid_path = root / "eval.cid"
        cid_path.write_text("container")

        def run(*_args, _failure=failure, **_kwargs):
            if _failure == "output_limit":
                raise ProcessOutputLimitError(stderr=b"operation=train console overflow")
            return subprocess.CompletedProcess(["eval"], 1, "", "operation=train failed")

        def prepare(*_args, _root=root, _cid_path=cid_path, _run=run, **_kwargs):
            result = runtime._run_checked(
                ["eval"], task_dir=_root, deadline=time.monotonic() + 30,
                environment={}, run=_run, cid_path=_cid_path, stage="baseline_validation",
            )
            assert result.status is runtime._CheckedCommandStatus.COMMAND_FAILED
            return runtime.TaskRuntimePreparationResult.AUTHORED_FAILURE

        with monkeypatch.context() as patch:
            _fail_cid_unlink(patch, cid_path, errno.ENOSPC, "private-cleanup-details")
            patch.setattr(runtime, "prepare_task_runtime_result", prepare)
            report = runtime.prepare_task_runtime_report(root)
        assert report.result is runtime.TaskRuntimePreparationResult.AUTHORED_FAILURE, failure
        assert report.failure_category == (
            "output_limit_exceeded" if failure == "output_limit" else "evaluation_failed"
        )
        assert "operation=train" in report.output_evidence.stderr
        assert "secondary" in report.diagnostic
        assert "storage_space_exhausted" in report.diagnostic
        assert "private-cleanup" not in report.diagnostic


def test_event_and_qualification_storage_errors_are_infrastructure_without_leaks(
    tmp_path, monkeypatch, capsys
):
    # test_event_quota_warning_reaches_stderr_without_event_payload
    from src.investigation import event_logging

    def fail_append(*_args, **_kwargs):
        raise OSError(errno.EDQUOT, "private filename")

    events_root = tmp_path / "events"
    events_root.mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(event_logging.state_store, "append_jsonl", fail_append)
        stream = io.StringIO()
        logger = event_logging.EventLogger(
            log_format="json", verbose=False, heartbeat_seconds=0, results_dir=events_root,
            worker_index=0, stream=stream,
        )
        try:
            logger.emit("first", candidate_payload="private data")
            logger.emit("second")
        finally:
            logger.close()
    stderr = capsys.readouterr().err
    assert "storage_quota_exceeded" in stderr
    assert "private" not in stderr
    assert len(stderr.encode()) < 1024
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert records[0]["fields"]["failure_category"] == "storage_quota_exceeded"
    assert [record["event"] for record in records][-2:] == ["first", "second"]

    # test_qualification_storage_errors_are_not_authored_repairs
    from src.task_drafting import qualification
    from src.task_drafting.publication_fs import TaskPublicationError
    from tests.task_drafting.test_task_qualification import _receipt
    from tests.runtime.test_task_runtime import _write_contract_task

    def fail():
        try:
            raise OSError(errno.EDQUOT, "private-credential")
        except OSError as error:
            raise TaskPublicationError("wrapped storage failure") from error

    for stage in ("publication_plan", "qualified_plan", "receipt"):
        root = tmp_path / stage
        root.mkdir()
        task = _write_contract_task(root)
        output = root / "validation"
        output.mkdir()
        calls = 0

        def plan(_stage=stage, **_kwargs):
            nonlocal calls
            calls += 1
            if _stage == "publication_plan" or _stage == "qualified_plan" and calls == 2:
                fail()
            return "1" * 64

        with monkeypatch.context() as patch:
            patch.setattr(
                "src.task_drafting.publication.task_publication_plan_sha256",
                lambda **kwargs: plan(**kwargs),
            )
            patch.setattr(
                qualification, "qualification_receipt_from_runtime", lambda *_a, **_k: _receipt()
            )
            if stage == "receipt":
                patch.setattr(qualification, "write_task_qualification_receipt", lambda *_a: fail())
            attempt = qualification.qualify_task_for_publication(
                task_dir=task, publication_task_dir=task, output_dir=output,
                receipt_path=output / "receipt.json", timeout_seconds=30, container_image="image",
                container_network="none", install_cache_dir=root / "cache", max_runtime_bytes=1024,
                prepare=lambda *_a, **_k: runtime.TaskRuntimePreparationReport(
                    runtime.TaskRuntimePreparationResult.READY, None, None, "", "8" * 64, 1.0,
                ),
            )
        assert attempt.status == "infrastructure_failure", stage
        assert attempt.category == "storage_quota_exceeded", stage
        assert "private-credential" not in attempt.diagnostic, stage


@pytest.mark.parametrize("rollback_errno", [None, errno.EIO, errno.ENOSPC])
def test_runtime_publication_quota_keeps_original_failure_and_recovery_state(
    tmp_path, monkeypatch, rollback_errno,
):
    from tests.program_testutils import make_program_dir

    task = tmp_path / "task"
    task.mkdir()
    make_program_dir(task / "program_0", content="print('baseline')\n")
    (task / "evaluate.py").write_text("")
    image = "sha256:" + "a" * 64

    def fake_run(command, **_kwargs):
        if command[:4] == ["docker", "image", "inspect", "--format={{.Id}}"]:
            return subprocess.CompletedProcess(command, 0, image + "\n", "")
        if "venv" in command:
            mount = next(value for value in command if value.endswith(":/runtime:rw"))
            python = Path(mount.removesuffix(":/runtime:rw")) / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("")
        if "/task/evaluate.py" in command:
            mount = next(value for value in command if value.endswith(":/results:rw"))
            results = Path(mount.removesuffix(":/results:rw"))
            (results / "metrics.json").write_text('{"combined_score":1.0}')
            (results / "correct.json").write_text('{"correct":true}')
        return subprocess.CompletedProcess(command, 0, "", "")

    assert runtime.prepare_task_runtime_report(
        task, containerize=True, container_image=image, run=fake_run,
    ).ready
    (task / "requirements.txt").write_text("example==1.0\n")
    original_replace = Path.replace

    def fail_publish(path, target):
        if Path(target) == task / ".venv":
            if path.name == ".previous-venv":
                if rollback_errno is not None:
                    raise OSError(rollback_errno, "rollback failed")
            else:
                raise OSError(errno.EDQUOT, "private-credential")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publish)
    report = runtime.prepare_task_runtime_report(
        task, containerize=True, container_image=image, run=fake_run,
    )
    assert report.result is runtime.TaskRuntimePreparationResult.INFRASTRUCTURE_FAILURE
    assert report.failure_category == "storage_quota_exceeded"
    assert "private-credential" not in report.diagnostic
    if rollback_errno is not None:
        assert list(tmp_path.glob(".task.runtime-build-*/.previous-venv/bin/python"))
    else:
        assert (task / ".venv/bin/python").is_file()

"""Program variant cloning utilities (program_<idx-1> -> program_<idx>)."""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import shutil
import stat
from pathlib import Path

from .files import evaluation_results_complete
from .evaluation_evidence import EVIDENCE_FILENAME
from .program_spec import (
    PROGRAM_SPEC_FILENAME,
    ProgramSpecError,
    load_program_spec,
)

# Evaluator artifacts are managed by the pipeline and must not be carried forward.
_NONOVERRIDABLE_EXCLUDES: set[str] = {
    ".harness_results",
    ".evaluation_pending",
    ".metaevo_bbob_official",
    ".metaevo_official",
    "_analysis_extras.json",
    "_results.json",
    "analysis",
    "metrics.json",
    "correct.json",
    EVIDENCE_FILENAME,
    "rubric_status.json",
    "stdout.log",
    "stderr.log",
    "evaluation_manifest.json",
    "harness_evaluate_stdout.log",
    "harness_evaluate_stderr.log",
}
_PROGRAM_DIR_PATTERN = re.compile(r"program_(\d+)")
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_SUPPORTED_EVALUATION_PRODUCERS = {
    "harness_eval",
    "isolated_evaluator",
    "sandbox_host_eval",
    "native_host_eval",
}
_REQUIRED_SOURCE_ARTIFACTS = (
    "metrics.json",
    "correct.json",
    "evaluation_manifest.json",
    "rubric_status.json",
)


def _read_regular_json(path: Path) -> dict[str, object]:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(
            f"Candidate source not ready; missing required artifact: {path.name}"
        ) from exc
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(
            f"Candidate source not ready; artifact must be a regular file: {path.name}"
        )
    if path_stat.st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError(
            f"Candidate source not ready; artifact is too large: {path.name}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Candidate source not ready; invalid JSON artifact: {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"Candidate source not ready; artifact must contain a JSON object: {path.name}"
        )
    return value


def _require_candidate_source_ready(*, src_dir: Path, dst_dir: Path) -> None:
    src_dir = src_dir.expanduser().resolve()
    dst_dir = dst_dir.expanduser().resolve()
    dst_match = _PROGRAM_DIR_PATTERN.fullmatch(dst_dir.name)
    if dst_match is None:
        raise ValueError("Candidate clone destination must be named program_<index>")

    dst_index = int(dst_match.group(1))
    if dst_index == 0:
        return
    src_match = _PROGRAM_DIR_PATTERN.fullmatch(src_dir.name)
    if src_match is None:
        raise ValueError("Candidate clone source must be named program_<index>")
    if src_dir.parent != dst_dir.parent:
        raise ValueError("Candidate clones must remain in the same experiment directory")

    src_index = int(src_match.group(1))
    if src_index + 1 != dst_index:
        raise ValueError(
            "Candidate clone must use adjacent indices: "
            f"{src_dir.name} -> {dst_dir.name}"
        )
    if src_index == 0:
        return

    artifacts = {
        filename: _read_regular_json(src_dir / filename)
        for filename in _REQUIRED_SOURCE_ARTIFACTS
    }
    manifest = artifacts["evaluation_manifest.json"]
    if manifest.get("producer") not in _SUPPORTED_EVALUATION_PRODUCERS:
        raise ValueError(
            "Candidate source evaluation_manifest.json has an unsupported producer"
        )
    if manifest.get("program_id") != src_dir.name:
        raise ValueError(
            "Candidate source evaluation_manifest.json program_id must match "
            f"{src_dir.name}"
        )
    if not evaluation_results_complete(src_dir):
        raise ValueError("Candidate source not ready; evaluation manifest is incomplete")

    score = artifacts["metrics.json"].get("combined_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise ValueError(
            "Candidate source metrics.json combined_score must be finite and numeric"
        )
    correct = artifacts["correct.json"].get("correct")
    if not isinstance(correct, bool):
        raise ValueError("Candidate source correct.json correct must be boolean")

    rubric = artifacts["rubric_status.json"]
    attempt_id = rubric.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise ValueError("Candidate source rubric_status.json needs an attempt_id")
    idea_id = rubric.get("idea_id")
    if not isinstance(idea_id, str) or not idea_id.strip():
        raise ValueError("Candidate source rubric_status.json needs an idea_id")
    if not isinstance(rubric.get("overall_pass"), bool):
        raise ValueError("Candidate source rubric_status.json overall_pass must be boolean")
    rubric_entries = rubric.get("rubrics")
    if not isinstance(rubric_entries, list) or not rubric_entries:
        raise ValueError("Candidate source rubric_status.json needs rubric entries")
    for entry in rubric_entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or not entry["id"].strip()
            or not isinstance(entry.get("pass"), bool)
            or not isinstance(entry.get("evidence"), str)
        ):
            raise ValueError(
                "Candidate source rubric_status.json contains an invalid rubric entry"
            )
    derived_overall_pass = correct and all(entry["pass"] for entry in rubric_entries)
    if rubric["overall_pass"] != derived_overall_pass:
        raise ValueError(
            "Candidate source rubric_status.json has inconsistent overall_pass"
        )
    if derived_overall_pass:
        raise ValueError(
            "Candidate source already passes all rubrics; do not create another candidate"
        )

    latest_rubric = _read_regular_json(src_dir.parent / "rubric_status.json")
    if latest_rubric != artifacts["rubric_status.json"]:
        raise ValueError(
            "Candidate source not ready; root rubric_status.json does not match "
            f"{src_dir.name}/rubric_status.json"
        )


def _matches_any(patterns: tuple[str, ...], relpath: str) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatchcase(relpath, pattern):
            return True
    return False


def _should_copy(
    *,
    relpath: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> bool:
    # Non-overridable excludes only apply at program root (exact relpath).
    root_name = relpath.split("/", 1)[0]
    if root_name in _NONOVERRIDABLE_EXCLUDES or root_name.startswith(
        ".evaluation_stage_"
    ):
        return False
    if include and _matches_any(include, relpath):
        return True
    if exclude and _matches_any(exclude, relpath):
        return False
    return True


def _should_copy_dir(
    *,
    relpath: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> bool:
    # Allow directory-oriented patterns like `node_modules/**` to match `node_modules/`.
    relpath_dir = f"{relpath}/"
    root_name = relpath.split("/", 1)[0]
    if root_name in _NONOVERRIDABLE_EXCLUDES or root_name.startswith(
        ".evaluation_stage_"
    ):
        return False
    if include and (_matches_any(include, relpath) or _matches_any(include, relpath_dir)):
        return True
    if exclude and (_matches_any(exclude, relpath) or _matches_any(exclude, relpath_dir)):
        return False
    return True


def clone_program_dir(*, src_dir: Path, dst_dir: Path) -> None:
    """Clone a program directory into a new destination, respecting program_spec.toml copy rules."""
    src_dir = src_dir.expanduser().resolve()
    dst_dir = dst_dir.expanduser().resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source program directory not found: {src_dir}")
    if src_dir == dst_dir or src_dir in dst_dir.parents or dst_dir in src_dir.parents:
        raise ValueError("Source and destination program directories must not overlap")

    spec = load_program_spec(src_dir, required=True)
    if spec is None:
        raise ProgramSpecError(f"Missing {PROGRAM_SPEC_FILENAME} in {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    if any(dst_dir.iterdir()):
        raise FileExistsError(f"Destination directory is not empty: {dst_dir}")

    exclude = spec.copy.exclude
    include = spec.copy.include

    # Walk and copy deterministically. Prune common huge excluded dirs when includes are empty.
    followlinks = False
    for root, dirnames, filenames in os.walk(src_dir, topdown=True, followlinks=followlinks):
        root_path = Path(root)
        rel_root = root_path.relative_to(src_dir).as_posix()
        if rel_root == ".":
            rel_root = ""

        # Copy symlinked directories as symlinks (do not traverse into them).
        for dirname in sorted(dirnames):
            src_path = root_path / dirname
            if not src_path.is_symlink():
                continue
            relpath = f"{rel_root}/{dirname}".lstrip("/")
            if not _should_copy_dir(relpath=relpath, include=include, exclude=exclude):
                dirnames.remove(dirname)
                continue
            dst_path = dst_dir / relpath
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            link_target = os.readlink(src_path)
            dst_path.symlink_to(link_target, target_is_directory=True)
            dirnames.remove(dirname)

        if not include:
            pruned: list[str] = []
            for dirname in list(dirnames):
                candidate_rel = f"{rel_root}/{dirname}".lstrip("/")
                # If directory is excluded, prune traversal.
                if _matches_any(exclude, f"{candidate_rel}/") or _matches_any(exclude, candidate_rel):
                    dirnames.remove(dirname)
                    pruned.append(dirname)
            if pruned:
                # Keep deterministic ordering after pruning.
                dirnames.sort()

        dirnames.sort()
        filenames.sort()

        for filename in filenames:
            src_path = root_path / filename
            relpath = f"{rel_root}/{filename}".lstrip("/")

            if not _should_copy(relpath=relpath, include=include, exclude=exclude):
                continue

            dst_path = dst_dir / relpath
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_symlink():
                # v1 policy: copy symlinks as symlinks (do not dereference targets).
                link_target = os.readlink(src_path)
                dst_path.symlink_to(link_target, target_is_directory=src_path.is_dir())
            elif src_path.is_file():
                shutil.copy2(src_path, dst_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clone a program_<idx> directory into a new variant.")
    parser.add_argument("--from", dest="src", required=True, help="Source program directory")
    parser.add_argument("--to", dest="dst", required=True, help="Destination program directory")
    args = parser.parse_args(argv)
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    _require_candidate_source_ready(src_dir=src_dir, dst_dir=dst_dir)
    clone_program_dir(src_dir=src_dir, dst_dir=dst_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate the exact TeX and BibTeX inputs observed during compilation."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

AGENT_SKILL_DIR_PREFIXES = (
    (".claude", "skills"),
    (".codex", "skills"),
    (".cursor", "skills"),
    (".opencode", "skills"),
    (".gemini", "skills"),
    (".pi", "agent", "skills"),
)
LATEX_SYSTEM_INPUT_ROOTS = tuple(
    Path(path)
    for path in (
        "/etc/texmf",
        "/opt/texlive",
        "/usr/local/share/miktex-texmf",
        "/usr/local/share/texmf",
        "/usr/local/texlive",
        "/usr/share/texlive",
        "/usr/share/texmf",
        "/usr/share/texmf-dist",
        "/var/lib/miktex-texmf",
        "/var/lib/texmf",
        "/nix/store",
    )
)
BIBTEX_AUX_DATA_RE = re.compile(r"\\bibdata\s*\{(?P<targets>[^{}]+)\}")
BIBTEX_AUX_DATA_COMMAND_RE = re.compile(r"\\bibdata(?![A-Za-z@])")
BIBTEX_AUX_STYLE_RE = re.compile(r"\\bibstyle\s*\{(?P<target>[^{}]+)\}")
BIBTEX_AUX_STYLE_COMMAND_RE = re.compile(r"\\bibstyle(?![A-Za-z@])")
BIBTEX_AUX_INPUT_RE = re.compile(r"\\@input\s*\{(?P<target>[^{}]+)\}")
BIBTEX_AUX_INPUT_COMMAND_RE = re.compile(r"\\@input(?![A-Za-z@])")
LATEX_ENV_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SOURCE_DATE_EPOCH",
    "FORCE_SOURCE_DATE",
)


def _uses_agent_skill_path(path: Path) -> bool:
    return any(
        path.parts[: len(prefix)] == prefix for prefix in AGENT_SKILL_DIR_PREFIXES
    )


def _is_approved_latex_system_input(input_path: Path) -> bool:
    """Accept a TeX system input whose real path stays inside any approved root.

    Distributions alias one approved root into another (Debian/Ubuntu ship
    ``/usr/share/texmf/web2c/texmf.cnf`` as a symlink into
    ``/usr/share/texlive/texmf-dist``), so a resolved input is checked against
    every approved root rather than only the first one it lexically matches.
    """
    if not any(input_path.is_relative_to(root) for root in LATEX_SYSTEM_INPUT_ROOTS):
        return False
    if not input_path.exists():
        return True
    try:
        resolved_input = input_path.resolve(strict=True)
    except OSError:
        return False
    for system_root in LATEX_SYSTEM_INPUT_ROOTS:
        try:
            resolved_root = system_root.resolve(strict=True)
        except OSError:
            continue
        if resolved_input.is_relative_to(resolved_root):
            return True
    return False


def validate_latex_recorder_inputs(paper_dir: Path, fls_name: str) -> None:
    root = paper_dir.resolve()
    fls_path = root / fls_name
    if not fls_path.is_file():
        raise RuntimeError(f"LaTeX recorder output missing: {fls_path}")
    recorder_lines = fls_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    recorded_pwd = next(
        (
            Path(line.removeprefix("PWD ").strip())
            for line in recorder_lines
            if line.startswith("PWD ") and line.removeprefix("PWD ").strip()
        ),
        root,
    )
    for line in recorder_lines:
        if not line.startswith("INPUT "):
            continue
        input_text = line.removeprefix("INPUT ").strip()
        if not input_text:
            continue
        input_path = Path(input_text)
        relative_candidate: Path | None = None
        if input_path.is_absolute():
            input_path = Path(os.path.abspath(input_path))
            for workspace_root in (root, recorded_pwd):
                try:
                    relative_candidate = input_path.relative_to(workspace_root)
                    break
                except ValueError:
                    continue
            if relative_candidate is None and _is_approved_latex_system_input(input_path):
                continue
        else:
            recorded_candidate = Path(os.path.abspath(recorded_pwd / input_path))
            try:
                relative_candidate = recorded_candidate.relative_to(recorded_pwd)
            except ValueError:
                relative_candidate = None
        if relative_candidate is None:
            raise RuntimeError(
                "compiled paper dependency is outside approved TeX roots: "
                f"{input_path.as_posix()}"
            )
        workspace_candidates = [relative_candidate]
        host_candidate = root / relative_candidate
        if host_candidate.exists():
            resolved_candidate = host_candidate.resolve(strict=True)
            try:
                resolved_relative = resolved_candidate.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(
                    "compiled paper dependency escapes the paper workspace: "
                    f"{relative_candidate.as_posix()}"
                ) from exc
            workspace_candidates.append(resolved_relative)
        blocked_candidate = next(
            (candidate for candidate in workspace_candidates if _uses_agent_skill_path(candidate)),
            None,
        )
        if blocked_candidate is not None:
            raise RuntimeError(
                "compiled paper must not read staged agent skill files: "
                f"{blocked_candidate.as_posix()}"
            )


def _strip_aux_comments(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = len(line[:index]) - len(line[:index].rstrip("\\"))
            if backslashes % 2 == 0:
                line = line[:index]
                break
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _reject_unparsed_commands(
    text: str,
    command_pattern: re.Pattern[str],
    parsed_matches: list[re.Match[str]],
    label: str,
) -> None:
    parsed_starts = {match.start() for match in parsed_matches}
    if any(match.start() not in parsed_starts for match in command_pattern.finditer(text)):
        raise RuntimeError(f"paper.aux uses a dynamic {label} command")


def _read_aux_graph(root: Path, initial_path: Path) -> list[str]:
    pending = [initial_path]
    visited: set[Path] = set()
    aux_texts: list[str] = []
    while pending:
        candidate = Path(os.path.abspath(pending.pop()))
        try:
            relative_candidate = candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("BibTeX aux input escapes the paper workspace") from exc
        if _uses_agent_skill_path(relative_candidate):
            raise RuntimeError("BibTeX must not read agent skill aux files")
        try:
            resolved_candidate = candidate.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"BibTeX aux input is missing or unsafe: {candidate}") from exc
        if not resolved_candidate.is_file() or resolved_candidate != candidate:
            raise RuntimeError(f"BibTeX aux input is missing or unsafe: {candidate}")
        if candidate in visited:
            continue
        visited.add(candidate)
        aux_text = _strip_aux_comments(
            candidate.read_text(encoding="utf-8", errors="replace")
        )
        input_matches = list(BIBTEX_AUX_INPUT_RE.finditer(aux_text))
        _reject_unparsed_commands(
            aux_text, BIBTEX_AUX_INPUT_COMMAND_RE, input_matches, "BibTeX aux input"
        )
        aux_texts.append(aux_text)
        pending.extend(root / match.group("target").strip() for match in input_matches)
    return aux_texts


def _validate_canonical_bibliography_target(root: Path, target_text: str) -> None:
    if "\\" in target_text or "#" in target_text:
        raise RuntimeError("paper.aux uses a dynamic bibliography target")
    target = Path(target_text)
    if not target.suffix:
        target = target.with_suffix(".bib")
    candidate = Path(os.path.abspath(root / target))
    canonical = root / "references.bib"
    if candidate != canonical:
        raise RuntimeError("compiled paper must use the canonical references.bib bibliography")
    if candidate.exists() and candidate.resolve(strict=True) != candidate:
        raise RuntimeError("compiled paper bibliography must not use symlinks")


def _validate_bibliography_style_target(root: Path, target_text: str) -> None:
    if "\\" in target_text or "#" in target_text:
        raise RuntimeError("paper.aux uses a dynamic bibliography style target")
    target = Path(target_text)
    if not target.suffix:
        target = target.with_suffix(".bst")
    candidate = Path(os.path.abspath(root / target))
    if target.parent == Path(".") and not candidate.exists():
        return
    try:
        relative_candidate = candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("compiled bibliography style escapes the paper workspace") from exc
    if _uses_agent_skill_path(relative_candidate):
        raise RuntimeError("compiled paper must not use agent skill bibliography styles")
    if not candidate.is_file():
        raise RuntimeError(f"compiled local bibliography style is missing: {candidate}")
    if candidate.resolve(strict=True) != candidate:
        raise RuntimeError("compiled bibliography style must not use symlinks")


def validate_bibtex_aux_dependencies(paper_dir: Path, aux_name: str) -> None:
    root = paper_dir.resolve()
    aux_texts = _read_aux_graph(root, root / aux_name)
    data_matches: list[re.Match[str]] = []
    style_matches: list[re.Match[str]] = []
    for aux_text in aux_texts:
        current_data_matches = list(BIBTEX_AUX_DATA_RE.finditer(aux_text))
        current_style_matches = list(BIBTEX_AUX_STYLE_RE.finditer(aux_text))
        _reject_unparsed_commands(
            aux_text, BIBTEX_AUX_DATA_COMMAND_RE, current_data_matches, "BibTeX data"
        )
        _reject_unparsed_commands(
            aux_text, BIBTEX_AUX_STYLE_COMMAND_RE, current_style_matches, "BibTeX style"
        )
        data_matches.extend(current_data_matches)
        style_matches.extend(current_style_matches)
    if len(data_matches) != 1 or len(style_matches) != 1:
        raise RuntimeError(
            "paper.aux must declare exactly one bibliography database and style"
        )
    for target in data_matches[0].group("targets").split(","):
        _validate_canonical_bibliography_target(root, target.strip())
    _validate_bibliography_style_target(root, style_matches[0].group("target").strip())


def run_with_clean_environment(command: list[str]) -> int:
    environment = {
        name: os.environ[name] for name in LATEX_ENV_ALLOWLIST if name in os.environ
    }
    environment["HOME"] = "/nonexistent"
    environment["TEXMFHOME"] = "/nonexistent"
    return subprocess.run(command, check=False, env=environment).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, option in (("recorder", "--fls"), ("bibtex", "--aux")):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--paper-dir", type=Path, required=True)
        subparser.add_argument(option, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("command_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.command == "recorder":
            validate_latex_recorder_inputs(args.paper_dir, args.fls)
        elif args.command == "bibtex":
            validate_bibtex_aux_dependencies(args.paper_dir, args.aux)
        else:
            command_args = args.command_args
            if command_args[:1] == ["--"]:
                command_args = command_args[1:]
            if not command_args:
                raise RuntimeError("missing command for clean-environment runner")
            return run_with_clean_environment(command_args)
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evidence-bundle verification, TeX/BibTeX dependency guards, and compile_latex."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable
from unittest.mock import Mock, patch

import pytest

from src.orchestrator import paper_step
from tests.paper_step_support import (
    SKILL_CLAIM,
    read_json,
    valid_tex,
    write_text,
    write_valid_artifacts,
)

MISSING_TOKEN = "@evidence[[generated/missing.json#acc]]"
SKILL_IMPORT_ERROR = "must not import agent skill files"
RECORDER_SKILL_ERROR = "must not read staged agent skill"
TEXLIVE_CLS = "/usr/share/texlive/texmf-dist/tex/latex/base/article.cls"


def _sub(root: Path, name: str) -> Path:
    """Fresh isolated workspace for one section of a merged test."""
    path = root / name
    path.mkdir()
    return path


def _write_paper(root: Path, tex: str) -> None:
    (root / "paper.tex").write_text(tex, encoding="utf-8")


def _write_files(root: Path, files: dict[str, str]) -> None:
    for rel_path, content in files.items():
        write_text(root / rel_path, content)


def _emit_bundle(paper_dir: Path) -> None:
    subprocess.check_call(
        [
            sys.executable,
            str(paper_step.SCIENTIFIC_EVIDENCE_SCRIPT),
            "emit-bundle",
            "--root-dir",
            str(paper_dir),
        ]
    )


def _which(name: str) -> str:
    return f"/usr/bin/{name}"


# --------------------------------------------------------------------------------------------
# validate_evidence_bundle: token scanning and operational-file exclusions
# --------------------------------------------------------------------------------------------

AGENT_SKILL_ROOTS = (
    ".claude/skills",
    ".codex/skills",
    ".cursor/skills",
    ".opencode/skills",
    ".gemini/skills",
    ".pi/agent/skills",
)

SILENT_CASES: list[tuple[str, dict[str, str]]] = [
    ("no_tokens", {"paper.tex": "no evidence tokens here\n"}),
    (
        "claude_support_files",
        {
            ".claude/skills/validate-paper/SKILL.md": (
                f"This doc mentions {MISSING_TOKEN} but is not the paper.\n"
            ),
            "paper.tex": "No evidence tokens in the paper.\n",
        },
    ),
    (
        "pi_support_and_root_report",
        {
            ".pi/agent/skills/validate-paper/SKILL.md": f"Operational {MISSING_TOKEN}.\n",
            "validation_report.json": f'{{"prior_error":"{MISSING_TOKEN}"}}\n',
            "paper.tex": "Clean paper.\n",
        },
    ),
    (
        "generated_prompt_files",
        {
            "paper_prompt.md": "Prompt boilerplate with placeholder @evidence[[...]].\n",
            "paper_repair_prompt_round_1.md": f"Another prompt with {MISSING_TOKEN}.\n",
            "validation_report.json": (
                '{"hard_errors": ["Fix dangling `@evidence[[...]]` links."]}\n'
            ),
            "paper.tex": "No evidence tokens in the actual paper.\n",
        },
    ),
    (
        "previous_paper_snapshot",
        {
            "previous_paper/paper.tex": (
                "Historical @evidence[[scientific-evidence/inputs/missing.csv]].\n"
            ),
            "paper.tex": "Current paper has no evidence tokens.\n",
        },
    ),
    *[
        (
            f"agent_support_{skill_root}",
            {
                f"{skill_root}/validate-paper/SKILL.md": (
                    "Example @evidence[[results/missing.json#accuracy]].\n"
                ),
                "paper.tex": "No evidence tokens in the actual paper.\n",
            },
        )
        for skill_root in AGENT_SKILL_ROOTS
    ],
]

MISSING_BUNDLE_CASES: list[tuple[str, dict[str, str], int]] = [
    (
        "paper_tokens",
        {"paper.tex": "Accuracy was 96.7\\%. % @evidence[[results/metrics.json#acc]]\n"},
        1,
    ),
    (
        "nested_operational_names",
        {
            "results/validation_report.json": f'{{"claim":"{MISSING_TOKEN}"}}\n',
            "results/.pi/data.md": f"Claim {MISSING_TOKEN}.\n",
        },
        2,
    ),
    (
        "nested_validation_report",
        {
            "results/validation_report.json": (
                '{"source": "@evidence[[results/missing.json#accuracy]]"}\n'
            ),
            "paper.tex": "No evidence tokens.\n",
        },
        1,
    ),
]


def test_validate_evidence_bundle_without_bundle_counts_only_paper_tokens(tmp_path: Path) -> None:
    # silently passes without paper tokens (operational files are excluded from the scan)
    for index, (name, files) in enumerate(SILENT_CASES):
        root = _sub(tmp_path, f"silent_{index}")
        _write_files(root, files)
        report = paper_step.validate_evidence_bundle(root)
        assert not report["checked"], name
        assert not report["bundle_present"], name
        assert report["paper_token_count"] == 0, name
        assert report["hard_errors"] == [], name
        assert report["dangling"] == [], name
    # fails when tokens exist but bundle missing
    for name, files, token_count in MISSING_BUNDLE_CASES:
        root = _sub(tmp_path, name)
        _write_files(root, files)
        report = paper_step.validate_evidence_bundle(root)
        assert not report["checked"], name
        assert not report["bundle_present"], name
        assert report["paper_token_count"] == token_count, name
        assert len(report["hard_errors"]) == 1, name
        assert "scientific-evidence/ bundle missing" in report["hard_errors"][0], name


# --------------------------------------------------------------------------------------------
# validate_evidence_bundle: agent skill import guards
# --------------------------------------------------------------------------------------------

SKILL_IMPORT_REJECT_CASES: list[tuple[str, str, dict[str, str], str]] = [
    (
        "imported_agent_skill_tex",
        f"\\input {SKILL_CLAIM}\n",
        {SKILL_CLAIM: "Unsupported @evidence[[results/missing.json#accuracy]].\n"},
        SKILL_IMPORT_ERROR,
    ),
    *[
        (f"normalized:{target}", f"\\input{{{target}}}\n", {}, SKILL_IMPORT_ERROR)
        for target in (
            ".codex/./skills/payload/claim.tex",
            ".codex//skills/payload/claim.tex",
            ".codex/x/../skills/payload/claim.tex",
        )
    ],
    (
        "macro_wrapped_import",
        f"\\newcommand{{\\loadfile}}[1]{{\\input{{#1}}}}\n\\loadfile{{{SKILL_CLAIM}}}\n",
        {},
        "dynamic import target",
    ),
    (
        "zero_argument_import_alias",
        f"\\newcommand{{\\loadfile}}{{\\input}}\n\\loadfile{{{SKILL_CLAIM}}}\n",
        {},
        "dynamic import command",
    ),
    (
        "workspace_local_package",
        "\\usepackage{loader}\n\\loadclaim\n",
        {"loader.sty": f"\\newcommand{{\\loadclaim}}{{\\input{{{SKILL_CLAIM}}}}}\n"},
        SKILL_IMPORT_ERROR,
    ),
    (
        "dynamic_local_package",
        "\\newcommand{\\paperpkg}{loader}\n\\usepackage{\\paperpkg}\n",
        {},
        "dynamic support import",
    ),
    ("percent_in_verb", f"\\verb|%| \\input{{{SKILL_CLAIM}}}\n", {}, SKILL_IMPORT_ERROR),
    (
        "false_literal_before_real_import",
        f"\\verb|\\iffalse| \\input{{{SKILL_CLAIM}}}\n",
        {},
        SKILL_IMPORT_ERROR,
    ),
    (
        "commented_verbatim_delimiters",
        f"% \\begin{{verbatim}}\n\\input{{{SKILL_CLAIM}}}\n% \\end{{verbatim}}\n",
        {},
        SKILL_IMPORT_ERROR,
    ),
    (
        "inline_literal_verbatim_delimiters",
        f"\\verb|\\begin{{verbatim}}|\n\\input{{{SKILL_CLAIM}}}\n\\verb|\\end{{verbatim}}|\n",
        {},
        SKILL_IMPORT_ERROR,
    ),
    (
        "agent_skill_input_in_else_branch",
        f"\\iffalse\nDisabled.\n\\else\n\\input{{{SKILL_CLAIM}}}\n\\fi\n",
        {},
        SKILL_IMPORT_ERROR,
    ),
]

SKILL_IMPORT_IGNORE_CASES: list[tuple[str, str, dict[str, str]]] = [
    ("import_literal_in_verb", f"\\verb|\\input{{{SKILL_CLAIM}}}|\n", {}),
    (
        "import_in_verbatim_environment",
        f"\\begin{{verbatim}}\n\\input{{{SKILL_CLAIM}}}\n\\end{{verbatim}}\n",
        {},
    ),
    (
        "commented_agent_skill_input",
        f"% \\input{{{SKILL_CLAIM}}}\n",
        {SKILL_CLAIM: "Example @evidence[[results/missing.json#accuracy]].\n"},
    ),
    ("disabled_agent_skill_input", f"\\iffalse\n\\input{{{SKILL_CLAIM}}}\n\\fi\n", {}),
    ("agent_skill_path_in_prose", "The staged path is \\verb|.codex/skills|.\n", {}),
]


def test_validate_evidence_bundle_agent_skill_import_guards(tmp_path: Path) -> None:
    # ignores inactive agent skill references
    for name, tex, extra_files in SKILL_IMPORT_IGNORE_CASES:
        root = _sub(tmp_path, name)
        _write_files(root, extra_files)
        _write_paper(root, tex)
        report = paper_step.validate_evidence_bundle(root)
        assert report["paper_token_count"] == 0, name
        assert report["hard_errors"] == [], name
    # rejects active agent skill imports
    for index, (name, tex, extra_files, match) in enumerate(SKILL_IMPORT_REJECT_CASES):
        root = _sub(tmp_path, f"reject_{index}")
        _write_files(root, extra_files)
        _write_paper(root, tex)
        with pytest.raises(RuntimeError, match=match):
            paper_step.validate_evidence_bundle(root)


def test_paper_tex_dependency_paths_guards(tmp_path: Path) -> None:
    # ignores includegraphics prefix
    root = _sub(tmp_path, "graphics_prefix")
    _write_paper(root, "\\includegraphics{graphics}\n")
    (root / "graphics.tex").write_text("Not imported.\n", encoding="utf-8")
    assert paper_step._paper_tex_dependency_paths(root) == set()
    # rejects macro import target
    root = _sub(tmp_path, "macro_target")
    _write_paper(root, f"\\def\\hiddenclaim{{{SKILL_CLAIM}}}\n\\input{{\\hiddenclaim}}\n")
    with pytest.raises(RuntimeError, match="dynamic import target"):
        paper_step._paper_tex_dependency_paths(root)
    # rejects outside-workspace imports (relative / absolute)
    for name, target in (("relative", "../external.tex"), ("absolute", "{absolute}")):
        base = _sub(tmp_path, name)
        root = _sub(base, "paper")
        external = base / "external.tex"
        external.write_text("External content.\n", encoding="utf-8")
        _write_paper(root, f"\\input{{{target.format(absolute=external)}}}\n")
        with pytest.raises(RuntimeError, match="escapes the paper workspace"):
            paper_step._paper_tex_dependency_paths(root)
    # rejects symlink escape
    base = _sub(tmp_path, "symlink")
    root = _sub(base, "paper")
    external = base / "external.tex"
    external.write_text("External content.\n", encoding="utf-8")
    (root / "section.tex").symlink_to(external)
    _write_paper(root, "\\input{section.tex}\n")
    with pytest.raises(RuntimeError, match="must not use symlinks|escapes"):
        paper_step._paper_tex_dependency_paths(root)


# --------------------------------------------------------------------------------------------
# validate_evidence_bundle: verifier integration
# --------------------------------------------------------------------------------------------


def test_validate_evidence_bundle_with_emitted_bundle(tmp_path: Path) -> None:
    # passes on anchored links
    root = _sub(tmp_path, "anchored")
    _emit_bundle(root)
    write_text(root / "results" / "metrics.json", json.dumps({"acc": 0.967}))
    _write_paper(root, "Accuracy was 96.7\\%. % @evidence[[results/metrics.json#acc]]\n")
    report = paper_step.validate_evidence_bundle(root)
    assert report["checked"]
    assert report["bundle_present"]
    assert report["summary"]["dangling_references"] == 0
    assert report["summary"]["resolved_references"] + report["summary"]["anchored_references"] >= 1
    assert report["hard_errors"] == []
    assert report["dangling"] == []
    # reports dangling refs
    root = _sub(tmp_path, "dangling")
    _emit_bundle(root)
    _write_paper(
        root,
        "Accuracy was 96.7\\%. % @evidence[[results/missing.json#acc]]\n"
        "See also @evidence[[figures/ghost.png]]\n",
    )
    report = paper_step.validate_evidence_bundle(root)
    assert report["checked"]
    assert report["summary"]["dangling_references"] == 2
    assert len(report["dangling"]) == 2
    first = report["dangling"][0]
    assert first["subject_path"] == "paper.tex"
    assert first["raw_reference"].startswith("@evidence[[")
    assert isinstance(first["messages"], list) and first["messages"]
    assert len(report["hard_errors"]) == 1
    assert "dangling" in report["hard_errors"][0]
    # uses durable verification report path
    root = _sub(tmp_path, "durable")
    _emit_bundle(root)
    _write_paper(
        root,
        "\n".join(f"Missing {idx}. @evidence[[results/missing_{idx}.json#acc]]" for idx in range(6))
        + "\n",
    )
    report = paper_step.validate_evidence_bundle(root)
    expected_report_path = root / "scientific-evidence" / "verification-report.json"
    assert report["verification_report_path"] == str(expected_report_path.resolve())
    assert expected_report_path.is_file()
    assert report["summary"]["dangling_references"] == 6
    assert str(expected_report_path) in report["hard_errors"][0]


CLEAN_WITH_DANGLING_EXIT = """
import json
import sys
from pathlib import Path

args = sys.argv[1:]
root_dir = args[args.index('--root-dir') + 1]
output_path = Path(args[args.index('--output') + 1])
output_path.write_text(json.dumps({
    'schema_name': 'scientific-evidence.verification-report',
    'verified_at': '2026-01-01T00:00:00Z',
    'root_dir': root_dir,
    'summary': {
        'total_references': 0,
        'resolved_references': 0,
        'anchored_references': 0,
        'dangling_references': 0,
        'manifests_checked': 0,
    },
    'results': [],
}), encoding='utf-8')
raise SystemExit(1)
""".strip() + "\n"


def test_validate_evidence_bundle_verifier_failures_and_payload_validation(tmp_path: Path) -> None:
    # keeps stale report on verifier failure (crash / clean report with dangling exit status)
    for name, script, expected_error in (
        ("stale_report_after_verifier_crash", "raise SystemExit(2)\n", "rc=2"),
        ("clean_report_with_dangling_exit", CLEAN_WITH_DANGLING_EXIT, "exit status disagrees"),
    ):
        base = _sub(tmp_path, name)
        root = base / "paper"
        bundle_dir = root / "scientific-evidence"
        bundle_dir.mkdir(parents=True)
        _write_paper(root, "No evidence links.\n")
        report_path = bundle_dir / "verification-report.json"
        stale_report = {"stale": True}
        report_path.write_text(json.dumps(stale_report), encoding="utf-8")
        verifier_script = base / "verifier.py"
        verifier_script.write_text(script, encoding="utf-8")
        with patch.object(paper_step, "SCIENTIFIC_EVIDENCE_SCRIPT", verifier_script):
            report = paper_step.validate_evidence_bundle(root)
        assert len(report["hard_errors"]) == 1, name
        assert expected_error in report["hard_errors"][0], name
        assert read_json(report_path) == stale_report, name
        assert list(bundle_dir.glob(".verification-report.*.json")) == [], name
    # verification payload validator rejects unusable payloads
    for name, root_dir, total_references in (
        ("inconsistent_counters", "/tmp/paper", 1),
        ("another_workspace", "/tmp/another-paper", 0),
    ):
        payload = {
            "schema_name": "scientific-evidence.verification-report",
            "verified_at": "2026-01-01T00:00:00Z",
            "root_dir": root_dir,
            "summary": {
                "total_references": total_references,
                "resolved_references": total_references,
                "anchored_references": 0,
                "dangling_references": 0,
                "manifests_checked": 0,
            },
            "results": [],
        }
        assert not paper_step._is_usable_verification_payload(
            payload, expected_root=Path("/tmp/paper")
        ), name


def test_validate_and_compile_paper_fails_on_dangling_evidence_links(tmp_path: Path) -> None:
    write_valid_artifacts(tmp_path)
    _emit_bundle(tmp_path)
    _write_paper(
        tmp_path,
        valid_tex().replace(
            "\\section{Results}\n",
            "\\section{Results}\nMetric 0.9. % @evidence[[results/ghost.json#acc]]\n",
        ),
    )
    with pytest.raises(RuntimeError) as ctx:
        paper_step.validate_and_compile_paper(tmp_path)
    assert "scientific-evidence validation failed" in str(ctx.value)
    evidence = read_json(tmp_path / "validation_report.json").get("evidence_validation", {})
    assert evidence.get("checked")
    assert evidence.get("summary", {}).get("dangling_references") == 1
    assert evidence.get("hard_errors")


# --------------------------------------------------------------------------------------------
# validate_latex_recorder_inputs
# --------------------------------------------------------------------------------------------


def _check_recorder(root: Path, fls: str) -> None:
    (root / "paper.fls").write_text(fls, encoding="utf-8")
    paper_step.validate_latex_recorder_inputs(root)


RECORDER_REJECT_CASES: list[tuple[str, str, str]] = [
    (
        "compiled_agent_skill_input",
        f"INPUT paper.tex\nINPUT {SKILL_CLAIM}\nINPUT {TEXLIVE_CLS}\n",
        RECORDER_SKILL_ERROR,
    ),
    (
        "container_workspace",
        f"PWD /workspace\nINPUT /workspace/paper.tex\nINPUT /workspace/{SKILL_CLAIM}\n",
        RECORDER_SKILL_ERROR,
    ),
    (
        "container_self_reentry_absolute",
        "PWD /workspace\nINPUT /workspace/../workspace/.codex/skills/payload/absolute.tex\n",
        RECORDER_SKILL_ERROR,
    ),
    (
        "container_self_reentry_relative",
        "PWD /workspace\nINPUT subdir/../../workspace/.codex/skills/payload/relative.tex\n",
        RECORDER_SKILL_ERROR,
    ),
    ("relative_workspace_escape", "PWD {root}\nINPUT ../outside-secret.tex\n", "outside approved TeX roots"),
    ("absolute_workspace_escape", "PWD {root}\nINPUT {tmp}/outside-secret.tex\n", "outside approved TeX roots"),
]


def test_validate_latex_recorder_inputs(tmp_path: Path) -> None:
    # allows approved inputs (system + manuscript / arch texlive)
    for name, fls in (
        (
            "system_and_manuscript_inputs",
            f"INPUT paper.tex\nINPUT sections/results.tex\nINPUT {TEXLIVE_CLS}\n",
        ),
        ("arch_texlive_inputs", "INPUT /usr/share/texmf-dist/tex/latex/base/article.cls\n"),
    ):
        _check_recorder(_sub(tmp_path, name), fls)
    # rejects unapproved inputs
    for name, fls, match in RECORDER_REJECT_CASES:
        base = _sub(tmp_path, name)
        root = _sub(base, "paper")
        with pytest.raises(RuntimeError, match=match):
            _check_recorder(root, fls.format(root=root, tmp=base))
    # rejects system-root symlink escape
    base = _sub(tmp_path, "system_root_symlink")
    root = _sub(base, "paper")
    system_root = base / "texmf"
    system_root.mkdir()
    external_path = base / "outside-secret.tex"
    external_path.write_text("secret", encoding="utf-8")
    (system_root / "article.cls").symlink_to(external_path)
    with (
        patch.object(paper_step, "LATEX_SYSTEM_INPUT_ROOTS", (system_root,)),
        pytest.raises(RuntimeError, match="outside approved TeX roots"),
    ):
        _check_recorder(root, f"PWD {root}\nINPUT {system_root / 'article.cls'}\n")
    # resolves symlink alias to the staged skill
    root = _sub(tmp_path, "symlink_alias")
    skill_file = root / SKILL_CLAIM
    write_text(skill_file, "Claim.\n")
    (root / "alias").symlink_to(skill_file.parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match=RECORDER_SKILL_ERROR):
        _check_recorder(root, f"PWD {root}\nINPUT alias/claim.tex\n")


VALIDATE_COMPILE_INPUTS_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/agents/skills/validate-paper/scripts/validate_compile_inputs.py"
)


def _load_validate_compile_inputs() -> Any:
    spec = importlib.util.spec_from_file_location(
        "validate_compile_inputs_under_test", VALIDATE_COMPILE_INPUTS_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("implementation", ["paper_step", "skill_script"])
def test_latex_system_input_symlink_between_approved_roots(
    tmp_path: Path, implementation: str
) -> None:
    # Debian/Ubuntu layout: /usr/share/texmf/web2c/texmf.cnf links into texlive.
    module = paper_step if implementation == "paper_step" else _load_validate_compile_inputs()
    texlive = tmp_path / "texlive"
    texmf = tmp_path / "texmf"
    (texlive / "web2c").mkdir(parents=True)
    (texmf / "web2c").mkdir(parents=True)
    real_cnf = texlive / "web2c" / "texmf.cnf"
    real_cnf.write_text("% cnf\n", encoding="utf-8")
    alias_cnf = texmf / "web2c" / "texmf.cnf"
    alias_cnf.symlink_to(real_cnf)
    outside = tmp_path / "outside.tex"
    outside.write_text("secret", encoding="utf-8")
    escape = texmf / "escape.tex"
    escape.symlink_to(outside)
    with patch.object(module, "LATEX_SYSTEM_INPUT_ROOTS", (texlive, texmf)):
        assert module._is_approved_latex_system_input(alias_cnf)
        assert module._is_approved_latex_system_input(real_cnf)
        assert not module._is_approved_latex_system_input(escape)
        assert not module._is_approved_latex_system_input(outside)
    if implementation == "paper_step":
        root = _sub(tmp_path, "paper")
        with patch.object(paper_step, "LATEX_SYSTEM_INPUT_ROOTS", (texlive, texmf)):
            _check_recorder(root, f"PWD {root}\nINPUT paper.tex\nINPUT {alias_cnf}\n")
            with pytest.raises(RuntimeError, match="outside approved TeX roots"):
                _check_recorder(root, f"PWD {root}\nINPUT {escape}\n")


# --------------------------------------------------------------------------------------------
# validate_bibtex_dependencies / validate_bibtex_aux_dependencies
# --------------------------------------------------------------------------------------------

BIBTEX_CASES: list[tuple[str, str, bool, str | None]] = [
    (
        "rejects_alternate_bibliography",
        "\\bibliographystyle{plain}\n\\bibliography{.codex/skills/payload/references}\n",
        False,
        "canonical references.bib",
    ),
    (
        "rejects_agent_skill_style",
        "\\bibliographystyle{.codex/skills/payload/evil}\n\\bibliography{references}\n",
        False,
        "agent skill bibliography styles",
    ),
    (
        "rejects_alternate_biblatex_resource",
        "\\addbibresource{.codex/skills/payload/references.bib}\n",
        False,
        "canonical references.bib",
    ),
    ("allows_system_style", "\\bibliographystyle{plainnat}\n\\bibliography{references}\n", False, None),
    ("allows_local_style", "\\bibliographystyle{paper}\n\\bibliography{references.bib}\n", True, None),
]


def test_validate_bibtex_dependencies(tmp_path: Path) -> None:
    for name, tex, local_style, match in BIBTEX_CASES:
        root = _sub(tmp_path, name)
        if local_style:
            (root / "paper.bst").write_text("ENTRY {} {} {}", encoding="utf-8")
        if match is None:
            paper_step.validate_bibtex_dependencies(root, tex)
        else:
            with pytest.raises(RuntimeError, match=match):
                paper_step.validate_bibtex_dependencies(root, tex)
    # rejects style symlink escape
    base = _sub(tmp_path, "style_symlink")
    root = _sub(base, "paper")
    external_style = base / "external.bst"
    external_style.write_text("ENTRY {} {} {}", encoding="utf-8")
    (root / "paper.bst").symlink_to(external_style)
    with pytest.raises(RuntimeError, match="must not use symlinks"):
        paper_step.validate_bibtex_dependencies(
            root, "\\bibliographystyle{paper}\n\\bibliography{references}\n"
        )


BIBTEX_AUX_CASES: list[tuple[str, dict[str, str], str | None]] = [
    (
        "rejects_alternate_database",
        {"paper.aux": "\\bibstyle{plain}\n\\bibdata{.codex/skills/payload/references}\n"},
        "canonical references.bib",
    ),
    (
        "rejects_agent_skill_style",
        {"paper.aux": "\\bibstyle{.codex/skills/payload/evil}\n\\bibdata{references}\n"},
        "agent skill bibliography styles",
    ),
    ("allows_canonical_inputs", {"paper.aux": "\\bibstyle{plainnat}\n\\bibdata{references}\n"}, None),
    (
        "ignores_comments_and_checks_inputs",
        {
            "paper.aux": "%\\bibstyle{plain}\n%\\bibdata{references}\n\\@input{sections.aux}\n",
            "sections.aux": "\\bibstyle{plain}\n\\bibdata{.codex/skills/payload/references}\n",
        },
        "canonical references.bib",
    ),
    (
        "allows_canonical_child_input",
        {
            "paper.aux": "\\@input{sections.aux}\n",
            "sections.aux": "\\bibstyle{plain}\n\\bibdata{references}\n",
        },
        None,
    ),
    (
        "rejects_child_escape",
        {
            "paper.aux": "\\@input{../outside.aux}\n",
            "../outside.aux": "\\bibstyle{plain}\n\\bibdata{references}\n",
        },
        "escapes the paper workspace",
    ),
]


def test_validate_bibtex_aux_dependencies(tmp_path: Path) -> None:
    for name, files, match in BIBTEX_AUX_CASES:
        base = _sub(tmp_path, name)
        root = _sub(base, "paper")
        _write_files(root, files)
        if match is None:
            paper_step.validate_bibtex_aux_dependencies(root)
        else:
            with pytest.raises(RuntimeError, match=match):
                paper_step.validate_bibtex_aux_dependencies(root)
    # rejects symlinked parent
    base = _sub(tmp_path, "symlinked_parent")
    root = _sub(base, "paper")
    outside = _sub(base, "outside")
    (root / "alias").symlink_to(outside, target_is_directory=True)
    (root / "paper.aux").write_text("\\@input{alias/child.aux}\n", encoding="utf-8")
    (outside / "child.aux").write_text("\\bibstyle{plain}\n\\bibdata{references}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must not use symlinks"):
        paper_step.validate_bibtex_aux_dependencies(root)


# --------------------------------------------------------------------------------------------
# compile_latex
# --------------------------------------------------------------------------------------------


def _recorder_run(
    root: Path,
    *,
    recorder_input: str | Callable[[int], str] = "paper.tex",
    aux: str | None = "\\bibstyle{plain}\n\\bibdata{references}\n",
    returncode: int = 0,
    on_run: Callable[[list[str], dict[str, Any]], None] | None = None,
) -> Callable[..., Mock]:
    """Fake subprocess.run that writes recorder/aux outputs on each pdflatex -recorder pass."""
    passes = 0

    def _fake_run(command: list[str], **kwargs: Any) -> Mock:
        nonlocal passes
        if "-recorder" in command:
            passes += 1
            current = recorder_input(passes) if callable(recorder_input) else recorder_input
            (root / "paper.fls").write_text(f"PWD {root}\nINPUT {current}\n", encoding="utf-8")
            if aux is not None:
                (root / "paper.aux").write_text(aux, encoding="utf-8")
        if on_run is not None:
            on_run(command, kwargs)
        if returncode:
            return Mock(returncode=returncode, stdout="", stderr="compile failed")
        return Mock(returncode=0, stdout="ok", stderr="")

    return _fake_run


UNTRUSTED_TEX_ENV = {
    "TEXMFHOME": "/tmp/untrusted-texmf",
    "BIBINPUTS": "/tmp/untrusted-bib",
    "BIBINPUTS_bibtex": "/tmp/untrusted-qualified-bib",
    "BSTINPUTS": "/tmp/untrusted-bst",
    "BSTINPUTS.bibtex": "/tmp/untrusted-qualified-bst",
    "TEXINPUTS": "/tmp/untrusted-tex-inputs",
    "TEXINPUTS.pdflatex": "/tmp/untrusted-qualified-inputs",
    "TEXFORMATS": "/tmp/untrusted-formats",
    "TEXMFLOCAL": "/tmp/untrusted-local-tree",
    "KPSE_DOT": "/tmp/untrusted-dot",
    "shell_escape": "t",
    "openout_any": "a",
    "UNRELATED_SECRET": "must-not-reach-tex",
    "TEXMF": "/tmp/untrusted-texmf-root",
    "TEXMFCNF": "/tmp/untrusted-texmf-config",
    "TEXMFHOME.bibtex": "/tmp/untrusted-qualified-home",
    "TEXMFCONFIG_bibtex": "/tmp/untrusted-qualified-config",
}


def test_compile_latex_native_passes_validate_recorder_aux_and_environment(tmp_path: Path) -> None:
    # sanitizes native tex search environment
    root = _sub(tmp_path, "environment")
    bibtex_environment: dict[str, str] | None = None

    def capture_bibtex_env(command: list[str], kwargs: dict[str, Any]) -> None:
        nonlocal bibtex_environment
        if Path(command[0]).name == "bibtex" and isinstance(kwargs.get("env"), dict):
            bibtex_environment = kwargs["env"]

    with (
        patch.dict(os.environ, UNTRUSTED_TEX_ENV),
        patch.object(paper_step.shutil, "which", side_effect=_which),
        patch.object(
            paper_step.subprocess, "run", side_effect=_recorder_run(root, on_run=capture_bibtex_env)
        ),
    ):
        paper_step.compile_latex(paper_dir=root)
    assert bibtex_environment is not None
    assert bibtex_environment["TEXMFHOME"] == "/nonexistent"
    assert bibtex_environment["HOME"] == "/nonexistent"
    for name in UNTRUSTED_TEX_ENV:
        if name != "TEXMFHOME":
            assert name not in bibtex_environment, name
    # validates aux before bibtex
    root = _sub(tmp_path, "aux")
    fake_run = _recorder_run(root, aux="\\bibstyle{plain}\n\\bibdata{.codex/skills/payload/references}\n")
    with (
        patch.object(paper_step.shutil, "which", side_effect=_which),
        patch.object(paper_step.subprocess, "run", side_effect=fake_run) as run_mock,
        pytest.raises(RuntimeError, match="canonical references.bib"),
    ):
        paper_step.compile_latex(paper_dir=root)
    assert run_mock.call_count == 1
    # validates each recorder pass
    root = _sub(tmp_path, "each_pass")
    fake_run = _recorder_run(root, recorder_input=lambda n: SKILL_CLAIM if n == 2 else "paper.tex")
    with (
        patch.object(paper_step.shutil, "which", side_effect=_which),
        patch.object(paper_step.subprocess, "run", side_effect=fake_run) as run_mock,
        pytest.raises(RuntimeError, match=RECORDER_SKILL_ERROR),
    ):
        paper_step.compile_latex(paper_dir=root)
    assert run_mock.call_count == 3
    # validates recorder output from a failed pass
    root = _sub(tmp_path, "failed_pass")
    fake_run = _recorder_run(root, recorder_input=SKILL_CLAIM, aux=None, returncode=1)
    with (
        patch.object(paper_step.shutil, "which", side_effect=_which),
        patch.object(paper_step.subprocess, "run", side_effect=fake_run),
        pytest.raises(RuntimeError, match=RECORDER_SKILL_ERROR),
    ):
        paper_step.compile_latex(paper_dir=root)


def test_compile_latex_container_execution_and_infrastructure_errors(tmp_path: Path) -> None:
    # runs in container (prefers container when image provided / falls back when local tex missing)
    for name, local_tex in (
        ("prefers_container_when_image_provided", "/usr/bin/{name}"),
        ("falls_back_to_container_when_local_tex_missing", None),
    ):
        root = _sub(tmp_path, name)
        (root / "paper.tex").write_text(valid_tex(), encoding="utf-8")
        which_map = {
            "pdflatex": local_tex and local_tex.format(name="pdflatex"),
            "bibtex": local_tex and local_tex.format(name="bibtex"),
            "docker": "/usr/bin/docker",
        }
        with (
            patch.object(paper_step.shutil, "which", side_effect=which_map.get),
            patch.object(paper_step.subprocess, "run") as run_mock,
            patch.object(paper_step, "validate_latex_recorder_inputs") as recorder_validation,
            patch.object(paper_step, "validate_bibtex_aux_dependencies"),
        ):
            run_mock.return_value = Mock(returncode=0, stdout="ok", stderr="")
            logs = paper_step.compile_latex(
                paper_dir=root,
                tex_name="paper.tex",
                container_image="test-image:latest",
                container_network="host",
            )
        assert len(logs) == 4, name
        assert run_mock.call_count == 4, name
        assert recorder_validation.call_count == 3, name
        expected_user_flags = (
            ["--user", f"{os.getuid()}:{os.getgid()}"]
            if hasattr(os, "getuid") and hasattr(os, "getgid")
            else []
        )
        assert run_mock.call_args_list[0].args[0] == [
            "/usr/bin/docker",
            "run",
            "--rm",
            "--network",
            "host",
            *expected_user_flags,
            "-v",
            f"{root.resolve()}:/workspace",
            "-w",
            "/workspace",
            "test-image:latest",
            *paper_step.CONTAINER_CLEAN_ENV_COMMAND,
            "pdflatex",
            "-recorder",
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "paper.tex",
        ], name
    # raises clear error when tex missing and no container
    root = _sub(tmp_path, "no_tex")
    with (
        patch.object(paper_step.shutil, "which", return_value=None),
        pytest.raises(RuntimeError, match="Missing required command\\(s\\): pdflatex, bibtex"),
    ):
        paper_step.compile_latex(paper_dir=root, tex_name="paper.tex")
    # classifies missing docker as infrastructure
    root = _sub(tmp_path, "no_docker")
    with (
        patch.object(paper_step.shutil, "which", return_value=None),
        pytest.raises(paper_step.PaperValidationInfrastructureError),
    ):
        paper_step.compile_latex(paper_dir=root, container_image="image@sha256:digest")
    # classifies docker run failure as infrastructure
    root = _sub(tmp_path, "docker_failure")
    with (
        patch.object(paper_step.shutil, "which", return_value="/usr/bin/docker"),
        patch.object(
            paper_step.subprocess,
            "run",
            return_value=Mock(returncode=125, stdout="", stderr="daemon failure"),
        ),
        pytest.raises(paper_step.PaperValidationInfrastructureError),
    ):
        paper_step.compile_latex(paper_dir=root, container_image="image@sha256:digest")

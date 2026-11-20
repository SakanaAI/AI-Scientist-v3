"""Venue catalog and briefs, prompt workspace context, prompt families, and grounding policy."""

from __future__ import annotations

import copy
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.investigation.files import load_instructions
from src.investigation.paper_prompt_context import (
    author_block_context_text,
    contributions_context_text,
    paper_workspace_context_replacements,
    stage_optional_rendered_doc,
    venue_context_text,
)
from src.investigation.paper_prompt_family import (
    infer_prompt_family_from_path,
    resolve_final_paper_stage,
)
from src.orchestrator import paper_step, paper_venues
from src.orchestrator.paper_step import VENUE_REAL_AUTHOR_MARKERS, _infer_venue_profile
from src.orchestrator.paper_venues import (
    KNOWN_CHECKS,
    REPO_ROOT,
    VenueSpecError,
    load_venue,
    page_budget_sentence,
    parse_venue_spec,
    venue_catalog,
    venue_names,
)
from src.pipeline.cli import load_pipeline_config
from src.pipeline.config import PAPER_FORMATS, validate_pipeline_config, venue_file_for_format
from src.pipeline.paths import derive_paths
from src.pipeline.phases import build_paper_cmd

TEMPLATES_DIR = REPO_ROOT / "src/agents/skills/write-paper"
PAPER_SCRIPTS = ("paper.sh", "paper_edit.sh")
ALIFE_TERMS = (
    "artificial life", "life-like", "lifelike", "living and lifelike complex adaptive systems",
)
STAGE_VENUE_SCRIPTS = (
    ("paper.sh", "stage_contributions_file"),
    ("paper_edit.sh", "stage_author_block_file"),
)


def _raw(name: str) -> dict:
    return yaml.safe_load(paper_venues.venue_spec_path(name).read_text(encoding="utf-8"))


def _brief(paper_format: str) -> str:
    return (REPO_ROOT / "src/templates/paper_venues" / f"{paper_format}.md").read_text(
        encoding="utf-8"
    )


def _script(name: str) -> str:
    return (REPO_ROOT / "src/scripts" / name).read_text(encoding="utf-8")


# ── venue catalog ──


def test_venue_catalog_declares_formats_assets_page_limits_and_rejects_unknown() -> None:
    # declared venues are the paper formats
    assert venue_names() == ("alife", "iclr")
    assert PAPER_FORMATS == venue_names()

    for name in venue_names():
        venue = venue_catalog()[name]
        # each venue template detects as its own format
        tex = (TEMPLATES_DIR / venue.template_dir / "paper.tex").read_text(encoding="utf-8")
        assert paper_step.detect_paper_format(tex) == name, name
        # venue assets resolve
        assert venue.brief_path.is_file(), name
        assert venue_file_for_format(name) == venue.brief, name
        assert venue_file_for_format(name) == f"src/templates/paper_venues/{name}.md", name
        assert (REPO_ROOT / venue_file_for_format(name)).is_file(), name
        if venue.author_block_path is not None:
            assert venue.author_block_path.is_file(), name
        assert venue.checks <= KNOWN_CHECKS, name

    # page limits mirror the catalog
    assert paper_step.PAGE_LIMITS == paper_venues.page_limits()
    assert set(paper_step.PAGE_LIMITS) == {
        name for name, venue in venue_catalog().items() if venue.page_budget is not None
    }

    # unknown format has no brief and no detection
    assert venue_file_for_format("not-a-venue") == ""
    assert paper_step.detect_paper_format(r"\documentclass{article}") is None
    with pytest.raises(VenueSpecError, match="does not exist"):
        load_venue("neurips")
    with pytest.raises(VenueSpecError, match="must match"):
        load_venue("../etc")


MALFORMED_SPEC_CASES = [
    ("unknown-key", lambda raw: raw.update(colour="blue"), "unknown keys"),
    ("missing-brief", lambda raw: raw.pop("brief"), "missing keys"),
    ("name-mismatch", lambda raw: raw.update(name="other"), "must match the file stem"),
    ("absolute-brief", lambda raw: raw.update(brief="/abs/brief.md"), "repo-relative"),
    ("missing-brief-file", lambda raw: raw.update(brief="src/templates/paper_venues/missing.md"), "is not a file"),
    ("invalid-marker-regex", lambda raw: raw.update(latex_markers=["("]), "latex_markers"),
    ("empty-markers", lambda raw: raw.update(latex_markers=[]), "non-empty list"),
    ("unknown-check", lambda raw: raw.update(checks=["sparkle"]), "unknown checks"),
    ("min-exceeds-max", lambda raw: raw["page_budget"].update(min_body_pages=9), "must not exceed"),
    ("fill-ratio-out-of-range", lambda raw: raw["page_budget"].update(min_last_body_page_fill_ratio=1.5), "must lie in"),
    ("missing-max-pages", lambda raw: raw["page_budget"].pop("max_body_pages"), "max_body_pages is required"),
    ("unknown-budget-key", lambda raw: raw["page_budget"].update(columns=2), "unknown keys"),
]


def test_malformed_specs_are_rejected() -> None:
    for name, mutate, message in MALFORMED_SPEC_CASES:
        raw = copy.deepcopy(_raw("alife"))
        mutate(raw)
        with pytest.raises(VenueSpecError, match=message):
            parse_venue_spec("alife", raw)
            pytest.fail(name)


def test_page_budget_sentence_and_venue_has_check() -> None:
    # page budget sentence states what the validator enforces
    assert page_budget_sentence("alife") == (
        "exactly 8 pages, excluding references. A body of 8 pages counts only when its final"
        " page is at least 75% full; a shorter or sparser body fails validation."
    )
    assert page_budget_sentence("iclr").startswith("8 to 9 pages, excluding references.")
    assert page_budget_sentence(None).startswith("no enforced page budget")

    spec = _raw("alife")
    spec["page_budget"] = {"max_body_pages": 9}
    venue = parse_venue_spec("alife", spec)
    assert venue.page_budget is not None
    assert venue.page_budget.as_limits() == {
        "max_body_pages": 9, "body_columns": 1, "exclude_references": False,
    }

    # venue_has_check rejects unknown check names
    assert paper_venues.venue_has_check("alife", "alife_framing")
    assert not paper_venues.venue_has_check("iclr", "alife_framing")
    assert not paper_venues.venue_has_check(None, "alife_framing")
    with pytest.raises(ValueError, match="unknown venue check"):
        paper_venues.venue_has_check("alife", "sparkle")


def test_a_new_venue_file_is_picked_up_without_code_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding configs/venue/<name>.yaml is the whole registration step."""
    venue_dir = tmp_path / "venue"
    venue_dir.mkdir()
    spec = _raw("alife")
    spec.update(
        name="neurips", author_block=None, template_dir="templates/neurips",
        latex_markers=[r"\\usepackage\{neurips_2026\}"],
        page_budget={"max_body_pages": 9, "exclude_references": True}, checks=[],
    )
    (venue_dir / "neurips.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    monkeypatch.setattr(paper_venues, "VENUE_DIR", venue_dir)
    venue_catalog.cache_clear()
    try:
        assert venue_names() == ("neurips",)
        assert paper_step.detect_paper_format(r"\usepackage{neurips_2026}") == "neurips"
        assert paper_venues.page_limits() == {
            "neurips": {"max_body_pages": 9, "body_columns": 1, "exclude_references": True}
        }
        assert not paper_venues.venue_has_check("neurips", "alife_framing")
    finally:
        venue_catalog.cache_clear()


# ── venue briefs and their staging ──


def _venue_args(paper_format: str, **overrides: str) -> list[str]:
    cfg = load_pipeline_config(
        ["task.dir=examples/circpack", f"paper.paper_format={paper_format}", *overrides.values()]
    )
    return build_paper_cmd(cfg, derive_paths(cfg))


def _run_stage_venue_file(
    tmp_path: Path, script: str, next_function: str, venue_source: Path | None
) -> Path:
    text = _script(script)
    start = text.index("stage_venue_file() {")
    function = text[start:text.index(f"{next_function}() {{", start)]
    output_dir = tmp_path / f"output-{script}"
    (output_dir / "task").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    (output_dir / "task" / "task.md").write_text("# Goal\n- test\n", encoding="utf-8")
    source_arg = shlex.quote(str(venue_source)) if venue_source is not None else "''"
    harness = tmp_path / f"stage-{script}"
    harness.write_text(
        "\n".join((
            "set -euo pipefail",
            f"PYTHON_BIN={shlex.quote(sys.executable)}",
            f"REPO_ROOT={shlex.quote(str(REPO_ROOT))}",
            f"OUTPUT_DIR={shlex.quote(str(output_dir))}",
            function,
            f"stage_venue_file {source_arg} iclr 8 15",
        )),
        encoding="utf-8",
    )
    subprocess.run(["bash", str(harness)], cwd=REPO_ROOT, check=True)
    return output_dir / "VENUE.md"


def test_paper_cmd_resolves_venue_file_from_format_override_or_opt_out() -> None:
    # paper cmd forwards the brief matching the format
    for paper_format in PAPER_FORMATS:
        cmd = _venue_args(paper_format)
        assert cmd[cmd.index("--venue-file") + 1] == f"src/templates/paper_venues/{paper_format}.md", paper_format

    # explicit venue file overrides the format default
    cmd = _venue_args("iclr", venue="paper.venue_file=README.md")
    assert cmd[cmd.index("--venue-file") + 1] == "README.md"

    # empty venue file opts out
    cmd = _venue_args("iclr", venue='paper.venue_file=""')
    assert cmd[cmd.index("--venue-file") + 1] == ""
    assert cmd.count("--venue-file") == 1

    # default config validates without a venue path
    cfg = load_pipeline_config(["task.dir=examples/circpack"])
    assert cfg.paper.venue_file is None
    assert validate_pipeline_config(cfg) == []


def test_venue_briefs_author_policy_llm_disclosure_and_page_budget_placeholder() -> None:
    # ICLR brief does not trip the real author requirement:
    # VENUE_REAL_AUTHOR_MARKERS contains "include author names"; ICLR anonymity must not match.
    brief = _brief("iclr").lower()
    assert [marker for marker in VENUE_REAL_AUTHOR_MARKERS if marker in brief] == []
    assert [term for term in ALIFE_TERMS if term in brief] == []
    profile = _infer_venue_profile(paper_format="iclr", venue_brief_text=_brief("iclr"))
    assert profile["require_real_authors"] is False
    assert profile["framing_terms"] == ()

    # ICLR brief requires LLM usage disclosure
    assert "llm usage" in brief
    assert "precise role" in brief

    # ALIFE brief still requires real authors
    profile = _infer_venue_profile(paper_format="alife", venue_brief_text=_brief("alife"))
    assert profile["require_real_authors"] is True
    assert profile["label"] == "ALIFE"

    # enforced page budgets reach the brief only through the placeholder:
    # an enforced budget must be rendered from the venue YAML rather than restated by hand.
    for paper_format in PAPER_FORMATS:
        venue = venue_catalog()[paper_format]
        budget_line = next(line for line in _brief(paper_format).splitlines() if "Body-page budget:" in line)
        if venue.page_budget is None:
            assert "{{PAPER_PAGE_BUDGET}}" not in budget_line, paper_format
        else:
            assert budget_line == "- Body-page budget: {{PAPER_PAGE_BUDGET}}", paper_format

    # anonymous formats have no author block: one for a double-blind venue would contradict its brief
    assert not (REPO_ROOT / "src/templates/paper_authors/iclr.tex").exists()


def test_paper_scripts_resolve_the_brief_from_the_format_and_drop_root_venue_file() -> None:
    # paper scripts resolve the brief from the format
    for script in PAPER_SCRIPTS:
        text = _script(script)
        assert "from src.orchestrator.paper_venues import VenueSpecError, load_venue, venue_names" in text, script
        assert 'VENUE_SOURCE_PATH="$REPO_ROOT/$VENUE_CATALOG_BRIEF"' in text, script
        assert "src/templates/paper_venues/" not in text, script
        assert "iclr|alife" not in text, script
        assert "--no-venue-file" not in text, script
        assert 'VENUE_SOURCE_PATH="$2"\n      VENUE_SOURCE_EXPLICIT=1' in text, script

    # paper resume removes disabled venue before early finalization
    text = _script("paper.sh")
    cleanup = 'rm -f -- "$OUTPUT_DIR/VENUE.md"'
    resume_finalization = 'if [ "$RESUME_EXISTING" -eq 1 ] && [ -z "$POSTHOC_DIR" ]'
    assert text.index(cleanup) < text.index(resume_finalization)

    # no caller pins the old root venue file
    stale = [
        path.name for path in sorted((REPO_ROOT / "scripts").glob("*.sh"))
        if "VENUE.md" in path.read_text(encoding="utf-8")
    ]
    assert stale == []
    assert not (REPO_ROOT / "VENUE.md").exists()


@pytest.mark.parametrize(("script", "next_function"), STAGE_VENUE_SCRIPTS)
def test_paper_scripts_stage_supplied_venue_and_remove_inherited_when_disabled(
    tmp_path: Path, script: str, next_function: str
) -> None:
    # stage supplied venue, rendering the format and the enforced page budget
    venue_source = tmp_path / "venue.md"
    venue_source.write_text("Format={{PAPER_FORMAT}}\nBudget={{PAPER_PAGE_BUDGET}}", encoding="utf-8")
    staged_venue = _run_stage_venue_file(tmp_path, script, next_function, venue_source)
    rendered = staged_venue.read_text(encoding="utf-8")
    assert rendered == f"Format=iclr\nBudget={page_budget_sentence('iclr')}"
    assert "8 to 9 pages" in rendered
    assert "75% full" in rendered

    # remove inherited venue when disabled (the staged file above now plays the stale role)
    staged_venue.write_text("stale venue guidance", encoding="utf-8")
    staged_venue = _run_stage_venue_file(tmp_path, script, next_function, None)
    assert not staged_venue.exists()


# ── prompt workspace context ──

CONTEXT_TEXT_CASES = [
    ("venue", venue_context_text, "./VENUE.md", "Read", "No venue-specific brief is provided"),
    ("author_block", author_block_context_text, "./paper_authors.tex", "hard contract",
     "No canonical author-block file is provided"),
    ("contributions", contributions_context_text, "./contributions.md", "base for contribution framing",
     "No contributions scaffold is provided"),
]


def test_context_text_references_workspace_file_or_falls_back() -> None:
    for name, render, path, present_marker, absent_marker in CONTEXT_TEXT_CASES:
        rendered = render(path)
        assert path in rendered, name
        assert present_marker in rendered, name
        fallback = render(None)
        assert absent_marker in fallback, name
        assert path not in fallback, name


def test_stage_optional_rendered_doc_renders_template_or_removes_inherited_file(tmp_path: Path) -> None:
    # renders template into workspace
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.md").write_text(
        "# Goal\n- optimize objective\n\n# Rubrics\n- pass threshold\n", encoding="utf-8"
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    venue_template = tmp_path / "venue.md"
    venue_template.write_text("Goal={{TASK_GOAL}}\nFormat={{PAPER_FORMAT}}\n", encoding="utf-8")

    staged = stage_optional_rendered_doc(
        source_path=venue_template, workspace_dir=workspace_dir, workspace_name="VENUE.md",
        task_dir=task_dir, results_dir=results_dir, extra_replacements={"PAPER_FORMAT": "alife"},
    )

    assert staged == workspace_dir / "VENUE.md"
    assert staged.is_file()
    assert "Goal=- optimize objective" in staged.read_text(encoding="utf-8")
    assert "Format=alife" in staged.read_text(encoding="utf-8")

    # removes inherited file when the source is absent
    inherited_dir = tmp_path / "inherited"
    inherited_dir.mkdir()
    inherited_venue = inherited_dir / "VENUE.md"
    inherited_venue.write_text("stale venue guidance", encoding="utf-8")
    staged = stage_optional_rendered_doc(
        source_path=None, workspace_dir=inherited_dir, workspace_name="VENUE.md",
        task_dir=tmp_path / "missing-task", results_dir=tmp_path / "missing-results",
    )
    assert staged is None
    assert not inherited_venue.exists()


def test_paper_workspace_context_replacements_cover_optional_inputs_gaps_and_fallbacks(
    tmp_path: Path,
) -> None:
    # include optional workspace inputs
    full = tmp_path / "full"
    full.mkdir()
    (full / "results_readme.md").write_text("Campaign overview", encoding="utf-8")
    aggregated_target = full / "real_aggregated"
    aggregated_target.mkdir()
    (full / "aggregated").symlink_to(aggregated_target, target_is_directory=True)
    (full / "posthoc_analyses_context.json").write_text(
        '[{"analysis_id":"alpha","title":"Alpha","family":"fam","summary_text":"Summary",'
        '"rubric":{"overall":"high"}}]',
        encoding="utf-8",
    )
    (full / "contributions.md").write_text("Contribution base", encoding="utf-8")

    replacements = paper_workspace_context_replacements(full)

    assert replacements["RESULTS_README"] == "Campaign overview"
    assert replacements["PAPER_AGGREGATED_DIR"] == "./aggregated"
    assert replacements["POSTHOC_RANKING_PATH"] == "./posthoc_ranking.json"
    assert replacements["DRAFT_REFERENCE_GAPS_PATH"] == "./draft_reference_gaps.md"
    assert replacements["POSTDRAFT_REFERENCE_QUERY_PLAN_JSON_PATH"] == "./postdraft_reference_query_plan.json"
    assert replacements["CONTRIBUTIONS_FILE_PATH"] == "./contributions.md"
    assert "./contributions.md" in replacements["CONTRIBUTIONS_CONTEXT"]
    assert "## Alpha" in replacements["POSTHOC_ANALYSES_CONTEXT"]
    assert "Interestingness Rubric" in replacements["POSTHOC_ANALYSES_CONTEXT"]

    # render draft reference gap context
    gaps = tmp_path / "gaps"
    gaps.mkdir()
    (gaps / "draft_reference_gaps.json").write_text(
        """
        {
          "status": "ready",
          "summary": "Two sections need stronger comparator support.",
          "missing_literatures": ["feedback alignment comparisons"],
          "rewrite_targets": [
            {
              "target_id": "related-work-comparators",
              "section": "related work",
              "priority": "high",
              "citation_intent": "comparator",
              "reason": "Comparator coverage is thin.",
              "anchor_text": "Related work paragraph about local learning rules.",
              "search_topics": ["direct feedback alignment biologically plausible learning"]
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    context = paper_workspace_context_replacements(gaps)["DRAFT_REFERENCE_GAPS_CONTEXT"]
    assert "Two sections need stronger comparator support" in context
    assert "related-work-comparators" in context
    assert "direct feedback alignment biologically plausible learning" in context

    # tolerate invalid draft reference gap JSON
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "draft_reference_gaps.json").write_text("{invalid", encoding="utf-8")
    replacements = paper_workspace_context_replacements(invalid)
    assert replacements["DRAFT_REFERENCE_GAPS_CONTEXT"] == (
        "Draft reference gap artifact is unreadable; ignore it unless the file is repaired."
    )

    # fall back when optional inputs are missing
    empty = tmp_path / "empty"
    empty.mkdir()
    replacements = paper_workspace_context_replacements(empty)
    assert "No campaign README available" in replacements["RESULTS_README"]
    assert replacements["PAPER_AGGREGATED_DIR"] == ""
    assert replacements["CONTRIBUTIONS_FILE_PATH"] == "./contributions.md"
    assert "No contributions scaffold is provided" in replacements["CONTRIBUTIONS_CONTEXT"]
    assert replacements["POSTHOC_ANALYSES_CONTEXT"] == "No posthoc analyses are available for this run."
    assert replacements["DRAFT_REFERENCE_GAPS_CONTEXT"] == (
        "No post-draft reference gaps have been identified for this run."
    )


# ── prompt family resolution ──

FAKE_REPO = Path("/tmp/repo")


def _skills(writer: str) -> tuple[str, ...]:
    return ("retrieve-references", writer, "validate-paper", "memory-log")


def test_prompt_family_inference_and_final_stage_resolution(tmp_path: Path) -> None:
    # infer prompt family from a known prompt
    family = infer_prompt_family_from_path(
        FAKE_REPO / "src" / "prompts" / "paper_standalone.md", repo_root=FAKE_REPO
    )
    assert family == "paper_standalone"

    # keeps the current prompt for the same family
    current_prompt_path = FAKE_REPO / "custom" / "paper_override.md"
    resolved = resolve_final_paper_stage(
        repo_root=FAKE_REPO, current_prompt_path=current_prompt_path,
        current_selected_skills=_skills("write-paper-standalone"),
        selected_prompt_family="paper_standalone",
    )
    assert resolved.prompt_path.resolve() == current_prompt_path.resolve()
    assert resolved.prompt_family == "paper_standalone"
    assert resolved.writer_skill == "write-paper-standalone"
    assert resolved.selected_skills == _skills("write-paper-standalone")

    # switches prompt and writer for a new family
    resolved = resolve_final_paper_stage(
        repo_root=FAKE_REPO, current_prompt_path=FAKE_REPO / "src" / "prompts" / "paper_standalone.md",
        current_selected_skills=_skills("write-paper-standalone"), selected_prompt_family="paper",
    )
    assert resolved.prompt_family == "paper"
    assert resolved.prompt_path.resolve() == (FAKE_REPO / "src" / "prompts" / "paper.md").resolve()
    assert resolved.writer_skill == "write-paper"
    assert resolved.selected_skills == _skills("write-paper")

    # persists the resolution artifact
    resolution_path = tmp_path / "paper_strategy_resolution.json"
    resolved = resolve_final_paper_stage(
        repo_root=FAKE_REPO, current_prompt_path=FAKE_REPO / "src" / "prompts" / "paper.md",
        current_selected_skills=_skills("write-paper"), selected_prompt_family="paper_standalone",
        resolution_path=resolution_path,
    )
    assert resolution_path.is_file()
    assert resolved.prompt_family == "paper_standalone"

    # rejects an unknown family
    with pytest.raises(ValueError, match="Unknown selected_prompt_family"):
        resolve_final_paper_stage(
            repo_root=FAKE_REPO, current_prompt_path=FAKE_REPO / "src" / "prompts" / "paper.md",
            current_selected_skills=_skills("write-paper"), selected_prompt_family="paper_transfer",
        )


# ── grounding policy survives every manuscript entry point ──

WRITER_TEMPLATES = (
    "paper.md", "paper_standalone.md", "paper_detailed_grouped.md", "paper_edit.md",
    "paper_edit_standalone.md", "paper_revision.md",
)
POLICY = (
    "Implementation-specific claims require inspectable source evidence",
    "Hashes and previous audit verdicts identify provenance, not substantive support",
    "Sequential optimization results support comparisons between evaluated configurations",
    "not isolated causal attribution to a component",
    "When ablations are missing or failed, narrow claims and state the limitation",
    "Claims supported by actual controlled comparisons remain valid within their measured scope",
)


def _render(template_name: str, tmp_path: Path) -> str:
    template = Path("src/prompts") / template_name
    task = tmp_path / "task"
    task.mkdir(exist_ok=True)
    (task / "task.md").write_text("# Goal\nSynthetic study\n\n# Rubrics\nCorrectness\n")
    keys = re.findall(r"\{\{([A-Z_]+)\}\}", template.read_text())
    rendered = load_instructions(
        template, task, tmp_path, {key: f"./fixture/{key.lower()}" for key in keys}
    )
    assert "{{" not in rendered
    return rendered


def test_rendered_prompts_require_inspectable_and_scope_matched_evidence(tmp_path: Path) -> None:
    for template in (*WRITER_TEMPLATES, "paper_review/evidence_audit.md"):
        rendered = _render(template, tmp_path)
        for requirement in POLICY:
            assert requirement in rendered, template


@pytest.mark.parametrize("script_name", PAPER_SCRIPTS)
@pytest.mark.parametrize("template", WRITER_TEMPLATES)
def test_generated_manuscript_repair_preserves_grounding_policy(
    script_name: str, template: str, tmp_path: Path
) -> None:
    rendered_path = tmp_path / "rendered.md"
    rendered_path.write_text(_render(template, tmp_path))
    script = (Path("src/scripts") / script_name).read_text()
    loop_start = script.rindex("\nrepair_round=0\n")
    stage_start = script.index("\n  if run_prompt_stage_with_feedback", loop_start)
    selection = script[loop_start:stage_start].replace("repair_round=0\n", "repair_round=1\n", 1)
    shell = f"""
set -euo pipefail
OUTPUT_DIR={shlex.quote(str(tmp_path))}
CURRENT_PAPER_INSTRUCT_PATH={shlex.quote(template)}
AGENT=synthetic
TIMESTAMP=test
PAPER_MAX_REPAIR_ROUNDS=2
PAPER_REVIEW_ROUNDS=0
restore_revision_references() {{ :; }}
paper_gate_start() {{ :; }}
paper_gate_finish() {{ :; }}
render_prompt() {{
  test "$1" = "$CURRENT_PAPER_INSTRUCT_PATH"
  test "$2" = "$CURRENT_PAPER_INSTRUCT_PATH"
  cat {shlex.quote(str(rendered_path))}
}}
{selection}
  printf '%s' "$PROMPT"
  exit 0
done
"""
    result = subprocess.run(
        ["bash", "-c", shell], capture_output=True, text=True, timeout=10, check=True
    )
    assert "paper_repair_prompt_round_1.md" in result.stdout
    for requirement in POLICY:
        assert requirement in result.stdout


# ── shared writing-style rules reach every manuscript writer ──

STYLE_TEMPLATES = WRITER_TEMPLATES


@pytest.mark.parametrize("template", STYLE_TEMPLATES)
def test_rendered_writer_prompts_include_shared_writing_style_once(
    template: str, tmp_path: Path
) -> None:
    rendered = _render(template, tmp_path)
    assert rendered.count("# Writing Style\n") == 1
    assert "PAPER_WRITING_STYLE" not in rendered

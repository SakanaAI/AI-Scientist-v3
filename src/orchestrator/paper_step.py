"""Paper validation and LaTeX compile orchestration helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import signal
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pypdf import PdfReader

from src.orchestrator.figure_manifest_schema import (
    NormalizedFigureManifestFigure,
    SUPPORTED_FIGURE_GENERATORS,
    SUPPORTED_FIGURE_TYPES,
    format_figure_manifest_validation_error,
    normalize_figure_manifest,
)
from src.orchestrator.paper_venues import (
    detect_venue,
    page_limits,
    venue_has_check,
)
from src.orchestrator.paper_supplement import (
    RENDERED_REFERENCE_HEADING_RE as REFERENCE_HEADING_RE,
    find_supplement_start_page,
    mask_tex_verbatim,
    strip_false_conditionals,
    validate_rendered_supplementary_information,
    validate_supplementary_information,
)
from src.orchestrator.paper_watermark import (
    WATERMARK_TEXT,
    PaperWatermarkError,
    apply_paper_watermark,
    has_managed_watermark,
)
from src.orchestrator.tex_graphics import (
    FIGURE_INCLUDE_RE,
    extract_graphics_paths as _extract_graphics_paths,
    read_tex_file_bounded,
    resolve_graphics_references,
)

SECTION_PATTERNS = {
    "abstract": re.compile(
        r"\\begin\{abstract\}|\\section\*?\{[^}]*abstract[^}]*\}",
        re.IGNORECASE,
    ),
    "introduction": re.compile(r"\\section\*?\{[^}]*introduction[^}]*\}", re.IGNORECASE),
    "related work": re.compile(r"\\section\*?\{[^}]*related work[^}]*\}", re.IGNORECASE),
    "method": re.compile(r"\\section\*?\{[^}]*method[^}]*\}", re.IGNORECASE),
    "results": re.compile(r"\\section\*?\{[^}]*results[^}]*\}", re.IGNORECASE),
    "conclusion": re.compile(r"\\section\*?\{[^}]*conclusion[^}]*\}", re.IGNORECASE),
    "references": re.compile(
        r"\\bibliography\{|\\printbibliography|\\begin\{thebibliography\}|\\section\*?\{[^}]*references[^}]*\}",
        re.IGNORECASE,
    ),
}
CITATION_KEY_RE = re.compile(r"@\w+\{([^,]+),")
TEX_CITATION_RE = re.compile(r"\\cite[a-zA-Z*]*\{([^}]+)\}")
TEX_INPUT_RE = re.compile(
    r"\\(?:input|include)(?![A-Za-z@])\s*"
    r"(?:\{(?P<braced>[^{}]+)\}|(?P<unbraced>[^\s%{}]+))"
)
TEX_IMPORT_COMMAND_RE = re.compile(r"\\(?:input|include)(?![A-Za-z@])")
TEX_LOCAL_SUPPORT_RE = re.compile(
    r"\\(?P<command>usepackage|RequirePackage|documentclass|LoadClass)"
    r"(?![A-Za-z@])(?:\s*\[[^\]]*\])?\s*\{(?P<targets>[^{}]+)\}"
)
TEX_LOCAL_SUPPORT_COMMAND_RE = re.compile(
    r"\\(?:usepackage|RequirePackage|documentclass|LoadClass)(?![A-Za-z@])"
)
DYNAMIC_TEX_IMPORT_RE = re.compile(r"\\csname\s*(?:input|include)\s*\\endcsname")
TEX_BIBLIOGRAPHY_RE = re.compile(
    r"\\bibliography(?![A-Za-z@])\s*\{(?P<targets>[^{}]+)\}"
)
TEX_BIBLIOGRAPHY_COMMAND_RE = re.compile(r"\\bibliography(?![A-Za-z@])")
TEX_BIB_RESOURCE_RE = re.compile(
    r"\\addbibresource(?![A-Za-z@])(?:\s*\[[^\]]*\])?\s*\{(?P<target>[^{}]+)\}"
)
TEX_BIB_RESOURCE_COMMAND_RE = re.compile(r"\\addbibresource(?![A-Za-z@])")
TEX_BIBLIOGRAPHY_STYLE_RE = re.compile(
    r"\\bibliographystyle(?![A-Za-z@])\s*\{(?P<target>[^{}]+)\}"
)
TEX_BIBLIOGRAPHY_STYLE_COMMAND_RE = re.compile(r"\\bibliographystyle(?![A-Za-z@])")
BIBTEX_AUX_DATA_RE = re.compile(r"\\bibdata\s*\{(?P<targets>[^{}]+)\}")
BIBTEX_AUX_DATA_COMMAND_RE = re.compile(r"\\bibdata(?![A-Za-z@])")
BIBTEX_AUX_STYLE_RE = re.compile(r"\\bibstyle\s*\{(?P<target>[^{}]+)\}")
BIBTEX_AUX_STYLE_COMMAND_RE = re.compile(r"\\bibstyle(?![A-Za-z@])")
BIBTEX_AUX_INPUT_RE = re.compile(r"\\@input\s*\{(?P<target>[^{}]+)\}")
BIBTEX_AUX_INPUT_COMMAND_RE = re.compile(r"\\@input(?![A-Za-z@])")
EVIDENCE_REF_RE = re.compile(r"@evidence\[\[[^\]]+\]\]")
PROMPT_ARTIFACT_RE = re.compile(r".+_prompt(?:_round_\d+(?:_contract_retry_\d+)?)?\.md$")
GENERATED_EVIDENCE_SCAN_ARTIFACT_NAMES = {
    "validation_feedback.json",
    "validation_feedback.md",
    "verification-report.json",
}
ROOT_GENERATED_EVIDENCE_SCAN_ARTIFACT_NAMES = {"validation_report.json"}
ROOT_EVIDENCE_ARCHIVE_DIR_PREFIXES = (("previous_paper",),)
GENERATED_EVIDENCE_SCAN_ARTIFACT_PREFIXES = ("paper_evidence_audit",)
EVIDENCE_BUNDLE_DIRNAME = "scientific-evidence"
EVIDENCE_SCAN_SUFFIXES = (
    ".tex",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".sh",
    ".r",
    ".csv",
    ".tsv",
)
EVIDENCE_SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    EVIDENCE_BUNDLE_DIRNAME,
}
AGENT_SKILL_DIR_PREFIXES = (
    (".claude", "skills"),
    (".codex", "skills"),
    (".cursor", "skills"),
    (".opencode", "skills"),
    (".gemini", "skills"),
    (".pi", "agent", "skills"),
)
VERIFICATION_SUMMARY_KEYS = (
    "total_references",
    "resolved_references",
    "anchored_references",
    "dangling_references",
    "manifests_checked",
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
CONTAINER_CLEAN_ENV_COMMAND = [
    "sh",
    "-c",
    'exec env -i PATH="$PATH" HOME=/nonexistent LANG="${LANG:-C.UTF-8}" '
    'TEXMFHOME=/nonexistent "$@"',
    "paper-compile",
]
LATEX_ENV_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SOURCE_DATE_EPOCH",
    "FORCE_SOURCE_DATE",
)
SCIENTIFIC_EVIDENCE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "agents"
    / "skills"
    / "scientific-evidence"
    / "scripts"
    / "scientific_evidence.py"
)
DEFAULT_CATEGORY_BUDGET = {
    "grouped_bar": 6,
    "bar": 8,
    "line": 8,
    "box": 6,
    "violin": 6,
    "scatter": 8,
    "heatmap": 8,
    "table_matrix": 8,
    "table": 8,
    "composite": 6,
    "diagram": 1,
}
DEFAULT_SERIES_BUDGET = {
    "grouped_bar": 4,
    "bar": 4,
    "line": 4,
    "box": 4,
    "violin": 4,
    "scatter": 4,
    "heatmap": 8,
    "table_matrix": 8,
    "table": 8,
    "composite": 4,
    "diagram": 1,
}
STRATEGY_FIGURE_TYPE_ALIASES = {
    "flow_diagram": "diagram",
    "horizontal_bar": "bar",
}


def _normalized_strategy_figure_type(value: object) -> str:
    figure_type = str(value).strip()
    return STRATEGY_FIGURE_TYPE_ALIASES.get(figure_type, figure_type)


def extract_graphics_paths(tex_text: str) -> list[str]:
    """Preserve the paper-step graphics extraction API."""
    return _extract_graphics_paths(tex_text)


STANDALONE_LANGUAGE_SOFT_WARNING_TERMS = (
    "task target",
    "aspirational target",
    "implementations",
)
GENERIC_FORBIDDEN_AUTHOR_TERMS = (
    "Anonymous Authors",
    "Anonymous Author",
    "Automated Research System",
    "Institution",
    "contact@institution.edu",
)
VENUE_REAL_AUTHOR_MARKERS = (
    "single-blind",
    "single blind",
    "not anonymous",
    "include real author names",
    "include author names",
    "real author names and affiliations",
)
ALIFE_REQUIRED_FRAMING_TERMS = (
    "artificial life",
    "life-like",
    "lifelike",
    "complex adaptive",
    "adaptive computation",
    "self-organization",
    "self-organisation",
    "robustness",
    "plasticity",
    "biologically plausible",
    "dale",
)
ALIFE_GENERIC_DRIFT_TERMS = (
    "edge devices",
    "deployment",
    "resource-constrained deployment",
    "scientific computing",
    "socio-technical",
)
#: Enforced page budgets, declared per venue under ``configs/venue/``.
PAGE_LIMITS: dict[str, dict[str, int | float | bool]] = page_limits()


class PaperValidationInfrastructureError(RuntimeError):
    """Raised when paper validation cannot run because its tooling is unavailable."""


_DOCKER_INFRASTRUCTURE_ERROR_MARKERS = (
    "cannot connect to the docker daemon",
    "error during connect",
    "is the docker daemon running",
    "permission denied while trying to connect",
    "no such image",
    "pull access denied",
)
CANONICAL_AUTHOR_INPUT_RE = re.compile(r"\\input\s*\{(?:\./)?paper_authors\.tex\}")


def read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    """Read JSON data from disk when available."""
    if not path.exists():
        if required:
            raise RuntimeError(f"Missing required artifact: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def extract_bibtex_keys(bibtex_text: str) -> set[str]:
    """Extract bibtex keys."""
    return {
        match.group(1).strip()
        for match in CITATION_KEY_RE.finditer(bibtex_text)
        if match.group(1).strip()
    }


def extract_tex_citations(tex_text: str) -> set[str]:
    """Extract tex citations."""
    keys: set[str] = set()
    for match in TEX_CITATION_RE.finditer(tex_text):
        raw = match.group(1).strip()
        if not raw:
            continue
        for part in raw.split(","):
            key = part.strip()
            if key:
                keys.add(key)
    return keys


def missing_required_sections(tex_text: str) -> list[str]:
    """Handle missing required sections logic for this module."""
    missing: list[str] = []
    for section, pattern in SECTION_PATTERNS.items():
        if not pattern.search(tex_text):
            missing.append(section)
    return missing


def _strip_tex_comments(tex_text: str) -> str:
    """Remove LaTeX comments while preserving escaped percent signs."""
    cleaned_lines: list[str] = []
    for line in tex_text.splitlines():
        kept: list[str] = []
        i = 0
        while i < len(line):
            if line.startswith(r"\verb", i):
                delimiter_index = i + len(r"\verb")
                if delimiter_index < len(line) and line[delimiter_index] == "*":
                    delimiter_index += 1
                if (
                    delimiter_index < len(line)
                    and not line[delimiter_index].isspace()
                    and not line[delimiter_index].isalpha()
                ):
                    closing_index = line.find(line[delimiter_index], delimiter_index + 1)
                    if closing_index >= 0:
                        kept.append(line[i : closing_index + 1])
                        i = closing_index + 1
                        continue
            char = line[i]
            if char == "%" and not _is_escaped_tex_character(line, i):
                break
            kept.append(char)
            i += 1
        cleaned_lines.append("".join(kept))
    return "\n".join(cleaned_lines)


def _is_escaped_tex_character(text: str, index: int) -> bool:
    backslash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslash_count += 1
        cursor -= 1
    return backslash_count % 2 == 1


def visible_tex_text(tex_text: str) -> str:
    """Reduce LaTeX source to a closer approximation of visible manuscript text."""
    visible = _strip_tex_comments(tex_text)
    visible = FIGURE_INCLUDE_RE.sub(" ", visible)
    visible = re.sub(
        r"\\(?:bibliography|bibliographystyle|label|ref|eqref|pageref|cite[a-zA-Z*]*)\*?(?:\[[^\]]*\])?\{[^}]*\}",
        " ",
        visible,
    )
    visible = re.sub(r"\\(?:input|include)\{[^}]*\}", " ", visible)
    visible = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", visible)
    visible = re.sub(r"[{}]", " ", visible)
    visible = re.sub(r"\s+", " ", visible)
    return visible.strip()


def _visible_tex_text(tex_text: str) -> str:
    """Backward-compatible private alias for visible manuscript text extraction."""
    return visible_tex_text(tex_text)


def _standalone_language_patterns(forbidden_terms: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    """Compile visible-language checks for standalone paper mode."""
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for term in forbidden_terms:
        clean = term.strip()
        if not clean:
            continue
        escaped = re.escape(clean)
        if re.fullmatch(r"[A-Za-z0-9_]+", clean):
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", re.IGNORECASE)
        else:
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", re.IGNORECASE)
        patterns.append((clean, pattern))
    return patterns


def validate_visible_language(
    tex_text: str,
    *,
    paper_profile: str,
    presentation_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate visible standalone-paper wording without changing default behavior."""
    report = {
        "checked": paper_profile == "standalone",
        "paper_profile": paper_profile,
        "hard_errors": [],
        "soft_warnings": [],
        "forbidden_terms": [],
    }
    if paper_profile != "standalone":
        return report

    forbidden_terms_raw = presentation_contract.get("forbidden_visible_terms")
    forbidden_terms = (
        [
            str(term).strip()
            for term in forbidden_terms_raw
            if isinstance(term, str) and term.strip()
        ]
        if isinstance(forbidden_terms_raw, list)
        else []
    )
    for extra_term in ("first eligible", "correctness threshold"):
        if extra_term not in forbidden_terms:
            forbidden_terms.append(extra_term)
    report["forbidden_terms"] = forbidden_terms

    visible_text = _visible_tex_text(tex_text)
    hard_errors: list[str] = []
    for term, pattern in _standalone_language_patterns(forbidden_terms):
        match = pattern.search(visible_text)
        if match is None:
            continue
        excerpt_start = max(0, match.start() - 40)
        excerpt_end = min(len(visible_text), match.end() + 40)
        excerpt = visible_text[excerpt_start:excerpt_end].strip()
        hard_errors.append(
            f"visible standalone paper text uses forbidden pipeline term {term!r}: ...{excerpt}..."
        )

    soft_warnings: list[str] = []
    for term in STANDALONE_LANGUAGE_SOFT_WARNING_TERMS:
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", visible_text, re.IGNORECASE
        ):
            soft_warnings.append(
                f"visible standalone paper text uses discouraged phrasing {term!r}; prefer reader-facing language"
            )

    report["hard_errors"] = hard_errors
    report["soft_warnings"] = soft_warnings
    return report


def _extract_tex_command_body(tex_text: str, command: str) -> str:
    """Extract a LaTeX command body with balanced-brace parsing."""
    match = re.search(rf"\\{re.escape(command)}\s*\{{", tex_text)
    if match is None:
        return ""

    result = _extract_balanced_tex_body(tex_text, match.end())
    return result[0].strip() if result is not None else ""


def _extract_balanced_tex_body(tex_text: str, start: int) -> tuple[str, int] | None:
    """Extract content after an opening brace and return its closing position."""
    depth = 1
    idx = start
    while idx < len(tex_text):
        char = tex_text[idx]
        if char == "{" and not _is_escaped_tex_character(tex_text, idx):
            depth += 1
        elif char == "}" and not _is_escaped_tex_character(tex_text, idx):
            depth -= 1
            if depth == 0:
                return tex_text[start:idx], idx + 1
        idx += 1
    return None


def _extract_tex_title(tex_text: str) -> str:
    return _extract_tex_command_body(tex_text, "title")


def _extract_tex_section_body(tex_text: str, heading: str) -> str:
    pattern = re.compile(r"\\section\*?\{(?P<title>[^}]*)\}", re.IGNORECASE)
    matches = list(pattern.finditer(tex_text))
    aliases = {
        "introduction": ("introduction", "introduction and motivation"),
        "discussion": ("discussion", "general discussion"),
        "conclusion": ("conclusion", "conclusions"),
    }
    allowed_titles = aliases.get(heading.strip().lower(), (heading.strip().lower(),))
    for idx, match in enumerate(matches):
        normalized_title = match.group("title").strip().lower()
        if not any(alias in normalized_title for alias in allowed_titles):
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(tex_text)
        return tex_text[start:end].strip()
    return ""


def _contains_any_term(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _normalize_tex_whitespace(text: str) -> str:
    return " ".join(text.split())


def _matches_canonical_author_block(author_text: str, canonical_text: str) -> bool:
    normalized_author = _normalize_tex_whitespace(_strip_tex_comments(author_text))
    if CANONICAL_AUTHOR_INPUT_RE.fullmatch(normalized_author):
        return True
    normalized_canonical = _normalize_tex_whitespace(_strip_tex_comments(canonical_text))
    return normalized_author == normalized_canonical


def _infer_venue_profile(*, paper_format: str | None, venue_brief_text: str) -> dict[str, Any]:
    """Infer validation expectations from paper format and venue brief."""
    venue_lower = venue_brief_text.lower()
    is_alife = (
        venue_has_check(paper_format, "alife_framing")
        or "artificial life" in venue_lower
        or "living and lifelike complex adaptive systems" in venue_lower
        or "life-like" in venue_lower
        or "lifelike" in venue_lower
    )
    require_real_authors = is_alife or any(
        marker in venue_lower for marker in VENUE_REAL_AUTHOR_MARKERS
    )
    return {
        "checked": require_real_authors or is_alife,
        "label": "ALIFE" if is_alife else "venue",
        "require_real_authors": require_real_authors,
        "framing_terms": ALIFE_REQUIRED_FRAMING_TERMS if is_alife else (),
        "generic_drift_terms": ALIFE_GENERIC_DRIFT_TERMS if is_alife else (),
    }


def validate_venue_alignment(
    tex_text: str,
    *,
    venue_brief_text: str,
    paper_format: str | None,
    paper_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate venue-specific author and framing expectations."""
    profile = _infer_venue_profile(paper_format=paper_format, venue_brief_text=venue_brief_text)
    report: dict[str, Any] = {
        "checked": profile["checked"],
        "paper_format": paper_format,
        "hard_errors": [],
        "soft_warnings": [],
    }
    if not profile["checked"]:
        return report

    hard_errors: list[str] = []
    soft_warnings: list[str] = []
    label = str(profile["label"])

    if bool(profile["require_real_authors"]):
        author_text = _extract_tex_command_body(tex_text, "author")
        canonical_author_text: str | None = None
        if paper_dir is not None:
            canonical_author_path = paper_dir / "paper_authors.tex"
            if canonical_author_path.is_file():
                canonical_author_text = _strip_tex_comments(
                    canonical_author_path.read_text(encoding="utf-8")
                ).strip()
        if not author_text:
            hard_errors.append(
                f"{label} paper is missing an author block; submissions are not anonymous"
            )
        for term in GENERIC_FORBIDDEN_AUTHOR_TERMS:
            if term.lower() in author_text.lower():
                hard_errors.append(
                    f"{label} author block uses forbidden anonymous/generic placeholder {term!r}"
                )
        if (
            canonical_author_text is not None
            and author_text
            and not _matches_canonical_author_block(author_text, canonical_author_text)
        ):
            hard_errors.append(
                f"{label} author block does not match canonical `./paper_authors.tex`"
            )

    title_text = _extract_tex_title(tex_text)
    abstract_match = re.search(
        r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}", tex_text, re.DOTALL
    )
    introduction_text = _extract_tex_section_body(tex_text, "Introduction")
    conclusion_text = _extract_tex_section_body(tex_text, "Conclusion")
    section_texts = {
        "title": title_text,
        "abstract": abstract_match.group("body").strip() if abstract_match is not None else "",
        "introduction": introduction_text,
        "conclusion": conclusion_text,
    }
    framing_terms = tuple(profile["framing_terms"])
    if framing_terms:
        for section_name, section_text in section_texts.items():
            if not section_text:
                continue
            if not _contains_any_term(section_text, framing_terms):
                soft_warnings.append(
                    f"{label} paper {section_name} lacks explicit {label}-facing framing language"
                )

        discussion_text = _extract_tex_section_body(tex_text, "Discussion")
        drift_scope = "\n".join(
            part for part in (discussion_text, section_texts["conclusion"]) if part
        )
        drift_terms = tuple(profile["generic_drift_terms"])
        if drift_scope and drift_terms and _contains_any_term(drift_scope, drift_terms):
            soft_warnings.append(
                f"{label} paper discussion/conclusion drifts into generic deployment or application framing without clear venue-specific relevance"
            )

    report["hard_errors"] = hard_errors
    report["soft_warnings"] = soft_warnings
    return report


def detect_paper_format(tex_text: str) -> str | None:
    """Infer the paper format from the venue whose LaTeX markers match."""
    venue = detect_venue(tex_text)
    return venue.name if venue is not None else None


def read_pdf_page_texts(pdf_path: Path) -> list[str]:
    """Extract page text from a compiled PDF."""
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def _fragment_y_position(*, user_matrix: Any, text_matrix: Any) -> float | None:
    """Prefer text-space y coordinates when user-space coordinates collapse to zero."""
    candidates: list[float] = []
    for matrix in (text_matrix, user_matrix):
        if not isinstance(matrix, (list, tuple)) or len(matrix) <= 5:
            continue
        try:
            candidate = float(matrix[5])
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate):
            candidates.append(candidate)
    if not candidates:
        return None

    for candidate in candidates:
        if abs(candidate) > 1e-6:
            return candidate
    return candidates[0]


def _fragment_x_position(*, user_matrix: Any, text_matrix: Any) -> float | None:
    """Prefer text-space x coordinates when user-space coordinates collapse to zero."""
    candidates: list[float] = []
    for matrix in (text_matrix, user_matrix):
        if not isinstance(matrix, (list, tuple)) or len(matrix) <= 4:
            continue
        try:
            candidate = float(matrix[4])
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate):
            candidates.append(candidate)
    if not candidates:
        return None

    for candidate in candidates:
        if abs(candidate) > 1e-6:
            return candidate
    return candidates[0]


def _graphic_bounds_from_matrix(matrix: Any) -> dict[str, float] | None:
    """Return the page-space bounds of a unit-square PDF graphic."""
    if not isinstance(matrix, (list, tuple)) or len(matrix) < 6:
        return None
    try:
        a, b, c, d, e, f = (float(matrix[index]) for index in range(6))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (a, b, c, d, e, f)):
        return None
    linear_scale = max(abs(a), abs(b), abs(c), abs(d), 1.0)
    normalized_determinant = (a / linear_scale) * (d / linear_scale) - (
        b / linear_scale
    ) * (c / linear_scale)
    if not math.isfinite(normalized_determinant) or abs(normalized_determinant) <= 1e-12:
        return None

    corners = (
        (e, f),
        (a + e, b + f),
        (c + e, d + f),
        (a + c + e, b + d + f),
    )
    if not all(math.isfinite(value) for point in corners for value in point):
        return None
    x_values = [point[0] for point in corners]
    y_values = [point[1] for point in corners]
    bounds = {
        "x_min": min(x_values),
        "x_max": max(x_values),
        "y_min": min(y_values),
        "y_max": max(y_values),
    }
    if bounds["x_max"] <= bounds["x_min"] or bounds["y_max"] <= bounds["y_min"]:
        return None
    return bounds


def _page_image_names(page: Any) -> set[str]:
    """Return image XObject names declared by a PDF page."""
    try:
        resource_owner = page
        visited: set[int] = set()
        resources = None
        while resource_owner is not None and id(resource_owner) not in visited:
            visited.add(id(resource_owner))
            resources = resource_owner.get("/Resources")
            if resources is not None:
                break
            parent = resource_owner.get("/Parent")
            resource_owner = parent.get_object() if hasattr(parent, "get_object") else parent
        resources = resources or {}
        if hasattr(resources, "get_object"):
            resources = resources.get_object()
        xobjects = resources.get("/XObject") or {}
        if hasattr(xobjects, "get_object"):
            xobjects = xobjects.get_object()
        return {
            str(name)
            for name, reference in xobjects.items()
            if str(reference.get_object().get("/Subtype")) == "/Image"
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return set()


def _multiply_pdf_matrices(left: Any, right: Any) -> list[float] | None:
    """Multiply two six-value PDF transformation matrices."""
    if not all(isinstance(matrix, (list, tuple)) and len(matrix) >= 6 for matrix in (left, right)):
        return None
    try:
        m = [float(value) for value in left[:6]]
        n = [float(value) for value in right[:6]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (*m, *n)):
        return None
    return [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]


def _read_page_graphics(
    page: Any,
    *,
    max_operations: int = 100_000,
    max_state_depth: int = 1024,
    max_unique_graphics: int = 1024,
) -> list[dict[str, float]]:
    """Read top-level image placements without recursing into Form XObjects."""
    image_names = _page_image_names(page)
    if (
        not image_names
        or max_operations <= 0
        or max_state_depth <= 0
        or max_unique_graphics <= 0
    ):
        return []
    try:
        operations = page.get_contents().operations
    except (AttributeError, KeyError, TypeError, ValueError):
        return []
    if len(operations) > max_operations:
        return []

    current_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    matrix_stack: list[list[float]] = []
    unique_bounds: set[tuple[float, float, float, float]] = set()
    for operands, operator in operations:
        if operator == b"q":
            if len(matrix_stack) >= max_state_depth:
                return []
            matrix_stack.append(current_matrix.copy())
            continue
        if operator == b"Q":
            current_matrix = matrix_stack.pop() if matrix_stack else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            continue
        if operator == b"cm":
            combined = _multiply_pdf_matrices(operands, current_matrix)
            current_matrix = combined or [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            continue
        if operator != b"Do" or not isinstance(operands, (list, tuple)) or not operands:
            continue
        if str(operands[0]) not in image_names:
            continue
        bounds = _graphic_bounds_from_matrix(current_matrix)
        if bounds is None:
            continue
        bounds_key = (
            bounds["x_min"],
            bounds["x_max"],
            bounds["y_min"],
            bounds["y_max"],
        )
        if bounds_key not in unique_bounds and len(unique_bounds) >= max_unique_graphics:
            return []
        unique_bounds.add(bounds_key)

    return [
        {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
        for x_min, x_max, y_min, y_max in sorted(unique_bounds)
    ]


def read_pdf_page_layouts(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract per-page text and image bounds for layout-aware validation."""
    reader = PdfReader(str(pdf_path))
    page_layouts: list[dict[str, Any]] = []

    for page in reader.pages:
        fragments: list[dict[str, float | str]] = []
        graphics = _read_page_graphics(page)

        def _visitor_text(
            text: Any,
            user_matrix: Any,
            text_matrix: Any,
            font_dict: Any,
            font_size: Any,
        ) -> None:
            del font_dict
            if not isinstance(text, str) or not text.strip():
                return

            y_value = _fragment_y_position(
                user_matrix=user_matrix,
                text_matrix=text_matrix,
            )
            if y_value is None:
                return
            x_value = _fragment_x_position(
                user_matrix=user_matrix,
                text_matrix=text_matrix,
            )
            if x_value is None:
                return

            try:
                font_size_value = float(font_size)
            except (TypeError, ValueError):
                font_size_value = 0.0
            if not math.isfinite(font_size_value):
                font_size_value = 0.0

            fragments.append(
                {
                    "text": text,
                    "x": x_value,
                    "y": y_value,
                    "font_size": max(font_size_value, 0.0),
                }
            )

        page_text = page.extract_text(visitor_text=_visitor_text) or ""
        page_x_min = float(page.mediabox.left)
        page_x_max = float(page.mediabox.right)
        page_y_min = float(page.mediabox.bottom)
        page_y_max = float(page.mediabox.top)
        page_layouts.append(
            {
                "text": page_text,
                "page_width": page_x_max - page_x_min,
                "page_height": page_y_max - page_y_min,
                "page_x_min": page_x_min,
                "page_x_max": page_x_max,
                "page_y_min": page_y_min,
                "page_y_max": page_y_max,
                "fragments": fragments,
                "graphics": graphics,
            }
        )

    return page_layouts


def _is_meaningful_text_fragment(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and bool(re.search(r"[A-Za-z]", stripped))


def _reference_heading_y(page_layout: dict[str, Any]) -> float | None:
    fragments = page_layout.get("fragments", [])
    if not isinstance(fragments, list):
        return None
    heading_positions: list[float] = []
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        text = fragment.get("text")
        y_value = fragment.get("y")
        if not isinstance(text, str):
            continue
        if y_value is None:
            continue
        try:
            y_float = float(y_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(y_float):
            continue
        if REFERENCE_HEADING_RE.fullmatch(text.strip()):
            heading_positions.append(y_float)
    if not heading_positions:
        return None
    return max(heading_positions)


def _body_text_bounds(
    page_layout: dict[str, Any],
    *,
    exclude_reference_region: bool,
    column_index: int | None = None,
    body_columns: int = 1,
) -> tuple[float, float] | None:
    fragments = page_layout.get("fragments", [])
    if not isinstance(fragments, list):
        return None

    reference_heading_y = _reference_heading_y(page_layout) if exclude_reference_region else None
    page_width_raw = page_layout.get("page_width")
    if page_width_raw is None:
        page_width_raw = 0.0
    try:
        page_width = float(page_width_raw)
    except (TypeError, ValueError):
        page_width = 0.0
    try:
        page_x_min = float(page_layout.get("page_x_min", 0.0))
        page_x_max = float(page_layout.get("page_x_max", page_x_min + page_width))
        page_y_min = float(page_layout.get("page_y_min", 0.0))
        page_y_max = float(
            page_layout.get("page_y_max", page_y_min + float(page_layout.get("page_height", 0.0)))
        )
    except (TypeError, ValueError):
        page_x_min = 0.0
        page_x_max = page_width
        page_y_min = 0.0
        page_y_max = float(page_layout.get("page_height", 0.0) or 0.0)
    column_boundary = page_width / body_columns if body_columns > 0 else 0.0
    y_tops: list[float] = []
    y_bottoms: list[float] = []

    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        text = fragment.get("text")
        if not isinstance(text, str) or not _is_meaningful_text_fragment(text):
            continue
        x_raw = fragment.get("x")
        y_raw = fragment.get("y")
        if x_raw is None or y_raw is None:
            continue
        try:
            x_value = float(x_raw)
            y_value = float(y_raw)
            font_size = float(fragment.get("font_size", 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x_value):
            continue
        if not math.isfinite(y_value):
            continue
        if column_index is not None and body_columns > 1 and page_width > 0:
            column_start = column_boundary * column_index
            column_end = column_boundary * (column_index + 1)
            if not (column_start <= x_value < column_end):
                continue
        if (
            exclude_reference_region
            and reference_heading_y is not None
            and y_value <= reference_heading_y
        ):
            continue
        if not math.isfinite(font_size) or font_size < 0:
            font_size = 0.0
        y_bottoms.append(y_value)
        y_tops.append(y_value + font_size)

    graphics = page_layout.get("graphics", [])
    if isinstance(graphics, list):
        for graphic in graphics:
            if not isinstance(graphic, dict):
                continue
            try:
                x_min = float(graphic["x_min"])
                x_max = float(graphic["x_max"])
                y_min = float(graphic["y_min"])
                y_max = float(graphic["y_max"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
                continue
            if x_max <= x_min or y_max <= y_min:
                continue
            x_min = max(x_min, page_x_min)
            x_max = min(x_max, page_x_max)
            y_min = max(y_min, page_y_min)
            y_max = min(y_max, page_y_max)
            if x_max <= x_min or y_max <= y_min:
                continue
            if column_index is not None and body_columns > 1 and page_width > 0:
                column_start = page_x_min + column_boundary * column_index
                column_end = page_x_min + column_boundary * (column_index + 1)
                if x_max <= column_start or x_min >= column_end:
                    continue
                x_min = max(x_min, column_start)
                x_max = min(x_max, column_end)
            if exclude_reference_region and reference_heading_y is not None:
                if y_max <= reference_heading_y:
                    continue
                y_min = max(y_min, reference_heading_y)
            y_bottoms.append(y_min)
            y_tops.append(y_max)

    if not y_tops or not y_bottoms:
        return None
    return max(y_tops), min(y_bottoms)


def compute_last_body_page_fill_ratio(
    page_layouts: list[dict[str, Any]],
    *,
    last_body_page_number: int,
    reference_start_page: int | None,
    body_columns: int = 1,
) -> float | None:
    """Estimate how much of the usable text span the last counted body page occupies."""
    if last_body_page_number < 1 or last_body_page_number > len(page_layouts):
        return None

    usable_bounds: list[tuple[float, float]] = []
    for page_number in range(1, last_body_page_number + 1):
        bounds = _body_text_bounds(
            page_layouts[page_number - 1],
            exclude_reference_region=page_number == reference_start_page,
            body_columns=body_columns,
        )
        if bounds is not None:
            usable_bounds.append(bounds)

    if not usable_bounds:
        return None

    usable_top = max(top for top, _ in usable_bounds)
    usable_bottom = min(bottom for _, bottom in usable_bounds)
    usable_height = usable_top - usable_bottom
    if usable_height <= 0:
        return None

    last_page = page_layouts[last_body_page_number - 1]
    if body_columns <= 1:
        last_bounds = _body_text_bounds(
            last_page,
            exclude_reference_region=last_body_page_number == reference_start_page,
            body_columns=body_columns,
        )
        if last_bounds is None:
            return None
        last_top, last_bottom = last_bounds
        fill_ratio = (last_top - last_bottom) / usable_height
        return max(0.0, min(fill_ratio, 1.0))

    column_fill_ratios: list[float] = []
    for column_index in range(body_columns):
        last_bounds = _body_text_bounds(
            last_page,
            exclude_reference_region=last_body_page_number == reference_start_page,
            column_index=column_index,
            body_columns=body_columns,
        )
        if last_bounds is None:
            column_fill_ratios.append(0.0)
            continue
        last_top, last_bottom = last_bounds
        fill_ratio = (last_top - last_bottom) / usable_height
        column_fill_ratios.append(max(0.0, min(fill_ratio, 1.0)))
    if not column_fill_ratios:
        return None
    return sum(column_fill_ratios) / len(column_fill_ratios)


def measure_last_body_page_fill_ratio(
    pdf_path: Path,
    *,
    last_body_page_number: int,
    reference_start_page: int | None,
    body_columns: int = 1,
) -> float | None:
    """Read the compiled PDF and estimate last-page body-text occupancy."""
    page_layouts = read_pdf_page_layouts(pdf_path)
    return compute_last_body_page_fill_ratio(
        page_layouts,
        last_body_page_number=last_body_page_number,
        reference_start_page=reference_start_page,
        body_columns=body_columns,
    )


def find_reference_start_page(page_texts: list[str]) -> int | None:
    """Return the first 1-indexed page that visibly starts the references section."""
    for page_index, page_text in enumerate(page_texts, start=1):
        if REFERENCE_HEADING_RE.search(page_text):
            return page_index
    return None


def _is_watermark_text_line(line: str) -> bool:
    """True for extracted-text lines contributed by the mandatory page watermark.

    Extraction may keep the overlay as one line or split it into fragments, so any
    line whose letters form a contiguous chunk of the watermark text is treated as
    watermark output rather than manuscript body text.
    """
    normalized = re.sub(r"[^A-Za-z]", "", line).casefold()
    if not normalized:
        return False
    return normalized in re.sub(r"[^A-Za-z]", "", WATERMARK_TEXT).casefold()


def reference_page_has_body_prefix(page_text: str) -> bool:
    """Heuristically detect substantive body text before a references heading on the same page."""
    match = REFERENCE_HEADING_RE.search(page_text)
    if match is None:
        return False

    prefix = page_text[: match.start()]
    meaningful_lines: list[str] = []
    for raw_line in prefix.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not re.search(r"[A-Za-z]", line):
            continue
        if _is_watermark_text_line(line):
            continue
        meaningful_lines.append(line)

    if not meaningful_lines:
        return False
    if len(meaningful_lines) >= 2:
        return True

    only_line = meaningful_lines[0]
    if re.search(r"[.!?;]", only_line):
        return True
    if len(re.findall(r"[A-Za-z]+", only_line)) >= 4:
        return True
    return len(only_line) >= 20


def validate_page_budget(
    *, tex_text: str, pdf_path: Path, page_texts: list[str] | None = None
) -> dict[str, Any]:
    """Validate template-coupled page budgets from the compiled PDF."""
    paper_format = detect_paper_format(tex_text)
    budget = PAGE_LIMITS.get(paper_format) if paper_format is not None else None
    report: dict[str, Any] = {
        "checked": budget is not None,
        "paper_format": paper_format,
        "min_body_pages": None,
        "min_last_body_page_fill_ratio": None,
        "max_body_pages": None,
        "exclude_references": False,
        "total_pages": None,
        "reference_start_page": None,
        "supplement_start_page": None,
        "supplement_pages": 0,
        "body_pages": None,
        "last_body_page_number": None,
        "last_body_page_fill_ratio": None,
        "last_body_page_counted": None,
        "errors": [],
    }
    if page_texts is None and budget is None:
        return report

    if page_texts is None:
        page_texts = read_pdf_page_texts(pdf_path)
    total_pages = len(page_texts)
    reference_start_page = find_reference_start_page(page_texts)
    supplement_start_page = find_supplement_start_page(page_texts)
    report["total_pages"] = total_pages
    report["reference_start_page"] = reference_start_page
    report["supplement_start_page"] = supplement_start_page
    report["supplement_pages"] = (
        total_pages - supplement_start_page + 1 if supplement_start_page is not None else 0
    )

    if budget is None:
        return report

    min_body_pages = int(budget["min_body_pages"]) if "min_body_pages" in budget else None
    body_columns = int(budget.get("body_columns", 1))
    min_last_body_page_fill_ratio = (
        float(budget["min_last_body_page_fill_ratio"])
        if "min_last_body_page_fill_ratio" in budget
        else None
    )
    max_body_pages = int(budget["max_body_pages"])
    exclude_references = bool(budget.get("exclude_references", False))

    report["min_body_pages"] = min_body_pages
    report["min_last_body_page_fill_ratio"] = min_last_body_page_fill_ratio
    report["max_body_pages"] = max_body_pages
    report["exclude_references"] = exclude_references

    if exclude_references:
        if reference_start_page is None:
            if total_pages <= max_body_pages:
                raw_body_pages = total_pages
            else:
                report["errors"].append(
                    f"{paper_format} page-limit validation could not locate the references section in paper.pdf; total pages = {total_pages}, limit = {max_body_pages} excluding references"
                )
                return report
        else:
            reference_page_text = page_texts[reference_start_page - 1]
            raw_body_pages = max(reference_start_page - 1, 0)
            if reference_page_has_body_prefix(reference_page_text):
                raw_body_pages += 1
    else:
        raw_body_pages = total_pages

    body_pages = raw_body_pages
    last_body_page_number = raw_body_pages if raw_body_pages > 0 else None
    report["last_body_page_number"] = last_body_page_number

    if (
        min_body_pages is not None
        and min_last_body_page_fill_ratio is not None
        and raw_body_pages == min_body_pages
        and last_body_page_number is not None
    ):
        fill_ratio = measure_last_body_page_fill_ratio(
            pdf_path,
            last_body_page_number=last_body_page_number,
            reference_start_page=reference_start_page,
            body_columns=body_columns,
        )
        report["last_body_page_fill_ratio"] = fill_ratio
        if fill_ratio is None:
            report["last_body_page_counted"] = False
            report["errors"].append(
                f"{paper_format} page minimum validation could not determine the last counted body page fill ratio for page {last_body_page_number}"
            )
            report["body_pages"] = body_pages
            return report
        if fill_ratio < min_last_body_page_fill_ratio:
            body_pages = max(raw_body_pages - 1, 0)
            report["last_body_page_counted"] = False
            report["errors"].append(
                f"{paper_format} last counted body page fill ratio {fill_ratio:.2f} is below required {min_last_body_page_fill_ratio:.2f}; page {last_body_page_number} does not count toward minimum page requirement"
            )
        else:
            report["last_body_page_counted"] = True

    report["body_pages"] = body_pages

    if min_body_pages is not None and body_pages < min_body_pages:
        qualifier = " excluding references" if exclude_references else ""
        if min_last_body_page_fill_ratio is not None:
            qualifier += (
                f"; the final body page counts only when at least "
                f"{min_last_body_page_fill_ratio:.0%} full"
            )
        report["errors"].append(
            f"{paper_format} page minimum not met: body uses {body_pages} pages (min {min_body_pages}{qualifier})"
        )

    if raw_body_pages > max_body_pages:
        qualifier = " excluding references" if exclude_references else ""
        report["errors"].append(
            f"{paper_format} page limit exceeded: body uses {raw_body_pages} pages (max {max_body_pages}{qualifier})"
        )
    return report


def read_png_size(path: Path) -> tuple[int, int]:
    """Read PNG width and height from the IHDR chunk."""
    with path.open("rb") as handle:
        header = handle.read(24)
    return _read_png_size_header(header, str(path))


def _read_png_size_header(header: bytes, path_label: str) -> tuple[int, int]:
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Invalid PNG file: {path_label}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


CONCEPT_FIGURE_GENERATOR = "generate-concept-figure"
CONCEPT_FIGURE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def is_valid_concept_figure_id(figure_id: str) -> bool:
    return CONCEPT_FIGURE_ID_PATTERN.fullmatch(figure_id) is not None


def _read_staged_concept_artifact(
    render_dir: Path,
    artifact_rel: str,
    expected_suffix: str,
    max_bytes: int,
    label: str,
) -> bytes:
    data = _read_regular_figure_artifact_nofollow(
        render_dir,
        artifact_rel,
        expected_suffix,
        max_bytes + 1,
    )
    if data is None or len(data) > max_bytes:
        raise RuntimeError(f"Invalid staged concept figure {label}: {artifact_rel}")
    return data


def _validate_concept_figure_png(data: bytes, path_label: str) -> None:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"Invalid staged concept figure PNG: {path_label}")
    offset = 8
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(data):
        if len(data) - offset < 12:
            raise RuntimeError(f"Truncated staged concept figure PNG: {path_label}")
        chunk_length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_end = offset + 12 + chunk_length
        if chunk_end > len(data):
            raise RuntimeError(f"Truncated staged concept figure PNG: {path_label}")
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + chunk_length]
        expected_crc = struct.unpack(">I", data[offset + 8 + chunk_length : chunk_end])[0]
        actual_crc = zlib.crc32(chunk_data, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise RuntimeError(f"Invalid staged concept figure PNG CRC: {path_label}")
        chunks.append((chunk_type, chunk_data))
        offset = chunk_end
        if chunk_type == b"IEND":
            break

    if offset != len(data) or not chunks or chunks[-1] != (b"IEND", b""):
        raise RuntimeError(f"Invalid staged concept figure PNG ending: {path_label}")
    if chunks[0][0] != b"IHDR" or len(chunks[0][1]) != 13:
        raise RuntimeError(f"Invalid staged concept figure PNG header: {path_label}")
    if sum(chunk_type == b"IHDR" for chunk_type, _ in chunks) != 1:
        raise RuntimeError(f"Invalid staged concept figure PNG header: {path_label}")
    known_critical_chunks = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}
    if any(
        chunk_type[0] & 0x20 == 0 and chunk_type not in known_critical_chunks
        for chunk_type, _ in chunks
    ):
        raise RuntimeError(f"Unknown critical staged concept figure PNG chunk: {path_label}")

    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", chunks[0][1]
    )
    valid_bit_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    if (
        not 0 < width <= 16_384
        or not 0 < height <= 16_384
        or width * height > 40_000_000
        or color_type == 3
        or bit_depth not in valid_bit_depths.get(color_type, set())
        or compression != 0
        or filtering != 0
        or interlace != 0
    ):
        raise RuntimeError(f"Unsafe staged concept figure PNG header: {path_label}")

    idat_indexes = [index for index, (chunk_type, _) in enumerate(chunks) if chunk_type == b"IDAT"]
    if not idat_indexes or idat_indexes != list(range(idat_indexes[0], idat_indexes[-1] + 1)):
        raise RuntimeError(f"Invalid staged concept figure PNG data: {path_label}")
    plte_indexes = [index for index, (chunk_type, _) in enumerate(chunks) if chunk_type == b"PLTE"]
    if len(plte_indexes) > 1 or (plte_indexes and plte_indexes[0] > idat_indexes[0]):
        raise RuntimeError(f"Invalid staged concept figure PNG palette: {path_label}")
    if color_type == 3 and not plte_indexes:
        raise RuntimeError(f"Missing staged concept figure PNG palette: {path_label}")
    if color_type in {0, 4} and plte_indexes:
        raise RuntimeError(f"Invalid staged concept figure PNG palette: {path_label}")
    if plte_indexes:
        palette = chunks[plte_indexes[0]][1]
        if not palette or len(palette) % 3 != 0 or len(palette) > 3 * 256:
            raise RuntimeError(f"Invalid staged concept figure PNG palette: {path_label}")
    compressed = b"".join(chunks[index][1] for index in idat_indexes)
    samples_per_pixel = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    row_bytes = (width * samples_per_pixel * bit_depth + 7) // 8
    expected_decoded_bytes = height * (row_bytes + 1)
    if expected_decoded_bytes > 64 * 1024 * 1024:
        raise RuntimeError(f"Unsafe staged concept figure PNG size: {path_label}")
    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(compressed, expected_decoded_bytes + 1)
    except zlib.error as exc:
        raise RuntimeError(f"Invalid staged concept figure PNG data: {path_label}") from exc
    if (
        not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or len(decoded) != expected_decoded_bytes
    ):
        raise RuntimeError(f"Invalid staged concept figure PNG data: {path_label}")
    row_stride = row_bytes + 1
    if any(decoded[row_offset] > 4 for row_offset in range(0, len(decoded), row_stride)):
        raise RuntimeError(f"Invalid staged concept figure PNG filter: {path_label}")


def _validate_concept_prompt_log(prompts_data: bytes, prompts_rel: str) -> None:
    try:
        payload = json.loads(prompts_data)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeError(f"Invalid staged concept figure prompt log: {prompts_rel}") from exc
    _reject_unfinished_concept_prompt_log(payload, prompts_rel)
    panels = payload.get("panels") if isinstance(payload, dict) else None
    panel_id_pattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
    if (
        not isinstance(panels, list)
        or not panels
        or len(panels) > 64
        or any(
            not isinstance(panel, dict)
            or not isinstance(panel.get("id"), str)
            or panel_id_pattern.fullmatch(panel["id"]) is None
            for panel in panels
        )
    ):
        raise RuntimeError(f"Invalid staged concept figure prompt log: {prompts_rel}")
    panel_ids = [panel["id"] for panel in panels]
    if len(panel_ids) != len(set(panel_ids)):
        raise RuntimeError(f"Invalid staged concept figure prompt log: {prompts_rel}")
    placement = payload.get("placement")
    if placement is not None and placement not in {"half-width", "full-width"}:
        raise RuntimeError(f"Invalid staged concept figure prompt log: {prompts_rel}")


def _reject_unfinished_concept_prompt_log(payload: Any, path_label: str) -> None:
    if isinstance(payload, dict) and (
        payload.get("placeholder") is True
        or str(payload.get("status", "")).lower() in {"pending", "placeholder"}
    ):
        raise RuntimeError(f"Concept prompt log is unfinished: {path_label}")


CONCEPT_FIGURE_TRANSACTION_JOURNAL = ".concept-figure-transaction.json"


@contextmanager
def _defer_publication_signals() -> Iterator[None]:
    previous_handlers: dict[signal.Signals, Any] = {}
    deferred_signals: list[int] = []

    def defer_signal(signum: int, _frame: Any) -> None:
        deferred_signals.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.signal(signum, defer_signal)
        except ValueError:
            previous_handlers.clear()
            break
    try:
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if deferred_signals:
            os.kill(os.getpid(), deferred_signals[0])


def _open_concept_figures_dir(paper_dir: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(paper_dir / "figures", flags)
    except OSError as exc:
        raise RuntimeError(f"Unsafe concept figure destination: {paper_dir / 'figures'}") from exc


def _unlink_transaction_file(directory_descriptor: int, name: str) -> None:
    try:
        mode = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False).st_mode
    except FileNotFoundError:
        return
    if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
        raise RuntimeError(f"Unsafe concept figure transaction entry: {name}")
    os.unlink(name, dir_fd=directory_descriptor)


def _write_concept_transaction_journal(
    directory_descriptor: int,
    payload: dict[str, Any],
) -> None:
    temp_name = f".{CONCEPT_FIGURE_TRANSACTION_JOURNAL}.{secrets.token_hex(8)}.new"
    descriptor = os.open(
        temp_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temp_name,
            CONCEPT_FIGURE_TRANSACTION_JOURNAL,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temp_name, dir_fd=directory_descriptor)
        except OSError:
            pass


def _validate_concept_transaction(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("phase") not in {
        "prepared",
        "backed_up",
        "published",
        "committed",
    }:
        raise RuntimeError("Invalid concept figure transaction journal")
    token = payload.get("token")
    records = payload.get("records")
    if (
        not isinstance(token, str)
        or re.fullmatch(r"[a-f0-9]{16}", token) is None
        or not isinstance(records, list)
        or len(records) not in {2, 3}
    ):
        raise RuntimeError("Invalid concept figure transaction journal")
    destinations = [record.get("destination") for record in records if isinstance(record, dict)]
    if len(destinations) != len(records) or not isinstance(destinations[0], str):
        raise RuntimeError("Invalid concept figure transaction journal")
    figure_id = destinations[0].removesuffix(".png")
    expected_destinations = [
        f"{figure_id}.png",
        f"{figure_id}.prompts.json",
    ]
    if len(records) == 3:
        expected_destinations.append(f"{figure_id}.png.prov.json")
    if destinations != expected_destinations or not is_valid_concept_figure_id(figure_id):
        raise RuntimeError("Invalid concept figure transaction journal")
    for record in records:
        destination = record["destination"]
        identity = record.get("identity")
        temp_identity = record.get("temp_identity")
        digest = record.get("digest")
        size = record.get("size")
        existed = record.get("existed")
        if (
            not isinstance(existed, bool)
            or record.get("temp") != f".{destination}.{token}.new"
            or record.get("backup") != f".{destination}.{token}.old"
            or not isinstance(temp_identity, list)
            or len(temp_identity) != 3
            or any(not isinstance(value, int) for value in temp_identity)
            or not isinstance(digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            or not isinstance(size, int)
            or size < 0
            or (
                existed
                and (
                    not isinstance(identity, list)
                    or len(identity) != 3
                    or any(not isinstance(value, int) for value in identity)
                )
            )
            or (not existed and identity is not None)
        ):
            raise RuntimeError("Invalid concept figure transaction journal")
    return payload


def _read_concept_transaction_journal(
    directory_descriptor: int,
) -> dict[str, Any] | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(
            CONCEPT_FIGURE_TRANSACTION_JOURNAL,
            flags,
            dir_fd=directory_descriptor,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError("Unsafe concept figure transaction journal") from exc
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise RuntimeError("Unsafe concept figure transaction journal")
        journal_data = handle.read(65_537)
    if len(journal_data) > 65_536:
        raise RuntimeError("Oversized concept figure transaction journal")
    try:
        return _validate_concept_transaction(json.loads(journal_data))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeError("Invalid concept figure transaction journal") from exc


def _cleanup_concept_transaction(
    directory_descriptor: int,
    payload: dict[str, Any],
) -> None:
    for record in payload["records"]:
        _unlink_transaction_file(directory_descriptor, record["temp"])
        _unlink_transaction_file(directory_descriptor, record["backup"])
    _unlink_transaction_file(directory_descriptor, CONCEPT_FIGURE_TRANSACTION_JOURNAL)
    os.fsync(directory_descriptor)


def _rollback_concept_transaction(
    directory_descriptor: int,
    payload: dict[str, Any],
) -> None:
    if payload["phase"] == "committed":
        _cleanup_concept_transaction(directory_descriptor, payload)
        return
    errors: list[BaseException] = []
    for record in reversed(payload["records"]):
        try:
            backup_stat = os.stat(
                record["backup"],
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            backup_stat = None
        if backup_stat is not None:
            backup_identity = [backup_stat.st_dev, backup_stat.st_ino, backup_stat.st_mode]
            if backup_identity != record["identity"]:
                errors.append(RuntimeError(f"Replaced transaction backup: {record['backup']}"))
                continue
            if not (stat.S_ISREG(backup_stat.st_mode) or stat.S_ISLNK(backup_stat.st_mode)):
                errors.append(RuntimeError(f"Unsafe transaction backup: {record['backup']}"))
                continue
            try:
                try:
                    destination_stat = os.stat(
                        record["destination"],
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    destination_stat = None
                if destination_stat is not None:
                    destination_identity = [
                        destination_stat.st_dev,
                        destination_stat.st_ino,
                        destination_stat.st_mode,
                    ]
                    if destination_identity != record["temp_identity"]:
                        raise RuntimeError(
                            f"Concept figure destination changed before rollback: "
                            f"{record['destination']}"
                        )
                    os.unlink(record["destination"], dir_fd=directory_descriptor)
                os.link(
                    record["backup"],
                    record["destination"],
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                os.unlink(record["backup"], dir_fd=directory_descriptor)
            except (OSError, RuntimeError) as exc:
                errors.append(exc)
        elif not record["existed"] and payload["phase"] in {"backed_up", "published"}:
            try:
                destination_stat = os.stat(
                    record["destination"],
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                destination_identity = [
                    destination_stat.st_dev,
                    destination_stat.st_ino,
                    destination_stat.st_mode,
                ]
                if destination_identity != record["temp_identity"]:
                    raise RuntimeError(
                        f"Concept figure destination changed before rollback: "
                        f"{record['destination']}"
                    )
                _unlink_transaction_file(directory_descriptor, record["destination"])
            except FileNotFoundError:
                pass
            except (OSError, RuntimeError) as exc:
                errors.append(exc)
    try:
        os.fsync(directory_descriptor)
    except OSError as exc:
        errors.append(exc)
    if errors:
        raise RuntimeError("Could not roll back concept figure artifact publication") from errors[0]
    _cleanup_concept_transaction(directory_descriptor, payload)


def publish_concept_figure_marker(paper_dir: Path) -> None:
    parent_descriptor = _open_artifact_parent(paper_dir / ".concept_figure_complete")
    temp_name = f".concept-figure-marker.{secrets.token_hex(8)}.new"
    try:
        descriptor = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(
            temp_name,
            ".concept_figure_complete",
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        try:
            os.unlink(temp_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def _verify_published_concept_transaction(
    directory_descriptor: int,
    payload: dict[str, Any],
) -> dict[str, bytes]:
    verified_artifacts: dict[str, bytes] = {}
    for record in payload["records"]:
        try:
            descriptor = os.open(
                record["destination"],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            raise RuntimeError("Published concept figure transaction was modified") from exc
        with os.fdopen(descriptor, "rb") as handle:
            destination_stat = os.fstat(handle.fileno())
            destination_identity = [
                destination_stat.st_dev,
                destination_stat.st_ino,
                destination_stat.st_mode,
            ]
            data = handle.read(record["size"] + 1)
        if (
            destination_identity != record["temp_identity"]
            or len(data) != record["size"]
            or hashlib.sha256(data).hexdigest() != record["digest"]
        ):
            raise RuntimeError("Published concept figure transaction was modified")
        verified_artifacts[record["destination"]] = data
    return verified_artifacts


def _recover_concept_transaction(directory_descriptor: int, paper_dir: Path) -> None:
    payload = _read_concept_transaction_journal(directory_descriptor)
    if payload is None:
        orphan_pattern = re.compile(
            r"\.[A-Za-z0-9][A-Za-z0-9_-]{0,127}\."
            r"(?:png(?:\.prov\.json)?|prompts\.json)\.[a-f0-9]{16}\.new"
        )
        for name in os.listdir(directory_descriptor):
            if orphan_pattern.fullmatch(name):
                _unlink_transaction_file(directory_descriptor, name)
        os.fsync(directory_descriptor)
        return
    is_legacy_transaction = len(payload["records"]) == 2
    figure_id = payload["records"][0]["destination"].removesuffix(".png")
    if payload["phase"] == "published":
        _verify_published_concept_transaction(directory_descriptor, payload)
        record_concept_figure_success(paper_dir, figure_id)
        publish_concept_figure_marker(paper_dir)
        payload["phase"] = "committed"
        _write_concept_transaction_journal(directory_descriptor, payload)
    if payload["phase"] == "committed":
        if is_legacy_transaction:
            verified_artifacts = _verify_published_concept_transaction(
                directory_descriptor, payload
            )
            png_data = verified_artifacts[f"{figure_id}.png"]
            _write_json_atomically(
                paper_dir / "figures" / f"{figure_id}.png.prov.json",
                json.loads(_concept_figure_provenance_bytes(figure_id, png_data)),
            )
        try:
            _cleanup_concept_transaction(directory_descriptor, payload)
        except (OSError, RuntimeError):
            pass
        return
    _rollback_concept_transaction(directory_descriptor, payload)


def begin_concept_figure_artifact_publication(
    paper_dir: Path,
    artifacts: list[tuple[str, bytes]],
) -> None:
    directory_descriptor = _open_concept_figures_dir(paper_dir)
    token = secrets.token_hex(8)
    payload: dict[str, Any] = {"phase": "prepared", "token": token, "records": []}
    journal_written = False
    try:
        _recover_concept_transaction(directory_descriptor, paper_dir)
        for destination_name, data in artifacts:
            try:
                destination_stat = os.stat(
                    destination_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                destination_stat = None
            if destination_stat is not None and not (
                stat.S_ISREG(destination_stat.st_mode) or stat.S_ISLNK(destination_stat.st_mode)
            ):
                raise RuntimeError(
                    f"Concept figure destination must be a regular file or symlink: "
                    f"figures/{destination_name}"
                )
            temp_name = f".{destination_name}.{token}.new"
            backup_name = f".{destination_name}.{token}.old"
            record = {
                "destination": destination_name,
                "temp": temp_name,
                "backup": backup_name,
                "existed": destination_stat is not None,
                "identity": (
                    [destination_stat.st_dev, destination_stat.st_ino, destination_stat.st_mode]
                    if destination_stat is not None
                    else None
                ),
                "temp_identity": None,
                "digest": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
            payload["records"].append(record)
            descriptor = os.open(
                temp_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_descriptor,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
                temp_stat = os.fstat(handle.fileno())
                record["temp_identity"] = [
                    temp_stat.st_dev,
                    temp_stat.st_ino,
                    temp_stat.st_mode,
                ]
        _validate_concept_transaction(payload)
        _write_concept_transaction_journal(directory_descriptor, payload)
        journal_written = True
        with _defer_publication_signals():
            try:
                for record in payload["records"]:
                    if not record["existed"]:
                        continue
                    os.replace(
                        record["destination"],
                        record["backup"],
                        src_dir_fd=directory_descriptor,
                        dst_dir_fd=directory_descriptor,
                    )
                    backup_stat = os.stat(
                        record["backup"],
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if [backup_stat.st_dev, backup_stat.st_ino, backup_stat.st_mode] != record[
                        "identity"
                    ]:
                        os.link(
                            record["backup"],
                            record["destination"],
                            src_dir_fd=directory_descriptor,
                            dst_dir_fd=directory_descriptor,
                            follow_symlinks=False,
                        )
                        os.unlink(record["backup"], dir_fd=directory_descriptor)
                        raise RuntimeError(
                            f"Concept figure destination changed during publication: "
                            f"{record['destination']}"
                        )
                payload["phase"] = "backed_up"
                _write_concept_transaction_journal(directory_descriptor, payload)
                for record in payload["records"]:
                    try:
                        os.stat(
                            record["destination"],
                            dir_fd=directory_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        raise RuntimeError(
                            f"Concept figure destination appeared during publication: "
                            f"{record['destination']}"
                        )
                    os.link(
                        record["temp"],
                        record["destination"],
                        src_dir_fd=directory_descriptor,
                        dst_dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    os.unlink(record["temp"], dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
                payload["phase"] = "published"
                _write_concept_transaction_journal(directory_descriptor, payload)
            except BaseException as exc:
                try:
                    _rollback_concept_transaction(directory_descriptor, payload)
                except RuntimeError as rollback_exc:
                    raise rollback_exc from exc
                raise RuntimeError("Could not publish concept figure artifacts") from exc
    finally:
        if not journal_written:
            for record in payload["records"]:
                try:
                    _unlink_transaction_file(directory_descriptor, record["temp"])
                except (OSError, RuntimeError):
                    pass
        os.close(directory_descriptor)


def commit_concept_figure_artifact_publication(paper_dir: Path) -> None:
    directory_descriptor = _open_concept_figures_dir(paper_dir)
    try:
        payload = _read_concept_transaction_journal(directory_descriptor)
        if payload is None or payload["phase"] != "published":
            raise RuntimeError("No published concept figure artifact transaction to commit")
        _verify_published_concept_transaction(directory_descriptor, payload)
        payload["phase"] = "committed"
        _write_concept_transaction_journal(directory_descriptor, payload)
        try:
            _cleanup_concept_transaction(directory_descriptor, payload)
        except (OSError, RuntimeError):
            pass
    finally:
        os.close(directory_descriptor)


def rollback_concept_figure_artifact_publication(paper_dir: Path) -> None:
    directory_descriptor = _open_concept_figures_dir(paper_dir)
    try:
        payload = _read_concept_transaction_journal(directory_descriptor)
        if payload is not None:
            with _defer_publication_signals():
                _rollback_concept_transaction(directory_descriptor, payload)
    finally:
        os.close(directory_descriptor)


def recover_concept_figure_artifact_publication(paper_dir: Path) -> None:
    directory_descriptor = _open_concept_figures_dir(paper_dir)
    try:
        with _defer_publication_signals():
            _recover_concept_transaction(directory_descriptor, paper_dir)
    finally:
        os.close(directory_descriptor)


def _concept_figure_provenance_bytes(figure_id: str, png_data: bytes) -> bytes:
    payload = {
        "schema_name": "scientific-evidence.artifact-manifest",
        "artifact": {
            "path": f"figures/{figure_id}.png",
            "kind": "figure",
            "media_type": "image/png",
            "sha256": hashlib.sha256(png_data).hexdigest(),
        },
        "derived_from": [
            "@evidence[[paper.tex]]",
            f"@evidence[[figures/{figure_id}.prompts.json]]",
        ],
        "named_outputs": {
            "placeholder": False,
            "figure_id": figure_id,
            "renderer": "autoillustrate",
        },
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def concept_figure_provenance_matches(paper_dir: Path, figure_id: str) -> bool:
    if not is_valid_concept_figure_id(figure_id):
        return False
    try:
        figures_dir = paper_dir / "figures"
        png_data = _read_regular_leaf_nofollow(
            figures_dir, f"{figure_id}.png", 100 * 1024 * 1024, "PNG"
        )
        provenance_data = _read_regular_leaf_nofollow(
            figures_dir,
            f"{figure_id}.png.prov.json",
            10 * 1024 * 1024,
            "provenance",
        )
    except RuntimeError:
        return False
    return provenance_data == _concept_figure_provenance_bytes(figure_id, png_data)


def clear_concept_figure_provenance(paper_dir: Path, figure_id: str) -> None:
    """Remove stale renderer provenance before accepting a failed-render fallback."""
    if not is_valid_concept_figure_id(figure_id):
        raise RuntimeError(f"Invalid concept figure_id: {figure_id!r}")
    directory_descriptor = _open_concept_figures_dir(paper_dir)
    try:
        _unlink_transaction_file(
            directory_descriptor,
            f"{figure_id}.png.prov.json",
        )
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def publish_concept_figure_artifacts(
    paper_dir: Path,
    render_dir: Path,
    figure_id: str,
) -> None:
    """Validate staged Autoillustrate outputs and begin recoverable publication."""
    if not is_valid_concept_figure_id(figure_id):
        raise RuntimeError(f"Invalid concept figure_id: {figure_id!r}")
    output_rel = f"figures/{figure_id}.png"
    prompts_rel = f"figures/{figure_id}.prompts.json"
    png_data = _read_staged_concept_artifact(
        render_dir, output_rel, ".png", 100 * 1024 * 1024, "PNG"
    )
    prompts_data = _read_staged_concept_artifact(
        render_dir, prompts_rel, ".json", 10 * 1024 * 1024, "prompt log"
    )
    _validate_concept_figure_png(png_data, output_rel)
    _validate_concept_prompt_log(prompts_data, prompts_rel)
    begin_concept_figure_artifact_publication(
        paper_dir,
        [
            (f"{figure_id}.png", png_data),
            (f"{figure_id}.prompts.json", prompts_data),
            (f"{figure_id}.png.prov.json", _concept_figure_provenance_bytes(figure_id, png_data)),
        ],
    )


def publish_stashed_concept_figure_artifacts(
    paper_dir: Path,
    state_dir: Path,
    figure_id: str,
) -> None:
    if not is_valid_concept_figure_id(figure_id):
        raise RuntimeError(f"Invalid concept figure_id: {figure_id!r}")
    png_data = _read_regular_leaf_nofollow(state_dir, "figure.png", 100 * 1024 * 1024, "PNG")
    prompts_data = _read_regular_leaf_nofollow(
        state_dir, "figure.prompts.json", 10 * 1024 * 1024, "prompt log"
    )
    _validate_concept_figure_png(png_data, "figure.png")
    _validate_concept_prompt_log(prompts_data, "figure.prompts.json")
    begin_concept_figure_artifact_publication(
        paper_dir,
        [
            (f"{figure_id}.png", png_data),
            (f"{figure_id}.prompts.json", prompts_data),
            (f"{figure_id}.png.prov.json", _concept_figure_provenance_bytes(figure_id, png_data)),
        ],
    )


def _read_regular_leaf_nofollow(
    directory: Path,
    name: str,
    max_bytes: int,
    label: str,
) -> bytes:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        directory_descriptor = os.open(directory, directory_flags)
        try:
            descriptor = os.open(name, file_flags, dir_fd=directory_descriptor)
        finally:
            os.close(directory_descriptor)
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise RuntimeError(f"Invalid stashed concept figure {label}: {name}")
            data = handle.read(max_bytes + 1)
    except OSError as exc:
        raise RuntimeError(f"Invalid stashed concept figure {label}: {name}") from exc
    if len(data) > max_bytes:
        raise RuntimeError(f"Invalid stashed concept figure {label}: {name}")
    return data


def _open_artifact_parent(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(path.parent, flags)
    except OSError as exc:
        raise RuntimeError(f"Unsafe JSON artifact parent: {path.parent}") from exc


def _read_json_object_nofollow(path: Path) -> dict[str, Any]:
    parent_descriptor = _open_artifact_parent(path)
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        try:
            descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise RuntimeError(f"Could not safely open JSON artifact: {path}") from exc
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise RuntimeError(f"JSON artifact is not a regular file: {path}")
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc
    finally:
        os.close(parent_descriptor)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def validation_hard_error_fingerprint(feedback_path: Path) -> str | None:
    """Return a stable fingerprint for a validation feedback error set."""
    payload = _read_json_object_nofollow(feedback_path)
    hard_errors = payload.get("hard_errors")
    if not isinstance(hard_errors, list) or not hard_errors:
        return None
    if any(not isinstance(error, str) for error in hard_errors):
        return None
    normalized_errors = sorted(
        {
            normalized
            for error in hard_errors
            if (normalized := re.sub(r"\s+", " ", error).strip())
        }
    )
    if not normalized_errors:
        return None
    serialized = json.dumps(
        normalized_errors,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def validation_failure_fingerprint(
    feedback_path: Path,
    paper_dir: Path,
) -> str | None:
    """Fingerprint validation errors together with their current paper inputs."""
    error_fingerprint = validation_hard_error_fingerprint(feedback_path)
    if error_fingerprint is None:
        return None

    from src.orchestrator.paper_evidence_audit import validation_input_sha256

    try:
        input_hashes = validation_input_sha256(paper_dir)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise RuntimeError("Could not fingerprint paper validation inputs") from exc
    if not input_hashes:
        return None

    serialized = json.dumps(
        {
            "hard_errors": error_fingerprint,
            "paper_inputs": input_hashes,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    parent_descriptor = _open_artifact_parent(path)
    temp_name = f".{path.name}.{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temp_name, flags, 0o600, dir_fd=parent_descriptor)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temp_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        try:
            os.unlink(temp_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def _is_regular_file_nofollow(path: Path) -> bool:
    try:
        parent_descriptor = _open_artifact_parent(path)
    except RuntimeError:
        return False
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        try:
            descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        except OSError:
            return False
        try:
            return stat.S_ISREG(os.fstat(descriptor).st_mode)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _hash_regular_file_nofollow(path: Path) -> str | None:
    try:
        parent_descriptor = _open_artifact_parent(path)
    except RuntimeError:
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        try:
            descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        except OSError:
            return None
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return None
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                return hashlib.sha256(handle.read()).hexdigest()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _safe_direct_figure_artifact(
    paper_dir: Path,
    artifact_rel: str,
    expected_suffix: str,
) -> Path | None:
    relative_path = Path(artifact_rel)
    if (
        relative_path.is_absolute()
        or len(relative_path.parts) != 2
        or relative_path.parts[0] != "figures"
        or relative_path.suffix.lower() != expected_suffix
    ):
        return None
    return paper_dir / relative_path


def _read_regular_figure_artifact_nofollow(
    paper_dir: Path,
    artifact_rel: str,
    expected_suffix: str,
    read_limit: int | None,
) -> bytes | None:
    components = artifact_rel.split("/")
    if (
        len(components) < 2
        or components[0] != "figures"
        or any(component in {"", ".", ".."} or "\\" in component for component in components)
        or Path(components[-1]).suffix.lower() != expected_suffix
    ):
        return None
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        current_descriptor = os.open(paper_dir, directory_flags)
        descriptors.append(current_descriptor)
        for component in components[:-1]:
            current_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=current_descriptor,
            )
            descriptors.append(current_descriptor)
        file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        file_descriptor = os.open(
            components[-1],
            file_flags,
            dir_fd=current_descriptor,
        )
        try:
            if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                return None
            with os.fdopen(file_descriptor, "rb") as handle:
                file_descriptor = -1
                return handle.read() if read_limit is None else handle.read(read_limit)
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)
    except OSError:
        return None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _cleanup_stale_figure_quarantines(
    directory_descriptor: int,
    artifact_name: str,
) -> None:
    pattern = re.compile(rf"\.{re.escape(artifact_name)}\.[a-f0-9]{{16}}\.stale")
    for name in os.listdir(directory_descriptor):
        if pattern.fullmatch(name):
            _unlink_transaction_file(directory_descriptor, name)


@contextmanager
def _quarantine_figure_artifact_nofollow(
    paper_dir: Path,
    artifact_rel: str,
    expected_suffix: str,
) -> Iterator[None]:
    components = artifact_rel.split("/")
    if (
        len(components) < 2
        or components[0] != "figures"
        or any(component in {"", ".", ".."} or "\\" in component for component in components)
        or Path(components[-1]).suffix.lower() != expected_suffix
    ):
        raise RuntimeError(f"Unsafe figure artifact path: {artifact_rel}")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    quarantine_name = f".{components[-1]}.{secrets.token_hex(8)}.stale"
    quarantined = False
    try:
        try:
            current_descriptor = os.open(paper_dir, directory_flags)
            descriptors.append(current_descriptor)
            for component in components[:-1]:
                current_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=current_descriptor,
                )
                descriptors.append(current_descriptor)
            _cleanup_stale_figure_quarantines(
                current_descriptor,
                components[-1],
            )
            try:
                artifact_mode = os.stat(
                    components[-1],
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                ).st_mode
                if not (stat.S_ISREG(artifact_mode) or stat.S_ISLNK(artifact_mode)):
                    raise RuntimeError(
                        f"Figure artifact must be a regular file or symlink: {artifact_rel}"
                    )
                os.rename(
                    components[-1],
                    quarantine_name,
                    src_dir_fd=current_descriptor,
                    dst_dir_fd=current_descriptor,
                )
                quarantined = True
            except FileNotFoundError:
                pass
        except OSError as exc:
            raise RuntimeError(
                f"Could not safely quarantine figure artifact: {artifact_rel}"
            ) from exc
        try:
            yield
        except BaseException:
            if quarantined:
                try:
                    os.replace(
                        quarantine_name,
                        components[-1],
                        src_dir_fd=current_descriptor,
                        dst_dir_fd=current_descriptor,
                    )
                except OSError as exc:
                    raise RuntimeError(
                        f"Could not restore quarantined figure artifact: {artifact_rel}"
                    ) from exc
            raise
        else:
            if quarantined:
                try:
                    os.unlink(quarantine_name, dir_fd=current_descriptor)
                except OSError as exc:
                    raise RuntimeError(
                        f"Could not remove quarantined figure artifact: {artifact_rel}"
                    ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _concept_prompt_log_rel(figure_id: str) -> str:
    if not is_valid_concept_figure_id(figure_id):
        raise RuntimeError(f"Invalid concept figure_id: {figure_id!r}")
    return f"figures/{figure_id}.prompts.json"


def read_concept_figure_prompt_log(
    paper_dir: Path,
    figure_id: str,
    prompt_log_path: str,
) -> dict[str, Any]:
    """Read a non-placeholder prompt log from its canonical safe path."""
    expected_path = _concept_prompt_log_rel(figure_id)
    if prompt_log_path != expected_path:
        raise RuntimeError(
            f"Concept figure {figure_id!r} must use prompt_log_path {expected_path!r}"
        )
    path = paper_dir / prompt_log_path
    payload = _read_json_object_nofollow(path)
    if not payload:
        raise RuntimeError(f"Concept prompt log is an empty placeholder: {prompt_log_path}")
    _reject_unfinished_concept_prompt_log(payload, prompt_log_path)
    return payload


def _declared_manifest_pdf(row: dict[str, Any]) -> str | None:
    candidates: list[str] = []
    path_pdf = row.get("path_pdf")
    if isinstance(path_pdf, str) and path_pdf.strip():
        candidates.append(path_pdf.strip())
    files = row.get("files")
    if isinstance(files, dict):
        rich_pdf = files.get("pdf")
        if isinstance(rich_pdf, str) and rich_pdf.strip():
            candidates.append(rich_pdf.strip())
    if len(candidates) > 1:
        raise RuntimeError("Concept figure manifest row declares multiple PDF paths")
    return candidates[0] if candidates else None


def _stale_concept_pdf(
    figures: list[Any],
    row: dict[str, Any],
    figure_id: str,
) -> str | None:
    declared_pdf = _declared_manifest_pdf(row)
    if declared_pdf is None:
        return None
    canonical_pdf = f"figures/{figure_id}.pdf"
    if declared_pdf != canonical_pdf:
        raise RuntimeError(
            f"Concept figure {figure_id!r} must use canonical PDF path {canonical_pdf!r}"
        )
    for other_row in figures:
        if other_row is row or not isinstance(other_row, dict):
            continue
        if _declared_manifest_pdf(other_row) == declared_pdf:
            raise RuntimeError(
                f"Concept figure PDF {declared_pdf!r} is shared by another manifest row"
            )
    return declared_pdf


def _clear_manifest_pdf(row: dict[str, Any]) -> None:
    if "path_pdf" in row:
        row["path_pdf"] = ""
        return
    files = row.get("files")
    if isinstance(files, dict) and "pdf" in files:
        files["pdf"] = ""


def _apply_concept_figure_manifest_update(
    row: dict[str, Any],
    figure_id: str,
    *,
    generator: str | None,
    fallback_reason: str | None,
) -> None:
    row["generator"] = generator
    row["prompt_log_path"] = _concept_prompt_log_rel(figure_id)
    if fallback_reason is None:
        row.pop("attempted_generator", None)
        row.pop("fallback_reason", None)
        return
    row["attempted_generator"] = CONCEPT_FIGURE_GENERATOR
    row["fallback_reason"] = fallback_reason


def _update_concept_figure_manifest(
    paper_dir: Path,
    figure_id: str,
    *,
    generator: str | None,
    fallback_reason: str | None,
) -> None:
    manifest_path = paper_dir / "figure_manifest.json"
    manifest = _read_json_object_nofollow(manifest_path)
    figures = manifest.get("figures")
    if not isinstance(figures, list):
        raise RuntimeError("figure_manifest.json field must be a list: figures")
    matching_rows = [
        row
        for row in figures
        if isinstance(row, dict) and str(row.get("figure_id", "")).strip() == figure_id
    ]
    if len(matching_rows) != 1:
        raise RuntimeError(f"figure_manifest.json must contain exactly one row for {figure_id!r}")
    row = matching_rows[0]
    stale_pdf = (
        _stale_concept_pdf(figures, row, figure_id)
        if generator == CONCEPT_FIGURE_GENERATOR and fallback_reason is None
        else None
    )
    if stale_pdf is None:
        _apply_concept_figure_manifest_update(
            row,
            figure_id,
            generator=generator,
            fallback_reason=fallback_reason,
        )
        _write_json_atomically(manifest_path, manifest)
        return

    with _quarantine_figure_artifact_nofollow(paper_dir, stale_pdf, ".pdf"):
        _clear_manifest_pdf(row)
        _apply_concept_figure_manifest_update(
            row,
            figure_id,
            generator=generator,
            fallback_reason=fallback_reason,
        )
        _write_json_atomically(manifest_path, manifest)


def record_concept_figure_failure(
    paper_dir: Path,
    figure_id: str,
    reason: str,
) -> None:
    """Replace draft placeholder provenance with a structured render failure."""
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise RuntimeError("Concept figure failure reason must not be empty")
    prompt_log_rel = _concept_prompt_log_rel(figure_id)
    prompt_log_path = paper_dir / prompt_log_rel
    _write_json_atomically(
        prompt_log_path,
        {
            "status": "failed",
            "attempted_generator": CONCEPT_FIGURE_GENERATOR,
            "reason": normalized_reason,
            "run_log_path": "concept_figure_run.log",
        },
    )
    _update_concept_figure_manifest(
        paper_dir,
        figure_id,
        generator=None,
        fallback_reason=normalized_reason,
    )


def record_concept_figure_success(paper_dir: Path, figure_id: str) -> None:
    """Validate render provenance and restore the successful manifest state."""
    prompt_log_rel = _concept_prompt_log_rel(figure_id)
    prompt_log = read_concept_figure_prompt_log(paper_dir, figure_id, prompt_log_rel)
    if prompt_log.get("status") == "failed":
        raise RuntimeError(f"Concept prompt log still records failure: {prompt_log_rel}")
    _update_concept_figure_manifest(
        paper_dir,
        figure_id,
        generator=CONCEPT_FIGURE_GENERATOR,
        fallback_reason=None,
    )
    figures_descriptor = _open_concept_figures_dir(paper_dir)
    try:
        _cleanup_stale_figure_quarantines(
            figures_descriptor,
            f"{figure_id}.pdf",
        )
        os.fsync(figures_descriptor)
    finally:
        os.close(figures_descriptor)


def _concept_figure_fallback_row(
    paper_dir: Path,
    figure_id: str,
) -> NormalizedFigureManifestFigure:
    manifest_data = _read_regular_leaf_nofollow(
        paper_dir,
        "figure_manifest.json",
        10 * 1024 * 1024,
        "figure manifest",
    )
    try:
        manifest = json.loads(manifest_data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("Invalid figure_manifest.json") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("figure_manifest.json must contain an object")
    figures = manifest.get("figures")
    if not isinstance(figures, list):
        raise RuntimeError("figure_manifest.json field must be a list: figures")
    matching_rows = [
        row
        for row in figures
        if isinstance(row, dict) and str(row.get("figure_id", "")).strip() == figure_id
    ]
    if len(matching_rows) != 1:
        raise RuntimeError(
            f"figure_manifest.json must contain exactly one row for {figure_id!r}"
        )
    try:
        row = normalize_figure_manifest({"figures": matching_rows})[0]
    except ValidationError as exc:
        raise RuntimeError("Invalid concept figure fallback manifest row") from exc
    expected_png = f"figures/{figure_id}.png"
    expected_pdf = f"figures/{figure_id}.pdf"
    expected_prompts = _concept_prompt_log_rel(figure_id)
    if (
        row.generator is None
        or row.generator == CONCEPT_FIGURE_GENERATOR
        or row.attempted_generator != CONCEPT_FIGURE_GENERATOR
        or not isinstance(row.fallback_reason, str)
        or not row.fallback_reason.strip()
        or row.path_png != expected_png
        or row.path_pdf != expected_pdf
        or row.prompt_log_path != expected_prompts
    ):
        raise RuntimeError(f"Incomplete concept figure fallback for {figure_id!r}")
    return row


def _read_valid_concept_figure_fallback_artifacts(
    paper_dir: Path,
    figure_id: str,
) -> tuple[NormalizedFigureManifestFigure, bytes, bytes, bytes, dict[str, Any]]:
    if not is_valid_concept_figure_id(figure_id):
        raise RuntimeError(f"Invalid concept figure_id: {figure_id!r}")
    row = _concept_figure_fallback_row(paper_dir, figure_id)
    png_data = _read_regular_figure_artifact_nofollow(
        paper_dir, row.path_png, ".png", 100 * 1024 * 1024 + 1
    )
    pdf_data = _read_regular_figure_artifact_nofollow(
        paper_dir, row.path_pdf, ".pdf", 100 * 1024 * 1024 + 1
    )
    if png_data is None or len(png_data) > 100 * 1024 * 1024:
        raise RuntimeError(f"Invalid concept figure fallback PNG: {row.path_png}")
    if pdf_data is None or len(pdf_data) > 100 * 1024 * 1024:
        raise RuntimeError(f"Invalid concept figure fallback PDF: {row.path_pdf}")
    _validate_concept_figure_png(png_data, row.path_png)
    if not pdf_data.startswith(b"%PDF-") or b"%%EOF" not in pdf_data[-1024:]:
        raise RuntimeError(f"Invalid concept figure fallback PDF: {row.path_pdf}")
    assert row.prompt_log_path is not None
    prompt_data = _read_regular_figure_artifact_nofollow(
        paper_dir,
        row.prompt_log_path,
        ".json",
        10 * 1024 * 1024 + 1,
    )
    if prompt_data is None or len(prompt_data) > 10 * 1024 * 1024:
        raise RuntimeError(f"Invalid concept figure fallback prompt log: {row.prompt_log_path}")
    try:
        prompt_log = json.loads(prompt_data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid concept figure fallback prompt log: {row.prompt_log_path}"
        ) from exc
    if not isinstance(prompt_log, dict):
        raise RuntimeError(f"Invalid concept figure fallback prompt log: {row.prompt_log_path}")
    if prompt_log.get("status") == "failed":
        if (
            prompt_log.get("attempted_generator") != CONCEPT_FIGURE_GENERATOR
            or not isinstance(prompt_log.get("reason"), str)
            or not prompt_log["reason"].strip()
        ):
            raise RuntimeError(
                f"Invalid concept figure fallback prompt log: {row.prompt_log_path}"
            )
    else:
        _validate_concept_prompt_log(prompt_data, row.prompt_log_path)
    return row, png_data, pdf_data, prompt_data, prompt_log


def _read_concept_figure_repair_feedback(
    paper_dir: Path,
    figure_id: str,
) -> tuple[str, list[str]]:
    feedback_data = _read_regular_leaf_nofollow(
        paper_dir,
        "paper_evidence_audit_feedback.json",
        10 * 1024 * 1024,
        "evidence audit feedback",
    )
    try:
        feedback = json.loads(feedback_data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("Invalid evidence audit feedback") from exc
    if not isinstance(feedback, dict):
        raise RuntimeError("Invalid evidence audit feedback")
    findings = feedback.get("findings")
    if (
        not isinstance(findings, list)
        or not findings
        or (feedback.get("verdict") != "fail" and feedback.get("ok") is not False)
    ):
        raise RuntimeError("Evidence audit did not authorize a figure repair")
    canonical_paths = {
        f"figures/{figure_id}.png".lower(),
        f"figures/{figure_id}.pdf".lower(),
    }
    matching_ids: list[str] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        finding_id = finding.get("id")
        if (
            not isinstance(finding_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", finding_id) is None
        ):
            continue
        finding_text = json.dumps(finding, sort_keys=True).lower()
        if any(target in finding_text for target in canonical_paths):
            matching_ids.append(finding_id)
    if not matching_ids:
        raise RuntimeError(f"Evidence audit did not identify concept figure {figure_id!r}")
    return hashlib.sha256(feedback_data).hexdigest(), sorted(set(matching_ids))


def authorize_concept_figure_repair_feedback(
    paper_dir: Path,
    figure_id: str,
) -> tuple[str, list[str]]:
    """Return bounded, figure-specific audit authorization metadata."""
    if not is_valid_concept_figure_id(figure_id):
        raise RuntimeError(f"Invalid concept figure_id: {figure_id!r}")
    return _read_concept_figure_repair_feedback(paper_dir, figure_id)


def _concept_figure_repair_fallback_provenance_bytes(
    figure_id: str,
    png_data: bytes,
    pdf_data: bytes,
    prompt_data: bytes,
    row: NormalizedFigureManifestFigure,
    renderer: str,
    request_fingerprint: str,
    audit_feedback_sha256: str,
    audit_finding_ids: list[str],
) -> bytes:
    payload = {
        "schema_name": "scientific-evidence.artifact-manifest",
        "artifact": {
            "path": f"figures/{figure_id}.png",
            "kind": "figure",
            "media_type": "image/png",
            "sha256": hashlib.sha256(png_data).hexdigest(),
        },
        "derived_from": [
            "@evidence[[paper.tex]]",
            f"@evidence[[figures/{figure_id}.prompts.json]]",
        ],
        "named_outputs": {
            "placeholder": False,
            "figure_id": figure_id,
            "renderer": renderer,
            "attempted_generator": CONCEPT_FIGURE_GENERATOR,
            "request_fingerprint": request_fingerprint,
            "pdf_sha256": hashlib.sha256(pdf_data).hexdigest(),
            "prompt_log_sha256": hashlib.sha256(prompt_data).hexdigest(),
            "manifest_row_sha256": hashlib.sha256(
                json.dumps(asdict(row), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "audit_feedback_sha256": audit_feedback_sha256,
            "audit_finding_ids": audit_finding_ids,
        },
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def promote_concept_figure_repair_fallback(
    paper_dir: Path,
    figure_id: str,
    request_fingerprint: str,
    audit_feedback_sha256: str,
    audit_finding_ids: list[str],
) -> None:
    """Validate an audit repair fallback and issue fingerprint-bound provenance."""
    if re.fullmatch(r"[a-f0-9]{64}", request_fingerprint) is None:
        raise RuntimeError("Invalid concept figure request fingerprint")
    if re.fullmatch(r"[a-f0-9]{64}", audit_feedback_sha256) is None:
        raise RuntimeError("Invalid evidence audit feedback fingerprint")
    if not audit_finding_ids or any(
        re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", finding_id) is None
        for finding_id in audit_finding_ids
    ):
        raise RuntimeError("Invalid evidence audit finding ids")
    row, png_data, pdf_data, prompt_data, prompt_log = (
        _read_valid_concept_figure_fallback_artifacts(paper_dir, figure_id)
    )
    if prompt_log.get("status") == "failed":
        raise RuntimeError("Audit repair fallback retained a failed prompt log")
    current_feedback_sha256, current_finding_ids = _read_concept_figure_repair_feedback(
        paper_dir, figure_id
    )
    if (
        current_feedback_sha256 != audit_feedback_sha256
        or current_finding_ids != audit_finding_ids
    ):
        raise RuntimeError("Evidence audit repair authorization changed")
    provenance = json.loads(
        _concept_figure_repair_fallback_provenance_bytes(
            figure_id,
            png_data,
            pdf_data,
            prompt_data,
            row,
            row.generator or "",
            request_fingerprint,
            audit_feedback_sha256,
            audit_finding_ids,
        )
    )
    _write_json_atomically(
        paper_dir / "figures" / f"{figure_id}.png.prov.json",
        provenance,
    )


def concept_figure_repair_fallback_matches(
    paper_dir: Path,
    figure_id: str,
    request_fingerprint: str,
    audit_feedback_sha256: str,
    audit_finding_ids: list[str],
) -> bool:
    """Check a previously promoted fallback against its private request identity."""
    if (
        re.fullmatch(r"[a-f0-9]{64}", request_fingerprint) is None
        or re.fullmatch(r"[a-f0-9]{64}", audit_feedback_sha256) is None
        or not audit_finding_ids
        or any(
            re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", finding_id) is None
            for finding_id in audit_finding_ids
        )
    ):
        return False
    try:
        row, png_data, pdf_data, prompt_data, prompt_log = (
            _read_valid_concept_figure_fallback_artifacts(paper_dir, figure_id)
        )
        if prompt_log.get("status") == "failed":
            return False
        provenance_data = _read_regular_leaf_nofollow(
            paper_dir / "figures",
            f"{figure_id}.png.prov.json",
            10 * 1024 * 1024,
            "provenance",
        )
    except RuntimeError:
        return False
    expected = _concept_figure_repair_fallback_provenance_bytes(
        figure_id,
        png_data,
        pdf_data,
        prompt_data,
        row,
        row.generator or "",
        request_fingerprint,
        audit_feedback_sha256,
        audit_finding_ids,
    )
    return provenance_data == expected


def has_valid_concept_figure_fallback(paper_dir: Path, figure_id: str) -> bool:
    """Check for a complete non-concept fallback after a failed render."""
    try:
        _row, _png_data, _pdf_data, _prompt_data, _prompt_log = (
            _read_valid_concept_figure_fallback_artifacts(paper_dir, figure_id)
        )
    except RuntimeError:
        return False
    return True


def fingerprint_concept_figure_preserve_state(paper_dir: Path) -> str | None:
    """Fingerprint trusted pre-edit concept request, provenance, and artifacts."""
    request = resolve_concept_figure_request(paper_dir)
    if request is None:
        return hashlib.sha256(b'{"request":null}').hexdigest()
    figure_id = request["figure_id"]
    try:
        manifest = _read_json_object_nofollow(paper_dir / "figure_manifest.json")
    except RuntimeError:
        return None
    figures = manifest.get("figures")
    if not isinstance(figures, list):
        return None
    matching_rows = [
        row
        for row in figures
        if isinstance(row, dict) and str(row.get("figure_id", "")).strip() == figure_id
    ]
    if len(matching_rows) != 1:
        return None
    try:
        row = normalize_figure_manifest({"figures": matching_rows})[0]
    except ValidationError:
        return None

    artifact_paths: list[Path] = []
    artifact_hashes: dict[str, str] = {}
    prompt_log: dict[str, Any] | None = None
    if row.generator == CONCEPT_FIGURE_GENERATOR:
        if row.path_png != request["output_rel"] or row.prompt_log_path is None:
            return None
        try:
            prompt_log = read_concept_figure_prompt_log(
                paper_dir,
                figure_id,
                row.prompt_log_path,
            )
        except RuntimeError:
            return None
        if prompt_log.get("status") == "failed":
            return None
        png_path = _safe_direct_figure_artifact(paper_dir, row.path_png, ".png")
        if png_path is None:
            return None
        artifact_paths.append(png_path)
        if row.path_pdf.strip():
            pdf_path = _safe_direct_figure_artifact(paper_dir, row.path_pdf, ".pdf")
            if pdf_path is None:
                return None
            artifact_paths.append(pdf_path)
    elif has_valid_concept_figure_fallback(paper_dir, figure_id):
        png_data = _read_regular_figure_artifact_nofollow(
            paper_dir,
            row.path_png,
            ".png",
            None,
        )
        pdf_data = _read_regular_figure_artifact_nofollow(
            paper_dir,
            row.path_pdf,
            ".pdf",
            None,
        )
        if png_data is None or pdf_data is None:
            return None
        artifact_hashes[row.path_png] = hashlib.sha256(png_data).hexdigest()
        artifact_hashes[row.path_pdf] = hashlib.sha256(pdf_data).hexdigest()
    else:
        return None

    for artifact_path in artifact_paths:
        artifact_hash = _hash_regular_file_nofollow(artifact_path)
        if artifact_hash is None:
            return None
        artifact_hashes[artifact_path.name] = artifact_hash

    state = {
        "request": request,
        "manifest_row": matching_rows[0],
        "prompt_log": prompt_log,
        "artifact_hashes": artifact_hashes,
    }
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve_concept_figure_request(paper_dir: Path) -> dict[str, Any] | None:
    """Derive the drawio concept-figure pipeline arguments from the strategy.

    The orchestrator owns concept-figure generation as its own synchronous step
    (the writing agent no longer launches it), so every ``run.sh`` argument must
    be a pure function of artifacts that already exist on disk — no agent crafts
    them. ``paper_strategy.json`` is produced by the strategy step long before the
    paper step and already carries the full contract in ``figure_specs``:

      * ``figure_id``        -> ``--output figures/<figure_id>.png``
      * ``purpose``          -> ``--instructions`` (the strategy's own statement
                                of what the figure must convey)
      * ``preferred_span``   -> ``--placement {full-width|half-width}``

    Returns ``None`` when the strategy planned no ``generate-concept-figure``
    diagram (the figure step is then a clean no-op). The synchronous pipeline
    supports exactly one concept figure, so ambiguous multi-figure requests fail
    before any output is rendered.
    """
    strategy_path = paper_dir / "paper_strategy.json"
    try:
        strategy = read_json(strategy_path, required=False)
    except RuntimeError:
        return None

    specs = strategy.get("figure_specs")
    if not isinstance(specs, list):
        return None

    concept_specs = [
        spec
        for spec in specs
        if isinstance(spec, dict)
        and str(spec.get("preferred_generator", "")).strip() == CONCEPT_FIGURE_GENERATOR
        and spec.get("required") is not False
    ]
    if len(concept_specs) > 1:
        raise RuntimeError(
            f"paper_strategy.json may contain at most one {CONCEPT_FIGURE_GENERATOR} spec"
        )

    for spec in concept_specs:
        figure_id = str(spec.get("figure_id", "")).strip()
        if not figure_id:
            continue
        if not is_valid_concept_figure_id(figure_id):
            raise RuntimeError(
                "paper_strategy.json contains invalid figure_id for "
                f"{CONCEPT_FIGURE_GENERATOR}: {figure_id!r}"
            )

        purpose = str(spec.get("purpose", "")).strip()
        instructions = purpose or f"Overview diagram for the figure '{figure_id}'."

        span = str(spec.get("preferred_span", "")).strip().lower().replace("-", "_")
        placement = "full-width" if span == "full_width" else "half-width"

        return {
            "figure_id": figure_id,
            "output_rel": f"figures/{figure_id}.png",
            "instructions": instructions,
            "placement": placement,
        }

    return None


def classify_figure_validation_errors(errors: list[str]) -> tuple[list[str], list[str]]:
    """Split figure validation issues into hard blockers and soft readability warnings."""
    soft_patterns = (
        re.compile(r"^figures\[\d+\] exceeds max total panels:"),
        re.compile(r"^figures\[\d+\] exceeds max panels per row:"),
        re.compile(r"^figures\[\d+\] panels_per_row exceeds n_panels"),
        re.compile(r"^figures\[\d+\] exceeds category budget in panel"),
        re.compile(r"^figures\[\d+\] exceeds series budget in panel"),
        re.compile(r"^figures\[\d+\] labels too long without abbreviation"),
        re.compile(r"^figures\[\d+\]\.font_sizes\."),
        re.compile(r"^figures\[\d+\]\.min_font_size_pt below minimum:"),
        re.compile(r"^figures\[\d+\] missing font size metadata"),
        re.compile(r"^figures\[\d+\] PNG width too small:"),
        re.compile(r"^figures\[\d+\] PNG height too small:"),
        re.compile(
            r"^figures\[\d+\] paper\.tex references raster PNG "
            r"despite available vector PDF$"
        ),
    )
    hard_errors: list[str] = []
    soft_warnings: list[str] = []
    for error in errors:
        if any(pattern.match(error) for pattern in soft_patterns):
            soft_warnings.append(error)
        else:
            hard_errors.append(error)
    return hard_errors, soft_warnings


FIGURE_QUALITY_WARNING_MARKERS = (
    "PNG width too small",
    "PNG height too small",
    "references raster PNG despite available vector PDF",
)


def has_actionable_figure_quality_warnings(warnings: list[str]) -> bool:
    """Return whether validation found a figure issue suitable for targeted repair."""
    return any(
        marker in warning
        for warning in warnings
        for marker in FIGURE_QUALITY_WARNING_MARKERS
    )


def validate_figure_manifest(
    paper_dir: Path,
    *,
    tex_text: str,
    allowed_source_ids: set[str],
) -> dict[str, Any]:
    """Validate figure manifest artifacts and readability budgets."""
    strategy_path = paper_dir / "paper_strategy.json"
    manifest_path = paper_dir / "figure_manifest.json"
    errors: list[str] = []
    graphics_resolution_failed = False
    try:
        graphics_resolutions = resolve_graphics_references(tex_text, paper_dir)
    except ValueError as exc:
        errors.append(f"paper.tex graphics validation failed: {exc}")
        graphics_resolutions = []
        graphics_resolution_failed = True
    tex_graphics = [resolution.raw_path for resolution in graphics_resolutions]
    graphics_candidates: dict[str, set[str]] = {raw_path: set() for raw_path in tex_graphics}
    for resolution in graphics_resolutions:
        if resolution.resolved_path is not None:
            graphics_candidates[resolution.raw_path].add(resolution.resolved_path)
    explicitly_referenced_pngs = {
        resolution.resolved_path
        for resolution in graphics_resolutions
        if resolution.explicitly_png and resolution.resolved_path is not None
    }

    try:
        strategy = read_json(strategy_path, required=False)
    except RuntimeError as exc:
        strategy = {}
        errors.append(str(exc))

    figure_specs = strategy.get("figure_specs", [])
    if figure_specs and not isinstance(figure_specs, list):
        errors.append("paper_strategy.json field must be a list: figure_specs")
        figure_specs = []

    concept_spec_count = sum(
        1
        for spec in figure_specs
        if isinstance(spec, dict)
        and str(spec.get("preferred_generator", "")).strip() == CONCEPT_FIGURE_GENERATOR
        and spec.get("required") is not False
    )
    if concept_spec_count > 1:
        errors.append(
            f"paper_strategy.json may contain at most one {CONCEPT_FIGURE_GENERATOR} spec"
        )

    manifest_required = bool(tex_graphics) or bool(figure_specs) or manifest_path.exists() or bool(errors)
    if not manifest_required:
        return {
            "checked": False,
            "required": False,
            "figure_count": 0,
            "referenced_paths": [],
            "manifest_path": str(manifest_path.resolve()),
            "errors": errors,
            "hard_errors": [],
            "soft_warnings": [],
        }

    manifest: dict[str, Any] = {}
    figure_rows: list[NormalizedFigureManifestFigure] = []
    if not manifest_path.exists():
        errors.append(f"Missing required artifact: {manifest_path}")
    else:
        try:
            manifest = read_json(manifest_path)
        except RuntimeError as exc:
            errors.append(str(exc))
        else:
            try:
                figure_rows = normalize_figure_manifest(manifest)
            except ValidationError as exc:
                errors.append(
                    "figure_manifest.json failed schema validation: "
                    f"{format_figure_manifest_validation_error(exc)}"
                )

    spec_map: dict[str, dict[str, Any]] = {}
    for idx, spec in enumerate(figure_specs):
        if not isinstance(spec, dict):
            errors.append(f"figure_specs[{idx}] must be an object")
            continue
        figure_id = str(spec.get("figure_id", "")).strip()
        if not figure_id:
            errors.append(f"figure_specs[{idx}] missing non-empty figure_id")
            continue
        preferred_generator = str(spec.get("preferred_generator", "")).strip()
        if preferred_generator and preferred_generator not in SUPPORTED_FIGURE_GENERATORS:
            allowed = ", ".join(sorted(SUPPORTED_FIGURE_GENERATORS))
            errors.append(f"figure_specs[{idx}].preferred_generator must be one of: {allowed}")
        if preferred_generator == CONCEPT_FIGURE_GENERATOR and not is_valid_concept_figure_id(
            figure_id
        ):
            errors.append(f"figure_specs[{idx}] invalid figure_id for {CONCEPT_FIGURE_GENERATOR}")
            continue
        figure_type = _normalized_strategy_figure_type(spec.get("figure_type", ""))
        if (
            bool(spec.get("required", False))
            and figure_type == "diagram"
            and not preferred_generator
        ):
            errors.append(f"figure_specs[{idx}] missing preferred_generator for required diagram")
        spec_map[figure_id] = spec

    seen_ids: set[str] = set()
    manifest_paths: set[str] = set()

    for idx, row in enumerate(figure_rows):
        figure_id = row.figure_id
        if figure_id in seen_ids:
            errors.append(f"duplicate figure_id: {figure_id}")
        seen_ids.add(figure_id)

        path_pdf_raw = row.path_pdf.strip()
        path_png_raw = row.path_png.strip()
        figure_type = row.figure_type.strip()
        if figure_type not in SUPPORTED_FIGURE_TYPES:
            errors.append(f"figures[{idx}] unsupported figure_type: {figure_type!r}")

        png_header = _read_regular_figure_artifact_nofollow(
            paper_dir,
            path_png_raw,
            ".png",
            24,
        )
        manifest_paths.add(Path(path_png_raw).as_posix())
        # A PDF is optional for concept output and documented failed renders.
        # Validate it only when the row declares one.
        pdf_header = None
        if path_pdf_raw:
            pdf_header = _read_regular_figure_artifact_nofollow(
                paper_dir,
                path_pdf_raw,
                ".pdf",
                5,
            )
            manifest_paths.add(Path(path_pdf_raw).as_posix())
            if pdf_header is None:
                errors.append(f"figures[{idx}] missing PDF artifact: {path_pdf_raw}")
        if png_header is None:
            errors.append(f"figures[{idx}] missing PNG artifact: {path_png_raw}")

        n_panels = row.n_panels
        panels_per_row = row.panels_per_row
        categories_per_panel = row.categories_per_panel
        series_per_panel = row.series_per_panel
        max_label_chars = row.max_label_chars
        uses_abbrev_labels = row.uses_abbrev_labels
        uses_progression = row.uses_progression
        source_ids = row.source_ids
        generator = (
            row.generator.strip()
            if isinstance(row.generator, str) and row.generator.strip()
            else None
        )
        attempted_generator = (
            row.attempted_generator.strip()
            if isinstance(row.attempted_generator, str) and row.attempted_generator.strip()
            else None
        )
        prompt_log_path_raw = (
            row.prompt_log_path.strip()
            if isinstance(row.prompt_log_path, str) and row.prompt_log_path.strip()
            else None
        )
        fallback_reason = (
            row.fallback_reason.strip()
            if isinstance(row.fallback_reason, str) and row.fallback_reason.strip()
            else None
        )

        if n_panels > 4:
            errors.append(f"figures[{idx}] exceeds max total panels: {n_panels} > 4")
        if panels_per_row > 3:
            errors.append(f"figures[{idx}] exceeds max panels per row: {panels_per_row} > 3")
        elif panels_per_row > n_panels:
            errors.append(f"figures[{idx}] panels_per_row exceeds n_panels")

        category_budget = DEFAULT_CATEGORY_BUDGET.get(figure_type, 6)
        series_budget = DEFAULT_SERIES_BUDGET.get(figure_type, 4)
        spec = spec_map.get(figure_id)
        preferred_generator = ""
        if spec is not None:
            category_budget = int(spec.get("max_categories_per_panel", category_budget))
            series_budget = int(spec.get("max_series_per_panel", series_budget))
            strategy_figure_type = _normalized_strategy_figure_type(
                spec.get("figure_type", figure_type)
            )
            if figure_type != strategy_figure_type:
                errors.append(f"figures[{idx}] figure_type does not match strategy for {figure_id}")
            if n_panels > int(spec.get("max_panels", 4)):
                errors.append(f"figures[{idx}] exceeds strategy max_panels for {figure_id}")
            if uses_progression and not bool(spec.get("allow_progression", False)):
                errors.append(f"figures[{idx}] progression forbidden by strategy for {figure_id}")
            preferred_generator = str(spec.get("preferred_generator", "")).strip()
            if preferred_generator == "generate-concept-figure":
                if generator == "generate-concept-figure":
                    if prompt_log_path_raw is None:
                        errors.append(
                            f"figures[{idx}] missing prompt_log_path for generate-concept-figure output"
                        )
                    else:
                        try:
                            prompt_log = read_concept_figure_prompt_log(
                                paper_dir,
                                figure_id,
                                prompt_log_path_raw,
                            )
                        except RuntimeError as exc:
                            errors.append(f"figures[{idx}] invalid prompt log: {exc}")
                        else:
                            if prompt_log.get("status") == "failed":
                                errors.append(
                                    f"figures[{idx}] prompt log records a failed render "
                                    "but generator claims generate-concept-figure"
                                )
                elif fallback_reason is None:
                    errors.append(
                        f"figures[{idx}] bypassed preferred_generator generate-concept-figure without fallback_reason"
                    )
                elif generator is None:
                    if attempted_generator != CONCEPT_FIGURE_GENERATOR:
                        errors.append(
                            f"figures[{idx}] failed concept render missing attempted_generator"
                        )
                    if prompt_log_path_raw is None:
                        errors.append(
                            f"figures[{idx}] missing prompt_log_path for failed concept render"
                        )
                    else:
                        try:
                            prompt_log = read_concept_figure_prompt_log(
                                paper_dir,
                                figure_id,
                                prompt_log_path_raw,
                            )
                        except RuntimeError as exc:
                            errors.append(f"figures[{idx}] invalid failure prompt log: {exc}")
                        else:
                            if prompt_log.get("status") != "failed":
                                errors.append(
                                    f"figures[{idx}] fallback prompt log must record status failed"
                                )
                            if prompt_log.get("attempted_generator") != CONCEPT_FIGURE_GENERATOR:
                                errors.append(
                                    f"figures[{idx}] failure prompt log must record "
                                    f"attempted_generator {CONCEPT_FIGURE_GENERATOR}"
                                )
                            if prompt_log.get("reason") != fallback_reason:
                                errors.append(
                                    f"figures[{idx}] fallback_reason does not match prompt log"
                                )

        if attempted_generator == CONCEPT_FIGURE_GENERATOR and (
            spec is None or preferred_generator != CONCEPT_FIGURE_GENERATOR
        ):
            errors.append(
                f"figures[{idx}] records a concept render attempt without a matching strategy spec"
            )
        if generator == CONCEPT_FIGURE_GENERATOR and (
            spec is None or preferred_generator != CONCEPT_FIGURE_GENERATOR
        ):
            errors.append(
                f"figures[{idx}] claims {CONCEPT_FIGURE_GENERATOR} output without "
                "a matching strategy spec"
            )

        for panel_idx, value in enumerate(categories_per_panel):
            if value > category_budget:
                errors.append(
                    f"figures[{idx}] exceeds category budget in panel {panel_idx}: {value} > {category_budget}"
                )
        for panel_idx, value in enumerate(series_per_panel):
            if value > series_budget:
                errors.append(
                    f"figures[{idx}] exceeds series budget in panel {panel_idx}: {value} > {series_budget}"
                )

        if max_label_chars is not None and max_label_chars > 18 and uses_abbrev_labels is False:
            errors.append(f"figures[{idx}] labels too long without abbreviation")

        minimums = {"title": 10.0, "axis": 9.0, "ticks": 8.0, "legend": 8.0}
        if row.font_sizes is not None:
            for key, minimum in minimums.items():
                font_value = row.font_sizes.get(key)
                if font_value is None:
                    errors.append(f"figures[{idx}].font_sizes.{key} missing")
                elif float(font_value) < minimum:
                    errors.append(
                        f"figures[{idx}].font_sizes.{key} below minimum: {font_value} < {minimum}"
                    )
        elif row.min_font_size_pt is not None:
            minimum = 7.0 if figure_type == "diagram" else 8.0
            if row.min_font_size_pt < minimum:
                errors.append(
                    f"figures[{idx}].min_font_size_pt below minimum: {row.min_font_size_pt} < {minimum}"
                )
        else:
            errors.append(f"figures[{idx}] missing font size metadata")

        if png_header is not None:
            width, height = _read_png_size_header(png_header, path_png_raw)
            if row.placement == "full_width":
                min_width = 2400
            elif row.placement == "column":
                min_width = 1200
            else:
                min_width = 1600 if n_panels > 2 else 1200
            min_height = 900 if n_panels > 2 else 700
            if width < min_width:
                errors.append(f"figures[{idx}] PNG width too small: {width} < {min_width}")
            if height < min_height:
                errors.append(f"figures[{idx}] PNG height too small: {height} < {min_height}")

        normalized_png_path = Path(path_png_raw).as_posix()
        pdf_is_vector_candidate = pdf_header is not None and pdf_header.startswith(b"%PDF-")
        if (
            path_pdf_raw
            and pdf_is_vector_candidate
            and normalized_png_path in explicitly_referenced_pngs
        ):
            errors.append(
                f"figures[{idx}] paper.tex references raster PNG despite available vector PDF"
            )

        if not source_ids:
            if figure_type != "diagram":
                errors.append(f"figures[{idx}].source_ids must be non-empty list")
        else:
            for source_id in source_ids:
                if source_id not in allowed_source_ids:
                    errors.append(f"figures[{idx}] unknown source id: {source_id!r}")

    for figure_id, spec in spec_map.items():
        if bool(spec.get("required", False)) and figure_id not in seen_ids:
            errors.append(f"Missing required figure from strategy: {figure_id}")

    if not graphics_resolution_failed:
        referenced_manifest_paths: set[str] = set()
        for raw_path in tex_graphics:
            candidates = graphics_candidates[raw_path]
            matched = {candidate for candidate in candidates if candidate in manifest_paths}
            if not matched:
                errors.append(
                    f"paper.tex references figure missing from figure_manifest.json: {raw_path}"
                )
                continue
            referenced_manifest_paths.update(matched)

        for manifest_rel_path in sorted(manifest_paths):
            if manifest_rel_path.endswith(".png"):
                counterpart = Path(manifest_rel_path).with_suffix(".pdf").as_posix()
                if counterpart in referenced_manifest_paths:
                    continue
            if manifest_rel_path.endswith(".pdf"):
                counterpart = Path(manifest_rel_path).with_suffix(".png").as_posix()
                if counterpart in referenced_manifest_paths:
                    continue
            if manifest_rel_path not in referenced_manifest_paths and tex_graphics:
                errors.append(
                    f"figure_manifest entry not referenced by paper.tex: {manifest_rel_path}"
                )

    hard_errors, soft_warnings = classify_figure_validation_errors(errors)
    return {
        "checked": True,
        "required": manifest_required,
        "figure_count": len(figure_rows),
        "referenced_paths": sorted(manifest_paths),
        "manifest_path": str(manifest_path.resolve()),
        "errors": errors,
        "hard_errors": hard_errors,
        "soft_warnings": soft_warnings,
    }


def validate_reference_provenance(
    *,
    references_payload: dict[str, Any],
    bib_keys: set[str],
) -> tuple[set[str], list[str]]:
    """Validate references_provenance.json against references.bib."""
    errors: list[str] = []
    refs = references_payload.get("references")
    if not isinstance(refs, list):
        return set(), ["references_provenance.json must contain a `references` list"]

    provenance_keys: set[str] = set()
    provenance_ids: set[str] = set()
    for idx, ref_row in enumerate(refs):
        if not isinstance(ref_row, dict):
            errors.append(f"references[{idx}] must be an object")
            continue

        source_id = ref_row.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            errors.append(f"references[{idx}].id missing")
        else:
            if source_id in provenance_ids:
                errors.append(f"references[{idx}].id duplicates existing id: {source_id!r}")
            provenance_ids.add(source_id)

        citation_key = ref_row.get("citation_key")
        if not isinstance(citation_key, str) or not citation_key.strip():
            errors.append(f"references[{idx}].citation_key missing")
            continue
        if citation_key in provenance_keys:
            errors.append(
                f"references[{idx}].citation_key duplicates existing key: {citation_key!r}"
            )
        provenance_keys.add(citation_key)

    missing_in_provenance = sorted(bib_keys - provenance_keys)
    if missing_in_provenance:
        errors.append(
            "references.bib keys missing from references_provenance.json: "
            + ", ".join(repr(key) for key in missing_in_provenance)
        )

    missing_in_bib = sorted(provenance_keys - bib_keys)
    if missing_in_bib:
        errors.append(
            "references_provenance.json keys missing from references.bib: "
            + ", ".join(repr(key) for key in missing_in_bib)
        )

    return provenance_ids, errors


def validate_claims_grounding(
    *,
    claims_payload: dict[str, Any],
    allowed_source_ids: set[str],
    valid_citation_keys: set[str],
) -> list[str]:
    """Validate claims grounding."""
    errors: list[str] = []
    claims = claims_payload.get("claims")
    if not isinstance(claims, list) or not claims:
        return ["claims_grounding.json must contain non-empty `claims` array"]

    for idx, claim_row in enumerate(claims):
        if not isinstance(claim_row, dict):
            errors.append(f"claims[{idx}] must be an object")
            continue
        claim_text = claim_row.get("claim")
        if not isinstance(claim_text, str) or not claim_text.strip():
            errors.append(f"claims[{idx}].claim must be non-empty string")
        source_ids = claim_row.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            errors.append(f"claims[{idx}].source_ids must be non-empty list")
        else:
            for source_id in source_ids:
                if not isinstance(source_id, str) or source_id not in allowed_source_ids:
                    errors.append(f"claims[{idx}].source_ids contains unknown id: {source_id!r}")

        citation_keys = claim_row.get("citation_keys")
        if citation_keys is None:
            continue
        if not isinstance(citation_keys, list):
            errors.append(f"claims[{idx}].citation_keys must be a list when provided")
            continue
        for citation_key in citation_keys:
            if not isinstance(citation_key, str) or citation_key not in valid_citation_keys:
                errors.append(f"claims[{idx}].citation_keys contains unknown key: {citation_key!r}")
    return errors


def _is_excluded_evidence_scan_path(path: Path, *, paper_dir: Path) -> bool:
    """Return True for operational artifacts that should not enter evidence scans."""
    rel_parts = path.relative_to(paper_dir).parts
    if any(rel_parts[: len(prefix)] == prefix for prefix in AGENT_SKILL_DIR_PREFIXES):
        return True
    if any(
        rel_parts[: len(prefix)] == prefix
        for prefix in ROOT_EVIDENCE_ARCHIVE_DIR_PREFIXES
    ):
        return True
    if any(part in EVIDENCE_SKIP_DIR_NAMES for part in rel_parts):
        return True
    if len(rel_parts) == 1 and rel_parts[0] in ROOT_GENERATED_EVIDENCE_SCAN_ARTIFACT_NAMES:
        return True
    if path.name in GENERATED_EVIDENCE_SCAN_ARTIFACT_NAMES:
        return True
    if any(path.name.startswith(prefix) for prefix in GENERATED_EVIDENCE_SCAN_ARTIFACT_PREFIXES):
        return True
    return bool(PROMPT_ARTIFACT_RE.fullmatch(path.name))


def _paper_tex_dependency_paths(paper_dir: Path) -> set[Path]:
    """Return existing local files recursively imported by paper.tex."""
    root = paper_dir.resolve()
    pending = [root / "paper.tex"]
    visited: set[Path] = set()
    dependencies: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in visited or not source.is_file():
            continue
        visited.add(source)
        try:
            text = strip_false_conditionals(
                mask_tex_verbatim(_strip_tex_comments(source.read_text(encoding="utf-8")))
            )
        except (OSError, UnicodeDecodeError):
            continue
        if DYNAMIC_TEX_IMPORT_RE.search(text):
            raise RuntimeError(f"paper.tex dependency uses a dynamic import command: {source}")
        input_matches = list(TEX_INPUT_RE.finditer(text))
        parsed_import_starts = {match.start() for match in input_matches}
        if any(
            match.start() not in parsed_import_starts
            for match in TEX_IMPORT_COMMAND_RE.finditer(text)
        ):
            raise RuntimeError(f"paper.tex dependency uses a dynamic import command: {source}")
        for match in input_matches:
            target_text = (match.group("braced") or match.group("unbraced")).strip()
            if "\\" in target_text or "#" in target_text:
                raise RuntimeError(f"paper.tex dependency uses a dynamic import target: {source}")
            target = Path(target_text)
            candidates: list[Path] = []
            for base in (root, source.parent):
                candidate = Path(os.path.abspath(base / target))
                try:
                    relative_candidate = candidate.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"paper.tex dependency escapes the paper workspace: {candidate}"
                    ) from exc
                if any(
                    relative_candidate.parts[: len(prefix)] == prefix
                    for prefix in AGENT_SKILL_DIR_PREFIXES
                ):
                    raise RuntimeError(f"paper.tex must not import agent skill files: {source}")
                candidates.append(candidate)
                if not candidate.suffix:
                    candidates.append(candidate.with_suffix(".tex"))
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve(strict=True)
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"paper.tex dependency escapes the paper workspace: {candidate}"
                    ) from exc
                if resolved != candidate:
                    raise RuntimeError(f"paper.tex dependency must not use symlinks: {candidate}")
                dependencies.add(resolved)
                pending.append(resolved)
                break
        support_matches = list(TEX_LOCAL_SUPPORT_RE.finditer(text))
        parsed_support_starts = {match.start() for match in support_matches}
        if any(
            match.start() not in parsed_support_starts
            for match in TEX_LOCAL_SUPPORT_COMMAND_RE.finditer(text)
        ):
            raise RuntimeError(f"paper.tex dependency uses a dynamic support import: {source}")
        for match in support_matches:
            suffix = ".cls" if "class" in match.group("command").lower() else ".sty"
            for raw_target in match.group("targets").split(","):
                target_text = raw_target.strip()
                if "\\" in target_text or "#" in target_text:
                    raise RuntimeError(
                        f"paper.tex dependency uses a dynamic support import: {source}"
                    )
                target = Path(target_text)
                if not target.name:
                    continue
                if not target.suffix:
                    target = target.with_suffix(suffix)
                for base in (root, source.parent):
                    candidate = Path(os.path.abspath(base / target))
                    try:
                        relative_candidate = candidate.relative_to(root)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"paper.tex dependency escapes the paper workspace: {candidate}"
                        ) from exc
                    if any(
                        relative_candidate.parts[: len(prefix)] == prefix
                        for prefix in AGENT_SKILL_DIR_PREFIXES
                    ):
                        raise RuntimeError(
                            f"paper.tex must not import agent skill files: {source}"
                        )
                    if not candidate.is_file():
                        continue
                    resolved = candidate.resolve(strict=True)
                    if resolved != candidate:
                        raise RuntimeError(
                            f"paper.tex dependency must not use symlinks: {candidate}"
                        )
                    dependencies.add(resolved)
                    pending.append(resolved)
                    break
    return dependencies


def _scan_paper_dir_for_evidence_tokens(paper_dir: Path) -> list[dict[str, Any]]:
    """Return every `@evidence[[...]]` occurrence found in paper text-like files."""
    occurrences: list[dict[str, Any]] = []
    imported_paths = _paper_tex_dependency_paths(paper_dir)
    for path in sorted(paper_dir.rglob("*")):
        if not path.is_file():
            continue
        if path not in imported_paths and _is_excluded_evidence_scan_path(
            path, paper_dir=paper_dir
        ):
            continue
        if path not in imported_paths and path.suffix.lower() not in EVIDENCE_SCAN_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in EVIDENCE_REF_RE.finditer(line):
                occurrences.append(
                    {
                        "file": str(path.relative_to(paper_dir)),
                        "line": line_no,
                        "token": match.group(0),
                    }
                )
    return occurrences


def validate_evidence_bundle(paper_dir: Path) -> dict[str, Any]:
    """Trace every `@evidence[[...]]` link in the scientific-evidence bundle.

    Delegates to the canonical scientific-evidence skill verifier so the skill
    stays the single source of truth for classification semantics (`resolved`,
    `anchored`, `dangling`). The bundle is considered opt-in: when it is missing
    and no `@evidence[[...]]` tokens exist anywhere under `paper_dir`, the
    check stays silent. When it is missing but tokens exist, it fails hard
    because the agent wrote links without the canonical trail.

    Returns a structured report that callers can stash under
    `report['evidence_validation']` and surface to the paper-writing agent.
    """
    bundle_dir = paper_dir / EVIDENCE_BUNDLE_DIRNAME
    tokens_in_paper = _scan_paper_dir_for_evidence_tokens(paper_dir)
    report: dict[str, Any] = {
        "checked": False,
        "bundle_path": str(bundle_dir.resolve()),
        "bundle_present": bundle_dir.is_dir(),
        "verification_report_path": None,
        "summary": {
            "total_references": 0,
            "resolved_references": 0,
            "anchored_references": 0,
            "dangling_references": 0,
            "manifests_checked": 0,
        },
        "dangling": [],
        "paper_token_count": len(tokens_in_paper),
        "hard_errors": [],
    }

    if bundle_dir.is_symlink():
        report["hard_errors"].append(
            f"scientific-evidence bundle must be a real directory: {bundle_dir}"
        )
        return report
    if not bundle_dir.is_dir():
        if tokens_in_paper:
            first = tokens_in_paper[0]
            report["hard_errors"].append(
                f"scientific-evidence/ bundle missing under {paper_dir} but paper text carries "
                f"{len(tokens_in_paper)} `@evidence[[...]]` token(s); first occurrence: "
                f"{first['file']}:{first['line']} {first['token']}. "
                "Generate the bundle with `scientific_evidence.sh emit-bundle --root-dir <paper_dir>` "
                "or remove the evidence tokens."
            )
        return report

    if not SCIENTIFIC_EVIDENCE_SCRIPT.is_file():
        report["hard_errors"].append(
            f"scientific-evidence verifier script missing at {SCIENTIFIC_EVIDENCE_SCRIPT}"
        )
        return report

    verification_report_path = bundle_dir / "verification-report.json"
    report["checked"] = True
    report["verification_report_path"] = str(verification_report_path.resolve())
    payload, verification_error = _run_evidence_verifier_atomically(
        paper_dir=paper_dir,
        bundle_dir=bundle_dir,
        verification_report_path=verification_report_path,
    )
    if verification_error is not None or payload is None:
        report["hard_errors"].append(verification_error or "evidence verifier failed")
        return report

    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in report["summary"]:
            value = summary.get(key)
            if isinstance(value, int):
                report["summary"][key] = value

    raw_results = payload.get("results")
    results: list[Any] = raw_results if isinstance(raw_results, list) else []
    dangling: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "dangling":
            continue
        messages = item.get("messages")
        if not isinstance(messages, list):
            messages = []
        dangling.append(
            {
                "subject_path": item.get("subject_path"),
                "subject_kind": item.get("subject_kind"),
                "raw_reference": item.get("raw_reference"),
                "target": item.get("target"),
                "locator": item.get("locator"),
                "messages": [str(message) for message in messages],
            }
        )
    report["dangling"] = dangling

    if dangling:
        preview_limit = 5
        parts = [
            f"scientific-evidence verifier found {len(dangling)} dangling `@evidence[[...]]` link(s).",
        ]
        for item in dangling[:preview_limit]:
            subject = item.get("subject_path") or "<unknown-subject>"
            token = item.get("raw_reference") or item.get("target") or "<unknown-token>"
            detail = "; ".join(item.get("messages") or []) or "dangling link"
            parts.append(f"{subject} {token} -> {detail}")
        if len(dangling) > preview_limit:
            parts.append(f"(+{len(dangling) - preview_limit} more; see {verification_report_path})")
        parts.append(
            "Fix or remove every dangling link, rerun "
            "`scientific_evidence.sh verify --root-dir <paper_dir>`, and repeat "
            "until zero dangling references remain."
        )
        report["hard_errors"].append(" ".join(parts))

    return report


def _is_usable_verification_payload(
    payload: object, *, expected_root: Path
) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_name") != "scientific-evidence.verification-report":
        return False
    if not all(isinstance(payload.get(key), str) and payload[key] for key in ("verified_at", "root_dir")):
        return False
    if Path(payload["root_dir"]).expanduser().resolve() != expected_root.resolve():
        return False
    summary = payload.get("summary")
    results = payload.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        return False
    if any(
        not isinstance(summary.get(key), int)
        or isinstance(summary.get(key), bool)
        or summary[key] < 0
        for key in VERIFICATION_SUMMARY_KEYS
    ):
        return False
    statuses: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            return False
        required_strings = ("subject_path", "subject_kind", "raw_reference", "target", "status")
        if any(not isinstance(result.get(key), str) or not result[key] for key in required_strings):
            return False
        status_value = result["status"]
        if status_value not in {"resolved", "anchored", "dangling"}:
            return False
        if result["subject_kind"] not in {"text-artifact", "manifest", "ledger-entry", "other"}:
            return False
        if any(
            key in result and (not isinstance(result[key], str) or not result[key])
            for key in ("locator", "resolved_path")
        ):
            return False
        if "hash_checked" in result and not isinstance(result["hash_checked"], bool):
            return False
        messages = result.get("messages")
        if not isinstance(messages, list) or any(not isinstance(message, str) for message in messages):
            return False
        statuses.append(status_value)
    return (
        summary["total_references"] == len(results)
        and summary["resolved_references"] == statuses.count("resolved")
        and summary["anchored_references"] == statuses.count("anchored")
        and summary["dangling_references"] == statuses.count("dangling")
    )


def _run_evidence_verifier_atomically(
    *,
    paper_dir: Path,
    bundle_dir: Path,
    verification_report_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".verification-report.", suffix=".json", dir=bundle_dir
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCIENTIFIC_EVIDENCE_SCRIPT),
                "verify",
                "--root-dir",
                str(paper_dir),
                "--output",
                str(temporary_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_tail = (proc.stdout or "")[-1000:]
        stderr_tail = (proc.stderr or "")[-1000:]
        if proc.returncode not in {0, 1}:
            return None, (
                "scientific-evidence verifier failed "
                f"(rc={proc.returncode}). stdout_tail={stdout_tail!r} stderr_tail={stderr_tail!r}"
            )
        try:
            temporary_stat = temporary_path.lstat()
            payload = json.loads(temporary_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return None, f"scientific-evidence verifier produced an invalid report: {exc}"
        if not stat.S_ISREG(temporary_stat.st_mode) or temporary_stat.st_uid != os.getuid():
            return None, "scientific-evidence verifier produced an untrusted report file"
        if not _is_usable_verification_payload(payload, expected_root=paper_dir):
            return None, "scientific-evidence verifier produced a schema-invalid report"
        summary = payload["summary"]
        assert isinstance(summary, dict)
        has_dangling = summary["dangling_references"] > 0
        if (proc.returncode == 1) != has_dangling:
            return None, (
                "scientific-evidence verifier exit status disagrees with "
                "dangling reference count"
            )
        os.replace(temporary_path, verification_report_path)
        return payload, None
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_latex_recorder_inputs(
    paper_dir: Path, *, fls_name: str = "paper.fls"
) -> None:
    """Reject compiled TeX dependencies that enter staged agent skill trees."""
    root = paper_dir.resolve()
    fls_path = root / fls_name
    if not fls_path.is_file():
        raise RuntimeError(f"LaTeX recorder output missing: {fls_path}")
    try:
        recorder_lines = fls_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Could not read LaTeX recorder output: {fls_path}") from exc
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
            (
                candidate
                for candidate in workspace_candidates
                if any(
                    candidate.parts[: len(prefix)] == prefix
                    for prefix in AGENT_SKILL_DIR_PREFIXES
                )
            ),
            None,
        )
        if blocked_candidate is not None:
            raise RuntimeError(
                "compiled paper must not read staged agent skill files: "
                f"{blocked_candidate.as_posix()}"
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


def validate_bibtex_dependencies(paper_dir: Path, tex_text: str) -> None:
    """Restrict BibTeX data and local styles to trusted manuscript files."""
    root = paper_dir.resolve()
    structural_tex = strip_false_conditionals(
        mask_tex_verbatim(_strip_tex_comments(tex_text))
    )
    bibliography_matches = list(TEX_BIBLIOGRAPHY_RE.finditer(structural_tex))
    resource_matches = list(TEX_BIB_RESOURCE_RE.finditer(structural_tex))
    style_matches = list(TEX_BIBLIOGRAPHY_STYLE_RE.finditer(structural_tex))
    _reject_unparsed_tex_commands(
        structural_tex, TEX_BIBLIOGRAPHY_COMMAND_RE, bibliography_matches, "bibliography"
    )
    _reject_unparsed_tex_commands(
        structural_tex, TEX_BIB_RESOURCE_COMMAND_RE, resource_matches, "bibliography resource"
    )
    _reject_unparsed_tex_commands(
        structural_tex,
        TEX_BIBLIOGRAPHY_STYLE_COMMAND_RE,
        style_matches,
        "bibliography style",
    )
    bibliography_targets = [
        target.strip()
        for match in bibliography_matches
        for target in match.group("targets").split(",")
    ]
    bibliography_targets.extend(match.group("target").strip() for match in resource_matches)
    for target in bibliography_targets:
        _validate_canonical_bibliography_target(root, target)
    for match in style_matches:
        _validate_bibliography_style_target(root, match.group("target").strip())


def validate_bibtex_aux_dependencies(
    paper_dir: Path, *, aux_name: str = "paper.aux"
) -> None:
    """Validate the exact bibliography inputs emitted for BibTeX."""
    root = paper_dir.resolve()
    aux_texts = _read_bibtex_aux_graph(root, root / aux_name)
    data_matches: list[re.Match[str]] = []
    style_matches: list[re.Match[str]] = []
    for aux_text in aux_texts:
        current_data_matches = list(BIBTEX_AUX_DATA_RE.finditer(aux_text))
        current_style_matches = list(BIBTEX_AUX_STYLE_RE.finditer(aux_text))
        _reject_unparsed_tex_commands(
            aux_text,
            BIBTEX_AUX_DATA_COMMAND_RE,
            current_data_matches,
            "BibTeX data",
        )
        _reject_unparsed_tex_commands(
            aux_text,
            BIBTEX_AUX_STYLE_COMMAND_RE,
            current_style_matches,
            "BibTeX style",
        )
        data_matches.extend(current_data_matches)
        style_matches.extend(current_style_matches)
    if len(data_matches) != 1 or len(style_matches) != 1:
        raise RuntimeError(
            "paper.aux must declare exactly one bibliography database and style"
        )
    for target in data_matches[0].group("targets").split(","):
        _validate_canonical_bibliography_target(root, target.strip())
    _validate_bibliography_style_target(
        root, style_matches[0].group("target").strip()
    )


def _read_bibtex_aux_graph(root: Path, initial_path: Path) -> list[str]:
    pending = [initial_path]
    visited: set[Path] = set()
    aux_texts: list[str] = []
    while pending:
        aux_path = pending.pop()
        candidate = Path(os.path.abspath(aux_path))
        try:
            relative_candidate = candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("BibTeX aux input escapes the paper workspace") from exc
        if any(
            relative_candidate.parts[: len(prefix)] == prefix
            for prefix in AGENT_SKILL_DIR_PREFIXES
        ):
            raise RuntimeError("BibTeX must not read agent skill aux files")
        try:
            resolved_candidate = candidate.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"BibTeX aux input is missing or unsafe: {candidate}") from exc
        if not resolved_candidate.is_file():
            raise RuntimeError(f"BibTeX aux input is missing or unsafe: {candidate}")
        if resolved_candidate != candidate:
            raise RuntimeError(f"BibTeX aux input must not use symlinks: {candidate}")
        if candidate in visited:
            continue
        visited.add(candidate)
        aux_text = _strip_tex_comments(
            candidate.read_text(encoding="utf-8", errors="replace")
        )
        input_matches = list(BIBTEX_AUX_INPUT_RE.finditer(aux_text))
        _reject_unparsed_tex_commands(
            aux_text, BIBTEX_AUX_INPUT_COMMAND_RE, input_matches, "BibTeX aux input"
        )
        aux_texts.append(aux_text)
        pending.extend(root / match.group("target").strip() for match in input_matches)
    return aux_texts


def _reject_unparsed_tex_commands(
    tex_text: str,
    command_pattern: re.Pattern[str],
    parsed_matches: list[re.Match[str]],
    label: str,
) -> None:
    parsed_starts = {match.start() for match in parsed_matches}
    if any(match.start() not in parsed_starts for match in command_pattern.finditer(tex_text)):
        raise RuntimeError(f"paper.tex uses a dynamic {label} command")


def _validate_canonical_bibliography_target(root: Path, target_text: str) -> None:
    if "\\" in target_text or "#" in target_text:
        raise RuntimeError("paper.tex uses a dynamic bibliography target")
    target = Path(target_text)
    if not target.suffix:
        target = target.with_suffix(".bib")
    candidate = Path(os.path.abspath(root / target))
    canonical = root / "references.bib"
    if candidate != canonical:
        raise RuntimeError("paper.tex must use the canonical references.bib bibliography")
    if candidate.exists() and candidate.resolve(strict=True) != candidate:
        raise RuntimeError("paper.tex bibliography must not use symlinks")


def _validate_bibliography_style_target(root: Path, target_text: str) -> None:
    if "\\" in target_text or "#" in target_text:
        raise RuntimeError("paper.tex uses a dynamic bibliography style target")
    target = Path(target_text)
    if not target.suffix:
        target = target.with_suffix(".bst")
    candidate = Path(os.path.abspath(root / target))
    if target.parent == Path(".") and not candidate.exists():
        return
    try:
        relative_candidate = candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("paper.tex bibliography style escapes the paper workspace") from exc
    if any(
        relative_candidate.parts[: len(prefix)] == prefix
        for prefix in AGENT_SKILL_DIR_PREFIXES
    ):
        raise RuntimeError("paper.tex must not use agent skill bibliography styles")
    if not candidate.is_file():
        raise RuntimeError(f"paper.tex local bibliography style is missing: {candidate}")
    if candidate.resolve(strict=True) != candidate:
        raise RuntimeError("paper.tex bibliography style must not use symlinks")


def compile_latex(
    *,
    paper_dir: Path,
    tex_name: str = "paper.tex",
    container_image: str | None = None,
    container_network: str = "bridge",
) -> list[dict[str, Any]]:
    """Handle compile latex logic for this module."""
    tex_commands = [
        [
            "pdflatex",
            "-recorder",
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_name,
        ],
        ["bibtex", Path(tex_name).stem],
        [
            "pdflatex",
            "-recorder",
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_name,
        ],
        [
            "pdflatex",
            "-recorder",
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_name,
        ],
    ]
    commands: list[list[str]]
    if container_image:
        docker = shutil.which("docker")
        if docker is None:
            raise PaperValidationInfrastructureError(
                "Containerized LaTeX compilation requested, but docker is not available on PATH."
            )

        mount_spec = f"{paper_dir.resolve()}:/workspace"
        # Run the container as the host user so pdflatex/bibtex can write
        # paper.log/.aux/.pdf into the host-owned workspace. Without this the
        # container's default image user (uid ~1000) cannot write a workspace
        # owned by a different host uid (e.g. LDAP uids on shared hosts), so the
        # very first `pdflatex` aborts with "I can't write on file `paper.log'"
        # and every validation/compile after the concept figure is rendered
        # fails -- leaving a stale PDF that still embeds the placeholder figure.
        # Mirrors the --user flag that run_agent_prompt already passes.
        user_flags: list[str] = []
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            user_flags = ["--user", f"{os.getuid()}:{os.getgid()}"]
        commands = [
            [
                docker,
                "run",
                "--rm",
                "--network",
                container_network,
                *user_flags,
                "-v",
                mount_spec,
                "-w",
                "/workspace",
                container_image,
                *CONTAINER_CLEAN_ENV_COMMAND,
                *tex_cmd,
            ]
            for tex_cmd in tex_commands
        ]
    else:
        pdflatex = shutil.which("pdflatex")
        bibtex = shutil.which("bibtex")
        missing = []
        if pdflatex is None:
            missing.append("pdflatex")
        if bibtex is None:
            missing.append("bibtex")
        if missing:
            missing_csv = ", ".join(missing)
            raise PaperValidationInfrastructureError(
                "Missing required command(s): "
                f"{missing_csv}. Install TeX Live (pdflatex + bibtex) "
                "or rerun with container.enabled=true and set container.image."
            )
        assert pdflatex is not None
        assert bibtex is not None
        commands = [
            [
                pdflatex,
                "-recorder",
                "-no-shell-escape",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_name,
            ],
            [bibtex, Path(tex_name).stem],
            [
                pdflatex,
                "-recorder",
                "-no-shell-escape",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_name,
            ],
            [
                pdflatex,
                "-recorder",
                "-no-shell-escape",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_name,
            ],
        ]

    logs: list[dict[str, Any]] = []
    recorder_path = paper_dir / f"{Path(tex_name).stem}.fls"
    if container_image:
        command_environment = os.environ.copy()
    else:
        command_environment = {
            name: os.environ[name] for name in LATEX_ENV_ALLOWLIST if name in os.environ
        }
        command_environment["HOME"] = "/nonexistent"
        command_environment["TEXMFHOME"] = "/nonexistent"
    for cmd, tex_command in zip(commands, tex_commands, strict=True):
        uses_recorder = "-recorder" in tex_command
        if tex_command[0] == "bibtex":
            validate_bibtex_aux_dependencies(
                paper_dir, aux_name=f"{Path(tex_name).stem}.aux"
            )
        if uses_recorder:
            recorder_path.unlink(missing_ok=True)
        proc = subprocess.run(
            cmd,
            cwd=str(paper_dir),
            text=True,
            capture_output=True,
            check=False,
            env=command_environment,
        )
        logs.append(
            {
                "command": cmd,
                "returncode": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-2000:],
                "stderr_tail": (proc.stderr or "")[-2000:],
            }
        )
        if uses_recorder and (proc.returncode == 0 or recorder_path.is_file()):
            validate_latex_recorder_inputs(paper_dir, fls_name=recorder_path.name)
        if proc.returncode != 0:
            diagnostic = f"{proc.stdout}\n{proc.stderr}".lower()
            if container_image and (
                proc.returncode in {125, 126, 127, 137}
                or any(marker in diagnostic for marker in _DOCKER_INFRASTRUCTURE_ERROR_MARKERS)
            ):
                raise PaperValidationInfrastructureError(
                    f"Containerized LaTeX command could not run: {' '.join(cmd)}"
                )
            raise RuntimeError(f"LaTeX command failed: {' '.join(cmd)}")
    return logs


def validate_and_compile_paper(
    paper_dir: Path,
    *,
    container_image: str | None = None,
    container_network: str = "bridge",
) -> dict[str, Any]:
    """Validate and compile paper."""
    tex_path = paper_dir / "paper.tex"
    bib_path = paper_dir / "references.bib"
    claims_path = paper_dir / "claims_grounding.json"
    data_path = paper_dir / "paper_data.json"
    provenance_path = paper_dir / "references_provenance.json"
    report_path = paper_dir / "validation_report.json"

    report: dict[str, Any] = {
        "ok": False,
        "paper_dir": str(paper_dir.resolve()),
        "paper_profile": "default",
        "missing_sections": [],
        "citation_count": 0,
        "bibtex_key_count": 0,
        "claim_count": 0,
        "feedback": {
            "figure_validation_errors": [],
            "figure_validation_warnings": [],
            "language_validation_warnings": [],
        },
        "figure_validation": {
            "checked": False,
            "required": False,
            "figure_count": 0,
            "referenced_paths": [],
            "manifest_path": str((paper_dir / "figure_manifest.json").resolve()),
            "errors": [],
            "hard_errors": [],
            "soft_warnings": [],
        },
        "reference_validation": {
            "checked": False,
            "provenance_path": str((paper_dir / "references_provenance.json").resolve()),
            "provenance_reference_count": 0,
            "errors": [],
        },
        "evidence_validation": {
            "checked": False,
            "bundle_path": str((paper_dir / EVIDENCE_BUNDLE_DIRNAME).resolve()),
            "bundle_present": False,
            "verification_report_path": None,
            "summary": {
                "total_references": 0,
                "resolved_references": 0,
                "anchored_references": 0,
                "dangling_references": 0,
                "manifests_checked": 0,
            },
            "dangling": [],
            "paper_token_count": 0,
            "hard_errors": [],
        },
        "standalone_language_validation": {
            "checked": False,
            "paper_profile": "default",
            "hard_errors": [],
            "soft_warnings": [],
            "forbidden_terms": [],
        },
        "venue_validation": {
            "checked": False,
            "paper_format": None,
            "hard_errors": [],
            "soft_warnings": [],
        },
        "supplementary_information_validation": {
            "checked": False,
            "appendix_present": False,
            "heading_present": False,
            "placed_after_references": False,
            "pre_bibliography_clearpage_present": False,
            "clearpage_present": False,
            "visible_character_count": 0,
            "substantive_element_count": 0,
            "rendered_checked": False,
            "rendered_heading_present": False,
            "rendered_visible_character_count": 0,
            "hard_errors": [],
        },
        "page_validation": {
            "checked": False,
            "paper_format": None,
            "min_body_pages": None,
            "min_last_body_page_fill_ratio": None,
            "max_body_pages": None,
            "exclude_references": False,
            "total_pages": None,
            "reference_start_page": None,
            "supplement_start_page": None,
            "supplement_pages": 0,
            "body_pages": None,
            "last_body_page_number": None,
            "last_body_page_fill_ratio": None,
            "last_body_page_counted": None,
            "errors": [],
        },
        "watermark_validation": {
            "checked": False,
            "text": WATERMARK_TEXT,
            "applied": False,
            "changed": False,
            "hard_errors": [],
        },
        "latex_logs": [],
        "pdf_path": None,
        "error": None,
    }

    try:
        if not tex_path.exists():
            raise RuntimeError(f"Missing required artifact: {tex_path}")
        if not bib_path.exists():
            raise RuntimeError(f"Missing required artifact: {bib_path}")

        # Mandatory, host-owned watermark: (re)inserted before the manuscript is read,
        # hashed, or compiled so every published PDF carries it.
        watermark_report = report["watermark_validation"]
        watermark_report["checked"] = True
        try:
            watermark_report["changed"] = apply_paper_watermark(paper_dir)
        except PaperWatermarkError as exc:
            watermark_report["hard_errors"].append(str(exc))
            raise RuntimeError(f"paper watermark failed: {exc}") from exc

        tex_text = read_tex_file_bounded(tex_path)
        if not has_managed_watermark(tex_text):
            watermark_report["hard_errors"].append(
                "managed watermark block missing from paper.tex after insertion"
            )
            raise RuntimeError("paper watermark failed: managed block missing after insertion")
        watermark_report["applied"] = True
        bib_text = bib_path.read_text(encoding="utf-8")
        venue_brief_path = paper_dir / "VENUE.md"
        venue_brief_text = (
            venue_brief_path.read_text(encoding="utf-8") if venue_brief_path.exists() else ""
        )
        missing_sections = missing_required_sections(tex_text)
        report["missing_sections"] = missing_sections
        if missing_sections:
            raise RuntimeError(f"paper.tex missing required sections: {missing_sections}")

        supplementary_report = validate_supplementary_information(tex_text)
        report["supplementary_information_validation"] = supplementary_report
        supplementary_errors = supplementary_report.get("hard_errors", [])
        if not isinstance(supplementary_errors, list):
            raise RuntimeError(
                "Supplementary Information validation returned invalid hard_errors payload"
            )
        if supplementary_errors:
            raise RuntimeError(
                "Supplementary Information validation failed: "
                + "; ".join(str(error) for error in supplementary_errors)
            )

        validate_bibtex_dependencies(paper_dir, tex_text)

        bib_keys = extract_bibtex_keys(bib_text)
        report["bibtex_key_count"] = len(bib_keys)
        if not bib_keys:
            raise RuntimeError("references.bib did not contain any BibTeX keys")
        cited_keys = extract_tex_citations(tex_text)
        report["citation_count"] = len(cited_keys)
        unknown_citations = sorted(key for key in cited_keys if key not in bib_keys)
        if unknown_citations:
            raise RuntimeError(f"paper.tex cites unknown BibTeX keys: {unknown_citations}")

        claims_payload = read_json(claims_path)
        report["claim_count"] = len(claims_payload.get("claims", []))
        paper_data = read_json(data_path)
        references_provenance = read_json(provenance_path)
        paper_profile = str(paper_data.get("paper_profile", "default"))
        report["paper_profile"] = paper_profile
        grounding_sources = paper_data.get("grounding_sources")
        if not isinstance(grounding_sources, dict):
            raise RuntimeError("paper_data.json missing object field: grounding_sources")
        allowed_source_ids = {str(key) for key in grounding_sources}

        provenance_ids, reference_errors = validate_reference_provenance(
            references_payload=references_provenance,
            bib_keys=bib_keys,
        )
        report["reference_validation"] = {
            "checked": True,
            "provenance_path": str(provenance_path.resolve()),
            "provenance_reference_count": len(provenance_ids),
            "errors": [str(error) for error in reference_errors],
        }
        if reference_errors:
            raise RuntimeError(
                "references provenance validation failed: " + "; ".join(reference_errors)
            )
        allowed_source_ids.update(provenance_ids)

        claim_errors = validate_claims_grounding(
            claims_payload=claims_payload,
            allowed_source_ids=allowed_source_ids,
            valid_citation_keys=bib_keys,
        )
        if claim_errors:
            raise RuntimeError(
                "claims_grounding.json validation failed: " + "; ".join(claim_errors)
            )

        evidence_report = validate_evidence_bundle(paper_dir)
        report["evidence_validation"] = evidence_report
        evidence_errors = evidence_report.get("hard_errors", [])
        if not isinstance(evidence_errors, list):
            raise RuntimeError("evidence validation returned invalid hard_errors payload")
        if evidence_errors:
            raise RuntimeError(
                "scientific-evidence validation failed: "
                + "; ".join(str(error) for error in evidence_errors)
            )

        language_report = validate_visible_language(
            tex_text,
            paper_profile=paper_profile,
            presentation_contract=paper_data.get("presentation_contract", {}),
        )
        report["standalone_language_validation"] = language_report
        report["feedback"]["language_validation_warnings"] = [
            str(message) for message in language_report.get("soft_warnings", [])
        ]
        language_errors = language_report.get("hard_errors", [])
        if not isinstance(language_errors, list):
            raise RuntimeError(
                "standalone language validation returned invalid hard_errors payload"
            )
        if language_errors:
            raise RuntimeError(
                "standalone paper language validation failed: "
                + "; ".join(str(error) for error in language_errors)
            )

        paper_format = detect_paper_format(tex_text)
        venue_report = validate_venue_alignment(
            tex_text,
            venue_brief_text=venue_brief_text,
            paper_format=paper_format,
            paper_dir=paper_dir,
        )
        report["venue_validation"] = venue_report
        venue_warnings = venue_report.get("soft_warnings", [])
        if not isinstance(venue_warnings, list):
            raise RuntimeError("venue validation returned invalid soft_warnings payload")
        report["feedback"]["language_validation_warnings"].extend(
            str(message) for message in venue_warnings
        )
        venue_errors = venue_report.get("hard_errors", [])
        if not isinstance(venue_errors, list):
            raise RuntimeError("venue validation returned invalid hard_errors payload")
        if venue_errors:
            raise RuntimeError(
                "venue validation failed: " + "; ".join(str(error) for error in venue_errors)
            )

        figure_report = validate_figure_manifest(
            paper_dir,
            tex_text=tex_text,
            allowed_source_ids=allowed_source_ids,
        )
        report["figure_validation"] = figure_report
        figure_errors = figure_report.get("errors", [])
        if not isinstance(figure_errors, list):
            raise RuntimeError("figure_manifest validation returned invalid errors payload")
        report["feedback"]["figure_validation_errors"] = [str(error) for error in figure_errors]
        figure_hard_errors = figure_report.get("hard_errors", [])
        if not isinstance(figure_hard_errors, list):
            raise RuntimeError("figure_manifest validation returned invalid hard_errors payload")
        figure_soft_warnings = figure_report.get("soft_warnings", [])
        if not isinstance(figure_soft_warnings, list):
            raise RuntimeError("figure_manifest validation returned invalid soft_warnings payload")
        report["feedback"]["figure_validation_warnings"] = [
            str(message) for message in figure_soft_warnings
        ]
        if figure_hard_errors:
            raise RuntimeError(
                "figure validation failed: " + "; ".join(str(error) for error in figure_hard_errors)
            )

        latex_logs = compile_latex(
            paper_dir=paper_dir,
            tex_name="paper.tex",
            container_image=container_image,
            container_network=container_network,
        )
        report["latex_logs"] = latex_logs
        pdf_path = paper_dir / "paper.pdf"
        if not pdf_path.exists():
            raise RuntimeError(f"Expected compiled PDF missing: {pdf_path}")

        compiled_page_texts: list[str] | None = None
        try:
            compiled_page_texts = read_pdf_page_texts(pdf_path)
        except Exception as exc:
            if pdf_path.read_bytes()[:4] == b"%PDF":
                raise RuntimeError(
                    "could not inspect compiled paper.pdf for rendered "
                    f"Supplementary Information: {exc}"
                ) from exc
        if compiled_page_texts is not None:
            rendered_supplement_report = validate_rendered_supplementary_information(
                compiled_page_texts
            )
            rendered_supplement_errors = rendered_supplement_report.pop("hard_errors")
            supplementary_report.update(rendered_supplement_report)
            supplementary_report["hard_errors"].extend(rendered_supplement_errors)
            if rendered_supplement_errors:
                raise RuntimeError(
                    "rendered Supplementary Information validation failed: "
                    + "; ".join(str(error) for error in rendered_supplement_errors)
                )

        if compiled_page_texts is None:
            page_report = validate_page_budget(tex_text=tex_text, pdf_path=pdf_path)
        else:
            page_report = validate_page_budget(
                tex_text=tex_text,
                pdf_path=pdf_path,
                page_texts=compiled_page_texts,
            )
        report["page_validation"] = page_report
        page_errors = page_report.get("errors", [])
        if not isinstance(page_errors, list):
            raise RuntimeError("page validation returned invalid errors payload")
        if page_errors:
            raise RuntimeError(
                "page-limit validation failed: " + "; ".join(str(error) for error in page_errors)
            )

        report["ok"] = True
        report["pdf_path"] = str(pdf_path.resolve())
    except Exception as exc:
        report["error"] = str(exc)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        raise

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report

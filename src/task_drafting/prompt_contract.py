"""Expand repository-owned scaffold guidance before untrusted prompt substitutions."""

from pathlib import Path

EVIDENCE_CONTRACT_MARKER = "{{EVALUATOR_EVIDENCE_CONTRACT}}"
EVIDENCE_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "prompts/shared/evaluator_evidence.md"
CANDIDATE_ENVIRONMENT_MARKER = "{{CANDIDATE_ENVIRONMENT_RECIPE}}"
CANDIDATE_ENVIRONMENT_PATH = EVIDENCE_CONTRACT_PATH.with_name("candidate_environment.md")


def expand_evidence_contract(template: str) -> str:
    for marker, path in (
        (EVIDENCE_CONTRACT_MARKER, EVIDENCE_CONTRACT_PATH),
        (CANDIDATE_ENVIRONMENT_MARKER, CANDIDATE_ENVIRONMENT_PATH),
        ("{{BASELINE_FAILURE_RECIPE}}", EVIDENCE_CONTRACT_PATH.with_name("baseline_failures.md")),
        (
            "{{EVALUATOR_FEASIBILITY_RECIPE}}",
            EVIDENCE_CONTRACT_PATH.with_name("evaluator_feasibility.md"),
        ),
        ("{{WORKER_DIAGNOSTICS_RECIPE}}", EVIDENCE_CONTRACT_PATH.with_name("worker_diagnostics.md")),
        ("{{REPAIR_DIAGNOSTICS_POLICY}}", EVIDENCE_CONTRACT_PATH.with_name("repair_diagnostics.md")),
        (
            "{{EVALUATOR_REPRODUCIBILITY_RULE}}",
            EVIDENCE_CONTRACT_PATH.with_name("evaluator_reproducibility.md"),
        ),
    ):
        if marker in template:
            template = template.replace(marker, path.read_text("utf-8"))
    return template

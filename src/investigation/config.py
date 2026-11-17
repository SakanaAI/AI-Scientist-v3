"""Structured configuration dataclasses for wrapper orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .constants import (
    DEFAULT_HARNESS_EVALUATION_TIMEOUT_SECONDS,
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_CONTAINER_NETWORK,
    DEFAULT_GIT_DIR_NAME,
    DEFAULT_IDEA_TIMEOUT_SECONDS,
    DEFAULT_MAX_IDEA_GENERATION_ATTEMPTS,
    DEFAULT_MAX_PROGRAM_EVALS_PER_ATTEMPT,
    DEFAULT_PAPER_MAX_REFS,
    DEFAULT_PAPER_MAX_REPAIR_ROUNDS,
    DEFAULT_PAPER_MIN_REFS,
    DEFAULT_PAPER_TIMEOUT_GRACE_SECONDS,
    DEFAULT_PAPER_TIMEOUT_SECONDS,
    DEFAULT_RUN_TIMEOUT_GRACE_SECONDS,
    DEFAULT_RUN_TIMEOUT_SECONDS,
)
from .checkpoint_paths import validate_git_dir_name
from .files import missing_task_scaffold_files

logger = logging.getLogger(__name__)

VALID_AGENTS = ("claude", "codex", "cursor", "opencode", "gemini", "pi")
VALID_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")
VALID_PAPER_PROFILES = ("default", "standalone")
VALID_FINAL_AI_REVIEWS = ("disabled", "openai")
VALID_LOG_FORMATS = ("human", "json")
PROFILE_MODEL_ERROR = "agent.model is required when agent.profile is provided"
GEMINI_REASONING_EFFORT_ERROR = (
    "agent.reasoning_effort is not supported for gemini, including agent.pool entries"
)


@dataclass
class AgentConfig:
    name: str = field(
        default="claude",
        metadata={"help": "Agent backend (claude|codex|cursor|opencode|gemini|pi)"},
    )
    model: Optional[str] = field(
        default=None,
        metadata={"help": "Model override for the agent CLI"},
    )
    profile: Optional[str] = field(
        default=None,
        metadata={"help": "Codex CLI profile name"},
    )
    reasoning_effort: Optional[str] = field(
        default=None,
        metadata={"help": "Agent reasoning effort (unsupported by Gemini)"},
    )
    pool: list[str] = field(
        default_factory=list,
        metadata={"help": "Optional agent harness pool sampled uniformly per cycle"},
    )
    models: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Per-agent model overrides used before agent.model"},
    )


@dataclass
class PathsConfig:
    task_dir: str = field(
        default="examples/circpack",
        metadata={"help": "Task directory"},
    )
    results_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Results directory (auto-computed if omitted)"},
    )
    global_memory_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Shared memory directory across runs"},
    )
    instruct: Optional[str] = field(
        default=None,
        metadata={"help": "Prompt template path"},
    )
    paper_instruct: Optional[str] = field(
        default=None,
        metadata={"help": "Paper generation prompt template path"},
    )
    paper_contributions: Optional[str] = field(
        default=None,
        metadata={"help": "Optional contributions.md template/base path for paper generation"},
    )
    shared_memory_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Shared memory directory for cross-worker sync (set automatically by parallel runner)"},
    )
    conclusions_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Per-worker conclusions directory (auto-resolved by campaign loop / parallel runner)"},
    )
    seed_program: Optional[str] = field(
        default=None,
        metadata={"help": "Override seed program path (used internally by parallel pruning)"},
    )


@dataclass
class ExecutionConfig:
    max_idea_cycles: int = field(
        default=20,
        metadata={"help": "Max outer idea-generation cycles"},
    )
    workers: int = field(
        default=1,
        metadata={"help": "Parallel workers (1 = sequential)"},
    )
    resume: bool = field(
        default=False,
        metadata={"help": "Resume from existing state"},
    )
    draft_task_if_missing: bool = field(
        default=False,
        metadata={"help": "Draft task.md if not found"},
    )
    task_only: bool = field(
        default=False,
        metadata={"help": "Prepare and validate the task, then stop before ideation"},
    )
    max_program_evals_per_attempt: int = field(
        default=DEFAULT_MAX_PROGRAM_EVALS_PER_ATTEMPT,
        metadata={"help": "Max evaluations per attempt"},
    )
    harness_evaluated_attempts: bool = field(
        default=False,
        metadata={
            "help": (
                "When true, the agent only writes program_1 and the wrapper runs "
                "evaluate.py exactly once after the agent exits."
            )
        },
    )
    gpu_evaluation: bool = field(
        default=False,
        metadata={
            "help": "Allow baseline and harness evaluations to use explicit GPUs"
        },
    )
    run_timeout_seconds: int = field(
        default=DEFAULT_RUN_TIMEOUT_SECONDS,
        metadata={"help": "Per-run timeout in seconds"},
    )
    harness_evaluation_timeout_seconds: int = field(
        default=DEFAULT_HARNESS_EVALUATION_TIMEOUT_SECONDS,
        metadata={"help": "Host scoring timeout, separate from the agent attempt timeout"},
    )
    idea_timeout_seconds: int = field(
        default=DEFAULT_IDEA_TIMEOUT_SECONDS,
        metadata={"help": "Per-idea-generation timeout in seconds"},
    )
    max_idea_generation_attempts: int = field(
        default=DEFAULT_MAX_IDEA_GENERATION_ATTEMPTS,
        metadata={"help": "Maximum fresh idea-generation attempts per cycle"},
    )
    run_timeout_grace_seconds: int = field(
        default=DEFAULT_RUN_TIMEOUT_GRACE_SECONDS,
        metadata={"help": "Grace period after timeout in seconds"},
    )
    prune_every_n_cycles: int = field(
        default=0,
        metadata={"help": "Reset ideation context every N cycles, keeping only the best program. 0 to disable."},
    )
    parallel_fail_fast: bool = field(
        default=False,
        metadata={"help": "Stop scheduling new parallel leases after the first worker failure."},
    )
    worker_index: int = field(
        default=0,
        metadata={"help": "1-based worker index when running as a parallel worker. 0 for solo runs (set automatically by parallel runner)."},
    )
    cycle_offset: int = field(
        default=0,
        metadata={"help": "Added to the local cycle index when reporting campaign-wide cycle numbers (set automatically by parallel runner)."},
    )
    total_cycles: int = field(
        default=0,
        metadata={"help": "Campaign-wide cycle count shown in progress output. 0 = max_idea_cycles (set automatically by parallel runner)."},
    )
    isolate_worker_memory: bool = field(
        default=False,
        metadata={
            "help": (
                "Ablation: disable automatic cross-worker memory and conclusion synchronization. "
                "This does not restrict filesystem access between host workers."
            )
        },
    )


@dataclass
class TaskDraftingConfig:
    """Task-general drafting and repair limits."""

    workflow: str = "robust"
    design_max_repairs: int = 8
    contract_max_repairs: int = 3
    smoke_max_repairs: int = 3
    smoke_max_infrastructure_attempts: int = 3
    alignment_max_repairs: int = 10
    qualification_max_repairs: int = 4
    agent_transient_max_attempts: int = 4
    qualification_max_infrastructure_attempts: int = 3
    production_entry_timeout_seconds: int = 300
    max_gate_checks_per_repair: int = 12
    max_runtime_bytes: int = 16 * 1024 * 1024 * 1024
    draft_max_attempts: int = 3
    evaluator_smoke_timeout_seconds: int = 1800
    # Earlier draft directory (or "latest") whose design and scaffold seed a new draft.
    resume_from: str = ""


@dataclass
class PaperConfig:
    enabled: bool = field(
        default=False,
        metadata={"help": "Run paper generation after experiments"},
    )
    profile: str = field(
        default="default",
        metadata={"help": "Paper workflow profile (default|standalone)"},
    )
    two_stage: bool = field(
        default=False,
        metadata={"help": "Run paper framing first, then final paper drafting"},
    )
    ai_review: str = field(
        default="disabled",
        metadata={"help": "Post-publication AI review mode (disabled|openai)"},
    )
    review_rounds: int = field(
        default=0,
        metadata={"help": "Optional review/repair cycles during paper generation, even while validation is still failing"},
    )
    max_repair_rounds: int = field(
        default=DEFAULT_PAPER_MAX_REPAIR_ROUNDS,
        metadata={"help": "Maximum repair rounds for paper drafting and integrity gates"},
    )
    timeout_seconds: int = field(
        default=DEFAULT_PAPER_TIMEOUT_SECONDS,
        metadata={"help": "Paper generation timeout in seconds"},
    )
    timeout_grace_seconds: int = field(
        default=DEFAULT_PAPER_TIMEOUT_GRACE_SECONDS,
        metadata={"help": "Grace period after paper timeout in seconds"},
    )
    min_refs: int = field(
        default=DEFAULT_PAPER_MIN_REFS,
        metadata={"help": "Minimum references in paper"},
    )
    max_refs: int = field(
        default=DEFAULT_PAPER_MAX_REFS,
        metadata={"help": "Maximum references in paper"},
    )


@dataclass
class ContainerConfig:
    enabled: bool = field(
        default=False,
        metadata={"help": "Run inside Docker container"},
    )
    image: str = field(
        default=DEFAULT_CONTAINER_IMAGE,
        metadata={"help": "Docker image to use"},
    )
    network: str = field(
        default=DEFAULT_CONTAINER_NETWORK,
        metadata={"help": "Docker network mode"},
    )
    auth_root: str = field(
        default="~",
        metadata={"help": "Auth credentials root directory"},
    )
    env_file: str = field(
        default="",
        metadata={"help": "Path to .env file for container"},
    )
    gpu_count: int = field(
        default=0,
        metadata={
            "help": (
                "GPUs each container gets; with execution.workers>1 the run needs "
                "workers x gpu_count GPUs and each worker receives its own slice"
            )
        },
    )


@dataclass
class LoggingConfig:
    format: str = field(
        default="human",
        metadata={"help": "Terminal log format (human|json)"},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Show full paths, configuration, and rubric evidence"},
    )
    heartbeat_seconds: int = field(
        default=60,
        metadata={"help": "Progress heartbeat interval in seconds (0 = disabled)"},
    )
    show_run_header: bool = field(
        default=True,
        metadata={"help": "Render the user-facing run banner and configuration header"},
    )


@dataclass
class VersioningConfig:
    """Sidecar git checkpoints of the run directory."""

    enabled: bool = field(
        default=True,
        metadata={"help": "Commit the run directory to a hidden sidecar git repo"},
    )
    run_root: Optional[str] = field(
        default=None,
        metadata={"help": "Run root to checkpoint; null -> paths.results_dir"},
    )
    git_dir_name: str = field(
        default=DEFAULT_GIT_DIR_NAME,
        metadata={"help": "Suffix for the sibling sidecar git directory"},
    )
    max_file_bytes: int = field(
        default=0,
        metadata={"help": "Skip files larger than this; 0 disables the cap"},
    )
    max_total_bytes: int = field(
        default=16 * 1024**3,
        metadata={"help": "Degrade before Git when included files exceed this; 0 disables the cap"},
    )
    timeout_seconds: int = field(
        default=600,
        metadata={"help": "Shared deadline for checkpoint Git operations, in seconds"},
    )
    tag_checkpoints: bool = field(
        default=True,
        metadata={"help": "Create a lightweight checkpoint/NNNN-<slug> tag per commit"},
    )
    final_include_environments: bool = field(
        default=False,
        metadata={"help": "Final commit also includes .venv and cache directories"},
    )
    finalize_on_exit: bool = field(
        default=True,
        metadata={"help": "Mark terminal wrapper checkpoints final"},
    )


def _generate_run_id() -> str:
    """Generate a timestamp-based run identifier (e.g. 20260224_153042)."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass
class InvestigationConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    task_drafting: TaskDraftingConfig = field(default_factory=TaskDraftingConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    run_id: str = field(default_factory=_generate_run_id)
    hydra_overrides: list[str] = field(
        default_factory=list,
        repr=False,
        metadata={"internal": True},
    )
    hydra_config: dict[str, object] = field(
        default_factory=dict,
        repr=False,
        metadata={"internal": True},
    )
    hydra_runtime_config: dict[str, object] = field(
        default_factory=dict,
        repr=False,
        metadata={"internal": True},
    )


class ConfigValidationError(SystemExit):
    """Raised when configuration validation fails."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(2)


def _validate_versioning(cfg: VersioningConfig) -> None:
    """Validate checkpoint knobs, raising ConfigValidationError on problems."""
    try:
        validate_git_dir_name(cfg.git_dir_name)
    except ValueError as exc:
        raise ConfigValidationError(f"{exc}, got {cfg.git_dir_name!r}") from exc
    if cfg.max_file_bytes < 0:
        raise ConfigValidationError("versioning.max_file_bytes must be >= 0")
    if cfg.max_total_bytes < 0:
        raise ConfigValidationError("versioning.max_total_bytes must be >= 0")
    if cfg.timeout_seconds <= 0:
        raise ConfigValidationError("versioning.timeout_seconds must be > 0")
    if cfg.run_root is not None and not str(cfg.run_root).strip():
        raise ConfigValidationError("versioning.run_root must be non-empty when provided")


def validate(cfg: InvestigationConfig) -> None:
    """Validate configuration values, raising ConfigValidationError on problems."""
    if cfg.agent.name not in VALID_AGENTS:
        raise ConfigValidationError(
            f"agent.name must be one of {VALID_AGENTS}, got {cfg.agent.name!r}"
        )
    if cfg.agent.model is not None and not str(cfg.agent.model).strip():
        raise ConfigValidationError("agent.model must be non-empty when provided")
    if cfg.agent.profile is not None and not str(cfg.agent.profile).strip():
        raise ConfigValidationError("agent.profile must be non-empty when provided")
    if cfg.agent.profile is not None and cfg.agent.name != "codex":
        raise ConfigValidationError("agent.profile is only supported when agent.name='codex'")
    if cfg.agent.profile is not None and cfg.agent.model is None:
        raise ConfigValidationError(PROFILE_MODEL_ERROR)
    if (
        cfg.agent.reasoning_effort is not None
        and cfg.agent.reasoning_effort not in VALID_REASONING_EFFORTS
    ):
        raise ConfigValidationError(
            "agent.reasoning_effort must be one of "
            f"{VALID_REASONING_EFFORTS}, got {cfg.agent.reasoning_effort!r}"
        )
    if cfg.agent.reasoning_effort is not None and (
        cfg.agent.name == "gemini" or "gemini" in cfg.agent.pool
    ):
        raise ConfigValidationError(GEMINI_REASONING_EFFORT_ERROR)
    seen_pool: set[str] = set()
    for agent in cfg.agent.pool:
        if agent not in VALID_AGENTS:
            raise ConfigValidationError(
                f"agent.pool entries must be one of {VALID_AGENTS}, got {agent!r}"
            )
        if agent in seen_pool:
            raise ConfigValidationError(f"agent.pool entries must be unique, got duplicate {agent!r}")
        seen_pool.add(agent)
    for agent, model in cfg.agent.models.items():
        if agent not in VALID_AGENTS:
            raise ConfigValidationError(
                f"agent.models keys must be one of {VALID_AGENTS}, got {agent!r}"
            )
        if not isinstance(model, str) or not model.strip():
            raise ConfigValidationError(
                f"agent.models[{agent!r}] must be a non-empty string"
            )
    if cfg.execution.max_idea_cycles <= 0:
        raise ConfigValidationError("execution.max_idea_cycles must be > 0")
    if cfg.execution.workers <= 0:
        raise ConfigValidationError("execution.workers must be > 0")
    if cfg.execution.run_timeout_seconds <= 0:
        raise ConfigValidationError("execution.run_timeout_seconds must be > 0")
    if (
        type(cfg.execution.harness_evaluation_timeout_seconds) is not int
        or cfg.execution.harness_evaluation_timeout_seconds <= 0
    ):
        raise ConfigValidationError("execution.harness_evaluation_timeout_seconds must be > 0")
    if cfg.execution.idea_timeout_seconds <= 0:
        raise ConfigValidationError("execution.idea_timeout_seconds must be > 0")
    if cfg.execution.max_idea_generation_attempts <= 0:
        raise ConfigValidationError("execution.max_idea_generation_attempts must be > 0")
    if cfg.execution.run_timeout_grace_seconds <= 0:
        raise ConfigValidationError("execution.run_timeout_grace_seconds must be > 0")
    if cfg.execution.max_program_evals_per_attempt <= 0:
        raise ConfigValidationError("execution.max_program_evals_per_attempt must be > 0")
    if cfg.task_drafting.workflow not in {"robust", "legacy"}:
        raise ConfigValidationError("task_drafting.workflow must be robust or legacy")
    drafting_limits = {
        "design_max_repairs": (cfg.task_drafting.design_max_repairs, 0, 8),
        "contract_max_repairs": (cfg.task_drafting.contract_max_repairs, 0, 3),
        "smoke_max_repairs": (cfg.task_drafting.smoke_max_repairs, 0, 3),
        "smoke_max_infrastructure_attempts": (
            cfg.task_drafting.smoke_max_infrastructure_attempts,
            1,
            3,
        ),
        "alignment_max_repairs": (cfg.task_drafting.alignment_max_repairs, 0, 10),
        "draft_max_attempts": (cfg.task_drafting.draft_max_attempts, 1, 3),
        "evaluator_smoke_timeout_seconds": (
            cfg.task_drafting.evaluator_smoke_timeout_seconds,
            1,
            7200,
        ),
        "qualification_max_repairs": (
            cfg.task_drafting.qualification_max_repairs,
            0,
            8,
        ),
        "agent_transient_max_attempts": (
            cfg.task_drafting.agent_transient_max_attempts,
            1,
            4,
        ),
        "qualification_max_infrastructure_attempts": (
            cfg.task_drafting.qualification_max_infrastructure_attempts,
            1,
            3,
        ),
        "production_entry_timeout_seconds": (
            cfg.task_drafting.production_entry_timeout_seconds,
            30,
            1800,
        ),
        "max_gate_checks_per_repair": (
            cfg.task_drafting.max_gate_checks_per_repair,
            1,
            12,
        ),
        "max_runtime_bytes": (
            cfg.task_drafting.max_runtime_bytes,
            1,
            64 * 1024 * 1024 * 1024,
        ),
    }
    for name, (value, minimum, maximum) in drafting_limits.items():
        if type(value) is not int or not minimum <= value <= maximum:
            raise ConfigValidationError(
                f"task_drafting.{name} must be an integer from {minimum} to {maximum}"
            )
    task_dir = Path(str(cfg.paths.task_dir)).expanduser()
    contracted_task = (task_dir / "task_contract.json").is_file()
    drafting_pending = cfg.execution.draft_task_if_missing and bool(
        missing_task_scaffold_files(task_dir, require_contract_v1=False)
    )
    if (
        cfg.execution.gpu_evaluation
        and not cfg.execution.harness_evaluated_attempts
        and not contracted_task
        and not drafting_pending
    ):
        raise ConfigValidationError(
            "execution.gpu_evaluation requires execution.harness_evaluated_attempts=true"
        )
    if cfg.execution.prune_every_n_cycles < 0:
        raise ConfigValidationError("execution.prune_every_n_cycles must be >= 0")
    if cfg.execution.worker_index < 0:
        raise ConfigValidationError("execution.worker_index must be >= 0")
    if cfg.execution.cycle_offset < 0:
        raise ConfigValidationError("execution.cycle_offset must be >= 0")
    if cfg.execution.total_cycles < 0:
        raise ConfigValidationError("execution.total_cycles must be >= 0")
    if cfg.paper.profile not in VALID_PAPER_PROFILES:
        raise ConfigValidationError(
            f"paper.profile must be one of {VALID_PAPER_PROFILES}, got {cfg.paper.profile!r}"
        )
    if cfg.paper.ai_review not in VALID_FINAL_AI_REVIEWS:
        raise ConfigValidationError(
            "paper.ai_review must be one of "
            f"{VALID_FINAL_AI_REVIEWS}, got {cfg.paper.ai_review!r}"
        )
    if cfg.paper.review_rounds < 0:
        raise ConfigValidationError("paper.review_rounds must be >= 0")
    if cfg.paper.max_repair_rounds < 0:
        raise ConfigValidationError("paper.max_repair_rounds must be >= 0")
    if cfg.paper.timeout_seconds <= 0:
        raise ConfigValidationError("paper.timeout_seconds must be > 0")
    if cfg.paper.timeout_grace_seconds <= 0:
        raise ConfigValidationError("paper.timeout_grace_seconds must be > 0")
    if cfg.paper.min_refs <= 0:
        raise ConfigValidationError("paper.min_refs must be > 0")
    if cfg.paper.max_refs < cfg.paper.min_refs:
        raise ConfigValidationError("paper.max_refs must be >= paper.min_refs")
    if not str(cfg.container.image).strip():
        raise ConfigValidationError("container.image must be non-empty")
    if not str(cfg.container.network).strip():
        raise ConfigValidationError("container.network must be non-empty")
    if not str(cfg.container.auth_root).strip():
        raise ConfigValidationError("container.auth_root must be non-empty")
    if cfg.container.env_file and not str(cfg.container.env_file).strip():
        raise ConfigValidationError("container.env_file must be non-empty when provided")
    if isinstance(cfg.container.gpu_count, bool) or cfg.container.gpu_count < 0:
        raise ConfigValidationError("container.gpu_count must be a non-negative integer")
    if cfg.logging.format not in VALID_LOG_FORMATS:
        raise ConfigValidationError(
            f"logging.format must be one of {VALID_LOG_FORMATS}, got {cfg.logging.format!r}"
        )
    if cfg.logging.heartbeat_seconds < 0:
        raise ConfigValidationError("logging.heartbeat_seconds must be >= 0")
    _validate_versioning(cfg.versioning)
    if cfg.execution.resume:
        pruning = cfg.execution.prune_every_n_cycles
        resume_mode_supported = (
            cfg.execution.workers == 1 and pruning == 0
        ) or 0 < pruning < cfg.execution.max_idea_cycles
        if not resume_mode_supported:
            qualifier = "parallel " if cfg.execution.workers > 1 else "pruned "
            raise ConfigValidationError(
                f"{qualifier}resume requires workers=1 with pruning disabled or "
                "0 < execution.prune_every_n_cycles < execution.max_idea_cycles"
            )
    if cfg.execution.workers > 1 and cfg.execution.draft_task_if_missing:
        raise ConfigValidationError("draft_task_if_missing is only supported with workers=1")
    if cfg.execution.task_only and cfg.execution.resume:
        raise ConfigValidationError("execution.task_only cannot resume an investigation run")
    if (
        cfg.execution.prune_every_n_cycles > 0
        and cfg.execution.workers > 1
        and cfg.execution.prune_every_n_cycles % cfg.execution.workers != 0
    ):
        raise ConfigValidationError(
            f"execution.prune_every_n_cycles ({cfg.execution.prune_every_n_cycles}) "
            f"must be a multiple of execution.workers ({cfg.execution.workers}) "
            "when parallel pruning is enabled"
        )


def config_to_flat_dict(cfg: InvestigationConfig) -> dict[str, str]:
    """Flatten a InvestigationConfig into a dict with dotted keys for display.

    Example: {"agent.name": "claude", "execution.workers": "1", ...}
    """
    import dataclasses

    flat: dict[str, str] = {}
    for section_field in dataclasses.fields(cfg):
        if section_field.metadata.get("internal", False):
            continue
        section = getattr(cfg, section_field.name)
        if dataclasses.is_dataclass(section):
            for param_field in dataclasses.fields(section):
                key = f"{section_field.name}.{param_field.name}"
                value = getattr(section, param_field.name)
                flat[key] = str(value) if value is not None else "null"
        else:
            flat[section_field.name] = str(section) if section is not None else "null"
    return flat


def has_hydra_override(cfg: InvestigationConfig, dotted_key: str) -> bool:
    """Return true when the user explicitly supplied a Hydra override for a key."""
    override_prefixes = (
        f"{dotted_key}=",
        f"+{dotted_key}=",
        f"++{dotted_key}=",
    )
    deletion_forms = (f"~{dotted_key}", f"~{dotted_key}=")
    for raw_override in cfg.hydra_overrides:
        override = raw_override.strip()
        if override.startswith(override_prefixes):
            return True
        if override == deletion_forms[0] or override.startswith(deletion_forms[1]):
            return True
    return False


def should_use_harness_instruction(cfg: InvestigationConfig) -> bool:
    """Return true when harness mode should select its prompt template."""
    if not cfg.execution.harness_evaluated_attempts:
        return False
    if has_hydra_override(cfg, "paths.instruct"):
        return False
    if cfg.paths.instruct and not cfg.hydra_overrides:
        return False
    return True


def resolve_defaults(cfg: InvestigationConfig) -> InvestigationConfig:
    """Compute derived defaults (results_dir from task_dir + agent.name when null).

    Returns the same config object, mutated in place for convenience.
    """
    if cfg.paper.profile == "standalone" and cfg.paths.paper_instruct is None:
        cfg.paths.paper_instruct = str(Path("src") / "prompts" / "paper_standalone.md")

    if cfg.paths.results_dir is None:
        agent_dir = "ensemble" if cfg.agent.pool else cfg.agent.name
        task_name = Path(cfg.paths.task_dir).name.strip() or "task"
        cfg.paths.results_dir = str(
            Path("results") / task_name / agent_dir / cfg.run_id
        )
        logger.debug(
            "Resolved results_dir to %s (from task_dir=%s, agent=%s, run_id=%s)",
            cfg.paths.results_dir,
            cfg.paths.task_dir,
            agent_dir,
            cfg.run_id,
        )
    return cfg

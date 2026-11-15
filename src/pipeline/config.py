"""Configuration dataclasses for the end-to-end pipeline orchestrator."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import stat
import copy
import functools
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, make_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    Mapping,
    Optional,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from src.investigation.config import (
    VALID_AGENTS,
    PROFILE_MODEL_ERROR,
    GEMINI_REASONING_EFFORT_ERROR,
    VALID_REASONING_EFFORTS,
    ContainerConfig,
    LoggingConfig,
    VersioningConfig,
)
from src.investigation.checkpoint_paths import validate_git_dir_name
from src.investigation.program_spec import entrypoint_relpath_from_spec
from src.investigation.task_contract import (
    TASK_CONTRACT_FILENAME,
    TaskContractError,
    load_task_contract,
)

from src.orchestrator.paper_venues import find_venue, venue_names
from .safe_fs import UnsafeGluePathError, open_directory
from .investigation_layout import investigation_workspace
from .portable_publication import portable_replacement_is_recoverable

logger = logging.getLogger(__name__)
_INVESTIGATION_COMPLETION_RECEIPT = ".investigation_completion.json"
_DOWNSTREAM_PHASES = {"ablations", "final_evaluation", "posthoc", "paper", "review", "revision"}
_BASELINE_SOURCE_ATTEMPT_IDS = frozenset({"bootstrap", "parallel_bootstrap"})

# Ordered phases in the end-to-end pipeline. These are distinct from the
# variants pipeline's internal numbered stages.
PIPELINE_PHASES = (
    "task_drafting",
    "investigation",
    "ablations",
    "final_evaluation",
    "posthoc",
    "paper",
    "review",
    "revision",
)
PAPER_FORMATS = venue_names()
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MAX_INVESTIGATION_STATE_BYTES = 1024 * 1024


def _generate_run_id() -> str:
    """Generate a timestamp-based run identifier (e.g. 20260224_153042)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass
class PipelineTaskConfig:
    """The task under study; the single source of truth for the task name."""

    dir: str = field(
        default="examples/circpack",
        metadata={"help": "Immutable task source directory; task_name = its basename"},
    )


@dataclass
class TaskDraftingConfig:
    """Pipeline-owned task drafting settings.

    Repair budgets and the workflow are investigation CLI keys, reachable as
    ``investigation.task_drafting.<key>``; only the overall deadline is the
    orchestrator's own (it becomes the drafting command's run timeout).
    """

    timeout_seconds: int = field(
        default=57600,
        metadata={"help": "Overall task drafting, smoke, review, and repair deadline"},
    )


@dataclass
class PipelineAgentConfig:
    """Agent selection shared by every phase."""

    name: str = field(default="claude", metadata={"help": f"Agent harness: {VALID_AGENTS}"})
    model: Optional[str] = field(
        default=None,
        metadata={"help": "Model name; null uses the harness's selected/configured default"},
    )
    profile: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Codex CLI profile"},
    )
    reasoning_effort: Optional[str] = field(
        default=None,
        metadata={"help": "Optional effort: low|medium|high|xhigh (not Gemini)"},
    )


@dataclass
class PipelineContainerConfig(ContainerConfig):
    """Container defaults for the orchestrator (stricter than the investigation CLI's)."""

    enabled: bool = field(
        default=True,
        metadata={"help": "Run every agent phase inside Docker"},
    )
    network: str = field(
        default="host",
        metadata={"help": "Docker network mode; the post-hoc phase uses its own narrower network"},
    )
    env_file: str = field(
        default=".env",
        metadata={"help": "Path to the .env file mounted into the container"},
    )
    gpu_count: int = field(
        default=0,
        metadata={
            "help": (
                "GPUs the task requires; compute phases need AGENT_CONTAINER_GPUS to "
                "expose exactly this many GPU UUIDs (0 requires it unset)"
            )
        },
    )
    paper_enabled: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Run the paper phase in Docker; null follows container.enabled. true keeps "
                "a native campaign's paper in the image, which has pdflatex/bibtex and runs "
                "the read-only evidence audit"
            )
        },
    )


def phase_uses_container(cfg: "PipelineConfig", phase: str) -> bool:
    """Whether ``phase`` runs its agents in Docker; only the paper phase can differ from the run."""
    if phase == "paper" and cfg.container.paper_enabled is not None:
        return cfg.container.paper_enabled
    return cfg.container.enabled


PAPER_NEEDS_CONTAINER_ERROR = (
    "the paper phase needs a container: its evidence audit refuses native agents and the "
    "image provides pdflatex/bibtex; set container.paper_enabled=true to run only the paper "
    "in Docker"
)


@dataclass
class PipelinePathsConfig:
    """Path configuration and resume anchors for the orchestrator."""

    results_root: str = field(
        default="results",
        metadata={"help": "Root under which investigation results are written"},
    )
    run_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit pipeline run directory / resume anchor"},
    )
    ablations_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit ablations directory"},
    )
    posthoc_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit post-hoc directory / optional paper input"},
    )
    final_evaluation_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit one-time final evaluation directory"},
    )
    paper_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit paper output directory"},
    )


@dataclass
class ReviewConfig:
    """Execution bounds for the external final-manuscript review phase."""

    timeout_seconds: int = field(
        default=1800, metadata={"help": "Review subprocess deadline in seconds"}
    )
    timeout_grace_seconds: int = field(
        default=30, metadata={"help": "Grace window before the review subprocess is killed"}
    )


@dataclass
class FinalConfig:
    """Execution bound for the contract-declared final evaluator."""

    timeout_seconds: int = field(
        default=21600, metadata={"help": "Final evaluation deadline in seconds"}
    )


PHASE_SECTIONS = ("investigation", "ablations", "posthoc", "paper")
_CONFIGS_DIR = str(REPO_ROOT / "configs")
T = TypeVar("T")


def flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping into dotted keys; empty mappings stay leaves."""
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            flat.update(flatten_mapping(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


@functools.lru_cache(maxsize=None)
def _composed_plain(config_name: str) -> Mapping[str, Any]:
    """Compose one config from ``configs/`` with no overrides (cached, read-only)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    if config_name == "pipeline_config":
        register_pipeline_schema()
    with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base="1.3"):
        raw = compose(config_name=config_name)
    plain = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(plain, dict)
    return cast(Mapping[str, Any], plain)


def stage_baseline(config_name: str) -> dict[str, Any]:
    """The stage's own defaults, flattened: the diff baseline for forwarding."""
    return copy.deepcopy(flatten_mapping(_composed_plain(config_name)))


def pipeline_phase_defaults(prefix: str) -> dict[str, Any]:
    """The composed phase subtree of ``pipeline_config.yaml`` with no overrides."""
    section = _composed_plain("pipeline_config")[prefix]
    assert isinstance(section, Mapping)
    return copy.deepcopy(flatten_mapping(section))


_BOOL_WORDS = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "on": True,
    "off": False,
    "1": True,
    "0": False,
}


def _coerce_like(value: Any, reference: Any) -> Any:
    """Coerce a string to the type of the stage's own default, OmegaConf-style.

    Stage subtrees are free-form in the schema, so an ``${oc.env:...}``
    interpolation or a quoted override arrives as a string. The stage default
    tells us the intended scalar type; anything that does not convert is left
    alone for validation to report.
    """
    if not isinstance(value, str) or reference is None or isinstance(reference, str):
        return value
    text = value.strip()
    if isinstance(reference, bool):
        return _BOOL_WORDS.get(text.lower(), value)
    if isinstance(reference, int):
        try:
            return int(text)
        except ValueError:
            return value
    if isinstance(reference, float):
        try:
            return float(text)
        except ValueError:
            return value
    return value


def _scalar_type_errors(prefix: str, values: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """Report values whose type disagrees with the stage's own default."""
    errors: list[str] = []
    for key, value in sorted(values.items()):
        reference = baseline.get(key)
        if reference is None or isinstance(reference, str) or value is None:
            continue
        if isinstance(reference, bool):
            if type(value) is not bool:
                errors.append(f"{prefix}.{key} must be a boolean")
        elif isinstance(reference, int):
            if type(value) is not int:
                errors.append(f"{prefix}.{key} must be an integer")
        elif isinstance(reference, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{prefix}.{key} must be a number")
        elif isinstance(reference, list):
            if not isinstance(value, list):
                errors.append(f"{prefix}.{key} must be a list")
    return errors


class _Key(Generic[T]):
    """Typed read/write view of one flattened stage key on a :class:`PhasePassthrough`."""

    def __init__(self, key: str) -> None:
        self.key = key

    @overload
    def __get__(self, obj: None, owner: type | None = None) -> "_Key[T]": ...

    @overload
    def __get__(self, obj: "PhasePassthrough", owner: type | None = None) -> T: ...

    def __get__(self, obj: "PhasePassthrough | None", owner: type | None = None) -> "_Key[T] | T":
        if obj is None:
            return self
        return cast(T, obj.get(self.key))

    def __set__(self, obj: "PhasePassthrough", value: T) -> None:
        obj.set(self.key, value)


class PhasePassthrough:
    """One pipeline phase, backed entirely by its downstream stage's composed config.

    A phase declares no config keys of its own: ``investigation.execution.workers``
    is the investigation CLI's key, ``ablations.execution.max_ablations`` is the variants
    pipeline's, ``paper.paper_review_rounds`` is a paper.sh flag. ``values`` is
    the whole flattened subtree and :attr:`forwarded` the subset that differs
    from the stage's own defaults. The typed ``_Key`` views on the subclasses
    are read/write windows for orchestrator logic, so using one never creates a
    second name for a setting. Adding a key to a stage therefore needs no change
    in this file.
    """

    #: Config prefix in error messages and overrides (``"investigation"``).
    prefix: ClassVar[str] = ""
    #: The stage's own config name under ``configs/``.
    stage_config_name: ClassVar[str] = ""
    #: Keys (or ``"section."`` prefixes) the orchestrator derives; setting one
    #: to a value that differs from the stage default is a config error.
    owned: ClassVar[tuple[str, ...]] = ()
    #: User-settable keys whose value the builder derives itself, so they are
    #: never forwarded by the generic rule.
    rewritten: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        if values is None:
            self.values: dict[str, Any] = pipeline_phase_defaults(self.prefix)
        else:
            baseline = stage_baseline(self.stage_config_name)
            self.values = {
                key: _coerce_like(value, baseline.get(key)) for key, value in values.items()
            }

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PhasePassthrough) and (
            type(other) is type(self) and other.values == self.values
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.values!r})"

    @property
    def baseline(self) -> dict[str, Any]:
        """Flattened stage defaults (the stage config composed alone)."""
        return stage_baseline(self.stage_config_name)

    def get(self, key: str) -> Any:
        if key in self.values:
            return self.values[key]
        baseline = self.baseline
        if key in baseline:
            return baseline[key]
        raise KeyError(f"{self.prefix}.{key} is not a {self.stage_config_name} key")

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    @classmethod
    def owned_entry(cls, key: str) -> str | None:
        for entry in cls.owned:
            if key == entry or (entry.endswith(".") and key.startswith(entry)):
                return entry
        return None

    @property
    def changed(self) -> dict[str, Any]:
        """Keys whose composed value differs from the stage's own default."""
        baseline = self.baseline
        return {
            key: value
            for key, value in sorted(self.values.items())
            if key not in baseline or baseline[key] != value
        }

    @property
    def forwarded(self) -> dict[str, Any]:
        """Stage overrides to emit: changed, not owned, not rewritten."""
        return {
            key: value
            for key, value in self.changed.items()
            if self.owned_entry(key) is None and key not in self.rewritten
        }

    def forwarded_excluding(self, *prefixes: str) -> dict[str, Any]:
        """:attr:`forwarded` without keys under the given dotted prefixes."""
        return {
            key: value
            for key, value in self.forwarded.items()
            if not key.startswith(prefixes)
        }

    @property
    def conflicting(self) -> list[str]:
        """Changed keys the orchestrator owns."""
        return [key for key in self.changed if self.owned_entry(key) is not None]

    def type_errors(self) -> list[str]:
        """Values whose type disagrees with the stage's own default."""
        return _scalar_type_errors(self.prefix, self.values, self.baseline)

    def conflict_errors(self) -> list[str]:
        return [
            f"{self.prefix}.{key} is set by the pipeline and cannot be overridden"
            for key in self.conflicting
        ]


class InvestigationConfig(PhasePassthrough):
    """The investigation phase: ``src.investigation.cli``, ``configs/config.yaml``."""

    prefix = "investigation"
    stage_config_name = "config"
    owned = (
        "paths.task_dir",
        "paths.results_dir",
        "run_id",
        "agent.",
        "container.",
        "logging.",
        "versioning.",
        "execution.task_only",
        "execution.worker_index",
        "paper.",
    )
    rewritten = frozenset({"execution.draft_task_if_missing"})

    resume = _Key[bool]("execution.resume")
    draft_task_if_missing = _Key[bool]("execution.draft_task_if_missing")
    max_idea_cycles = _Key[int]("execution.max_idea_cycles")
    workers = _Key[int]("execution.workers")
    prune_every_n_cycles = _Key[int]("execution.prune_every_n_cycles")
    max_program_evals_per_attempt = _Key[int]("execution.max_program_evals_per_attempt")
    harness_evaluated_attempts = _Key[bool]("execution.harness_evaluated_attempts")
    gpu_evaluation = _Key[bool]("execution.gpu_evaluation")
    max_idea_generation_attempts = _Key[int]("execution.max_idea_generation_attempts")
    idea_timeout_seconds = _Key[int]("execution.idea_timeout_seconds")
    run_timeout_seconds = _Key[int]("execution.run_timeout_seconds")
    harness_evaluation_timeout_seconds = _Key[int](
        "execution.harness_evaluation_timeout_seconds"
    )
    drafting_workflow = _Key[str]("task_drafting.workflow")
    production_entry_timeout_seconds = _Key[int](
        "task_drafting.production_entry_timeout_seconds"
    )
    max_runtime_bytes = _Key[int]("task_drafting.max_runtime_bytes")


class AblationsConfig(PhasePassthrough):
    """The ablations phase: variants pipeline, ``configs/variants_config.yaml``."""

    prefix = "ablations"
    stage_config_name = "variants_config"
    owned = (
        "paths.task_dir",
        "paths.task_name",
        "paths.results_dir",
        "paths.output_dir",
        "agent.",
        "container.",
        "logging.",
    )
    #: ``null`` variant deadlines keep the variants default, so the builder
    #: derives the forwarded value.
    rewritten = frozenset(
        {
            "execution.sandbox_image",
            "execution.variant_timeout_seconds",
            "execution.cross_task_timeout_seconds",
        }
    )

    num_replications = _Key[int]("execution.num_replications")
    evaluation_timeout_seconds = _Key[Optional[int]]("execution.evaluation_timeout_seconds")
    variant_timeout_seconds = _Key[Optional[int]]("execution.variant_timeout_seconds")
    cross_task_timeout_seconds = _Key[Optional[int]]("execution.cross_task_timeout_seconds")
    max_evals_per_variant = _Key[int]("execution.max_evals_per_variant")
    sandbox_image = _Key[Optional[str]]("execution.sandbox_image")
    steps = _Key[Optional[str]]("execution.stages")
    custom_only = _Key[bool]("execution.custom_only")
    custom_baselines = _Key[list[str]]("paths.custom_baselines")
    extra_tasks = _Key[list[str]]("paths.extra_tasks")


class PosthocConfig(PhasePassthrough):
    """The post-hoc phase: ``configs/posthoc_config.yaml``."""

    prefix = "posthoc"
    stage_config_name = "posthoc_config"
    owned = (
        "paths.",
        "run_id",
        "agent.",
        "container.enabled",
        "container.image",
        "container.auth_root",
        "container.env_file",
        "logging.",
        "execution.checkpoint_mode",
        "execution.retrain_if_checkpoint_missing",
    )
    rewritten = frozenset({"container.network"})

    bootstrap_timeout_seconds = _Key[int]("execution.bootstrap_timeout_seconds")
    bootstrap_max_attempts = _Key[int]("execution.bootstrap_max_attempts")
    generation_timeout_seconds = _Key[int]("execution.generation_timeout_seconds")
    selection_timeout_seconds = _Key[int]("execution.selection_timeout_seconds")
    analysis_timeout_seconds = _Key[int]("execution.analysis_timeout_seconds")
    analysis_timeout_grace_seconds = _Key[int]("execution.analysis_timeout_grace_seconds")
    max_selected_analyses = _Key[int]("execution.max_selected_analyses")
    max_attempts_per_analysis = _Key[int]("execution.max_attempts_per_analysis")
    primary_metric_key = _Key[Optional[str]]("execution.primary_metric_key")
    primary_metric_tolerance = _Key[float]("execution.primary_metric_tolerance")
    container_network = _Key[str]("container.network")


PAPER_OWNED_KEYS: frozenset[str] = frozenset(
    {
        "agent",
        "agent_model",
        "agent_profile",
        "reasoning_effort",
        "task_dir",
        "results_dir",
        "ablations_dir",
        "posthoc_dir",
        "final_evaluation_dir",
        "no_final_evaluation",
        "contract_task_dir",
        "investigation_results_dir",
        "global_memory_dir",
        "output_dir",
        "manifest_dir",
        "resume_existing",
        "containerize",
        "container_image",
        "container_network",
        "container_auth_root",
        "container_env_file",
        "log_format",
        "verbose",
        "heartbeat_seconds",
        "event_transport",
    }
)


class PaperConfig(PhasePassthrough):
    """The paper phase: paper.sh flags mirrored by ``configs/paper_config.yaml``."""

    prefix = "paper"
    stage_config_name = "paper_config"
    owned = tuple(sorted(PAPER_OWNED_KEYS))
    #: Venue selection is resolved by the builder (null -> the format's brief,
    #: "" -> no brief), and the publication policy is always passed
    #: explicitly so the pipeline's intent never depends on paper.sh's default.
    rewritten = frozenset({"venue_file", "final_outcome_policy"})
    #: Passed on every run even when equal to the mirrored default, so the
    #: paper command always states its format and budgets. paper.sh used to
    #: initialise these from environment variables; it no longer reads any flag
    #: from the environment, and tests/test_paper_config_matches_script.py
    #: fails if one comes back without being listed here.
    always_forwarded = frozenset(
        {"paper_format", "paper_profile", "paper_max_repair_rounds", "paper_review_rounds"}
    )

    final_outcome_policy = _Key[str]("final_outcome_policy")
    format = _Key[str]("paper_format")
    min_refs = _Key[int]("paper_min_refs")
    max_refs = _Key[int]("paper_max_refs")
    venue_file = _Key[Optional[str]]("venue_file")
    standalone = _Key[bool]("standalone_paper")


PHASE_CLASSES: dict[str, type[PhasePassthrough]] = {
    "investigation": InvestigationConfig,
    "ablations": AblationsConfig,
    "posthoc": PosthocConfig,
    "paper": PaperConfig,
}


@dataclass
class PipelineExecutionConfig:
    """Which stages to run and how failures are handled."""

    phases: str = field(
        default="task_drafting,investigation,ablations,final_evaluation,posthoc,paper",
        metadata={
            "help": (
                "Ordered phases to run (task_drafting, investigation, ablations, final_evaluation, posthoc, "
                "paper, review, "
                "revision, "
                "or comma subset)"
            )
        },
    )
    dry_run: bool = field(
        default=False,
        metadata={"help": "Print resolved commands without executing them"},
    )
    fail_fast: bool = field(
        default=True,
        metadata={"help": "Stop at the first failing stage"},
    )
    no_winner_policy: str = field(
        default="best_available",
        metadata={"help": "best_available: best valid candidate, then baseline; skip opts out"},
    )


@dataclass
class PipelineConfig:
    """Top-level configuration for the pipeline orchestrator."""

    task: PipelineTaskConfig = field(default_factory=PipelineTaskConfig)
    task_drafting: TaskDraftingConfig = field(default_factory=TaskDraftingConfig)
    agent: PipelineAgentConfig = field(default_factory=PipelineAgentConfig)
    paths: PipelinePathsConfig = field(default_factory=PipelinePathsConfig)
    investigation: InvestigationConfig = field(default_factory=InvestigationConfig)
    ablations: AblationsConfig = field(default_factory=AblationsConfig)
    posthoc: PosthocConfig = field(default_factory=PosthocConfig)
    final: FinalConfig = field(default_factory=FinalConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    container: PipelineContainerConfig = field(default_factory=PipelineContainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    execution: PipelineExecutionConfig = field(default_factory=PipelineExecutionConfig)
    run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Shared run id; minted at run time if omitted"},
    )


PIPELINE_SCHEMA_NAME = "pipeline_schema"


@functools.lru_cache(maxsize=None)
def pipeline_config_schema() -> type:
    """The structured config Hydra composes ``configs/pipeline_config.yaml`` onto.

    The dataclass is the schema: every pipeline-owned key, type, default, and
    help string is declared once on :class:`PipelineConfig`, and Hydra
    type-checks overrides against it and rejects unknown keys. The four phase
    sections are free-form here because each is the downstream stage's own
    config, composed in by the YAML defaults list; the loader wraps them into
    the :class:`PhasePassthrough` subclasses afterwards.
    """
    hints = get_type_hints(PipelineConfig)
    specs: list[tuple[str, Any, Any]] = []
    for spec in fields(PipelineConfig):
        if spec.name in PHASE_SECTIONS:
            specs.append((spec.name, dict[str, Any], field(default_factory=dict)))
        elif spec.default is not MISSING:
            specs.append((spec.name, hints[spec.name], field(default=spec.default)))
        else:
            specs.append(
                (spec.name, hints[spec.name], field(default_factory=spec.default_factory))
            )
    return make_dataclass("PipelineConfigSchema", specs)


def register_pipeline_schema() -> None:
    """Register the schema node so the YAML defaults list can name it."""
    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(name=PIPELINE_SCHEMA_NAME, node=pipeline_config_schema())


@dataclass(frozen=True)
class ConfigKeyDescription:
    """One row of ``python -m src.pipeline.cli --help``."""

    key: str
    type_name: str
    default: str
    help: str


def _type_name(annotation: object) -> str:
    args = get_args(annotation)
    if get_origin(annotation) is Union:
        return " | ".join(_type_name(arg) for arg in args)
    if args:
        origin = get_origin(annotation)
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{', '.join(_type_name(arg) for arg in args)}]"
    if annotation is type(None):
        return "None"
    return getattr(annotation, "__name__", None) or str(annotation)


_STAGE_LABELS = {
    "config": "src.investigation.cli (configs/config.yaml)",
    "variants_config": "variants pipeline (configs/variants_config.yaml)",
    "posthoc_config": "post-hoc pipeline (configs/posthoc_config.yaml)",
    "paper_config": "paper.sh flag (configs/paper_config.yaml)",
}


def describe_pipeline_config() -> list[ConfigKeyDescription]:
    """Flatten the schema into ``(key, type, default, help)`` rows for help output."""
    rows: list[ConfigKeyDescription] = []

    def walk(cls: type, prefix: str) -> None:
        hints = get_type_hints(cls)
        for spec in fields(cls):
            annotation = hints[spec.name]
            key = f"{prefix}{spec.name}"
            if isinstance(annotation, type) and is_dataclass(annotation):
                walk(annotation, f"{key}.")
                continue
            if spec.default is not MISSING:
                default: Any = spec.default
            elif spec.default_factory is not MISSING:
                default = spec.default_factory()
            else:
                default = None
            rows.append(
                ConfigKeyDescription(
                    key=key,
                    type_name=_type_name(annotation),
                    default="null" if default is None else str(default),
                    help=str(spec.metadata.get("help", "")),
                )
            )

    walk(PipelineConfig, "")
    for prefix, phase_cls in PHASE_CLASSES.items():
        baseline = stage_baseline(phase_cls.stage_config_name)
        defaults = pipeline_phase_defaults(prefix)
        for key in sorted(defaults):
            default = defaults[key]
            owned = phase_cls.owned_entry(key) is not None
            note = f"{_STAGE_LABELS[phase_cls.stage_config_name]} key"
            if owned:
                note += "; set by the pipeline"
            elif key in baseline and baseline[key] != default:
                note += f" (stage default: {baseline[key]!r})"
            rows.append(
                ConfigKeyDescription(
                    key=f"{prefix}.{key}",
                    type_name="any" if default is None else type(default).__name__,
                    default="null" if default is None else str(default),
                    help=note,
                )
            )
    return rows


def venue_file_for_format(paper_format: str) -> str:
    """Return the repo-relative venue brief for *paper_format*, or "" when unknown."""
    venue = find_venue(paper_format)
    return venue.brief if venue is not None else ""


def parse_pipeline_phases(phases_str: str) -> set[str]:
    """Parse a comma-separated phase spec into a validated set of names."""
    result = {
        "task_drafting" if part.strip() == "task" else part.strip()
        for part in phases_str.split(",")
        if part.strip()
    }
    if not result:
        raise ValueError(f"No phases selected in {phases_str!r}")
    invalid = result - set(PIPELINE_PHASES)
    if invalid:
        raise ValueError(
            f"Invalid pipeline phases: {sorted(invalid)}. Valid: {', '.join(PIPELINE_PHASES)}"
        )
    return result


def task_requires_final_evaluation(task_dir: Path) -> bool:
    """Read publication policy, failing closed when a present contract is invalid."""
    contract_path = task_dir / TASK_CONTRACT_FILENAME
    try:
        metadata = contract_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if not stat.S_ISREG(metadata.st_mode):
        return True
    try:
        contract = load_task_contract(contract_path)
    except TaskContractError:
        return True
    return contract.publication.requires_final_evaluation


def downstream_study_available(cfg: PipelineConfig, investigation_dir: Path) -> bool:
    """Preview fallback availability without creating a frozen study."""
    if cfg.execution.no_winner_policy == "skip":
        return False
    from .downstream_selection import preview_downstream_selection

    return preview_downstream_selection(investigation_dir) is not None


# Which selected pipeline phases consume each composed stage section.
_PHASE_SECTION_USERS: dict[str, frozenset[str]] = {
    "investigation": frozenset({"task_drafting", "investigation"}),
    "ablations": frozenset({"ablations"}),
    "posthoc": frozenset({"posthoc"}),
    "paper": frozenset({"paper", "revision"}),
}


def validate_pipeline_config(cfg: PipelineConfig) -> list[str]:
    """Validate the config, returning a list of error messages (empty = valid)."""
    errors: list[str] = []
    if cfg.execution.no_winner_policy not in {"best_available", "skip"}:
        errors.append("execution.no_winner_policy must be best_available or skip")
    unexpandable_paths: set[str] = set()
    task_dir: Path | None = None
    try:
        selected = parse_pipeline_phases(cfg.execution.phases)
    except ValueError as exc:
        errors.append(f"execution.phases: {exc}")
        return errors

    for phase_name, phases_using_it in _PHASE_SECTION_USERS.items():
        if not selected.intersection(phases_using_it):
            continue
        phase = getattr(cfg, phase_name)
        errors.extend(phase.conflict_errors())
        errors.extend(error for error in phase.type_errors() if error not in errors)

    if not cfg.task.dir:
        errors.append("task.dir is required")
    else:
        try:
            task_dir = Path(cfg.task.dir).expanduser()
        except RuntimeError:
            errors.append(f"task.dir could not be expanded: {cfg.task.dir}")
        else:
            if not task_dir.name or task_dir.name in {".", ".."}:
                errors.append(f"task.dir must identify one named directory: {cfg.task.dir}")
            source_required = (
                bool(selected.intersection({"task_drafting", "investigation"}))
                and not cfg.investigation.resume
            )
            if source_required and (task_dir.is_symlink() or not task_dir.is_dir()):
                errors.append(f"task.dir is not a real directory: {cfg.task.dir}")

    path_values = {
        "paths.results_root": cfg.paths.results_root,
        "paths.run_dir": cfg.paths.run_dir,
        "paths.ablations_dir": cfg.paths.ablations_dir,
        "paths.posthoc_dir": cfg.paths.posthoc_dir,
        "paths.final_evaluation_dir": cfg.paths.final_evaluation_dir,
        "paths.paper_dir": cfg.paths.paper_dir,
    }
    for key, value in path_values.items():
        if not value:
            continue
        try:
            Path(value).expanduser()
        except RuntimeError:
            errors.append(f"{key} could not be expanded: {value}")
            unexpandable_paths.add(key)

    if (
        "investigation" not in selected
        and cfg.paths.run_dir
        and "paths.run_dir" not in unexpandable_paths
        and completed_investigation_outcome(
            investigation_workspace(Path(cfg.paths.run_dir).expanduser())
        )
        == "completed_no_winner"
    ):
        try:
            if not downstream_study_available(
                cfg, investigation_workspace(Path(cfg.paths.run_dir).expanduser())
            ):
                selected -= _DOWNSTREAM_PHASES
        except (OSError, ValueError) as exc:
            errors.append(f"Invalid downstream selection: {exc}")

    configured_paper_dir = cfg.paths.paper_dir
    pending_revision_recovery = (
        bool(selected.intersection({"review", "revision"}))
        and configured_paper_dir is not None
        and bool(configured_paper_dir)
        and "paths.paper_dir" not in unexpandable_paths
        and portable_replacement_is_recoverable(Path(configured_paper_dir).expanduser())
    )

    if cfg.run_id is not None and (
        not cfg.run_id
        or cfg.run_id in {".", ".."}
        or "/" in cfg.run_id
        or "\\" in cfg.run_id
        or "\0" in cfg.run_id
    ):
        errors.append(f"run_id must be one non-empty path component, got {cfg.run_id!r}")

    if cfg.agent.name not in VALID_AGENTS:
        errors.append(f"agent.name must be one of {VALID_AGENTS}, got {cfg.agent.name!r}")
    if cfg.logging.format not in {"human", "json"}:
        errors.append(
            f"logging.format must be one of ('human', 'json'), got {cfg.logging.format!r}"
        )
    if type(cfg.logging.heartbeat_seconds) is not int:
        errors.append("logging.heartbeat_seconds must be an integer")
    elif cfg.logging.heartbeat_seconds < 0:
        errors.append("logging.heartbeat_seconds must be >= 0")
    if cfg.agent.profile is not None and not str(cfg.agent.profile).strip():
        errors.append("agent.profile must be non-empty when provided")
    if cfg.agent.profile is not None and cfg.agent.name != "codex":
        errors.append("agent.profile is only supported when agent.name='codex'")
    if cfg.agent.profile is not None and not str(cfg.agent.model or "").strip():
        errors.append(PROFILE_MODEL_ERROR)
    if (
        cfg.agent.reasoning_effort is not None
        and cfg.agent.reasoning_effort not in VALID_REASONING_EFFORTS
    ):
        errors.append(f"agent.reasoning_effort must be one of {VALID_REASONING_EFFORTS}")
    if cfg.agent.name == "gemini" and cfg.agent.reasoning_effort is not None:
        errors.append(GEMINI_REASONING_EFFORT_ERROR)

    if cfg.container.enabled or ("paper" in selected and phase_uses_container(cfg, "paper")):
        auth_root_raw = str(cfg.container.auth_root)
        if not auth_root_raw.strip():
            errors.append("container.auth_root must be non-empty when container is enabled")
        else:
            try:
                auth_root = Path(auth_root_raw).expanduser()
            except RuntimeError:
                errors.append(f"container.auth_root could not be expanded: {auth_root_raw}")
            else:
                if auth_root.is_symlink() or not auth_root.is_dir():
                    errors.append(f"container.auth_root is not a real directory: {auth_root_raw}")
        if cfg.container.env_file:
            try:
                Path(str(cfg.container.env_file)).expanduser()
            except RuntimeError:
                errors.append(f"container.env_file could not be expanded: {cfg.container.env_file}")

    if type(cfg.task_drafting.timeout_seconds) is not int:
        errors.append("task_drafting.timeout_seconds must be an integer")
    elif cfg.task_drafting.timeout_seconds < 1:
        errors.append("task_drafting.timeout_seconds must be >= 1")
    if cfg.investigation.drafting_workflow not in {"robust", "legacy"}:
        errors.append("investigation.task_drafting.workflow must be robust or legacy")
    drafting_limits = {
        "design_max_repairs": (0, 8),
        "contract_max_repairs": (0, 3),
        "smoke_max_repairs": (0, 3),
        "smoke_max_infrastructure_attempts": (1, 3),
        "alignment_max_repairs": (0, 10),
        "qualification_max_repairs": (0, 8),
        "agent_transient_max_attempts": (1, 4),
        "qualification_max_infrastructure_attempts": (1, 3),
        "production_entry_timeout_seconds": (30, 1800),
        "max_gate_checks_per_repair": (1, 12),
        "max_runtime_bytes": (1, 64 * 1024 * 1024 * 1024),
        "draft_max_attempts": (1, 3),
        "evaluator_smoke_timeout_seconds": (1, 7200),
    }
    for name, (minimum, maximum) in drafting_limits.items():
        limit_value = cfg.investigation.get(f"task_drafting.{name}")
        if type(limit_value) is not int or not minimum <= limit_value <= maximum:
            errors.append(
                f"investigation.task_drafting.{name} must be an integer "
                f"from {minimum} to {maximum}"
            )
    if (
        "paper" in selected
        and "final_evaluation" not in selected
        and cfg.paper.final_outcome_policy != "allow_not_qualified"
        and task_dir is not None
        and task_requires_final_evaluation(task_dir)
    ):
        errors.append("contracted paper phase requires final phase")
    if cfg.paper.final_outcome_policy not in {"qualified_only", "allow_not_qualified"}:
        errors.append(
            "paper.final_outcome_policy must be qualified_only or allow_not_qualified"
        )
    investigation_counts_valid = True
    gpu_count_valid = type(cfg.container.gpu_count) is int and cfg.container.gpu_count >= 0
    if not gpu_count_valid:
        errors.append("container.gpu_count must be a non-negative integer")
    if "task_drafting" in selected and not cfg.container.enabled:
        errors.append("task_drafting requires container.enabled=true")
    if "task_drafting" in selected and cfg.investigation.resume:
        errors.append("task_drafting cannot be combined with investigation.execution.resume=true")
    if selected.intersection({"task_drafting", "investigation"}):
        minimums = {
            "max_idea_cycles": 1,
            "workers": 1,
            "prune_every_n_cycles": 0,
            "max_program_evals_per_attempt": 1,
            "max_idea_generation_attempts": 1,
            "idea_timeout_seconds": 1,
            "run_timeout_seconds": 1,
            "harness_evaluation_timeout_seconds": 1,
        }
        for field_name, minimum in minimums.items():
            value = getattr(cfg.investigation, field_name)
            if type(value) is not int:
                investigation_counts_valid = False
                errors.append(f"investigation.execution.{field_name} must be an integer")
            elif value < minimum:
                investigation_counts_valid = False
                errors.append(f"investigation.execution.{field_name} must be >= {minimum}")
        if type(cfg.investigation.harness_evaluated_attempts) is not bool:
            errors.append("investigation.execution.harness_evaluated_attempts must be a boolean")
        if type(cfg.investigation.gpu_evaluation) is not bool:
            errors.append("investigation.execution.gpu_evaluation must be a boolean")
        elif cfg.investigation.gpu_evaluation and not cfg.investigation.harness_evaluated_attempts:
            errors.append(
                "investigation.execution.gpu_evaluation requires "
                "investigation.execution.harness_evaluated_attempts=true"
            )
        elif cfg.investigation.gpu_evaluation and gpu_count_valid and cfg.container.gpu_count < 1:
            errors.append(
                "investigation.execution.gpu_evaluation requires container.gpu_count >= 1"
            )

    if "ablations" in selected:
        if not cfg.container.enabled:
            if cfg.ablations.extra_tasks:
                errors.append(
                    "ablations.paths.extra_tasks (cross-task variants) require container.enabled=true"
                )
            contract = task_dir / TASK_CONTRACT_FILENAME if task_dir is not None else None
            if contract is not None and (contract.exists() or contract.is_symlink()):
                errors.append(
                    "native ablations (container.enabled=false) cannot run a contract task: "
                    "its hidden final evaluator needs agent isolation"
                )
        evaluation_timeout = cfg.ablations.evaluation_timeout_seconds
        if evaluation_timeout is not None and (
            type(evaluation_timeout) is not int or evaluation_timeout < 1
        ):
            errors.append(
                "ablations.execution.evaluation_timeout_seconds must be a positive integer or null"
            )
        for field_name in ("variant_timeout_seconds", "cross_task_timeout_seconds"):
            timeout_value = getattr(cfg.ablations, field_name)
            key = f"ablations.execution.{field_name}"
            if timeout_value is None:
                continue
            if type(timeout_value) is not int:
                if f"{key} must be an integer" not in errors:
                    errors.append(f"{key} must be an integer")
            elif timeout_value < 1:
                errors.append(f"{key} must be >= 1")
        for field_name in ("num_replications", "max_evals_per_variant"):
            value = getattr(cfg.ablations, field_name)
            if type(value) is not int:
                errors.append(f"ablations.execution.{field_name} must be an integer")
            elif value < 1:
                errors.append(f"ablations.execution.{field_name} must be >= 1")

    if "posthoc" in selected:
        posthoc_minimums = {
            "bootstrap_timeout_seconds": 1,
            "bootstrap_max_attempts": 1,
            "generation_timeout_seconds": 1,
            "selection_timeout_seconds": 1,
            "analysis_timeout_seconds": 1,
            "analysis_timeout_grace_seconds": 1,
            "max_selected_analyses": 0,
            "max_attempts_per_analysis": 1,
        }
        for field_name, minimum in posthoc_minimums.items():
            value = getattr(cfg.posthoc, field_name)
            if type(value) is not int:
                errors.append(f"posthoc.execution.{field_name} must be an integer")
            elif value < minimum:
                errors.append(f"posthoc.execution.{field_name} must be >= {minimum}")
        tolerance = cfg.posthoc.primary_metric_tolerance
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            errors.append("posthoc.execution.primary_metric_tolerance must be a number")
        elif not math.isfinite(float(tolerance)) or tolerance < 0:
            errors.append("posthoc.execution.primary_metric_tolerance must be finite and >= 0")
        if (
            cfg.posthoc.primary_metric_key is not None
            and not str(cfg.posthoc.primary_metric_key).strip()
        ):
            errors.append("posthoc.execution.primary_metric_key must be non-empty when provided")
        if not str(cfg.posthoc.container_network).strip():
            errors.append("posthoc.container.network must be non-empty")

    if selected.intersection({"paper", "revision"}):
        if cfg.paper.format not in PAPER_FORMATS:
            errors.append(
                f"paper.paper_format must be one of {PAPER_FORMATS}, got {cfg.paper.format!r}"
            )
        refs_are_integers = True
        for field_name in ("min_refs", "max_refs"):
            value = getattr(cfg.paper, field_name)
            if type(value) is not int:
                refs_are_integers = False
                errors.append(f"paper.paper_{field_name} must be an integer")
        if refs_are_integers:
            if cfg.paper.min_refs < 1:
                errors.append("paper.paper_min_refs must be >= 1")
            if cfg.paper.max_refs < cfg.paper.min_refs:
                errors.append(
                    f"paper.paper_min_refs ({cfg.paper.min_refs}) must be <= "
                    f"paper.paper_max_refs ({cfg.paper.max_refs})"
                )
        if cfg.paper.venue_file:
            venue_file = Path(cfg.paper.venue_file)
            if not venue_file.is_absolute():
                venue_file = REPO_ROOT / venue_file
            if not venue_file.is_file():
                errors.append(f"paper.venue_file is not a file: {cfg.paper.venue_file}")

    if selected.intersection({"review", "revision"}):
        for field_name in ("timeout_seconds", "timeout_grace_seconds"):
            value = getattr(cfg.review, field_name)
            if type(value) is not int:
                errors.append(f"review.{field_name} must be an integer")
            elif value < 1:
                errors.append(f"review.{field_name} must be >= 1")

    if "final_evaluation" in selected:
        if not cfg.container.enabled:
            errors.append("final phase requires container.enabled=true")
        if type(cfg.final.timeout_seconds) is not int:
            errors.append("final.timeout_seconds must be an integer")
        elif cfg.final.timeout_seconds < 1:
            errors.append("final.timeout_seconds must be >= 1")

    if "investigation" in selected and cfg.investigation.resume:
        if not cfg.paths.run_dir:
            errors.append("paths.run_dir is required when resuming investigation")
        elif "paths.run_dir" not in unexpandable_paths:
            errors.extend(
                validate_investigation_resume_state(
                    investigation_workspace(Path(cfg.paths.run_dir).expanduser())
                )
            )

        if investigation_counts_valid:
            pruning = cfg.investigation.prune_every_n_cycles
            resume_mode_supported = (
                cfg.investigation.workers == 1 and pruning == 0
            ) or 0 < pruning < cfg.investigation.max_idea_cycles
            if not resume_mode_supported:
                qualifier = "parallel " if cfg.investigation.workers > 1 else ""
                errors.append(
                    f"{qualifier}investigation resume requires workers=1 with pruning "
                    "disabled or 0 < investigation.execution.prune_every_n_cycles "
                    "< investigation.execution.max_idea_cycles"
                )

    if (
        "ablations" in selected
        and "investigation" not in selected
        and "paths.run_dir" not in unexpandable_paths
    ):
        _require_existing_dir(
            cfg.paths.run_dir,
            "paths.run_dir",
            "running ablations without investigation",
            errors,
        )
        if cfg.paths.run_dir:
            investigation_dir = investigation_workspace(Path(cfg.paths.run_dir).expanduser())
            _require_existing_dir(
                str(investigation_dir),
                "investigation workspace",
                "running ablations without investigation",
                errors,
            )
            if investigation_dir.is_dir():
                errors.extend(validate_completed_investigation_for_downstream(investigation_dir))
    if (
        "final_evaluation" in selected
        and "investigation" not in selected
        and "paths.run_dir" not in unexpandable_paths
    ):
        _require_existing_dir(
            cfg.paths.run_dir,
            "paths.run_dir",
            "running final evaluation without investigation",
            errors,
        )
        if cfg.paths.run_dir:
            investigation_dir = investigation_workspace(Path(cfg.paths.run_dir).expanduser())
            _require_existing_dir(
                str(investigation_dir),
                "investigation workspace",
                "running final evaluation without investigation",
                errors,
            )
            if investigation_dir.is_dir():
                errors.extend(validate_completed_investigation_for_downstream(investigation_dir))
    if (
        "paper" in selected
        and "investigation" not in selected
        and "paths.run_dir" not in unexpandable_paths
    ):
        _require_existing_dir(
            cfg.paths.run_dir,
            "paths.run_dir",
            "running paper without investigation",
            errors,
        )
        if cfg.paths.run_dir:
            investigation_dir = investigation_workspace(Path(cfg.paths.run_dir).expanduser())
            _require_existing_dir(
                str(investigation_dir),
                "investigation workspace",
                "running paper without investigation",
                errors,
            )
            if investigation_dir.is_dir():
                errors.extend(validate_completed_investigation_for_downstream(investigation_dir))
    if selected.intersection({"review", "revision"}) and "paper" not in selected:
        if "paths.run_dir" not in unexpandable_paths:
            _require_existing_dir(
                cfg.paths.run_dir,
                "paths.run_dir",
                "running review without paper",
                errors,
            )
        if "paths.paper_dir" not in unexpandable_paths and not pending_revision_recovery:
            _require_existing_dir(
                cfg.paths.paper_dir,
                "paths.paper_dir",
                "running review without paper",
                errors,
            )
        if cfg.paths.run_dir and cfg.paths.paper_dir and not pending_revision_recovery:
            errors.extend(
                validate_review_input(
                    Path(cfg.paths.run_dir).expanduser(),
                    Path(cfg.paths.paper_dir).expanduser(),
                )
            )
    if "revision" in selected and "review" not in selected and "paper" in selected:
        errors.append("running paper and revision requires review in the same phase selection")
    if (
        "revision" in selected
        and "review" not in selected
        and "paper" not in selected
        and cfg.paths.paper_dir
        and not pending_revision_recovery
    ):
        paper_dir = Path(cfg.paths.paper_dir).expanduser()
        named_review = paper_dir / "ai_review.before_revision.json"
        legacy_review = paper_dir / "ai_review.json"
        selected_review = (
            (named_review, "before_revision")
            if named_review.is_file() and not named_review.is_symlink()
            else (legacy_review, "ai_review")
        )
        review_path, review_key = selected_review
        review_is_completed = False
        try:
            from src.investigation.final_ai_review import read_paper_manifest

            manifest = read_paper_manifest(paper_dir / "latest_manifest.json")
            review_state = manifest.get(review_key)
            manifest_review_path = (
                review_state.get("review_json") if isinstance(review_state, dict) else None
            )
            review_is_completed = (
                review_path.is_file()
                and not review_path.is_symlink()
                and isinstance(review_state, dict)
                and review_state.get("status") == "completed"
                and isinstance(manifest_review_path, str)
                and Path(manifest_review_path).expanduser().resolve() == review_path.resolve()
            )
        except (OSError, RuntimeError, ValueError):
            review_is_completed = False
        if not review_is_completed:
            errors.append(
                "revision without review requires a completed initial AI review "
                "in paths.paper_dir"
            )
    if "revision" in selected and not cfg.container.enabled:
        errors.append("revision phase requires container.enabled=true")
    if "paper" in selected and not phase_uses_container(cfg, "paper"):
        errors.append(PAPER_NEEDS_CONTAINER_ERROR)
    errors.extend(_versioning_errors(cfg.versioning))

    return errors


def _versioning_errors(cfg: VersioningConfig) -> list[str]:
    """Check the checkpoint knobs, mirroring the investigation CLI's own validation."""
    errors: list[str] = []
    try:
        validate_git_dir_name(cfg.git_dir_name)
    except ValueError as exc:
        errors.append(f"{exc}: {cfg.git_dir_name!r}")
    if cfg.max_file_bytes < 0:
        errors.append("versioning.max_file_bytes must be >= 0")
    if cfg.max_total_bytes < 0:
        errors.append("versioning.max_total_bytes must be >= 0")
    if cfg.timeout_seconds <= 0:
        errors.append("versioning.timeout_seconds must be > 0")
    if cfg.run_root is not None and not str(cfg.run_root).strip():
        errors.append("versioning.run_root must be non-empty when provided")
    return errors


def _require_existing_dir(value: Optional[str], key: str, context: str, errors: list[str]) -> None:
    if not value:
        errors.append(f"{key} is required when {context}")
    else:
        try:
            path = Path(value).expanduser()
        except RuntimeError:
            errors.append(f"{key} could not be expanded: {value}")
        else:
            if path.is_symlink() or not path.is_dir():
                errors.append(f"{key} is not a directory: {value}")


def validate_review_input(run_dir: Path, paper_dir: Path) -> list[str]:
    """Validate explicit review-only anchors without mutating paper artifacts."""
    from src.investigation.final_ai_review import read_paper_manifest

    errors = validate_review_output_path(run_dir, paper_dir)
    if errors:
        return errors

    paper_pdf = paper_dir / "paper.pdf"
    manifest_path = paper_dir / "latest_manifest.json"
    if paper_pdf.is_symlink() or not paper_pdf.is_file():
        errors.append(f"review paper PDF is not a regular file: {paper_pdf}")
        return errors
    if manifest_path.is_symlink() or not manifest_path.is_file():
        errors.append(f"review paper manifest is not a regular file: {manifest_path}")
        return errors
    try:
        manifest = read_paper_manifest(manifest_path)
    except (OSError, UnicodeError, ValueError):
        errors.append(f"review paper manifest is invalid: {manifest_path}")
        return errors
    manifest_pdf = manifest.get("paper_pdf") if isinstance(manifest, dict) else None
    try:
        manifest_pdf_path = (
            Path(manifest_pdf).expanduser().resolve() if isinstance(manifest_pdf, str) else None
        )
    except (OSError, RuntimeError):
        manifest_pdf_path = None
    if manifest_pdf_path != paper_pdf.resolve():
        errors.append(
            f"review paper manifest does not point to the review paper PDF: {manifest_path}"
        )
    return errors


def validate_review_output_path(run_dir: Path, paper_dir: Path) -> list[str]:
    """Reject symlinked or unresolvable paper roots before review."""
    del run_dir
    if paper_dir.is_symlink():
        return [f"paths.paper_dir must not be a symlink: {paper_dir}"]
    try:
        paper_dir.resolve()
    except (OSError, RuntimeError):
        return [f"paths.paper_dir could not be resolved: {paper_dir}"]
    return []


def validate_posthoc_paper_input(posthoc_dir: Path) -> list[str]:
    """Require an explicit post-hoc run to contain consumable, non-degraded output."""
    from src.orchestrator.paper_data_posthoc import collect_posthoc_analyses

    if posthoc_dir.is_symlink() or not posthoc_dir.is_dir():
        return [f"paths.posthoc_dir is not a directory: {posthoc_dir}"]
    status_path = posthoc_dir / "integration_status.json"
    if status_path.exists() or status_path.is_symlink():
        try:
            if status_path.is_symlink() or status_path.stat().st_size > 64 * 1024:
                raise OSError("unsafe status file")
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return [f"posthoc integration status is invalid: {status_path}"]
        if not isinstance(status, dict) or status.get("status") != "completed":
            return [f"posthoc integration status is not completed: {status_path}"]
    try:
        analyses = collect_posthoc_analyses(Path.cwd(), posthoc_dir=posthoc_dir)
    except (OSError, UnicodeError, ValueError, RecursionError):
        analyses = []
    if not analyses:
        return [f"paths.posthoc_dir has no consumable analyses: {posthoc_dir}"]
    return []


def _read_investigation_state_at(directory_fd: int, filename: str) -> dict[str, object] | None:
    try:
        state_fd = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    except OSError:
        return None
    try:
        before = os.fstat(state_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_size > _MAX_INVESTIGATION_STATE_BYTES
        ):
            return None
        chunks: list[bytes] = []
        remaining = _MAX_INVESTIGATION_STATE_BYTES + 1
        while remaining > 0:
            chunk = os.read(state_fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(state_fd)
    except OSError:
        return None
    finally:
        os.close(state_fd)
    if (
        len(payload) != before.st_size
        or len(payload) > _MAX_INVESTIGATION_STATE_BYTES
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        return None
    try:
        state = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def validate_investigation_resume_state(results_dir: Path) -> list[str]:
    """Return errors for a missing or unusable investigation checkpoint."""
    errors: list[str] = []
    states: dict[str, dict[str, object]] = {}
    try:
        with open_directory(results_dir, role="investigation workspace") as workspace_fd:
            for filename in ("outer_state.json", "best_program_manifest.json"):
                state_path = results_dir / filename
                state = _read_investigation_state_at(workspace_fd, filename)
                if state is None:
                    errors.append(f"investigation resume state is missing or invalid: {state_path}")
                else:
                    states[filename] = state
    except (FileNotFoundError, UnsafeGluePathError, OSError):
        return [f"paths.run_dir investigation workspace is not a directory: {results_dir}"]

    outer_state = states.get("outer_state.json")
    if outer_state is not None:
        current_cycle = outer_state.get("current_cycle")
        if (
            isinstance(current_cycle, bool)
            or not isinstance(current_cycle, int)
            or current_cycle < 0
        ):
            errors.append(
                "investigation resume state has invalid current_cycle: "
                f"{results_dir / 'outer_state.json'}"
            )

    manifest = states.get("best_program_manifest.json")
    if manifest is not None and not _manifest_program_is_usable(manifest, results_dir=results_dir):
        errors.append(
            "investigation resume state has no usable program_dir and entrypoint: "
            f"{results_dir / 'best_program_manifest.json'}"
        )

    return errors


def validate_completed_investigation(results_dir: Path) -> list[str]:
    """Require a real, scored winner before downstream scientific phases."""
    manifest_path = results_dir / "best_program_manifest.json"
    try:
        with open_directory(results_dir, role="investigation workspace") as workspace_fd:
            manifest = _read_investigation_state_at(workspace_fd, manifest_path.name)
    except (FileNotFoundError, UnsafeGluePathError, OSError):
        return [f"investigation produced no valid best-program manifest: {manifest_path}"]
    if manifest is None:
        return [f"investigation produced no valid best-program manifest: {manifest_path}"]
    if not _manifest_program_is_usable(manifest, results_dir=results_dir):
        return [f"investigation produced no usable best program: {manifest_path}"]
    if _manifest_retains_baseline(manifest):
        return [
            "investigation retained its baseline and produced no optimized winner: "
            f"{manifest_path}"
        ]

    score = manifest.get("combined_score")
    scored = (
        not isinstance(score, bool)
        and isinstance(score, (int, float))
        and math.isfinite(float(score))
    )
    if manifest.get("correct") is not True or not scored:
        return [f"investigation produced no scored correct program: {manifest_path}"]
    return []


def _investigation_manifest_outcome(results_dir: Path) -> tuple[str, str] | None:
    manifest_path = results_dir / "best_program_manifest.json"
    try:
        with open_directory(results_dir, role="investigation workspace") as workspace_fd:
            manifest = _read_investigation_state_at(workspace_fd, manifest_path.name)
    except (FileNotFoundError, UnsafeGluePathError, OSError):
        return None
    if manifest is None or not _manifest_program_is_usable(manifest, results_dir=results_dir):
        return None
    score = manifest.get("combined_score")
    scored = (
        not isinstance(score, bool)
        and isinstance(score, (int, float))
        and math.isfinite(float(score))
    )
    if manifest.get("correct") is True and scored:
        outcome = (
            "completed_no_winner"
            if _manifest_retains_baseline(manifest)
            else "winner"
        )
    elif manifest.get("correct") is False and scored and _manifest_retains_baseline(manifest):
        outcome = "completed_no_winner"
    elif (
        "correct" in manifest
        and "combined_score" in manifest
        and manifest["correct"] is None
        and manifest["combined_score"] is None
    ):
        outcome = "completed_no_winner"
    else:
        return None
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return outcome, hashlib.sha256(encoded).hexdigest()


def _manifest_retains_baseline(manifest: dict[str, object]) -> bool:
    source_attempt_id = manifest.get("source_attempt_id")
    return (
        isinstance(source_attempt_id, str)
        and source_attempt_id in _BASELINE_SOURCE_ATTEMPT_IDS
    )


def investigation_completion_receipt_path(results_dir: Path) -> Path:
    return results_dir.parent / _INVESTIGATION_COMPLETION_RECEIPT


def investigation_completion_payload(results_dir: Path) -> dict[str, object] | None:
    classified = _investigation_manifest_outcome(results_dir)
    if classified is None:
        return None
    outcome, manifest_sha256 = classified
    return {
        "schema_version": 1,
        "status": "completed",
        "outcome": outcome,
        "manifest_sha256": manifest_sha256,
    }


def completed_investigation_outcome(results_dir: Path) -> str | None:
    """Return a terminal outcome only when a harness completion receipt matches state."""
    expected = investigation_completion_payload(results_dir)
    if expected is None:
        return None
    receipt_path = investigation_completion_receipt_path(results_dir)
    try:
        with open_directory(receipt_path.parent, role="pipeline run envelope") as run_fd:
            receipt = _read_investigation_state_at(run_fd, receipt_path.name)
    except (FileNotFoundError, UnsafeGluePathError, OSError):
        return None
    if receipt == expected:
        return str(expected["outcome"])
    if expected["outcome"] == "completed_no_winner" and receipt == (
        expected | {"outcome": "winner"}
    ):
        return "completed_no_winner"
    return None


def validate_completed_investigation_for_downstream(results_dir: Path) -> list[str]:
    """Accept valid search exhaustion while rejecting malformed or unusable state."""
    if completed_investigation_outcome(results_dir) == "completed_no_winner":
        return []
    return validate_completed_investigation(results_dir)


def _manifest_program_is_usable(manifest: dict[str, object], *, results_dir: Path) -> bool:
    try:
        workspace = results_dir.expanduser().resolve()
    except RuntimeError:
        return False
    legacy_program = manifest.get("program_path")
    if isinstance(legacy_program, str) and legacy_program.strip():
        try:
            program_path = Path(legacy_program).expanduser().resolve()
        except RuntimeError:
            return False
        return program_path.is_relative_to(workspace) and program_path.is_file()

    program_dir_raw = manifest.get("program_dir")
    entrypoint_raw = manifest.get("entrypoint_path")
    if not isinstance(program_dir_raw, str) or not program_dir_raw.strip():
        return False

    try:
        program_dir = Path(program_dir_raw).expanduser().resolve()
    except RuntimeError:
        return False
    if not program_dir.is_relative_to(workspace):
        return False
    if not isinstance(entrypoint_raw, str) or not entrypoint_raw.strip():
        entrypoint_raw = (
            entrypoint_relpath_from_spec(program_dir, required=False) or "src/program.py"
        )
    entrypoint_rel = Path(entrypoint_raw)
    if entrypoint_rel.is_absolute() or ".." in entrypoint_rel.parts:
        return False
    entrypoint = (program_dir / entrypoint_rel).resolve()
    try:
        entrypoint.relative_to(program_dir)
    except ValueError:
        return False
    return program_dir.is_dir() and entrypoint.is_file()

#!/bin/bash
# Usage: bash src/scripts/run.sh [--agent claude|codex|cursor|opencode|gemini|pi] [--agent-model <name>] [--reasoning-effort low|medium|high|xhigh] [--task-dir <path>] [--results-dir <path>] [--global-memory-dir <path>] [--instruct <path>] [--seed-program <path>] [--rubrics-file <path>] [--idea-file <path>] [--attempt-id <id>] [--idea-dir-name <name>] [--max-program-evals-per-attempt <n>] [--harness-evaluated-attempt] [--gpu-evaluation] [--agent-timeout-seconds <n>] [--harness-evaluation-timeout-seconds <n>] [--containerize] [--container-image <name>] [--container-network <mode>] [--container-auth-root <path>] [--container-env-file <path>]

set -euo pipefail

# Execution time limits for CC
export BASH_DEFAULT_TIMEOUT_MS=600000    # 10 minutes
export BASH_MAX_TIMEOUT_MS=7200000       # 2 hours (upper limit)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/container_agent.sh"
source "$SCRIPT_DIR/lib/task_pathing.sh"

AGENT_LOG_MAX_BYTES=16777216
validate_agent_log_max_bytes "$AGENT_LOG_MAX_BYTES" || exit $?

AGENT="claude"
AGENT_MODEL=""
AGENT_PROFILE=""
AGENT_REASONING_EFFORT=""
TASK_DIR="examples/circpack"
RESULTS_DIR=""
GLOBAL_MEMORY_DIR=""
CONCLUSIONS_DIR=""
WORKER_INDEX=0
HARNESS_EVAL_TIMEOUT_SECONDS=7200
SEARCH_ONLY_TASK_SOURCE_DIR=""
CONTRACTED_CONTAINER_ENV_FILE=""
CONTRACTED_TASK=0
SCRIPT_TEMP_ROOT="${TMPDIR:-/tmp}"

cleanup_search_only_task_source() {
  [ -n "${SEARCH_ONLY_TASK_SOURCE_DIR:-}" ] || return 0
  if ! "$PYTHON_BIN" - "$SEARCH_ONLY_TASK_SOURCE_DIR" "$SCRIPT_TEMP_ROOT" "$REPO_ROOT" <<'PY'
import shutil
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[3])
from src.investigation.temp_cleanup import validate_temp_directory

path = Path(sys.argv[1])
if validate_temp_directory(path, Path(sys.argv[2]), "agentic-task-search-"):
    path.chmod(0o700, follow_symlinks=False)
    shutil.rmtree(path)
PY
  then
    echo "Warning: failed to remove search-only task source." >&2
  fi
  SEARCH_ONLY_TASK_SOURCE_DIR=""
}

cleanup_contracted_container_env() {
  [ -n "${CONTRACTED_CONTAINER_ENV_FILE:-}" ] || return 0
  "$PYTHON_BIN" - "$CONTRACTED_CONTAINER_ENV_FILE" "$SCRIPT_TEMP_ROOT" "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[3])
from src.investigation.temp_cleanup import validate_temp_file

path = Path(sys.argv[1])
if validate_temp_file(path, Path(sys.argv[2]), "agentic-run-provider-env-"):
    path.unlink(missing_ok=True)
PY
  CONTRACTED_CONTAINER_ENV_FILE=""
}
INSTRUCT_PATH=""
SEED_PROGRAM_DIR=""
RUBRICS_FILE=""
IDEA_FILE=""
ATTEMPT_ID=""
IDEA_DIR_NAME=""
ATTEMPT_DIR_NAME=""
MAX_PROGRAM_EVALS_PER_ATTEMPT=4
HARNESS_EVALUATED_ATTEMPT=0
GPU_EVALUATION=0
AGENT_TIMEOUT_SECONDS=""
CONTAINERIZE=0
CONTAINER_IMAGE="agentic-scientist/agent-cli:latest"
CONTAINER_NETWORK="bridge"
CONTAINER_AUTH_ROOT="$HOME"
CONTAINER_ENV_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent|-a)
      AGENT="$2"
      shift 2
      ;;
    --agent-model|--model)
      AGENT_MODEL="$2"
      shift 2
      ;;
    --agent-profile|--profile)
      AGENT_PROFILE="$2"
      shift 2
      ;;
    --reasoning-effort)
      AGENT_REASONING_EFFORT="$2"
      shift 2
      ;;
    --task-dir|-t)
      TASK_DIR="$2"
      shift 2
      ;;
    --results-dir|-r)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --global-memory-dir)
      GLOBAL_MEMORY_DIR="$2"
      shift 2
      ;;
    --conclusions-dir)
      CONCLUSIONS_DIR="$2"
      shift 2
      ;;
    --worker-index)
      WORKER_INDEX="$2"
      shift 2
      ;;
    --instruct|-i)
      INSTRUCT_PATH="$2"
      shift 2
      ;;
    --seed-program)
      SEED_PROGRAM_DIR="$2"
      shift 2
      ;;
    --rubrics-file)
      RUBRICS_FILE="$2"
      shift 2
      ;;
    --idea-file)
      IDEA_FILE="$2"
      shift 2
      ;;
    --attempt-id)
      ATTEMPT_ID="$2"
      shift 2
      ;;
    --idea-dir-name)
      IDEA_DIR_NAME="$2"
      shift 2
      ;;
    --attempt-dir-name)
      # Retained for callers that still pass an explicit attempt directory name.
      ATTEMPT_DIR_NAME="$2"
      shift 2
      ;;
    --max-program-evals-per-attempt|--max-program-evals)
      MAX_PROGRAM_EVALS_PER_ATTEMPT="$2"
      shift 2
      ;;
    --harness-evaluated-attempt|--harness-evaluated-attempts)
      HARNESS_EVALUATED_ATTEMPT=1
      shift
      ;;
    --gpu-evaluation)
      GPU_EVALUATION=1
      shift
      ;;
    --agent-timeout-seconds)
      AGENT_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --harness-evaluation-timeout-seconds)
      HARNESS_EVAL_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --containerize)
      CONTAINERIZE=1
      shift
      ;;
    --container-image)
      CONTAINER_IMAGE="$2"
      shift 2
      ;;
    --container-network)
      CONTAINER_NETWORK="$2"
      shift 2
      ;;
    --container-auth-root)
      CONTAINER_AUTH_ROOT="$2"
      shift 2
      ;;
    --container-env-file)
      CONTAINER_ENV_FILE="$2"
      shift 2
      ;;
    claude|codex|cursor|opencode|gemini|pi)
      AGENT="$1"
      shift
      ;;
    *)
      TASK_DIR="$1"
      shift
      ;;
  esac
done

if [ "$GPU_EVALUATION" -eq 1 ] && [ "$HARNESS_EVALUATED_ATTEMPT" -ne 1 ]; then
  echo "--gpu-evaluation requires --harness-evaluated-attempt" >&2
  exit 1
fi

if [ -n "$AGENT_MODEL" ]; then
  AGENT_MODEL_NORMALIZED="$(printf '%s' "$AGENT_MODEL" | tr '[:upper:]' '[:lower:]')"
  case "$AGENT_MODEL_NORMALIZED" in
    none|null)
      AGENT_MODEL=""
      ;;
  esac
fi

if [ -z "$RESULTS_DIR" ]; then
  TASK_NAME="$(basename "${TASK_DIR%/}")"
  if [ -z "$TASK_NAME" ] || [ "$TASK_NAME" = "." ] || [ "$TASK_NAME" = "/" ]; then
    TASK_NAME="task"
  fi
  RESULTS_DIR="results/$TASK_NAME/$AGENT"
fi

if [[ "$TASK_DIR" != /* ]]; then
  TASK_DIR="$REPO_ROOT/$TASK_DIR"
fi
if [[ "$RESULTS_DIR" != /* ]]; then
  RESULTS_DIR="$REPO_ROOT/$RESULTS_DIR"
fi
if [ -z "$GLOBAL_MEMORY_DIR" ]; then
  GLOBAL_MEMORY_DIR="$RESULTS_DIR"
fi
if [[ "$GLOBAL_MEMORY_DIR" != /* ]]; then
  GLOBAL_MEMORY_DIR="$REPO_ROOT/$GLOBAL_MEMORY_DIR"
fi
if [ -z "$CONCLUSIONS_DIR" ]; then
  CONCLUSIONS_DIR="$GLOBAL_MEMORY_DIR"
fi
if [[ "$CONCLUSIONS_DIR" != /* ]]; then
  CONCLUSIONS_DIR="$REPO_ROOT/$CONCLUSIONS_DIR"
fi
if ! [[ "$WORKER_INDEX" =~ ^[0-9]+$ ]]; then
  echo "--worker-index must be a non-negative integer (got: ${WORKER_INDEX:-<empty>})" >&2
  exit 1
fi
if [ -n "$INSTRUCT_PATH" ] && [[ "$INSTRUCT_PATH" != /* ]]; then
  INSTRUCT_PATH="$REPO_ROOT/$INSTRUCT_PATH"
fi
if [ -n "$SEED_PROGRAM_DIR" ] && [[ "$SEED_PROGRAM_DIR" != /* ]]; then
  SEED_PROGRAM_DIR="$REPO_ROOT/$SEED_PROGRAM_DIR"
fi
if [ -n "$RUBRICS_FILE" ] && [[ "$RUBRICS_FILE" != /* ]]; then
  RUBRICS_FILE="$REPO_ROOT/$RUBRICS_FILE"
fi
if [ -n "$IDEA_FILE" ] && [[ "$IDEA_FILE" != /* ]]; then
  IDEA_FILE="$REPO_ROOT/$IDEA_FILE"
fi
if [[ "$CONTAINER_AUTH_ROOT" == "~"* ]]; then
  CONTAINER_AUTH_ROOT="${CONTAINER_AUTH_ROOT/#\~/$HOME}"
fi
if [ -n "$CONTAINER_AUTH_ROOT" ] && [[ "$CONTAINER_AUTH_ROOT" != /* ]]; then
  CONTAINER_AUTH_ROOT="$REPO_ROOT/$CONTAINER_AUTH_ROOT"
fi
if [[ "$CONTAINER_ENV_FILE" == "~"* ]]; then
  CONTAINER_ENV_FILE="${CONTAINER_ENV_FILE/#\~/$HOME}"
fi
if [ -n "$CONTAINER_ENV_FILE" ] && [[ "$CONTAINER_ENV_FILE" != /* ]]; then
  CONTAINER_ENV_FILE="$REPO_ROOT/$CONTAINER_ENV_FILE"
fi
if [ -z "$CONTAINER_ENV_FILE" ] && [ -f "$REPO_ROOT/.env" ]; then
  CONTAINER_ENV_FILE="$REPO_ROOT/.env"
fi

if [ -z "$INSTRUCT_PATH" ]; then
  if [ "$HARNESS_EVALUATED_ATTEMPT" -eq 1 ]; then
    INSTRUCT_PATH="$REPO_ROOT/src/prompts/harness_edit.md"
  else
    INSTRUCT_PATH="$REPO_ROOT/src/prompts/general.md"
  fi
  if [ ! -f "$INSTRUCT_PATH" ]; then
    INSTRUCT_PATH="$TASK_DIR/goal.md"
  fi
fi
if [ -z "$SEED_PROGRAM_DIR" ]; then
  SEED_PROGRAM_DIR="$TASK_DIR/program_0"
fi
if [ -n "$AGENT_MODEL" ] && [ -z "${AGENT_MODEL//[[:space:]]/}" ]; then
  echo "--agent-model must be non-empty when provided" >&2
  exit 1
fi
if [ -n "$AGENT_PROFILE" ] && [ -z "${AGENT_PROFILE//[[:space:]]/}" ]; then
  echo "--agent-profile must be non-empty when provided" >&2
  exit 1
fi
if [ -n "$AGENT_PROFILE" ] && [ "$AGENT" != "codex" ]; then
  echo "--agent-profile is only supported with --agent codex" >&2
  exit 1
fi
if [ -n "$AGENT_TIMEOUT_SECONDS" ] && ! [[ "$AGENT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--agent-timeout-seconds must be a positive integer" >&2
  exit 1
fi
validate_agent_reasoning_effort "$AGENT" "$AGENT_REASONING_EFFORT" || exit 1
if ! [[ "$HARNESS_EVAL_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--harness-evaluation-timeout-seconds must be a positive integer" >&2
  exit 1
fi
export AGENT_REASONING_EFFORT

if [ ! -d "$TASK_DIR" ]; then
  echo "Missing task directory: $TASK_DIR" >&2
  exit 1
fi
UNRESOLVED_TASK_DIR="$TASK_DIR"
RESOLVED_BOUNDARY_PATHS=()
mapfile -d '' -t RESOLVED_BOUNDARY_PATHS < <(
  "$PYTHON_BIN" - "$TASK_DIR" "$RESULTS_DIR" <<'PY'
import sys
from pathlib import Path

task_source = Path(sys.argv[1]).resolve(strict=True)
results_root = Path(sys.argv[2]).resolve(strict=False)
if task_source == results_root or task_source.is_relative_to(results_root):
    raise SystemExit(
        "canonical task source is inside agent-writable results: "
        f"{task_source} (results: {results_root})"
    )
for path in (task_source, results_root):
    sys.stdout.buffer.write(str(path).encode() + b"\0")
PY
)
if [ "${#RESOLVED_BOUNDARY_PATHS[@]}" -ne 2 ]; then
  echo "Failed to resolve task/results security boundary" >&2
  exit 1
fi
CANONICAL_TASK_SOURCE_DIR="${RESOLVED_BOUNDARY_PATHS[0]}"
RESULTS_DIR="${RESOLVED_BOUNDARY_PATHS[1]}"
TASK_DIR="$CANONICAL_TASK_SOURCE_DIR"
if [ "$SEED_PROGRAM_DIR" = "$UNRESOLVED_TASK_DIR/program_0" ]; then
  SEED_PROGRAM_DIR="$TASK_DIR/program_0"
fi
if [ -e "$CANONICAL_TASK_SOURCE_DIR/task_contract.json" ] \
  || [ -L "$CANONICAL_TASK_SOURCE_DIR/task_contract.json" ]; then
  if [ -L "$CANONICAL_TASK_SOURCE_DIR/task_contract.json" ] \
    || [ ! -f "$CANONICAL_TASK_SOURCE_DIR/task_contract.json" ]; then
    echo "task_contract.json must be a regular file" >&2
    exit 1
  fi
  CONTRACTED_TASK=1
  export AGENT_CONTAINER_MOUNT_DOCKER_SOCK=0
  export AGENT_CONTAINER_SESSION_NAME=""
  export AGENT_CONTAINER_EXTRA_MOUNTS=""
  export AGENT_CONTAINER_MOUNT_ROOT=""
  export AGENT_CONTAINER_WRITABLE_FILE_SOURCE=""
  export AGENT_CONTAINER_WRITABLE_FILE_RELATIVE=""
  export AGENT_CONTAINER_TASK_PYTHON=""
  export AGENT_CONTAINER_PASSTHROUGH_KEYS=""
  export AGENT_CONTAINER_SECRET_PASSTHROUGH_KEYS=""
fi
if [ ! -d "$TASK_DIR/program_0" ]; then
  echo "Missing required directory: $TASK_DIR/program_0" >&2
  exit 1
fi
if [ ! -f "$TASK_DIR/program_0/program_spec.toml" ]; then
  echo "Missing required file: $TASK_DIR/program_0/program_spec.toml" >&2
  exit 1
fi
if [ ! -d "$SEED_PROGRAM_DIR" ]; then
  echo "Missing seed program directory: $SEED_PROGRAM_DIR" >&2
  exit 1
fi
if [ ! -f "$SEED_PROGRAM_DIR/program_spec.toml" ]; then
  echo "Missing seed program spec: $SEED_PROGRAM_DIR/program_spec.toml" >&2
  exit 1
fi
if [ ! -f "$TASK_DIR/capture.py" ]; then
  echo "Missing required file: $TASK_DIR/capture.py" >&2
  exit 1
fi
if [ ! -f "$TASK_DIR/evaluate.py" ]; then
  echo "Missing required file: $TASK_DIR/evaluate.py" >&2
  exit 1
fi
if [ ! -f "$INSTRUCT_PATH" ]; then
  echo "Missing instruction template: $INSTRUCT_PATH" >&2
  exit 1
fi
if [ -n "$RUBRICS_FILE" ] && [ ! -f "$RUBRICS_FILE" ]; then
  echo "Missing rubrics file: $RUBRICS_FILE" >&2
  exit 1
fi
if [ -n "$IDEA_FILE" ] && [ ! -f "$IDEA_FILE" ]; then
  echo "Missing idea file: $IDEA_FILE" >&2
  exit 1
fi
if ! [[ "$MAX_PROGRAM_EVALS_PER_ATTEMPT" =~ ^[0-9]+$ ]] || [ "$MAX_PROGRAM_EVALS_PER_ATTEMPT" -le 0 ]; then
  echo "--max-program-evals-per-attempt must be a positive integer (got: $MAX_PROGRAM_EVALS_PER_ATTEMPT)" >&2
  exit 1
fi
if [ "$CONTAINERIZE" -eq 1 ]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required when --containerize is enabled" >&2
    exit 1
  fi
  if [ ! -d "$CONTAINER_AUTH_ROOT" ]; then
    echo "Missing container auth root directory: $CONTAINER_AUTH_ROOT" >&2
    exit 1
  fi
  if [ -n "$CONTAINER_ENV_FILE" ] && [ ! -f "$CONTAINER_ENV_FILE" ]; then
    echo "Missing container env file: $CONTAINER_ENV_FILE" >&2
    exit 1
  fi
fi

validate_path_token() {
  local label="$1"
  local value="$2"
  if [[ "$value" == *"/"* ]] || [[ "$value" == *".."* ]]; then
    echo "Invalid ${label}: '${value}' (must not contain '/' or '..')" >&2
    exit 1
  fi
}

mkdir -p "$RESULTS_DIR"
configure_cost_tracking "$RESULTS_DIR" run
mkdir -p "$GLOBAL_MEMORY_DIR"
mkdir -p "$CONCLUSIONS_DIR"
TASK_SOURCE_DIR="$CANONICAL_TASK_SOURCE_DIR"
if [ "$CONTRACTED_TASK" -eq 1 ]; then
  if [ "$CONTAINERIZE" -ne 1 ]; then
    echo "Contracted tasks require container isolation to seal final evaluation artifacts" >&2
    exit 1
  fi
  SEALED_VALUES=()
  mapfile -d '' -t SEALED_VALUES < <(
    "$PYTHON_BIN" - "$REPO_ROOT" "$TASK_SOURCE_DIR" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
task_dir = Path(sys.argv[2])
sys.path.insert(0, str(repo_root))

from src.investigation.task_contract import load_task_contract  # noqa: E402

contract = load_task_contract(task_dir / "task_contract.json")
for value in (contract.evaluators.final, contract.resources.final_root):
    sys.stdout.buffer.write(value.encode("ascii") + b"\0")
PY
  )
  if [ "${#SEALED_VALUES[@]}" -ne 2 ]; then
    echo "Invalid sealed artifact declarations" >&2
    exit 1
  fi
  SEALED_FINAL_EVALUATOR="${SEALED_VALUES[0]}"
  SEALED_FINAL_RESOURCES="${SEALED_VALUES[1]}"
  export AGENT_TASK_SEALED_FILES="${SEALED_FINAL_EVALUATOR}"$'\n'"task_contract.json"$'\n'".agentic_task_validation.json"
  if [ -f "$TASK_SOURCE_DIR/manual_validation.json" ]; then
    AGENT_TASK_SEALED_FILES+=$'\n'"manual_validation.json"
  fi
  export AGENT_TASK_SEALED_DIRS="$SEALED_FINAL_RESOURCES"
  AGENT_TASK_SEAL_EMPTY_DIR="$RESULTS_DIR/.task-seal-empty"
  mkdir -p "$AGENT_TASK_SEAL_EMPTY_DIR"
  if find "$AGENT_TASK_SEAL_EMPTY_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "Task seal directory is not empty: $AGENT_TASK_SEAL_EMPTY_DIR" >&2
    exit 1
  fi
  export AGENT_TASK_SEAL_EMPTY_DIR
  export AGENT_CONTAINER_MOUNT_DOCKER_SOCK=0
fi
STAGED_TASK_DIR="$(ensure_staged_task_copy "$REPO_ROOT" "$TASK_SOURCE_DIR" "$RESULTS_DIR")"
TASK_DIR="$STAGED_TASK_DIR"
HARNESS_PYTHON_BIN="$PYTHON_BIN"
if [ "$HARNESS_EVALUATED_ATTEMPT" -eq 1 ] && { [ "$CONTRACTED_TASK" -eq 1 ] || [ "$CONTAINERIZE" -eq 1 ]; }; then
  # Sealed evaluation mounts the canonical task's managed venv; the staged copy's venv
  # would not match --task-source-dir.
  HARNESS_PYTHON_BIN="$(require_managed_task_python "$CANONICAL_TASK_SOURCE_DIR")"
else
  TASK_VENV_PYTHON="$TASK_DIR/.venv/bin/python"
  if [ -f "$TASK_VENV_PYTHON" ] && [ -x "$TASK_VENV_PYTHON" ]; then
    HARNESS_PYTHON_BIN="$TASK_VENV_PYTHON"
  fi
fi
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -z "$ATTEMPT_ID" ]; then
  ATTEMPT_ID="attempt_${TIMESTAMP}"
fi

IDEA_ID="$(
  "$PYTHON_BIN" - "$IDEA_FILE" <<'PY'
import json
import sys
from pathlib import Path

raw_path = sys.argv[1] if len(sys.argv) > 1 else ""
if not raw_path:
    print("idea_unknown")
    raise SystemExit(0)

path = Path(raw_path)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("idea_unknown")
    raise SystemExit(0)

idea_id = data.get("idea_id") if isinstance(data, dict) else None
if isinstance(idea_id, str) and idea_id.strip():
    print(idea_id.strip())
else:
    print("idea_unknown")
PY
)"

validate_path_token "attempt id" "$ATTEMPT_ID"
validate_path_token "idea id" "$IDEA_ID"

if [ -z "$IDEA_DIR_NAME" ]; then
  IDEA_DIR_NAME="$(
    "$PYTHON_BIN" - "$REPO_ROOT" "$IDEA_FILE" "$IDEA_ID" <<'PY'
import json
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
idea_file_raw = sys.argv[2]
idea_id = sys.argv[3]

sys.path.insert(0, str(script_dir))
from src.orchestrator import state as state_store

idea_title = None
if idea_file_raw:
    idea_file = Path(idea_file_raw)
    if idea_file.exists():
        try:
            parsed = json.loads(idea_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            raw_title = parsed.get("title")
            if isinstance(raw_title, str):
                idea_title = raw_title

print(
    state_store.idea_dir_name(
        cycle_index=None,
        idea_title=idea_title,
        idea_id=idea_id,
    )
)
PY
  )"
fi

validate_path_token "idea directory name" "$IDEA_DIR_NAME"
if [ -z "$ATTEMPT_DIR_NAME" ]; then
  ATTEMPT_DIR_NAME="idea_root"
fi
validate_path_token "attempt directory name" "$ATTEMPT_DIR_NAME"

EXPERIMENT_DIR="$RESULTS_DIR/ideas/$IDEA_DIR_NAME"
mkdir -p "$EXPERIMENT_DIR"
if ls "$EXPERIMENT_DIR"/program_* >/dev/null 2>&1 || [ -f "$EXPERIMENT_DIR/rubric_status.json" ]; then
  echo "Idea directory already contains run artifacts: $EXPERIMENT_DIR" >&2
  exit 1
fi
mkdir -p "$EXPERIMENT_DIR/program_0"

# Stage task artifacts into isolated experiment workspace.
(cd "$REPO_ROOT" && "$PYTHON_BIN" -m src.investigation.program_clone --from "$SEED_PROGRAM_DIR" --to "$EXPERIMENT_DIR/program_0")
SEARCH_EVALUATOR_SOURCE_DIR="$TASK_DIR"
if [ -f "$TASK_SOURCE_DIR/task_contract.json" ]; then
  SEARCH_EVALUATOR_SOURCE_DIR="$TASK_SOURCE_DIR"
fi
cp "$SEARCH_EVALUATOR_SOURCE_DIR/capture.py" "$EXPERIMENT_DIR/capture.py"
cp "$SEARCH_EVALUATOR_SOURCE_DIR/evaluate.py" "$EXPERIMENT_DIR/evaluate.py"
# Copy shared parent-level python helpers needed by legacy task evaluators.
TASK_PARENT_DIR="$(dirname "$TASK_SOURCE_DIR")"
SEARCH_HELPER_NAMES=()
if [ -f "$TASK_SOURCE_DIR/task_contract.json" ]; then
  SEARCH_HELPER_OUTPUT=""
  if ! SEARCH_HELPER_OUTPUT="$(
    "$PYTHON_BIN" - "$REPO_ROOT" "$TASK_SOURCE_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
source_task_dir = Path(sys.argv[2])
sys.path.insert(0, str(repo_root))

from src.investigation.task_contract import load_task_contract  # noqa: E402

contract = load_task_contract(source_task_dir / "task_contract.json")
excluded_names = {
    Path(contract.evaluators.search).name,
    Path(contract.evaluators.final).name,
    "capture.py",
    "__init__.py",
}
helper_names: list[str] = []
for artifact in contract.provenance.artifacts:
    if artifact.role != "search_helper":
        continue
    relative = Path(artifact.path)
    if relative.parent != Path(".") or relative.suffix != ".py":
        raise SystemExit(
            f"Contract search_helper must be a top-level Python file: {artifact.path}"
        )
    if relative.name in excluded_names:
        raise SystemExit(f"Reserved search_helper declaration: {artifact.path}")
    helper_path = source_task_dir / relative
    if helper_path.is_symlink() or not helper_path.is_file():
        raise SystemExit(f"Unsafe search_helper declaration: {artifact.path}")
    digest = hashlib.sha256(helper_path.read_bytes()).hexdigest()
    if digest != artifact.sha256:
        raise SystemExit(f"search_helper digest mismatch: {artifact.path}")
    helper_names.append(relative.name)
for helper_name in sorted(helper_names):
    print(helper_name)
PY
  )"; then
    echo "Failed to load contracted search helper allowlist" >&2
    exit 1
  fi
  if [ -n "$SEARCH_HELPER_OUTPUT" ]; then
    mapfile -t SEARCH_HELPER_NAMES <<< "$SEARCH_HELPER_OUTPUT"
  fi
fi

should_stage_python_helper() {
  local helper_name="$1"
  local allowed_name
  if [ ! -f "$TASK_SOURCE_DIR/task_contract.json" ]; then
    return 0
  fi
  for allowed_name in "${SEARCH_HELPER_NAMES[@]}"; do
    if [ "$helper_name" = "$allowed_name" ]; then
      return 0
    fi
  done
  return 1
}

if [ -f "$TASK_SOURCE_DIR/task_contract.json" ]; then
  "$PYTHON_BIN" - "$TASK_DIR" "$SEALED_FINAL_EVALUATOR" "$SEALED_FINAL_RESOURCES" \
    "${SEARCH_HELPER_NAMES[@]}" <<'PY'
import shutil
import sys
import hashlib
from pathlib import Path

task_dir = Path(sys.argv[1])
final_evaluator = sys.argv[2]
final_resources = sys.argv[3]
search_helper_names = set(sys.argv[4:])
kept_python_names = {"capture.py", "evaluate.py", *search_helper_names}
if task_dir.is_symlink() or not task_dir.is_dir():
    raise SystemExit(f"Unsafe staged task directory: {task_dir}")


def remove_staged_path(relative_path: str) -> None:
    path = task_dir / relative_path
    current = task_dir
    for part in Path(relative_path).parts:
        current /= part
        if current.is_symlink():
            raise SystemExit(f"Unsafe staged task artifact: {current}")
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        raise SystemExit(f"Unsafe staged task artifact: {path}")


for sealed_path in (
    final_evaluator,
    final_resources,
    "task_contract.json",
    ".agentic_task_validation.json",
    "manual_validation.json",
):
    remove_staged_path(sealed_path)

for helper_path in task_dir.glob("*.py"):
    if helper_path.name not in kept_python_names:
        remove_staged_path(helper_path.name)
PY
fi

AGENT_VISIBLE_TASK_SOURCE_DIR="$CANONICAL_TASK_SOURCE_DIR"
if [ "$CONTRACTED_TASK" -eq 1 ]; then
  SEARCH_ONLY_TASK_SOURCE_DIR="$(mktemp -d "$SCRIPT_TEMP_ROOT/agentic-task-search-XXXXXX")"
  trap cleanup_search_only_task_source EXIT
  "$PYTHON_BIN" - \
    "$REPO_ROOT" \
    "$CANONICAL_TASK_SOURCE_DIR" \
    "$SEARCH_ONLY_TASK_SOURCE_DIR" \
    "$SEALED_FINAL_EVALUATOR" \
    "$SEALED_FINAL_RESOURCES" \
    "${SEARCH_HELPER_NAMES[@]}" <<'PY'
import shutil
import sys
from pathlib import Path

from src.investigation.content_publication import ContentPublicationGuard
from src.investigation.search_runtime import package_search_runtime

repo_root = Path(sys.argv[1])
source = Path(sys.argv[2])
destination = Path(sys.argv[3])
final_evaluator = sys.argv[4]
final_resources = Path(sys.argv[5])
helper_names = sys.argv[6:]
if source.is_symlink() or destination.is_symlink():
    raise SystemExit("Unsafe search-only task source")


def copy_file(relative: Path) -> None:
    source_path = source / relative
    destination_path = destination / relative
    if source_path.is_symlink() or not source_path.is_file():
        raise SystemExit(f"Unsafe search-only task file: {relative.as_posix()}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


for relative in (Path("task.md"), Path("capture.py"), Path("evaluate.py")):
    copy_file(relative)
for helper_name in helper_names:
    copy_file(Path(helper_name))
search_resources = source / "resources" / "search"
if search_resources.is_symlink() or not search_resources.is_dir():
    raise SystemExit("Missing safe contracted search resources")
for path in search_resources.rglob("*"):
    relative = path.relative_to(source)
    if path.is_symlink() or not (path.is_file() or path.is_dir()):
        raise SystemExit(f"Unsafe contracted search resource: {relative.as_posix()}")
shutil.copytree(search_resources, destination / "resources" / "search")
if (destination / final_evaluator).exists() or (destination / final_resources).exists():
    raise SystemExit("Search-only task source contains final artifacts")

sealed_paths = [source / final_evaluator]
sealed_root = source / final_resources
if sealed_root.is_dir() and not sealed_root.is_symlink():
    sealed_paths.extend(path for path in sealed_root.rglob("*") if path.is_file())
publication_guard = ContentPublicationGuard.capture(
    sealed_files=sealed_paths,
    public_files=(path for path in destination.rglob("*") if path.is_file()),
)

package_search_runtime(repo_root, destination)
published_files = []
for path in destination.rglob("*"):
    if path.is_symlink() or not (path.is_file() or path.is_dir()):
        raise SystemExit("Search runtime packager created an unsafe artifact")
    if path.is_file():
        published_files.append(path)
if publication_guard.first_violation(published_files) is not None:
    raise SystemExit("Search runtime packager copied sealed content")

destination.chmod(0o555)
PY
  AGENT_VISIBLE_TASK_SOURCE_DIR="$SEARCH_ONLY_TASK_SOURCE_DIR"
  export AGENT_CONTAINER_TASK_PYTHON="/task-source/.search-runtime/bin/python"
  export AGENT_TASK_SEALED_FILES=""
  export AGENT_TASK_SEALED_DIRS=""
fi

for helper_path in "$TASK_PARENT_DIR"/*.py; do
  [ -e "$helper_path" ] || continue
  helper_name="$(basename "$helper_path")"
  case "$helper_name" in
    __init__.py|capture.py|evaluate.py)
      continue
      ;;
  esac
  should_stage_python_helper "$helper_name" || continue
  cp "$helper_path" "$EXPERIMENT_DIR/$helper_name"
done
# Copy additional task-local python helpers needed by evaluate.py
# (e.g., docker_eval.py for GPT speedrun tasks). Task-local helpers win
# over parent-level helpers with the same file name.
for helper_path in "$SEARCH_EVALUATOR_SOURCE_DIR"/*.py; do
  [ -e "$helper_path" ] || continue
  helper_name="$(basename "$helper_path")"
  case "$helper_name" in
    capture.py|evaluate.py)
      continue
      ;;
  esac
  should_stage_python_helper "$helper_name" || continue
  cp "$helper_path" "$EXPERIMENT_DIR/$helper_name"
done
if [ -f "$SEARCH_EVALUATOR_SOURCE_DIR/task.md" ]; then
  cp "$SEARCH_EVALUATOR_SOURCE_DIR/task.md" "$EXPERIMENT_DIR/task.md"
fi

if [ -n "$IDEA_FILE" ]; then
  cp "$IDEA_FILE" "$EXPERIMENT_DIR/active_idea.json"
else
  printf '%s\n' '{}' > "$EXPERIMENT_DIR/active_idea.json"
fi
if [ -n "$RUBRICS_FILE" ]; then
  cp "$RUBRICS_FILE" "$EXPERIMENT_DIR/active_rubrics.json"
else
  printf '%s\n' '[]' > "$EXPERIMENT_DIR/active_rubrics.json"
fi
export ATTEMPT_ID
export IDEA_ID
export IDEA_DIR_NAME
export ATTEMPT_DIR_NAME
export MAX_PROGRAM_EVALS_PER_ATTEMPT
export TASK_SOURCE_DIR="$AGENT_VISIBLE_TASK_SOURCE_DIR"
export GLOBAL_MEMORY_MD_FILE="$GLOBAL_MEMORY_DIR/memory.md"
export GLOBAL_MEMORY_JSONL_FILE="$GLOBAL_MEMORY_DIR/memory.jsonl"
export CONCLUSIONS_MD_FILE="$CONCLUSIONS_DIR/conclusions.md"
export CONCLUSIONS_HISTORY_FILE="$CONCLUSIONS_DIR/conclusions_history.jsonl"
export OTHER_WORKERS_CONCLUSIONS_FILE="$CONCLUSIONS_DIR/conclusions_other_workers.md"
export WORKER_INDEX

case "$AGENT" in
  claude) AGENT_CONFIG_DIR=".claude" ;;
  codex) AGENT_CONFIG_DIR=".codex" ;;
  cursor) AGENT_CONFIG_DIR=".cursor" ;;
  opencode) AGENT_CONFIG_DIR=".opencode" ;;
  gemini) AGENT_CONFIG_DIR=".gemini" ;;
  pi) AGENT_CONFIG_DIR=".pi/agent" ;;
  *)
    echo "Unsupported agent: $AGENT" >&2
    exit 1
    ;;
esac

FORMATTED_SKILLS_REL_DIR="$AGENT_CONFIG_DIR/skills"
RUN_SKILL_STEP="run"
if [ "$HARNESS_EVALUATED_ATTEMPT" -eq 1 ]; then
  RUN_SKILL_STEP="run_harness_eval"
fi
(
  cd "$EXPERIMENT_DIR"
  "$PYTHON_BIN" - "$REPO_ROOT" "$FORMATTED_SKILLS_REL_DIR" "$RESULTS_DIR" "$GLOBAL_MEMORY_DIR" "$ATTEMPT_ID" "$IDEA_ID" "$IDEA_DIR_NAME" "$ATTEMPT_DIR_NAME" "$MAX_PROGRAM_EVALS_PER_ATTEMPT" "$CONCLUSIONS_DIR" "$WORKER_INDEX" "$RUN_SKILL_STEP" <<'PY'
import json
import shutil
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
formatted_skills_dir = Path(sys.argv[2])
results_dir = Path(sys.argv[3])
global_memory_dir = Path(sys.argv[4])
attempt_id = sys.argv[5]
idea_id = sys.argv[6]
idea_dir_name = sys.argv[7]
attempt_dir_name = sys.argv[8]
max_program_evals_per_attempt = sys.argv[9]
conclusions_dir = Path(sys.argv[10])
worker_index = int(sys.argv[11])
run_skill_step = sys.argv[12]

sys.path.insert(0, str(script_dir))
from src.orchestrator import state as state_store
from src.investigation.skill_scope import skills_for_step, stage_skill_tree, validate_skill_set
from src.investigation.files import load_instructions, relative_display_path

skills_src = script_dir / "src" / "agents" / "skills"
if not skills_src.is_dir():
    raise SystemExit(f"Missing skills directory: {skills_src}")
selected_skills = skills_for_step(run_skill_step)
validate_skill_set(skills_src, selected_skills)

active_idea = Path("active_idea.json").read_text(encoding="utf-8").strip() or "{}"
active_rubrics = Path("active_rubrics.json").read_text(encoding="utf-8").strip() or "[]"

try:
    json.loads(active_idea)
except json.JSONDecodeError:
    active_idea = "{}"
try:
    json.loads(active_rubrics)
except json.JSONDecodeError:
    active_rubrics = "[]"

state_store.ensure_histories(
    results_dir,
    include_global_memory=results_dir == global_memory_dir,
)
state_store.ensure_global_memory_files(global_memory_dir)
state_store.ensure_conclusions_files(conclusions_dir)
global_memory_md = state_store.global_memory_md_path(global_memory_dir)
global_memory_jsonl = state_store.global_memory_jsonl_path(global_memory_dir)
global_memory_recent = state_store.tail_jsonl(global_memory_jsonl, max_lines=40)
conclusions_md = state_store.conclusions_md_path(conclusions_dir)
conclusions_history = state_store.conclusions_history_path(conclusions_dir)
other_workers_conclusions_md = state_store.other_workers_conclusions_md_path(conclusions_dir)


def _read_conclusions_body(path: Path, fallback: str) -> str:
    if not path.is_file():
        return fallback
    try:
        body = path.read_text(encoding="utf-8").strip()
    except OSError:
        return fallback
    return body or fallback


own_conclusions_text = _read_conclusions_body(
    conclusions_md, "_No conclusions yet._"
)
other_workers_conclusions_text = _read_conclusions_body(
    other_workers_conclusions_md, "_No conclusions from other workers yet._"
)

def summarize_jsonl_entries(entries, keys):
    if not entries:
        return "- none"
    lines = []
    for idx, entry in enumerate(entries[-5:], start=1):
        if not isinstance(entry, dict):
            lines.append(f"- item {idx}: <non-object entry>")
            continue
        parts = []
        for key in keys:
            value = entry.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            parts.append(f"{key}={value}")
        if not parts:
            parts.append("<no summarized fields>")
        lines.append(f"- item {idx}: " + "; ".join(parts))
    return "\n".join(lines)

base_dir = Path(".").resolve()

extra = {
    "REPO_ROOT": str(script_dir),
    "ACTIVE_IDEA_JSON": active_idea,
    "ACTIVE_RUBRICS_JSON": active_rubrics,
    "ACTIVE_IDEA_FILE": "./active_idea.json",
    "ACTIVE_RUBRICS_FILE": "./active_rubrics.json",
    "ATTEMPT_ID": attempt_id,
    "IDEA_ID": idea_id,
    "IDEA_DIR_NAME": idea_dir_name,
    "ATTEMPT_DIR_NAME": attempt_dir_name,
    "MAX_PROGRAM_EVALS_PER_ATTEMPT": max_program_evals_per_attempt,
    "WORKER_INDEX": str(worker_index),
    "GLOBAL_MEMORY_MD_FILE": relative_display_path(global_memory_md, base_dir=base_dir),
    "GLOBAL_MEMORY_JSONL_FILE": relative_display_path(global_memory_jsonl, base_dir=base_dir),
    "GLOBAL_MEMORY_RECENT_SUMMARY": summarize_jsonl_entries(
        global_memory_recent,
        ("step", "idea_id", "summary", "next_action"),
    ),
    "CONCLUSIONS_MD_FILE": relative_display_path(conclusions_md, base_dir=base_dir),
    "CONCLUSIONS_HISTORY_FILE": relative_display_path(conclusions_history, base_dir=base_dir),
    "OWN_CONCLUSIONS": own_conclusions_text,
    "OTHER_WORKERS_CONCLUSIONS": other_workers_conclusions_text,
}

for skill_name in selected_skills:
    src_dir = skills_src / skill_name
    dst_dir = formatted_skills_dir / src_dir.name
    stage_skill_tree(src_dir, dst_dir)

    src_skill = src_dir / "SKILL.md"
    if src_skill.is_file():
        rendered = load_instructions(src_skill, Path("."), Path("."), extra)
        (dst_dir / "SKILL.md").write_text(rendered, encoding="utf-8")
PY
)

PROMPT="$(
  cd "$EXPERIMENT_DIR"
  "$PYTHON_BIN" - "$REPO_ROOT" "$INSTRUCT_PATH" "$RESULTS_DIR" "$GLOBAL_MEMORY_DIR" "$ATTEMPT_ID" "$IDEA_ID" "$IDEA_DIR_NAME" "$ATTEMPT_DIR_NAME" "$MAX_PROGRAM_EVALS_PER_ATTEMPT" "$CONCLUSIONS_DIR" "$WORKER_INDEX" "$CANONICAL_TASK_SOURCE_DIR" <<'PY'
import json
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
instruct_path = Path(sys.argv[2])
results_dir = Path(sys.argv[3])
global_memory_dir = Path(sys.argv[4])
attempt_id = sys.argv[5]
idea_id = sys.argv[6]
idea_dir_name = sys.argv[7]
attempt_dir_name = sys.argv[8]
max_program_evals_per_attempt = sys.argv[9]
conclusions_dir = Path(sys.argv[10])
worker_index = int(sys.argv[11])
task_source_dir = Path(sys.argv[12])
sys.path.insert(0, str(script_dir))

from src.orchestrator import state as state_store
from src.investigation.public_task_contract import public_search_contract_section
from src.investigation.files import load_instructions, relative_display_path

active_idea = Path("active_idea.json").read_text(encoding="utf-8").strip() or "{}"
active_rubrics = Path("active_rubrics.json").read_text(encoding="utf-8").strip() or "[]"
contract_placeholder = "{{PUBLIC_SEARCH_CONTRACT_SECTION}}"
public_contract_section = (
    public_search_contract_section(task_source_dir)
    if contract_placeholder in instruct_path.read_text(encoding="utf-8")
    else ""
)

try:
    json.loads(active_idea)
except json.JSONDecodeError:
    active_idea = "{}"
try:
    json.loads(active_rubrics)
except json.JSONDecodeError:
    active_rubrics = "[]"

state_store.ensure_histories(
    results_dir,
    include_global_memory=results_dir == global_memory_dir,
)
state_store.ensure_global_memory_files(global_memory_dir)
state_store.ensure_conclusions_files(conclusions_dir)
global_memory_md = state_store.global_memory_md_path(global_memory_dir)
global_memory_jsonl = state_store.global_memory_jsonl_path(global_memory_dir)
global_memory_recent = state_store.tail_jsonl(global_memory_jsonl, max_lines=40)
conclusions_md = state_store.conclusions_md_path(conclusions_dir)
conclusions_history = state_store.conclusions_history_path(conclusions_dir)
other_workers_conclusions_md = state_store.other_workers_conclusions_md_path(conclusions_dir)


def _read_conclusions_body(path: Path, fallback: str) -> str:
    if not path.is_file():
        return fallback
    try:
        body = path.read_text(encoding="utf-8").strip()
    except OSError:
        return fallback
    return body or fallback


own_conclusions_text = _read_conclusions_body(
    conclusions_md, "_No conclusions yet._"
)
other_workers_conclusions_text = _read_conclusions_body(
    other_workers_conclusions_md, "_No conclusions from other workers yet._"
)

def summarize_jsonl_entries(entries, keys):
    if not entries:
        return "- none"
    lines = []
    for idx, entry in enumerate(entries[-5:], start=1):
        if not isinstance(entry, dict):
            lines.append(f"- item {idx}: <non-object entry>")
            continue
        parts = []
        for key in keys:
            value = entry.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            parts.append(f"{key}={value}")
        if not parts:
            parts.append("<no summarized fields>")
        lines.append(f"- item {idx}: " + "; ".join(parts))
    return "\n".join(lines)

base_dir = Path(".").resolve()

extra = {
    "REPO_ROOT": str(script_dir),
    "ACTIVE_IDEA_JSON": active_idea,
    "ACTIVE_RUBRICS_JSON": active_rubrics,
    "ACTIVE_IDEA_FILE": "./active_idea.json",
    "ACTIVE_RUBRICS_FILE": "./active_rubrics.json",
    "ATTEMPT_ID": attempt_id,
    "IDEA_ID": idea_id,
    "IDEA_DIR_NAME": idea_dir_name,
    "ATTEMPT_DIR_NAME": attempt_dir_name,
    "MAX_PROGRAM_EVALS_PER_ATTEMPT": max_program_evals_per_attempt,
    "WORKER_INDEX": str(worker_index),
    "PUBLIC_SEARCH_CONTRACT_SECTION": public_contract_section,
    "GLOBAL_MEMORY_MD_FILE": relative_display_path(global_memory_md, base_dir=base_dir),
    "GLOBAL_MEMORY_JSONL_FILE": relative_display_path(global_memory_jsonl, base_dir=base_dir),
    "GLOBAL_MEMORY_RECENT_SUMMARY": summarize_jsonl_entries(
        global_memory_recent,
        ("step", "idea_id", "summary", "next_action"),
    ),
    "CONCLUSIONS_MD_FILE": relative_display_path(conclusions_md, base_dir=base_dir),
    "CONCLUSIONS_HISTORY_FILE": relative_display_path(conclusions_history, base_dir=base_dir),
    "OWN_CONCLUSIONS": own_conclusions_text,
    "OTHER_WORKERS_CONCLUSIONS": other_workers_conclusions_text,
}

template = load_instructions(instruct_path, Path("."), Path("."), extra)
print(template)
PY
)"

write_agent_prompt_file "$EXPERIMENT_DIR/prompt.md" "$PROMPT"
LOG_FILE="$EXPERIMENT_DIR/agent_${AGENT}_${TIMESTAMP}.log"
LOG_PIPE="$EXPERIMENT_DIR/.agent_stream_${TIMESTAMP}.fifo"
LATEST_RUN_MANIFEST="$RESULTS_DIR/latest_run_manifest.json"
AGENT_PID=""
TEE_PID=""
INTERRUPTED=0
CLEANUP_IN_PROGRESS=0
FORCE_CLEANUP=0
INTERRUPT_COUNT=0
INTERRUPT_CLEANUP_TIMEOUT_SECONDS=8
INTERRUPT_COMMAND_TIMEOUT_SECONDS=3

"$PYTHON_BIN" - "$LATEST_RUN_MANIFEST" "$EXPERIMENT_DIR" "$LOG_FILE" "$AGENT" "$ATTEMPT_ID" "$IDEA_ID" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "agent": sys.argv[4],
    "attempt_id": sys.argv[5],
    "idea_id": sys.argv[6],
    "experiment_dir": sys.argv[2],
    "log_file": sys.argv[3],
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
PY

kill_workspace_processes() {
  local signal_name="$1"
  local pid

  if [ -z "${EXPERIMENT_DIR:-}" ] || [ ! -d "$EXPERIMENT_DIR" ]; then
    return 0
  fi

  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    if [ "$pid" = "$$" ]; then
      continue
    fi
    if [ -n "${AGENT_PID:-}" ] && [ "$pid" = "$AGENT_PID" ]; then
      continue
    fi
    if [ -n "${TEE_PID:-}" ] && [ "$pid" = "$TEE_PID" ]; then
      continue
    fi
    kill_pid_tree "$pid" "$signal_name"
  done < <(lsof -t +D "$EXPERIMENT_DIR" 2>/dev/null | sort -u)
}

command_with_timeout() {
  local timeout_seconds="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$timeout_seconds" "$@"
    return $?
  fi
  "$@"
}

wait_with_timeout() {
  local pid="$1"
  local timeout_seconds="$2"
  local start_ts
  start_ts="$(date +%s)"

  while kill -0 "$pid" 2>/dev/null; do
    if [ $(( $(date +%s) - start_ts )) -ge "$timeout_seconds" ]; then
      return 124
    fi
    sleep 0.1
  done

  wait "$pid"
}

cleanup_run_processes() {
  if [ "${CLEANUP_IN_PROGRESS:-0}" -eq 1 ]; then
    return 0
  fi
  CLEANUP_IN_PROGRESS=1
  local cleanup_timeout="${INTERRUPT_CLEANUP_TIMEOUT_SECONDS:-8}"
  local command_timeout="${INTERRUPT_COMMAND_TIMEOUT_SECONDS:-3}"

  if [ -n "${AGENT_PID:-}" ] && kill -0 "$AGENT_PID" 2>/dev/null; then
    if [ "${FORCE_CLEANUP:-0}" -eq 1 ]; then
      kill -KILL "$AGENT_PID" 2>/dev/null || true
      wait_with_timeout "$AGENT_PID" 1 || true
    else
      terminate_pid_tree "$AGENT_PID" 0.2
      if ! wait_with_timeout "$AGENT_PID" "$cleanup_timeout"; then
        echo "Agent shutdown exceeded ${cleanup_timeout}s; forcing kill." >&2
        kill -KILL "$AGENT_PID" 2>/dev/null || true
        wait_with_timeout "$AGENT_PID" 1 || true
      fi
    fi
  fi
  AGENT_PID=""

  if [ -n "${TEE_PID:-}" ] && kill -0 "$TEE_PID" 2>/dev/null; then
    kill -TERM "$TEE_PID" 2>/dev/null || true
    if ! wait_with_timeout "$TEE_PID" 1; then
      kill -KILL "$TEE_PID" 2>/dev/null || true
      wait_with_timeout "$TEE_PID" 1 || true
    fi
  fi
  TEE_PID=""

  cleanup_search_only_task_source

  cleanup_contracted_container_env || true

  if [ "${INTERRUPTED:-0}" -eq 1 ] && [ -f "${LOG_FILE:-}" ]; then
    record_headless_cost_log \
      "$PYTHON_BIN" "$LOG_FILE" attempt \
      "attempt:${ATTEMPT_ID}:${TIMESTAMP}" interrupted
  fi

  # Some agent tool executions may daemonize evaluate workers outside the direct
  # agent child tree. Sweep any process still touching this idea workspace.
  if [ "${FORCE_CLEANUP:-0}" -eq 0 ]; then
    command_with_timeout "$command_timeout" bash -c \
      "$(declare -f kill_pid_tree); $(declare -f kill_workspace_processes); EXPERIMENT_DIR=\"\$1\"; kill_workspace_processes TERM" \
      _ "$EXPERIMENT_DIR" >/dev/null 2>&1 || true
    sleep 0.2
    command_with_timeout "$command_timeout" bash -c \
      "$(declare -f kill_pid_tree); $(declare -f kill_workspace_processes); EXPERIMENT_DIR=\"\$1\"; kill_workspace_processes KILL" \
      _ "$EXPERIMENT_DIR" >/dev/null 2>&1 || true
  else
    echo "Skipping workspace lsof sweep during forced cleanup." >&2
  fi

  if [ "$CONTAINERIZE" -eq 1 ] && [ -n "${AGENT_CONTAINER_SESSION_NAME:-}" ]; then
    command_with_timeout "$command_timeout" bash -lc "source \"$SCRIPT_DIR/lib/container_agent.sh\"; stop_container_session_force \"$AGENT_CONTAINER_SESSION_NAME\"" >/dev/null 2>&1 || true
  fi

  if [ -n "${LOG_PIPE:-}" ] && [ -p "$LOG_PIPE" ]; then
    rm -f "$LOG_PIPE"
  fi
  LOG_PIPE=""
}

on_interrupt() {
  INTERRUPT_COUNT=$((INTERRUPT_COUNT + 1))
  if [ "$INTERRUPT_COUNT" -ge 2 ]; then
    FORCE_CLEANUP=1
    trap - INT TERM
    echo "Second interrupt received. Forcing immediate shutdown..." >&2
    cleanup_run_processes
    exit 130
  fi

  trap on_interrupt INT TERM
  INTERRUPTED=1
  echo "Interrupt received. Starting bounded shutdown (press Ctrl+C again to force)." >&2
  cleanup_run_processes
  exit 130
}

trap on_interrupt INT TERM
trap cleanup_run_processes EXIT

if [ "$CONTRACTED_TASK" -eq 1 ] \
  && [ "$CONTAINERIZE" -eq 1 ] \
  && [ -n "$CONTAINER_ENV_FILE" ]; then
  CONTRACTED_CONTAINER_ENV_FILE="$(mktemp "$SCRIPT_TEMP_ROOT/agentic-run-provider-env-XXXXXX")"
  sanitize_env_file_for_docker \
    "$CONTAINER_ENV_FILE" "$CONTRACTED_CONTAINER_ENV_FILE" \
    "$(agent_provider_allowed_env_keys "$AGENT" "$AGENT_MODEL" "$CONTAINER_ENV_FILE")"
  chmod 600 "$CONTRACTED_CONTAINER_ENV_FILE"
  CONTAINER_ENV_FILE="$CONTRACTED_CONTAINER_ENV_FILE"
fi

echo "Experiment directory: $EXPERIMENT_DIR"
echo "Agent config directory: $EXPERIMENT_DIR/$AGENT_CONFIG_DIR"
echo "Logging to: $LOG_FILE"
echo "Latest run manifest: $LATEST_RUN_MANIFEST"
echo "Max program evals per attempt: $MAX_PROGRAM_EVALS_PER_ATTEMPT"
if [ "$HARNESS_EVALUATED_ATTEMPT" -eq 1 ]; then
  echo "Harness evaluated attempt: true"
fi
if [ -n "$AGENT_MODEL" ]; then
  echo "Agent model override: $AGENT_MODEL"
fi
if [ -n "$AGENT_REASONING_EFFORT" ]; then
  echo "Agent reasoning effort override: $AGENT_REASONING_EFFORT"
fi
if [ "$CONTAINERIZE" -eq 1 ]; then
  CONTAINER_HOME_DIR="$(prepare_container_home "$RESULTS_DIR")"
  seed_auth_into_home "$AGENT" "$CONTAINER_AUTH_ROOT" "$CONTAINER_HOME_DIR"
  echo "Containerized run: image=$CONTAINER_IMAGE network=$CONTAINER_NETWORK"
  echo "Container home directory: $CONTAINER_HOME_DIR"
  if [ -n "${AGENT_CONTAINER_SESSION_NAME:-}" ]; then
    echo "Container session: ${AGENT_CONTAINER_SESSION_NAME}"
  fi
  if [ -n "$CONTAINER_ENV_FILE" ]; then
    echo "Container env file: $CONTAINER_ENV_FILE"
  fi
else
  CONTAINER_HOME_DIR=""
fi

sanitize_docker_name() {
  local raw="$1"
  printf '%s' "$raw" | tr -c 'a-zA-Z0-9_.-' '-' | sed -e 's/^-*//' -e 's/-*$//'
}

mkfifo "$LOG_PIPE"
bounded_agent_log_writer "$PYTHON_BIN" "$LOG_FILE" "$AGENT_LOG_MAX_BYTES" \
  < "$LOG_PIPE" &
TEE_PID=$!

(
  if [ -n "${AGENTIC_TASK_FAILURE_FD:-}" ]; then
    exec {AGENTIC_TASK_FAILURE_FD}>&-
    unset AGENTIC_TASK_FAILURE_FD
  fi
  AGENT_PROMPT_FILE="prompt.md" \
  run_agent_prompt \
    "$AGENT" \
    "$PROMPT" \
    "$AGENT_MODEL" \
    "$AGENT_PROFILE" \
    "$EXPERIMENT_DIR" \
    "$CONTAINERIZE" \
    "$CONTAINER_IMAGE" \
    "$CONTAINER_NETWORK" \
    "$CONTAINER_HOME_DIR" \
    "$CONTAINER_ENV_FILE" \
    "$RESULTS_DIR"
) > "$LOG_PIPE" 2>&1 &
AGENT_PID=$!

set +e
AGENT_EXIT_CODE=0
TEE_EXIT_CODE=0
if [ -n "${AGENT_PID:-}" ]; then
  if [ "$HARNESS_EVALUATED_ATTEMPT" -eq 1 ] && [ -n "$AGENT_TIMEOUT_SECONDS" ]; then
    wait_with_timeout "$AGENT_PID" "$AGENT_TIMEOUT_SECONDS"
    AGENT_EXIT_CODE=$?
    if [ "$AGENT_EXIT_CODE" -eq 124 ]; then
      AGENT_EXIT_CODE=124
      terminate_pid_tree "$AGENT_PID" 0.2
    fi
  else
    wait "$AGENT_PID"
    AGENT_EXIT_CODE=$?
  fi
fi
if [ -n "${TEE_PID:-}" ]; then
  wait "$TEE_PID"
  TEE_EXIT_CODE=$?
fi
set -e

if [ "$TEE_EXIT_CODE" -ne 0 ]; then
  echo "[run.sh] infrastructure_failure agent_exit=$AGENT_EXIT_CODE logger_exit=$TEE_EXIT_CODE" >&2 || true
  if [ "$INTERRUPTED" -ne 1 ] && [ "$AGENT_EXIT_CODE" -ne 130 ] \
    && [ -n "${AGENTIC_TASK_FAILURE_FD:-}" ]; then
    "$PYTHON_BIN" -m src.investigation.task_failure \
      --log-failure-fd "$AGENTIC_TASK_FAILURE_FD" \
      --attempt-id "$ATTEMPT_ID" --idea-id "$IDEA_ID"
  fi
fi
record_headless_cost_log \
  "$PYTHON_BIN" "$LOG_FILE" attempt \
  "attempt:${ATTEMPT_ID}:${TIMESTAMP}" \
  "$(headless_cost_status "$AGENT_EXIT_CODE" "$INTERRUPTED")"

# Reusable sessions belong to the investigation CLI, or to a direct caller's teardown lifecycle.
cleanup_search_only_task_source
cleanup_contracted_container_env
trap - INT TERM EXIT

if [ "$INTERRUPTED" -eq 1 ] || [ "$AGENT_EXIT_CODE" -eq 130 ]; then
  exit 130
fi
if [ "$TEE_EXIT_CODE" -ne 0 ]; then
  exit 4
fi
if [ "$AGENT_EXIT_CODE" -ne 0 ] && ! {
  [ "$HARNESS_EVALUATED_ATTEMPT" -eq 1 ] && [ "$AGENT_EXIT_CODE" -eq 124 ];
}; then
  exit "$AGENT_EXIT_CODE"
fi
if [ "$AGENT_EXIT_CODE" -eq 124 ]; then
  echo "Agent timed out; harness will evaluate the candidate left in program_1." >&2
fi

if [ "$HARNESS_EVALUATED_ATTEMPT" -eq 1 ]; then
  HARNESS_EVAL_ARGS=(
    --experiment-dir "$EXPERIMENT_DIR" \
    --attempt-id "$ATTEMPT_ID" \
    --idea-id "$IDEA_ID" \
    --rubrics-file "$EXPERIMENT_DIR/active_rubrics.json" \
    --python-bin "$HARNESS_PYTHON_BIN" \
    --global-memory-md "$GLOBAL_MEMORY_MD_FILE" \
    --global-memory-jsonl "$GLOBAL_MEMORY_JSONL_FILE" \
    --timeout-seconds "$HARNESS_EVAL_TIMEOUT_SECONDS"
  )
  if [ "$CONTAINERIZE" -eq 1 ]; then
    HARNESS_EVAL_ARGS+=(
      --task-source-dir "$CANONICAL_TASK_SOURCE_DIR"
      --sandbox-image "$CONTAINER_IMAGE"
    )
    if [ "$GPU_EVALUATION" -eq 1 ]; then
      [ -n "${AGENT_CONTAINER_GPUS:-}" ] || {
        echo "--gpu-evaluation requires AGENT_CONTAINER_GPUS for containerized evaluation" >&2
        exit 1
      }
      HARNESS_EVAL_ARGS+=(--sandbox-gpus "$AGENT_CONTAINER_GPUS")
    fi
  fi
  if [ "$GPU_EVALUATION" -eq 1 ]; then
    HARNESS_EVAL_ARGS+=(--gpu-evaluation)
  fi
  if [ "$CONTAINERIZE" -eq 1 ] && [ -n "${AGENTIC_TASK_FAILURE_FD:-}" ]; then
    HARNESS_EVAL_ARGS+=(--task-failure-fd "$AGENTIC_TASK_FAILURE_FD")
  fi
  if [ "$CONTAINERIZE" -eq 1 ]; then
    "$PYTHON_BIN" -m src.investigation.harness_eval "${HARNESS_EVAL_ARGS[@]}"
  else
    # Native evaluators run from the experiment copy, which holds only evaluate.py: like the
    # sandbox (/task-source), point them at the task source for resources they read.
    TASK_SOURCE_DIR="$CANONICAL_TASK_SOURCE_DIR" \
      "$PYTHON_BIN" -m src.investigation.harness_eval "${HARNESS_EVAL_ARGS[@]}"
  fi
fi

"$PYTHON_BIN" - "$EXPERIMENT_DIR" "$ATTEMPT_ID" "$IDEA_ID" "$EXPERIMENT_DIR/active_rubrics.json" "$MAX_PROGRAM_EVALS_PER_ATTEMPT" "$GLOBAL_MEMORY_JSONL_FILE" <<'PY'
import json
import re
import sys
from pathlib import Path

from src.investigation.files import evaluation_results_complete

experiment_dir = Path(sys.argv[1])
attempt_id = sys.argv[2]
idea_id = sys.argv[3]
rubrics_path = Path(sys.argv[4])
max_program_evals_per_attempt = int(sys.argv[5])
global_memory_jsonl_path = Path(sys.argv[6])

PROGRAM_DIR_RE = re.compile(r"^program_(\d+)$")


def load_json(path: Path, label: str):
    if not path.exists():
        raise SystemExit(f"Missing required artifact: {label}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {label}: {exc}") from exc


def extract_target_ids(raw):
    target_rubrics = []
    if isinstance(raw, list):
        target_rubrics = raw
    elif isinstance(raw, dict):
        maybe = raw.get("target_rubrics")
        if isinstance(maybe, list):
            target_rubrics = maybe

    return {
        item["id"]
        for item in target_rubrics
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
    }


def validate_status(path: Path, label: str, target_ids: set[str]):
    data = load_json(path, label)
    if not isinstance(data, dict):
        raise SystemExit(f"{label} must be a JSON object")

    required = ("attempt_id", "idea_id", "overall_pass", "rubrics")
    for key in required:
        if key not in data:
            raise SystemExit(f"{label} missing key: {key}")

    if data["attempt_id"] != attempt_id:
        raise SystemExit(f"{label} attempt_id mismatch: expected {attempt_id}")
    if data["idea_id"] != idea_id:
        raise SystemExit(f"{label} idea_id mismatch: expected {idea_id}")
    if not isinstance(data["overall_pass"], bool):
        raise SystemExit(f"{label} overall_pass must be boolean")
    if not isinstance(data["rubrics"], list):
        raise SystemExit(f"{label} rubrics must be an array")

    reported_ids: set[str] = set()
    all_pass = True
    for idx, item in enumerate(data["rubrics"]):
        if not isinstance(item, dict):
            raise SystemExit(f"{label} rubrics[{idx}] must be an object")
        if not isinstance(item.get("id"), str) or not item["id"].strip():
            raise SystemExit(f"{label} rubrics[{idx}].id must be non-empty string")
        if not isinstance(item.get("pass"), bool):
            raise SystemExit(f"{label} rubrics[{idx}].pass must be boolean")
        reported_ids.add(item["id"])
        if not item["pass"]:
            all_pass = False

    if target_ids and reported_ids != target_ids:
        raise SystemExit(
            f"{label} rubric ids mismatch: expected {sorted(target_ids)} got {sorted(reported_ids)}"
        )
    if data["overall_pass"] != all_pass:
        raise SystemExit(f"{label} overall_pass must equal all(rubrics[*].pass)")
    return data


def read_jsonl_entries(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise SystemExit(f"Missing required global memory artifact: {path}")
    entries: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries


try:
    target_data = json.loads(rubrics_path.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    raise SystemExit(f"Invalid active_rubrics.json: {exc}") from exc

target_ids = extract_target_ids(target_data)

evaluated_dirs: list[tuple[int, Path]] = []
for entry in experiment_dir.iterdir():
    if not entry.is_dir():
        continue
    match = PROGRAM_DIR_RE.match(entry.name)
    if not match:
        continue
    if (
        (entry / "metrics.json").exists()
        and (entry / "correct.json").exists()
        and evaluation_results_complete(entry)
    ):
        evaluated_dirs.append((int(match.group(1)), entry))

evaluated_dirs.sort(key=lambda item: item[0])
candidate_dirs = [(program_idx, program_dir) for program_idx, program_dir in evaluated_dirs if program_idx > 0]
candidate_eval_count = len(candidate_dirs)
if candidate_eval_count > max_program_evals_per_attempt:
    raise SystemExit(
        "evaluated candidate program count exceeds cap: "
        f"{candidate_eval_count} > {max_program_evals_per_attempt}"
    )

latest_status = None
latest_program = None
for program_idx, program_dir in candidate_dirs:
    label = f"./{program_dir.name}/rubric_status.json"
    status = validate_status(
        program_dir / "rubric_status.json",
        label,
        target_ids,
    )
    latest_status = status
    latest_program = program_idx

final_status = validate_status(
    experiment_dir / "rubric_status.json",
    "./rubric_status.json",
    target_ids,
)

if latest_status is not None:
    if final_status["overall_pass"] != latest_status["overall_pass"]:
        raise SystemExit(
            "root rubric_status.json overall_pass must match latest evaluated program artifact"
        )
    if final_status["rubrics"] != latest_status["rubrics"]:
        raise SystemExit(
            "root rubric_status.json rubrics must match latest evaluated program artifact"
        )
    print(
        "[run.sh] validated rubric artifacts for "
        f"{candidate_eval_count} evaluated candidate program(s) "
        f"(candidates={candidate_eval_count}/{max_program_evals_per_attempt}); "
        f"latest=program_{latest_program}"
    )
else:
    print("[run.sh] no evaluated candidate program directories detected; validated root rubric_status only")

memory_entries = read_jsonl_entries(global_memory_jsonl_path)
attempt_entries = [entry for entry in memory_entries if entry.get("attempt_id") == attempt_id]
if not attempt_entries:
    raise SystemExit(
        "memory.jsonl must contain at least one entry with current attempt_id="
        f"{attempt_id}"
    )
if latest_program is not None:
    expected_program_id = f"program_{latest_program}"
    if not any(entry.get("program_id") == expected_program_id for entry in attempt_entries):
        raise SystemExit(
            "memory.jsonl must contain entry for latest evaluated program_id="
            f"{expected_program_id}"
        )

print(
    f"[run.sh] rubric_status validated: attempt={attempt_id}, idea={idea_id}, overall_pass={final_status['overall_pass']}"
)
PY

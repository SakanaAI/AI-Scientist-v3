"""Create an AI-Scientist-v3 task scaffold from the setup-task templates.

    python new_task.py <task_dir> [--layout single|overlay] [--entrypoint program.py]

Never overwrites: it stops if any file it would write already exists.
"""

import argparse
import os
import sys
from pathlib import Path, PurePosixPath

SKILL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SKILL_DIR.parents[1]
TEMPLATES = SKILL_DIR / "templates"
CANONICAL_CAPTURE = REPO_ROOT / "examples" / "circpack" / "capture.py"
OVERLAY_UPSTREAM = "resources/upstream"

UPSTREAM_NOTE = """\
url: TODO
commit: TODO
license: TODO
# Copy the pinned checkout (without .git, data, or logs) into resources/upstream/.
# Put the editable files under program_0/src/ at the same relative paths.
"""

README = """\
# TODO task name

## Interview
- Objective (metric, direction, who computes it):
- Editable surface:
- Layout: {layout}
- Upstream and data (URL, commit, hashes, licenses):
- Rules (static / runtime / judgment):
- Leakage guards:
- Cost per evaluation (search / full):
- Baseline and its known score:
- Score variance (measured spread):

## Validation log
- [ ] static checks
- [ ] smoke run
- [ ] baseline at search profile: score =, wall time =
- [ ] negative programs give correct=false:
- [ ] agsci doctor / --dry-run

## Launch
```bash
agsci run <this dir> ...
```
"""


def safe_entrypoint(raw: str) -> str:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise SystemExit(f"--entrypoint must be a relative .py path inside src/: {raw}")
    return path.as_posix()


def evaluator_source(layout: str) -> str:
    source = (TEMPLATES / "evaluate.py").read_text(encoding="utf-8")
    if layout == "overlay":
        source = source.replace(
            'UPSTREAM_SUBDIR: str | None = None  # e.g. "resources/upstream"',
            f'UPSTREAM_SUBDIR: str | None = "{OVERLAY_UPSTREAM}"',
        )
    return source


def scaffold_files(layout: str, entrypoint: str) -> dict[str, str]:
    """Relative path -> content for every file the scaffold needs."""
    spec = (TEMPLATES / "program_spec.toml").read_text(encoding="utf-8")
    files = {
        "task.md": (TEMPLATES / "task.md").read_text(encoding="utf-8"),
        "README.md": README.format(layout=layout),
        "evaluate.py": evaluator_source(layout),
        "capture.py": CANONICAL_CAPTURE.read_text(encoding="utf-8"),
        ".agentic_task_ignore": (TEMPLATES / "agentic_task_ignore").read_text(encoding="utf-8"),
        "program_0/program_spec.toml": spec.replace("{entrypoint}", entrypoint),
        f"program_0/src/{entrypoint}": (TEMPLATES / "program.py").read_text(encoding="utf-8"),
    }
    if layout == "overlay":
        files["resources/UPSTREAM"] = UPSTREAM_NOTE
        files[f"{OVERLAY_UPSTREAM}/.keep"] = ""
    return files


def write_new_file(task_dir: Path, rel: str, content: str) -> None:
    """Create a file that did not exist, never through a symlink, never outside the task dir."""
    target = task_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.parent.resolve().is_relative_to(task_dir):
        raise SystemExit(f"refusing to write outside {task_dir}: {rel} (symlinked parent)")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--layout", choices=("single", "overlay"), default="single")
    parser.add_argument("--entrypoint", default="program.py", help="path inside program_0/src/")
    args = parser.parse_args()
    if not CANONICAL_CAPTURE.is_file():
        raise SystemExit(f"canonical capture.py not found at {CANONICAL_CAPTURE}; is the skill inside the repo?")

    files = scaffold_files(args.layout, safe_entrypoint(args.entrypoint))
    task_dir = args.task_dir.expanduser().resolve()
    existing = [rel for rel in files if (task_dir / rel).exists() or (task_dir / rel).is_symlink()]
    if existing:
        print(f"refusing to overwrite in {task_dir}: {', '.join(existing)}", file=sys.stderr)
        return 1
    for rel, content in files.items():
        write_new_file(task_dir, rel, content)
    print(f"created {args.layout} task scaffold at {task_dir}")
    for rel in files:
        print(f"  {rel}")
    print("next: fill the TODOs in task.md, evaluate.py and program_0, then run check_task.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

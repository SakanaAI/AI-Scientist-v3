"""Write a launch script for one `agsci run` campaign: a Slurm batch file or a local (tmux) script.

    python write_launch.py <task_dir> --out runs/launch.sbatch --scheduler slurm --partition gpu \
        --gpus 8 --mode native --eval-timeout 5400 -- preset=paper agent.name=claude

Hydra overrides after `--` are passed through and win over generated ones with the same key.
Phases are always explicit (default: investigation,ablations,posthoc,paper), so a preset can never
add the external `review` phase without the user asking for it. Nothing is submitted unless
`--submit` is given (Slurm only); after reviewing a script, submit it with `sbatch <file>`.
"""

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SKILL_DIR.parents[1]
TEMPLATES = {"slurm": SKILL_DIR / "templates" / "launch.sbatch", "local": SKILL_DIR / "templates" / "launch_local.sh"}
DEFAULT_PHASES = "investigation,ablations,posthoc,paper"
CONTAINER_ONLY_PHASES = {"task_drafting", "final_evaluation", "revision"}
SCHEDULER_FIELD = re.compile(r"[A-Za-z0-9._:@,+-]+")
VARIABLE_REFERENCE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})")


def override_key(override: str) -> str:
    return override.split("=", 1)[0].lstrip("+~")


def phases_override(phases: str) -> str:
    """Quoted so that both `agsci run` and `agsci doctor` parse the comma list as one string."""
    return f'execution.phases="{phases}"'


def generated_overrides(args: argparse.Namespace) -> list[str]:
    """Overrides implied by the launch options, before the user's own."""
    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    if args.mode == "native" and CONTAINER_ONLY_PHASES.intersection(phases):
        raise SystemExit(f"native mode cannot run {sorted(CONTAINER_ONLY_PHASES.intersection(phases))}")
    overrides = [phases_override(",".join(phases))]
    if args.mode == "native":
        overrides.append("container.enabled=false")
        if "paper" in phases:  # the paper's evidence audit and LaTeX build run only in the image
            overrides.append("container.paper_enabled=true")
    if args.gpus:
        if args.mode == "native" and args.workers > 1:
            raise SystemExit("native GPU runs (harness-evaluated) support --workers 1 only")
        if args.gpus % args.workers:
            raise SystemExit(f"--gpus {args.gpus} is not divisible by --workers {args.workers}")
        overrides += [
            f"container.gpu_count={args.gpus // args.workers}",
            "investigation.execution.gpu_evaluation=true",
            "investigation.execution.harness_evaluated_attempts=true",
        ]
    if args.workers > 1:
        overrides.append(f"investigation.execution.workers={args.workers}")
    if args.eval_timeout:
        overrides.append(f"investigation.execution.harness_evaluation_timeout_seconds={args.eval_timeout}")
    if args.results_root:
        overrides.append(f"paths.results_root={args.results_root.expanduser().resolve()}")
    if args.resume:
        run_dir = args.resume.expanduser().resolve()
        overrides += [f"paths.run_dir={run_dir}", f"run_id={run_dir.name}", "investigation.execution.resume=true"]
    return overrides


def merged_overrides(generated: list[str], user: list[str]) -> list[str]:
    user_keys = {override_key(item) for item in user}
    return [item for item in generated if override_key(item) not in user_keys] + user


def double_quote_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")


def expandable(value: str) -> str:
    """Double-quote a value so plain `$NAME` / `${NAME}` expand when the job runs; nothing else does."""
    if "\n" in value or "\r" in value:
        raise SystemExit(f"newlines are not allowed in exported values: {value!r}")
    parts, position = [], 0
    for match in VARIABLE_REFERENCE.finditer(value):
        parts += [double_quote_literal(value[position : match.start()]), match.group()]
        position = match.end()
    parts.append(double_quote_literal(value[position:]))
    return '"' + "".join(parts) + '"'


def env_exports(args: argparse.Namespace) -> str:
    lines = []
    if args.temp_root:
        root = expandable(args.temp_root)
        lines += [f"mkdir -p {root}", f"export AGENTIC_TEMP_ROOT={root}"]
    for item in args.env:
        name, sep, value = item.partition("=")
        if not sep or not name.isidentifier():
            raise SystemExit(f"--env expects NAME=VALUE, got {item!r}")
        lines.append(f"export {name}={expandable(value)}")
    return "\n".join(lines)


def scheduler_field(name: str, value: str) -> str:
    """Values that land in #SBATCH lines, where nothing can be quoted."""
    if not SCHEDULER_FIELD.fullmatch(value):
        raise SystemExit(f"--{name} may only contain letters, digits and ._:@,+-: {value!r}")
    return value


def log_directory(args: argparse.Namespace, out: Path) -> Path:
    log_dir = (args.log_dir or out.parent).expanduser().resolve()
    if any(char.isspace() for char in str(log_dir)):
        raise SystemExit(f"--log-dir must not contain whitespace (sbatch --output cannot be quoted): {log_dir}")
    return log_dir


def render(args: argparse.Namespace, task_dir: Path, out: Path, overrides: list[str]) -> str:
    job_name = scheduler_field("job-name", args.job_name or f"agsci-{task_dir.name}")
    log_dir = log_directory(args, out)
    replacements = {
        "@JOB_NAME@": job_name,
        "@PARTITION@": scheduler_field("partition", args.partition or "none"),
        "@GRES_LINE@": f"#SBATCH --gres=gpu:{args.gpus}" if args.gpus else "# CPU task: no GPUs requested",
        "@CPUS@": str(args.cpus),
        "@MEM@": scheduler_field("mem", args.mem),
        "@TIME_LINE@": f"#SBATCH --time={scheduler_field('time', args.time)}" if args.time
        else "# no --time: the partition default applies",
        "@LOG_DIR@": str(log_dir),
        "@START_COMMAND@": local_start_command(job_name, out, log_dir),
        "@ENV_EXPORTS@": env_exports(args),
        "@AGSCI_REPO@": shlex.quote(str(REPO_ROOT)),
        "@TASK_DIR@": shlex.quote(str(task_dir)),
        "@OVERRIDES@": " \\\n  ".join(shlex.quote(item) for item in overrides),
    }
    text = TEMPLATES[args.scheduler].read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


def local_start_command(job_name: str, out: Path, log_dir: Path) -> str:
    """tmux command that keeps stdout piped (not a TTY), so the pipeline writes journal heartbeats."""
    log_file = shlex.quote(str(log_dir / f"{job_name}.out"))
    inner = f"bash {shlex.quote(str(out))} 2>&1 | tee -a {log_file}"
    return shlex.join(["tmux", "new", "-d", "-s", job_name.replace(".", "-").replace(":", "-"), inner])


def preflight_commands(task_dir: Path, overrides: list[str]) -> list[str]:
    arguments = " ".join(shlex.quote(item) for item in [str(task_dir), *overrides])
    repo = shlex.quote(str(REPO_ROOT))
    return [f"cd {repo} && .venv/bin/agsci doctor {arguments}", f"cd {repo} && .venv/bin/agsci run {arguments} --dry-run"]


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="script to write (refuses to overwrite)")
    parser.add_argument("--scheduler", choices=("slurm", "local"), default="slurm")
    parser.add_argument("--mode", choices=("container", "native"), default="container")
    parser.add_argument("--phases", default=DEFAULT_PHASES, help="comma list; add 'review' only with consent")
    parser.add_argument("--partition", help="Slurm partition (required for slurm)")
    parser.add_argument("--gpus", type=int, default=0, help="GPUs on the node for this run (0 = CPU task)")
    parser.add_argument("--workers", type=int, default=1, help="investigation workers sharing those GPUs")
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--mem", default="400G")
    parser.add_argument("--time", help="Slurm time limit, e.g. 3-00:00:00 (default: partition default)")
    parser.add_argument("--job-name")
    parser.add_argument("--log-dir", type=Path, help="job log directory (default: next to --out)")
    parser.add_argument("--eval-timeout", type=int, help="harness evaluation timeout in seconds")
    parser.add_argument("--results-root", type=Path, help="paths.results_root for the run")
    parser.add_argument("--resume", type=Path, help="existing run dir to resume")
    parser.add_argument("--temp-root", help="AGENTIC_TEMP_ROOT scratch dir, e.g. '/tmp/agsci_$USER' ($NAME expands at run time)")
    parser.add_argument("--env", action="append", default=[], help="extra NAME=VALUE exports ($NAME expands at run time)")
    parser.add_argument("--submit", action="store_true", help="sbatch the script right after writing it")
    return parser.parse_args(argv[:split]), argv[split + 1 :]


def main(argv: list[str]) -> int:
    args, user_overrides = parse_args(argv)
    task_dir = args.task_dir.expanduser().resolve()
    if not (task_dir / "task.md").is_file():
        raise SystemExit(f"no task.md in {task_dir}")
    if args.scheduler == "slurm" and not args.partition:
        raise SystemExit("--partition is required for --scheduler slurm")
    if args.submit and args.scheduler != "slurm":
        raise SystemExit("--submit only applies to --scheduler slurm; start local scripts with the printed tmux command")
    out = args.out.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    overrides = merged_overrides(generated_overrides(args), user_overrides)
    script = render(args, task_dir, out, overrides)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_directory(args, out).mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="utf-8")
    out.chmod(0o750)
    print(f"wrote {out}")
    print("overrides:\n  " + "\n  ".join(overrides))
    print("preflight (run where the campaign will run):\n  " + "\n  ".join(preflight_commands(task_dir, overrides)))
    if args.submit:
        result = subprocess.run(["sbatch", str(out)], capture_output=True, text=True, check=False)
        print(result.stdout.strip() or result.stderr.strip())
        return result.returncode
    if args.scheduler == "slurm":
        print(f"submit after review: sbatch {shlex.quote(str(out))}")
    else:
        print(f"start after review: {local_start_command(args.job_name or f'agsci-{task_dir.name}', out, log_directory(args, out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

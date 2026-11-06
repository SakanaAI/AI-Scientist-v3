"""Checks for the run-campaign launch-script generator."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "skills" / "run-campaign" / "scripts"


def run_script(name: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *map(str, args)],
        capture_output=True, text=True, timeout=60, check=False,
    )


def make_task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    task.mkdir(exist_ok=True)
    (task / "task.md").write_text("# Goal\n- x\n# Rubrics\n- y\n", encoding="utf-8")
    return task


def launch_command(script: Path) -> list[str]:
    return shlex.split(script.read_text(encoding="utf-8").split("agsci run", 1)[1].replace("\\\n", " "))[1:]


def write_launch(tmp_path: Path, *args: object) -> tuple[subprocess.CompletedProcess[str], Path]:
    out = tmp_path / "runs" / "launch.sh"
    return run_script("write_launch.py", make_task(tmp_path), "--out", out, *args), out




def test_native_slurm_launch_merges_user_overrides_last(tmp_path: Path) -> None:
    result, out = write_launch(
        tmp_path, "--partition", "gpu", "--gpus", "8", "--mode", "native", "--eval-timeout", "5400",
        "--", 'execution.phases="investigation"', "agent.name=codex",
    )

    assert result.returncode == 0, result.stderr
    command = launch_command(out)
    assert "#SBATCH --gres=gpu:8" in out.read_text()
    assert "container.enabled=false" in command
    assert "investigation.execution.harness_evaluation_timeout_seconds=5400" in command
    assert [arg for arg in command if arg.startswith("execution.phases=")] == ['execution.phases="investigation"']
    assert command[-1] == "agent.name=codex"


def test_phases_are_explicit_so_presets_cannot_add_review(tmp_path: Path) -> None:
    result, out = write_launch(tmp_path, "--scheduler", "local", "--", "preset=paper")

    assert result.returncode == 0, result.stderr
    assert 'execution.phases="investigation,ablations,posthoc,paper"' in launch_command(out)


def test_native_mode_rejects_container_only_phases(tmp_path: Path) -> None:
    result, _ = write_launch(tmp_path, "--scheduler", "local", "--mode", "native",
                             "--phases", "investigation,final_evaluation")

    assert result.returncode != 0 and "native mode cannot run" in result.stderr


def test_resume_adds_run_dir_run_id_and_resume_flag(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "task" / "claude" / "20260101_000000"

    result, out = write_launch(tmp_path, "--scheduler", "local", "--resume", run_dir)

    assert result.returncode == 0, result.stderr
    command = launch_command(out)
    assert f"paths.run_dir={run_dir}" in command
    assert "run_id=20260101_000000" in command
    assert "investigation.execution.resume=true" in command


def test_cpu_slurm_job_requests_no_gres(tmp_path: Path) -> None:
    result, out = write_launch(tmp_path, "--partition", "cpu")

    assert result.returncode == 0, result.stderr
    assert "--gres" not in out.read_text()


def test_scheduler_fields_reject_injected_lines(tmp_path: Path) -> None:
    result, out = write_launch(tmp_path, "--partition", "gpu", "--job-name", "x\ntouch pwned")

    assert result.returncode != 0 and "--job-name may only contain" in result.stderr
    assert not out.exists()


def test_exports_expand_plain_variables_but_never_run_commands(tmp_path: Path) -> None:
    result, out = write_launch(
        tmp_path, "--scheduler", "local", "--temp-root", "/tmp/agsci_$USER",
        "--env", "A=x[$(touch pwned1)]", "--env", "B=${arr[A]}", "--env", "P=$(touch pwned2)", "--env", "Q=${P@P}",
    )

    assert result.returncode == 0, result.stderr
    exports = "\n".join(line for line in out.read_text().splitlines() if line.startswith("export"))
    shown = subprocess.run(
        ["bash", "-c", exports + '\nprintf "%s\\n" "$AGENTIC_TEMP_ROOT" "$B" "$Q"'],
        capture_output=True, text=True, cwd=tmp_path, env={"USER": "alice", "PATH": os.environ["PATH"]}, check=True,
    )
    assert shown.stdout.splitlines() == ["/tmp/agsci_alice", "${arr[A]}", "${P@P}"]
    assert not list(tmp_path.glob("pwned*"))


def test_launch_refuses_overwrite_and_uneven_gpu_split(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    out = tmp_path / "launch.sh"
    out.write_text("existing", encoding="utf-8")

    overwrite = run_script("write_launch.py", task, "--out", out, "--scheduler", "local")
    uneven = run_script("write_launch.py", task, "--out", tmp_path / "b.sh", "--scheduler", "local",
                        "--gpus", "8", "--workers", "3")

    assert overwrite.returncode != 0 and "refusing to overwrite" in overwrite.stderr
    assert uneven.returncode != 0 and "not divisible" in uneven.stderr

"""End-to-end checks for the setup-task skill scripts: scaffold, then evaluate good and bad programs."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "skills" / "setup-task" / "scripts"


def run_script(name: str, *args: object, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *map(str, args)],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def scaffold(tmp_path: Path, *args: str) -> Path:
    task = tmp_path / "task"
    result = run_script("new_task.py", task, *args)
    assert result.returncode == 0, result.stderr
    return task


def patch_evaluator(task: Path, old: str, new: str) -> None:
    evaluator = task / "evaluate.py"
    source = evaluator.read_text(encoding="utf-8")
    assert old in source
    evaluator.write_text(source.replace(old, new, 1), encoding="utf-8")


def program_with(task: Path, tmp_path: Path, source: str) -> Path:
    program = tmp_path / "candidates" / "program_0"
    shutil.copytree(task / "program_0", program)
    (program / "src" / "program.py").write_text(source, encoding="utf-8")
    return program


def process_alive(pid_file: Path) -> bool:
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"


SLEEPER = "import os, time\nopen({pid!r}, 'w').write(str(os.getpid()))\ntime.sleep(300)\n"


def test_new_task_refuses_to_overwrite(tmp_path: Path) -> None:
    task = scaffold(tmp_path)

    result = run_script("new_task.py", task)

    assert result.returncode == 1
    assert "refusing to overwrite" in result.stderr


def test_scaffolded_baseline_evaluates_correct_through_harness_wrapper(tmp_path: Path) -> None:
    task = scaffold(tmp_path)

    result = run_script("check_task.py", task, "--run", "--expect", "correct")

    assert result.returncode == 0, result.stdout
    assert "via harness wrapper" in result.stdout and "correct = True" in result.stdout


def test_expectation_mismatch_fails_the_check(tmp_path: Path) -> None:
    task = scaffold(tmp_path)

    result = run_script("check_task.py", task, "--run", "--expect", "incorrect")

    assert result.returncode == 1
    assert "expected incorrect" in result.stdout


def test_crashing_program_is_incorrect_not_an_evaluator_failure(tmp_path: Path) -> None:
    task = scaffold(tmp_path)
    program = program_with(task, tmp_path, 'raise RuntimeError("boom")\n')

    result = run_script("check_task.py", task, "--run", "--program", program, "--expect", "incorrect")

    assert result.returncode == 0, result.stdout
    assert "exit code 0" in result.stdout


def test_malformed_solution_is_incorrect_not_an_evaluator_failure(tmp_path: Path) -> None:
    task = scaffold(tmp_path)
    program = program_with(task, tmp_path, "open('solution.json', 'w').write('not json')\n")

    result = run_script("check_task.py", task, "--run", "--program", program, "--expect", "incorrect")

    assert result.returncode == 0, result.stdout
    assert "unreadable solution.json" in result.stdout


def test_static_rule_violation_is_incorrect(tmp_path: Path) -> None:
    task = scaffold(tmp_path)
    patch_evaluator(task, '    # "no_autograd"', '    "no_autograd"')
    program = program_with(task, tmp_path, "import torch.autograd\n")

    result = run_script("check_task.py", task, "--run", "--program", program, "--expect", "incorrect")

    assert result.returncode == 0, result.stdout
    assert "rule no_autograd" in result.stdout


def test_program_timeout_kills_the_candidate(tmp_path: Path) -> None:
    task = scaffold(tmp_path)
    patch_evaluator(task, '"search": 600', '"search": 2')
    pid_file = tmp_path / "candidate.pid"
    program = program_with(task, tmp_path, SLEEPER.format(pid=str(pid_file)))

    result = run_script("check_task.py", task, "--run", "--program", program, "--expect", "incorrect")

    assert result.returncode == 0, result.stdout
    assert "timed out after 2 s" in result.stdout
    assert not process_alive(pid_file)


def test_leftover_child_process_does_not_hold_the_evaluator(tmp_path: Path) -> None:
    task = scaffold(tmp_path)
    program = program_with(task, tmp_path, (
        "import json, subprocess\n"
        "subprocess.Popen(['sleep', '60'])\n"
        "json.dump({'value': 42}, open('solution.json', 'w'))\n"
    ))
    started = time.monotonic()

    result = run_script("check_task.py", task, "--run", "--program", program, "--expect", "correct")

    assert result.returncode == 0, result.stdout
    assert time.monotonic() - started < 30


def test_check_timeout_stops_the_whole_evaluation(tmp_path: Path) -> None:
    task = scaffold(tmp_path)
    pid_file = tmp_path / "candidate.pid"
    program = program_with(task, tmp_path, SLEEPER.format(pid=str(pid_file)))

    result = run_script("check_task.py", task, "--run", "--program", program, "--timeout", "3", "--profile", "search")

    assert result.returncode == 1
    assert "was stopped" in result.stdout
    assert not process_alive(pid_file)


def test_replications_are_run_and_averaged(tmp_path: Path) -> None:
    task = scaffold(tmp_path)
    results = tmp_path / "results"

    subprocess.run(
        [sys.executable, str(task / "evaluate.py"), "--program_path", str(task / "program_0" / "src" / "program.py"),
         "--results_dir", str(results), "--num_replications", "2"],
        cwd=task, capture_output=True, text=True, timeout=60, check=True,
    )

    metrics = json.loads((results / "metrics.json").read_text())
    assert metrics["num_replications"] == 2 and metrics["combined_score_std"] == 0.0


def test_overlay_program_runs_on_top_of_upstream(tmp_path: Path) -> None:
    task = scaffold(tmp_path, "--layout", "overlay", "--entrypoint", "train/run.py")
    (task / "resources" / "upstream" / "helper.py").write_text("VALUE = 42.0\n", encoding="utf-8")
    (task / "program_0" / "src" / "train" / "run.py").write_text(
        "import json, sys\nsys.path.insert(0, '.')\nfrom helper import VALUE\n"
        "json.dump({'value': VALUE}, open('solution.json', 'w'))\n",
        encoding="utf-8",
    )

    result = run_script("check_task.py", task, "--run", "--expect", "correct")

    assert result.returncode == 0, result.stdout
    assert "combined_score = 0.0" in result.stdout  # helper.py from upstream supplied the exact target


def test_evaluator_writing_into_task_dir_is_reported(tmp_path: Path) -> None:
    task = scaffold(tmp_path)
    patch_evaluator(task, "    scratch = Path(tempfile.mkdtemp(",
                    '    (task_dir() / "leak.txt").write_text("x")\n    scratch = Path(tempfile.mkdtemp(')

    result = run_script("check_task.py", task, "--run")

    assert result.returncode == 1
    assert "wrote into the task dir" in result.stdout


def test_unpinned_requirements_fail_static_checks(tmp_path: Path) -> None:
    task = scaffold(tmp_path)
    (task / "requirements.txt").write_text("numpy>=2\n", encoding="utf-8")

    result = run_script("check_task.py", task)

    assert result.returncode == 1
    assert "pinned name==version" in result.stdout

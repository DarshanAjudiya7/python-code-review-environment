import os
import re
import subprocess
import sys
from pathlib import Path

from tasks import task_ids


ROOT = Path(__file__).resolve().parents[1]
START_RE = re.compile(r"^\[START\] task=([a-z0-9-]+)$")
STEP_RE = re.compile(r"^\[STEP\] step=(\d+) reward=(-?\d+(?:\.\d+)?)$")
END_RE = re.compile(r"^\[END\] task=([a-z0-9-]+) score=(\d+(?:\.\d+)?) steps=(\d+)$")


def test_inference_emits_structured_stdout_for_all_tasks():
    env = os.environ.copy()
    env.pop("API_BASE_URL", None)
    env.pop("HF_TOKEN", None)
    env["MODEL_NAME"] = "mock-model"

    result = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "[START]" not in result.stderr
    assert "[STEP]" not in result.stderr
    assert "[END]" not in result.stderr

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    expected_tasks = task_ids()
    seen_tasks = []
    line_index = 0

    while line_index < len(lines):
        start_match = START_RE.match(lines[line_index])
        assert start_match, f"Invalid START line: {lines[line_index]}"
        task_id = start_match.group(1)
        seen_tasks.append(task_id)
        line_index += 1

        step_count = 0
        while line_index < len(lines) and STEP_RE.match(lines[line_index]):
            step_count += 1
            step_match = STEP_RE.match(lines[line_index])
            assert step_match is not None
            assert int(step_match.group(1)) == step_count
            reward = float(step_match.group(2))
            assert -1.0 <= reward <= 1.0
            line_index += 1

        assert step_count >= 1
        assert line_index < len(lines), "Missing END line"
        end_match = END_RE.match(lines[line_index])
        assert end_match, f"Invalid END line: {lines[line_index]}"
        assert end_match.group(1) == task_id
        assert 0.0 <= float(end_match.group(2)) <= 1.0
        assert int(end_match.group(3)) == step_count
        line_index += 1

    assert seen_tasks == expected_tasks

"""Helpers for deterministic pytest execution in temp sandboxes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class PytestExecution:
    passed: int
    failed: int
    total: int
    timed_out: bool
    output: str


def _test_module_source(test_cases: List[str]) -> str:
    lines = [
        "from candidate import *",
        "",
        "def raises(exc_type, func, *args, **kwargs):",
        "    try:",
        "        func(*args, **kwargs)",
        "    except exc_type:",
        "        return True",
        "    return False",
        "",
    ]
    for index, expr in enumerate(test_cases):
        lines.append(f"def test_case_{index}():")
        lines.append(f"    assert {expr}")
        lines.append("")
    return "\n".join(lines)


def _runner_script() -> str:
    return """import json
import pathlib
import pytest


class Collector:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return
        if report.passed:
            self.passed += 1
        elif report.failed:
            self.failed += 1


collector = Collector()
exit_code = pytest.main(["-q", "test_candidate.py"], plugins=[collector])
payload = {
    "passed": collector.passed,
    "failed": collector.failed,
    "exit_code": int(exit_code),
}
pathlib.Path("pytest_results.json").write_text(json.dumps(payload), encoding="utf-8")
"""


def run_pytest_suite(candidate_code: str, tests: Iterable[str], timeout_s: float = 120.0) -> PytestExecution:
    test_cases = list(tests)
    with tempfile.TemporaryDirectory(prefix="python-code-review-") as temp_dir:
        temp_path = Path(temp_dir)
        env = os.environ.copy()
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        (temp_path / "candidate.py").write_text(candidate_code, encoding="utf-8")
        (temp_path / "test_candidate.py").write_text(_test_module_source(test_cases), encoding="utf-8")
        (temp_path / "runner.py").write_text(_runner_script(), encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, "runner.py"],
                cwd=temp_path,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            total = max(len(test_cases), 1)
            return PytestExecution(0, total, total, True, (output or "pytest timed out").strip())

        result_path = temp_path / "pytest_results.json"
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        if not result_path.exists():
            total = max(len(test_cases), 1)
            return PytestExecution(0, total, total, False, output or "pytest runner did not produce results")

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        passed = int(payload.get("passed", 0))
        failed = int(payload.get("failed", 0))
        total = max(passed + failed, len(test_cases))
        return PytestExecution(passed, failed, total, False, output)

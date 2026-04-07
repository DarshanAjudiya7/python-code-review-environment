"""Helpers for deterministic pytest execution in temp sandboxes."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PytestExecution:
    """Exact pytest execution summary."""

    passed: int
    failed: int
    total: int
    timed_out: bool
    output: str


def _test_module_source(tests: Iterable[str]) -> str:
    """Build a valid pytest module from expression-style or full test snippets."""
    blocks: list[str] = ["from candidate import *  # noqa: F401,F403"]
    for index, test in enumerate(tests, start=1):
        snippet = str(test).strip()
        if not snippet:
            continue
        if snippet.startswith("def test_"):
            blocks.append(snippet)
            continue
        blocks.append(
            "\n".join(
                [
                    f"def test_case_{index:03d}():",
                    f"    assert {snippet}",
                ]
            )
        )
    return "\n\n".join(blocks) or "def test_placeholder():\n    assert True\n"


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


def run_pytest_suite(candidate_code: str, tests: Iterable[str], timeout_s: float = 3.0) -> PytestExecution:
    """Run a pytest suite against candidate.py and return structured results."""

    test_cases = list(tests)
    try:
        with tempfile.TemporaryDirectory(prefix="python-code-review-") as temp_dir:
            temp_path = Path(temp_dir)
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
                )
            except subprocess.TimeoutExpired as exc:
                output = (exc.stdout or "") + (exc.stderr or "")
                return PytestExecution(
                    passed=0,
                    failed=max(len(test_cases), 1),
                    total=max(len(test_cases), 1),
                    timed_out=True,
                    output=(output or "pytest timed out").strip(),
                )

            result_path = temp_path / "pytest_results.json"
            if not result_path.exists():
                output = (completed.stdout or "") + (completed.stderr or "")
                total = max(len(test_cases), 1)
                return PytestExecution(
                    passed=0,
                    failed=total,
                    total=total,
                    timed_out=False,
                    output=output.strip(),
                )

            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                output = ((completed.stdout or "") + (completed.stderr or "")).strip()
                return PytestExecution(
                    passed=0,
                    failed=max(len(test_cases), 1),
                    total=max(len(test_cases), 1),
                    timed_out=False,
                    output=(output or str(exc)).strip(),
                )

            passed = int(payload.get("passed", 0))
            failed = int(payload.get("failed", 0))
            total = max(passed + failed, len(test_cases))
            output = ((completed.stdout or "") + (completed.stderr or "")).strip()
            return PytestExecution(
                passed=passed,
                failed=failed,
                total=total,
                timed_out=False,
                output=output,
            )
    except Exception as exc:
        return PytestExecution(
            passed=0,
            failed=max(len(test_cases), 1),
            total=max(len(test_cases), 1),
            timed_out=False,
            output=str(exc),
        )

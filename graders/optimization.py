"""Deterministic grading for optimization and refactor tasks."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from graders.common import clamp_score, compile_tree, nested_loop_depth, style_score
from graders.pytest_runner import run_pytest_suite
from models import TaskGrade
from tasks.task_bank import TaskSpec


def _benchmark_script(task: TaskSpec) -> str:
    return f"""import json
import time
from candidate import {task.benchmark_entrypoint}

{task.benchmark_builder}

events = build_benchmark_events()
start = time.perf_counter()
for _ in range({task.benchmark_repeats}):
    result = {task.benchmark_entrypoint}(events)
elapsed = time.perf_counter() - start
Path = __import__("pathlib").Path
Path("benchmark.json").write_text(json.dumps({{"elapsed": elapsed, "rows": len(result)}}), encoding="utf-8")
"""


def benchmark_runtime(candidate_code: str, task: TaskSpec) -> tuple[float, bool, str]:
    """Benchmark runtime deterministically against the starter implementation."""

    assert task.benchmark_entrypoint is not None
    try:
        with tempfile.TemporaryDirectory(prefix="python-code-review-bench-") as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "candidate.py").write_text(candidate_code, encoding="utf-8")
            (temp_path / "starter.py").write_text(task.starter_code, encoding="utf-8")
            (temp_path / "candidate_runner.py").write_text(_benchmark_script(task), encoding="utf-8")

            starter_script = _benchmark_script(task).replace("from candidate import", "from starter import")
            (temp_path / "starter_runner.py").write_text(starter_script, encoding="utf-8")

            try:
                starter_run = subprocess.run(
                    [sys.executable, "starter_runner.py"],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=task.benchmark_timeout_s,
                    check=False,
                )
                starter_payload = json.loads((temp_path / "benchmark.json").read_text(encoding="utf-8"))

                candidate_run = subprocess.run(
                    [sys.executable, "candidate_runner.py"],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=task.benchmark_timeout_s,
                    check=False,
                )
                candidate_payload = json.loads((temp_path / "benchmark.json").read_text(encoding="utf-8"))
            except subprocess.TimeoutExpired as exc:
                output = (exc.stdout or "") + (exc.stderr or "")
                return 0.0, True, (output or "benchmark timed out").strip()
            except Exception as exc:  # pragma: no cover
                return 0.0, False, str(exc)

            starter_elapsed = max(float(starter_payload["elapsed"]), 1e-9)
            candidate_elapsed = max(float(candidate_payload["elapsed"]), 1e-9)
            speedup = starter_elapsed / candidate_elapsed
            runtime_score = clamp_score(min((speedup - 1.0) / 3.0, 1.0))
            output = "\n".join(
                part
                for part in [
                    starter_run.stdout.strip(),
                    starter_run.stderr.strip(),
                    candidate_run.stdout.strip(),
                    candidate_run.stderr.strip(),
                    f"starter={starter_elapsed:.6f}s candidate={candidate_elapsed:.6f}s speedup={speedup:.2f}x",
                ]
                if part
            )
            return runtime_score, False, output
    except Exception as exc:  # pragma: no cover
        return 0.0, False, str(exc)


def ast_quality_score(code: str, task: TaskSpec) -> float:
    """Score maintainability and algorithmic structure."""

    tree, parse_error = compile_tree(code)
    if tree is None:
        return 0.0

    import ast

    function_node = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    docstring_points = 0.2 if function_node and ast.get_docstring(function_node, clean=False) else 0.0
    nested_points = 0.4 if nested_loop_depth(tree) <= 1 else 0.0
    marker_points = 0.0
    for marker in task.expected_quality_markers:
        if marker in code:
            marker_points += 0.2
    return clamp_score(docstring_points + nested_points + marker_points)


def grade_optimization_task(candidate_code: str, task: TaskSpec) -> TaskGrade:
    """Grade optimization tasks using correctness, runtime, AST quality, and style."""

    execution = run_pytest_suite(
        candidate_code,
        [*task.visible_tests, *task.hidden_tests],
        timeout_s=task.benchmark_timeout_s,
    )
    test_fraction = execution.passed / execution.total if execution.total else 0.0

    if execution.timed_out:
        return TaskGrade(
            score=0.0,
            tests_passed=execution.passed,
            tests_total=execution.total,
            timed_out=True,
            details={"tests": execution.output},
        )

    runtime_score, timed_out, benchmark_output = benchmark_runtime(candidate_code, task)
    if timed_out:
        return TaskGrade(
            score=0.0,
            tests_passed=execution.passed,
            tests_total=execution.total,
            timed_out=True,
            details={"tests": execution.output, "benchmark": benchmark_output},
        )

    quality_score = ast_quality_score(candidate_code, task)
    pep8_score = style_score(candidate_code, task.style_max_line_length)
    score = clamp_score(
        (0.5 * test_fraction)
        + (0.3 * runtime_score)
        + (0.15 * quality_score)
        + (0.05 * pep8_score)
    )
    return TaskGrade(
        score=score,
        syntax_score=1.0,
        tests_passed=execution.passed,
        tests_total=execution.total,
        quality_score=quality_score,
        runtime_score=runtime_score,
        details={
            "tests": execution.output,
            "benchmark": benchmark_output,
            "test_fraction": round(test_fraction, 4),
            "runtime_score": round(runtime_score, 4),
            "style_score": round(pep8_score, 4),
        },
    )

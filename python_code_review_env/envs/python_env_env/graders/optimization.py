"""Deterministic grading for optimization tasks."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from python_code_review_env.envs.python_env_env.graders.common import (
    clamp_score,
    compile_tree,
    compiles,
    nested_loop_depth,
    style_score,
    syntax_error_message,
)
from python_code_review_env.envs.python_env_env.graders.pytest_runner import run_pytest_suite
from python_code_review_env.envs.python_env_env.models import TaskGrade
from python_code_review_env.envs.python_env_env.tasks.task_bank import TaskSpec


def _benchmark_script(module_name: str, task: TaskSpec) -> str:
    return f"""import copy
import json
import time
from pathlib import Path
from {module_name} import {task.benchmark_entrypoint}

benchmark_input = {task.benchmark_input_expr}
start = time.perf_counter()
for _ in range({task.benchmark_repeats}):
    result = {task.benchmark_entrypoint}(copy.copy(benchmark_input))
elapsed = time.perf_counter() - start
Path("benchmark.json").write_text(
    json.dumps({{"elapsed": elapsed, "rows": len(result)}}),
    encoding="utf-8",
)
"""


def benchmark_runtime(candidate_code: str, task: TaskSpec) -> tuple[float, bool, str]:
    assert task.benchmark_entrypoint is not None
    assert task.benchmark_input_expr is not None

    with tempfile.TemporaryDirectory(prefix="python-code-review-bench-") as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "candidate.py").write_text(candidate_code, encoding="utf-8")
        (temp_path / "starter.py").write_text(task.starter_code, encoding="utf-8")
        (temp_path / "candidate_runner.py").write_text(
            _benchmark_script("candidate", task),
            encoding="utf-8",
        )
        (temp_path / "starter_runner.py").write_text(
            _benchmark_script("starter", task),
            encoding="utf-8",
        )

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
        runtime_score = clamp_score((speedup - 1.0) / 1.5)
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


def ast_quality_score(code: str, task: TaskSpec) -> float:
    tree, _ = compile_tree(code)
    if tree is None:
        return 0.0

    function_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == task.benchmark_entrypoint
        ),
        None,
    )
    if function_node is None:
        return 0.0

    docstring_points = 0.15 if ast.get_docstring(function_node, clean=False) else 0.0
    annotation_points = 0.15 if function_node.returns is not None else 0.0
    if all(arg.annotation is not None for arg in function_node.args.args):
        annotation_points += 0.1

    set_usage = 0.0
    anti_pattern_found = False
    for node in ast.walk(function_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "set":
            set_usage = 0.25
        if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Subscript):
            if getattr(node.annotation.value, "id", "") == "set":
                set_usage = 0.25
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Name) and comparator.id == "result":
                    anti_pattern_found = True

    loop_points = 0.2 if nested_loop_depth(function_node) <= 1 else 0.0
    marker_points = 0.0
    for marker in task.expected_quality_markers:
        if marker in code:
            marker_points += 0.05

    quality = docstring_points + annotation_points + set_usage + loop_points + marker_points
    if anti_pattern_found and set_usage == 0.0:
        quality -= 0.2
    return clamp_score(quality)


def grade_optimization_task(candidate_code: str, task: TaskSpec, include_hidden: bool = True) -> TaskGrade:
    if not compiles(candidate_code):
        error = syntax_error_message(candidate_code)
        return TaskGrade(score=0.0, syntax_score=0.0, details={"compile_error": error})

    tests = list(task.visible_tests)
    if include_hidden:
        tests.extend(task.hidden_tests)

    execution = run_pytest_suite(candidate_code, tests, timeout_s=task.benchmark_timeout_s)
    test_fraction = execution.passed / execution.total if execution.total else 0.0
    if execution.timed_out:
        return TaskGrade(
            score=0.0,
            syntax_score=1.0,
            tests_passed=execution.passed,
            tests_total=execution.total,
            timed_out=True,
            details={"compile_error": "", "tests": execution.output},
        )

    runtime_score, timed_out, benchmark_output = benchmark_runtime(candidate_code, task)
    if timed_out:
        return TaskGrade(
            score=0.0,
            syntax_score=1.0,
            tests_passed=execution.passed,
            tests_total=execution.total,
            timed_out=True,
            details={"compile_error": "", "tests": execution.output, "benchmark": benchmark_output},
        )

    quality = ast_quality_score(candidate_code, task)
    style = style_score(candidate_code, task.style_max_line_length)
    score = clamp_score(
        (0.55 * test_fraction) + (0.2 * runtime_score) + (0.15 * quality) + (0.10 * style)
    )
    return TaskGrade(
        score=score,
        syntax_score=1.0,
        tests_passed=execution.passed,
        tests_total=execution.total,
        quality_score=quality,
        runtime_score=runtime_score,
        style_score=style,
        details={
            "compile_error": "",
            "tests": execution.output,
            "benchmark": benchmark_output,
            "test_fraction": round(test_fraction, 4),
        },
    )

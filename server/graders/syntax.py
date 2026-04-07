"""Deterministic graders for syntax and bug-fix tasks."""

from __future__ import annotations

try:
    from ..models import TaskGrade
    from ..task_bank import TaskSpec
    from .common import clamp_score, compiles, normalized_diff_score, style_score, syntax_error_message
    from .optimization import grade_optimization_task
    from .pytest_runner import run_pytest_suite
except ImportError:  # pragma: no cover
    from models import TaskGrade
    from task_bank import TaskSpec
    from graders.common import clamp_score, compiles, normalized_diff_score, style_score, syntax_error_message
    from graders.optimization import grade_optimization_task
    from graders.pytest_runner import run_pytest_suite


def grade_syntax_task(candidate_code: str, task: TaskSpec, include_hidden: bool = True) -> TaskGrade:
    del include_hidden
    error = syntax_error_message(candidate_code)
    diff_score = normalized_diff_score(candidate_code, task.reference_code)
    style = style_score(candidate_code, task.style_max_line_length)

    if not error:
        return TaskGrade(
            score=1.0,
            syntax_score=1.0,
            quality_score=style,
            style_score=style,
            details={"compile_error": ""},
        )

    partial = 0.0 if diff_score < 0.2 else 0.3 + (0.4 * ((diff_score - 0.2) / 0.8))
    return TaskGrade(
        score=clamp_score(partial),
        syntax_score=0.0,
        quality_score=style * diff_score,
        style_score=style,
        details={"compile_error": error},
    )


def grade_bug_fix_task(candidate_code: str, task: TaskSpec, include_hidden: bool = True) -> TaskGrade:
    if not compiles(candidate_code):
        error = syntax_error_message(candidate_code)
        return TaskGrade(score=0.0, syntax_score=0.0, details={"compile_error": error})

    tests = list(task.visible_tests)
    if include_hidden:
        tests.extend(task.hidden_tests)

    execution = run_pytest_suite(candidate_code, tests, timeout_s=120.0)
    if execution.timed_out:
        return TaskGrade(
            score=0.0,
            syntax_score=1.0,
            tests_passed=execution.passed,
            tests_total=execution.total,
            timed_out=True,
            details={"compile_error": "", "tests": execution.output},
        )

    pass_fraction = execution.passed / execution.total if execution.total else 0.0
    style = style_score(candidate_code, task.style_max_line_length)
    return TaskGrade(
        score=clamp_score(pass_fraction),
        syntax_score=1.0,
        tests_passed=execution.passed,
        tests_total=execution.total,
        quality_score=style,
        style_score=style,
        details={"compile_error": "", "tests": execution.output},
    )


def grade_task(candidate_code: str, task: TaskSpec, include_hidden: bool = True) -> TaskGrade:
    if task.task_kind == "syntax_fix":
        return grade_syntax_task(candidate_code, task, include_hidden=include_hidden)
    if task.task_kind == "bug_fix":
        return grade_bug_fix_task(candidate_code, task, include_hidden=include_hidden)
    return grade_optimization_task(candidate_code, task, include_hidden=include_hidden)

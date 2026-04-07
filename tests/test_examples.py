from python_code_review_env.envs.python_env_env.graders.optimization import grade_optimization_task
from python_code_review_env.envs.python_env_env.graders.syntax import (
    grade_bug_fix_task,
    grade_syntax_task,
)
from python_code_review_env.envs.python_env_env.tasks.task_bank import get_task


def test_syntax_grader_partial_score_is_bounded():
    task = get_task("syntax-fix-easy")
    grade = grade_syntax_task(task.starter_code, task)

    assert 0.0 <= grade.score < 1.0


def test_bug_fix_grader_reference_solution_reaches_one():
    task = get_task("bug-fix-medium")
    grade = grade_bug_fix_task(task.reference_code, task, include_hidden=True)

    assert grade.score == 1.0
    assert grade.tests_passed == grade.tests_total


def test_optimization_grader_scores_better_than_starter():
    task = get_task("optimization-hard")
    starter_grade = grade_optimization_task(task.starter_code, task)
    reference_grade = grade_optimization_task(task.reference_code, task)

    assert reference_grade.score > starter_grade.score
    assert reference_grade.runtime_score >= starter_grade.runtime_score

from python_code_review_env.envs.python_env_env.models import PythonCodeReviewAction
from python_code_review_env.envs.python_env_env.server.env import PythonCodeReviewEnvironment
from python_code_review_env.envs.python_env_env.tasks.task_bank import get_task


def test_reset_cycles_tasks_in_order():
    env = PythonCodeReviewEnvironment()

    first = env.reset()
    second = env.reset()
    third = env.reset()

    assert first.task_id == "syntax-fix-easy"
    assert second.task_id == "bug-fix-medium"
    assert third.task_id == "optimization-hard"


def test_invalid_edit_code_penalizes_action():
    env = PythonCodeReviewEnvironment()
    env.reset(task_id="syntax-fix-easy")

    observation = env.step(PythonCodeReviewAction(action_type="edit_code", code=""))

    assert observation.reward == -0.1
    assert observation.reward_details.invalid_action_penalty == 0.1
    assert "requires a non-empty code" in observation.last_action_status


def test_easy_task_gets_full_score_after_fix():
    env = PythonCodeReviewEnvironment()
    task = get_task("syntax-fix-easy")
    env.reset(task_id=task.task_id)

    env.step(PythonCodeReviewAction(action_type="edit_code", code=task.reference_code))
    observation = env.step(PythonCodeReviewAction(action_type="submit_solution"))

    assert observation.done is True
    assert observation.score == 1.0
    assert observation.reward_details.correctness_bonus == 0.5


def test_medium_task_reports_visible_progress():
    env = PythonCodeReviewEnvironment()
    env.reset(task_id="bug-fix-medium")

    observation = env.step(PythonCodeReviewAction(action_type="run_tests"))

    assert observation.score < 1.0
    assert "visible checks" in observation.test_results


def test_hard_task_reference_solution_scores_high():
    env = PythonCodeReviewEnvironment()
    task = get_task("optimization-hard")
    env.reset(task_id=task.task_id)

    env.step(PythonCodeReviewAction(action_type="edit_code", code=task.reference_code))
    observation = env.step(PythonCodeReviewAction(action_type="submit_solution"))

    assert observation.done is True
    assert observation.score >= 0.9

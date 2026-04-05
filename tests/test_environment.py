from models import PythonCodeReviewAction
from server.env import PythonCodeReviewEnvironment


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

    assert observation.reward < 0
    assert observation.reward_details.invalid_action_penalty == 0.1
    assert "requires code" in observation.last_action_status


def test_easy_task_gets_full_score_after_fix():
    env = PythonCodeReviewEnvironment()
    env.reset(task_id="syntax-fix-easy")

    env.step(
        PythonCodeReviewAction(
            action_type="edit_code",
            code="""def normalize_username(raw_name: str) -> str:
    cleaned = raw_name.strip().lower()
    if not cleaned:
        return "anonymous"
    return cleaned.replace(" ", "_")
""",
        )
    )
    observation = env.step(PythonCodeReviewAction(action_type="submit_solution"))

    assert observation.done is True
    assert observation.score == 1.0


def test_medium_task_reports_partial_visible_progress():
    env = PythonCodeReviewEnvironment()
    env.reset(task_id="bug-fix-medium")

    observation = env.step(PythonCodeReviewAction(action_type="run_tests"))

    assert observation.score < 1.0
    assert "visible checks" in observation.test_results


def test_hard_task_reference_solution_scores_high():
    env = PythonCodeReviewEnvironment()
    env.reset(task_id="optimization-hard")

    env.step(
        PythonCodeReviewAction(
            action_type="edit_code",
            code="""from collections import Counter
from typing import Iterable


def summarize_user_activity(events: Iterable[dict]) -> list[tuple[str, int]]:
    \"\"\"Aggregate user activity counts in one pass.\"\"\"

    counts = Counter(event["user_id"] for event in events)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))
""",
        )
    )
    observation = env.step(PythonCodeReviewAction(action_type="submit_solution"))

    assert observation.done is True
    assert observation.score >= 0.9

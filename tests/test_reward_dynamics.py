from python_code_review_env.envs.python_env_env.models import PythonCodeReviewAction
from python_code_review_env.envs.python_env_env.server.env import PythonCodeReviewEnvironment
from python_code_review_env.envs.python_env_env.tasks.task_bank import get_task


def test_reward_changes_across_deterministic_steps():
    env = PythonCodeReviewEnvironment(verbose=False)
    task = get_task("syntax-fix-easy")
    env.reset(task_id=task.task_id)

    actions = [
        PythonCodeReviewAction(action_type="analyze_code"),
        PythonCodeReviewAction(action_type="edit_code", code=""),
        PythonCodeReviewAction(action_type="edit_code", code=task.reference_code),
        PythonCodeReviewAction(action_type="submit_solution"),
    ]

    rewards = []
    for action in actions:
        observation = env.step(action)
        rewards.append(float(observation.reward or 0.0))

    assert all(-1.0 <= reward <= 1.0 for reward in rewards)
    assert rewards[0] == 0.0
    assert rewards[1] < 0.0
    assert rewards[2] > 0.0
    assert rewards[3] > rewards[2]

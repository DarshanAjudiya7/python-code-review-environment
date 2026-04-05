from models import PythonCodeReviewAction
from server.env import PythonCodeReviewEnvironment


FIXED_SYNTAX_CODE = """def normalize_username(raw_name: str) -> str:
    cleaned = raw_name.strip().lower()
    if not cleaned:
        return "anonymous"
    return cleaned.replace(" ", "_")
"""


def test_reward_changes_across_five_steps():
    env = PythonCodeReviewEnvironment(verbose=False)
    env.reset(task_id="syntax-fix-easy")

    actions = [
        PythonCodeReviewAction(action_type="analyze_code"),
        PythonCodeReviewAction(action_type="analyze_code"),
        PythonCodeReviewAction(action_type="run_tests"),
        PythonCodeReviewAction(action_type="edit_code", code=FIXED_SYNTAX_CODE),
        PythonCodeReviewAction(action_type="submit_solution"),
    ]

    rewards = []
    for action in actions:
        observation = env.step(action)
        rewards.append(float(observation.reward or 0.0))

    assert all(-1.0 <= reward <= 1.0 for reward in rewards)
    assert len(set(rewards)) > 1
    assert any(reward > 0 for reward in rewards)
    assert any(reward < 0 for reward in rewards)
    assert not any(
        rewards[index] == rewards[index + 1] == rewards[index + 2]
        for index in range(len(rewards) - 2)
    )

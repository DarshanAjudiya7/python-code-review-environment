"""Client for the Python code review environment."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import (
    HistoryEntry,
    PythonCodeReviewAction,
    PythonCodeReviewObservation,
    PythonCodeReviewState,
    RewardDetails,
)


class PythonEnv(
    EnvClient[PythonCodeReviewAction, PythonCodeReviewObservation, PythonCodeReviewState]
):
    """OpenEnv HTTP client for the Python code review benchmark."""

    def _step_payload(self, action: PythonCodeReviewAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[PythonCodeReviewObservation]:
        obs = payload.get("observation", {})
        observation = PythonCodeReviewObservation(
            task_id=obs["task_id"],
            title=obs["title"],
            difficulty=obs["difficulty"],
            task_kind=obs["task_kind"],
            task_description=obs["task_description"],
            current_code=obs.get("current_code", ""),
            errors=obs.get("errors", ""),
            test_results=obs.get("test_results", ""),
            history=[HistoryEntry(**entry) for entry in obs.get("history", [])],
            attempts_remaining=obs.get("attempts_remaining", 0),
            last_action_status=obs.get("last_action_status", ""),
            score=obs.get("score", 0.0),
            reward_details=RewardDetails(**obs.get("reward_details", {})),
            done=payload.get("done", obs.get("done", False)),
            reward=payload.get("reward", obs.get("reward")),
            metadata=obs.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs.get("reward")),
            done=payload.get("done", obs.get("done", False)),
        )

    def _parse_state(self, payload: Dict) -> PythonCodeReviewState:
        return PythonCodeReviewState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id"),
            difficulty=payload.get("difficulty"),
            task_kind=payload.get("task_kind"),
            attempts_remaining=payload.get("attempts_remaining", 0),
            current_code=payload.get("current_code", ""),
            errors=payload.get("errors", ""),
            test_results=payload.get("test_results", ""),
            history=[HistoryEntry(**entry) for entry in payload.get("history", [])],
            score=payload.get("score", 0.0),
            done=payload.get("done", False),
        )


CodeReviewEnv = PythonEnv
MyEnv = PythonEnv

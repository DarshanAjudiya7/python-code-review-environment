"""Typed models for Python code review and repair environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


Difficulty = Literal["easy", "medium", "hard"]
TaskKind = Literal["syntax_fix", "bug_fix", "optimization"]
ActionType = Literal["analyze_code", "edit_code", "run_tests", "submit_solution"]


class HistoryEntry(BaseModel):
    """Record of one action taken during an episode."""

    step: int = Field(..., ge=0)
    action_type: ActionType
    status: str = Field(..., description="Outcome message")
    reward: float = Field(...)


class RewardDetails(BaseModel):
    """Detailed reward breakdown for transparency."""

    value: float = Field(..., description="Net scalar reward for this step")
    syntax_reward: float = Field(default=0.0, description="Bonus for fixing syntax")
    test_reward: float = Field(default=0.0, description="Reward from passing tests")
    quality_bonus: float = Field(default=0.0, description="Bonus for code quality improvements")
    correctness_bonus: float = Field(default=0.0, description="Bonus for full correctness")
    invalid_action_penalty: float = Field(default=0.0, description="Penalty for invalid actions")
    timeout_penalty: float = Field(default=0.0, description="Penalty for timeouts")
    reason: str = Field(..., description="Explanation of reward")


class PythonCodeReviewAction(Action):
    """Action space for code review environment."""

    action_type: ActionType = Field(..., description="Type of action to perform")
    code: Optional[str] = Field(default=None, description="New code for edit_code actions")


class PythonCodeReviewObservation(Observation):
    """Observation returned by reset() and step()."""

    task_id: str = Field(..., description="Current task identifier")
    difficulty: Difficulty = Field(..., description="Task difficulty level")
    task_description: str = Field(..., description="Detailed task description")
    current_code: str = Field(..., description="Current code state")
    errors: str = Field(..., description="Syntax/compilation errors, if any")
    test_results: str = Field(..., description="Results from test execution")
    visible_tests: List[str] = Field(default_factory=list, description="Public test cases")
    history: List[HistoryEntry] = Field(default_factory=list, description="Action history")
    attempts_remaining: int = Field(..., ge=0, description="Actions left in episode")
    score: float = Field(..., ge=0.0, le=1.0, description="Current episode score")
    reward: RewardDetails = Field(default_factory=lambda: RewardDetails(value=0.0, reason="Reset"))


class PythonCodeReviewState(State):
    """Exposed environment state."""

    episode_id: str = Field(..., description="Unique episode identifier")
    step_count: int = Field(default=0, ge=0)
    task_id: Optional[str] = Field(default=None)
    difficulty: Optional[Difficulty] = Field(default=None)
    task_kind: Optional[TaskKind] = Field(default=None)
    attempts_remaining: int = Field(default=0, ge=0)
    current_code: str = Field(default="")
    errors: str = Field(default="")
    test_results: str = Field(default="")
    history: List[HistoryEntry] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    done: bool = Field(default=False)


class TaskDescriptor(BaseModel):
    """Public task metadata."""

    task_id: str = Field(..., description="Stable task identifier")
    title: str = Field(..., description="Human-readable title")
    difficulty: Difficulty = Field(..., description="Difficulty level")
    task_kind: TaskKind = Field(..., description="Type of task")
    task_description: str = Field(..., description="Full task description")
    starter_code: str = Field(..., description="Initial broken code")
    visible_tests: List[str] = Field(default_factory=list, description="Public test cases")
    max_steps: int = Field(..., ge=1, description="Maximum steps allowed")


class TaskGrade(BaseModel):
    """Grading result for task submission."""

    score: float = Field(..., ge=0.0, le=1.0, description="Overall score")
    syntax_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tests_passed: int = Field(default=0, ge=0)
    tests_total: int = Field(default=0, ge=0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    timed_out: bool = Field(default=False)
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["ok"] = "ok"
    environment: str = "python_code_review_env"
    task_count: int = Field(default=0, ge=0)
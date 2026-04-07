"""Pydantic models for the python_code_review_env benchmark."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


Difficulty = Literal["easy", "medium", "hard"]
TaskKind = Literal["syntax_fix", "bug_fix", "optimization"]
ActionType = Literal["analyze_code", "edit_code", "run_tests", "submit_solution"]


class HistoryEntry(BaseModel):
    """One action/feedback pair in the episode trace."""

    step: int = Field(..., ge=0)
    action_type: ActionType
    status: str
    reward: float


class RewardDetails(BaseModel):
    """Transparent reward breakdown for the last action."""

    value: float = Field(..., ge=-1.0, le=1.0)
    syntax_reward: float = Field(default=0.0)
    test_reward: float = Field(default=0.0)
    quality_bonus: float = Field(default=0.0)
    correctness_bonus: float = Field(default=0.0)
    progress_delta: float = Field(default=0.0)
    invalid_action_penalty: float = Field(default=0.0)
    timeout_penalty: float = Field(default=0.0)
    reason: str
    prev_score: float = Field(default=0.0, ge=0.0, le=1.0)
    curr_score: float = Field(default=0.0, ge=0.0, le=1.0)
    code_changed: bool = Field(default=False)


class PythonCodeReviewAction(Action):
    """Structured environment action."""

    action_type: ActionType
    code: Optional[str] = None
    reasoning: Optional[str] = None


class PythonCodeReviewObservation(Observation):
    """Structured observation returned by reset/step."""

    task_id: str
    title: str = ""
    difficulty: Difficulty
    task_kind: Optional[TaskKind] = None
    task_description: str
    current_code: str
    errors: str
    test_results: str
    visible_tests: List[str] = Field(default_factory=list)
    history: List[HistoryEntry] = Field(default_factory=list)
    attempts_remaining: int = Field(..., ge=0)
    last_action_status: str = ""
    score: float = Field(..., ge=0.0, le=1.0)
    reward_details: RewardDetails = Field(
        default_factory=lambda: RewardDetails(value=0.0, reason="Episode reset."),
    )


class PythonCodeReviewState(State):
    """Exposed environment state."""

    episode_id: str
    step_count: int = Field(default=0, ge=0)
    task_id: Optional[str] = None
    difficulty: Optional[Difficulty] = None
    task_kind: Optional[TaskKind] = None
    attempts_remaining: int = Field(default=0, ge=0)
    current_code: str = ""
    errors: str = ""
    test_results: str = ""
    history: List[HistoryEntry] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    done: bool = False


class TaskDescriptor(BaseModel):
    """Public task metadata."""

    task_id: str
    title: str
    difficulty: Difficulty
    task_kind: TaskKind
    task_description: str
    starter_code: str
    visible_tests: List[str] = Field(default_factory=list)
    max_steps: int = Field(..., ge=1)


class TaskGrade(BaseModel):
    """Deterministic grading output."""

    score: float = Field(..., ge=0.0, le=1.0)
    syntax_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tests_passed: int = Field(default=0, ge=0)
    tests_total: int = Field(default=0, ge=0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    runtime_score: float = Field(default=0.0, ge=0.0, le=1.0)
    style_score: float = Field(default=0.0, ge=0.0, le=1.0)
    timed_out: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Simple health response."""

    status: Literal["ok"] = "ok"
    environment: str = "python_code_review_env"
    task_count: int = Field(default=0, ge=0)

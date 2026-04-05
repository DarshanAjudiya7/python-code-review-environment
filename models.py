"""Typed models shared across the Python PR review environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


Difficulty = Literal["easy", "medium", "hard"]
Category = Literal["bug", "security", "performance", "maintainability", "testing"]
Severity = Literal["critical", "warning", "info"]
Operation = Literal["read_file", "add_finding", "submit_review", "finish"]


class ReviewFinding(BaseModel):
    """Structured review finding submitted by the agent."""

    file_path: str = Field(..., description="Repository-relative file path")
    line: Optional[int] = Field(default=None, ge=1, description="1-based line number")
    category: Category = Field(..., description="Finding category")
    severity: Severity = Field(default="warning", description="Finding severity")
    title: str = Field(..., min_length=3, description="Short finding title")
    explanation: str = Field(..., min_length=5, description="Why the issue matters")
    suggested_fix: str = Field(..., min_length=5, description="Concrete fix direction")


class ReviewHistoryEntry(BaseModel):
    """Compact step history exposed to the agent."""

    step: int = Field(..., ge=0)
    operation: Operation = Field(..., description="Action taken on that step")
    summary: str = Field(..., description="Human-readable result summary")


class PythonReviewReward(BaseModel):
    """Structured reward breakdown for one step."""

    value: float = Field(..., description="Net scalar reward used by the environment")
    matched_progress: float = Field(default=0.0, description="Positive reward for valid progress")
    false_positive_penalty: float = Field(
        default=0.0, description="Penalty applied to unsupported findings"
    )
    efficiency_penalty: float = Field(
        default=-0.3, description="Penalty applied to wasteful repeated actions"
    )
    invalid_action_penalty: float = Field(
        default=-0.5, description="Penalty applied to invalid actions"
    )
    reason: str = Field(default="", description="Short explanation of the reward outcome")


class TaskDescriptor(BaseModel):
    """Public description of one PR review task."""

    task_id: str = Field(..., description="Stable task identifier")
    difficulty: Difficulty = Field(..., description="Difficulty bucket")
    title: str = Field(..., description="Short task title")
    goal: str = Field(..., description="Objective the reviewer should accomplish")
    repo_summary: str = Field(..., description="High-level repository or PR context")
    changed_files: List[str] = Field(default_factory=list)
    available_files: List[str] = Field(default_factory=list)
    max_steps: int = Field(..., ge=1, description="Maximum number of actions in the episode")


class PythonReviewAction(Action):
    """Action accepted by the PR review environment."""

    operation: Operation = Field(..., description="The action to perform")
    path: Optional[str] = Field(
        default=None,
        description="Repository-relative path for read_file actions",
    )
    finding: Optional[ReviewFinding] = Field(
        default=None,
        description="Structured finding for add_finding actions",
    )


class PythonReviewObservation(Observation):
    """Observation returned by reset and step."""

    task_id: str = Field(..., description="Active task identifier")
    difficulty: Difficulty = Field(..., description="Task difficulty bucket")
    goal: str = Field(..., description="Task objective shown to the agent")
    repo_summary: str = Field(..., description="High-level PR or repo context")
    changed_files: List[str] = Field(default_factory=list)
    visible_diff: str = Field(..., description="Visible diff plus any opened file context")
    available_files: List[str] = Field(default_factory=list)
    review_history: List[ReviewHistoryEntry] = Field(default_factory=list)
    attempts_remaining: int = Field(default=0, ge=0)
    last_action_status: str = Field(default="", description="Feedback for the latest action")
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_details: PythonReviewReward = Field(default_factory=lambda: PythonReviewReward(value=0.0))


class PythonReviewState(State):
    """Current non-secret environment state."""

    task_id: Optional[str] = Field(default=None)
    difficulty: Optional[Difficulty] = Field(default=None)
    attempts_remaining: int = Field(default=0, ge=0)
    opened_files: List[str] = Field(default_factory=list)
    submitted_findings: List[ReviewFinding] = Field(default_factory=list)
    review_history: List[ReviewHistoryEntry] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    done: bool = Field(default=False)


class TaskSummary(BaseModel):
    """Compact task listing for API consumers."""

    task_id: str
    difficulty: Difficulty
    title: str
    goal: str


class TaskSubmission(BaseModel):
    """Offline grader input."""

    findings: List[ReviewFinding] = Field(default_factory=list)


class TaskGrade(BaseModel):
    """Offline grader result used by tests and helper routes."""

    score: float = Field(..., ge=0.0, le=1.0)
    matched_issue_ids: List[str] = Field(default_factory=list)
    false_positives: int = Field(default=0, ge=0)
    duplicate_findings: int = Field(default=0, ge=0)
    matched_weight: float = Field(default=0.0, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Health payload for deployment checks."""

    status: Literal["ok"] = "ok"
    environment: str = "python_pr_review_env"
    task_count: int = Field(default=0, ge=0)


class DeleteResponse(BaseModel):
    """Acknowledgement payload for cleanup routes."""

    detail: str


def reward_metadata(reward: PythonReviewReward) -> Dict[str, Any]:
    """Serialize reward details for OpenEnv metadata payloads."""

    return reward.model_dump()
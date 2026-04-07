"""Typed models for the self-contained server package."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .compat import Action, Observation, State


Difficulty = Literal["easy", "medium", "hard"]
TaskKind = Literal["syntax_fix", "bug_fix", "optimization"]
ActionType = Literal["analyze_code", "edit_code", "run_tests", "submit_solution"]
Category = Literal["bug", "security", "performance", "maintainability", "style", "testing"]
Severity = Literal["critical", "warning", "info"]


class HistoryEntry(BaseModel):
    step: int = Field(..., ge=0)
    action_type: ActionType
    status: str
    reward: float


class RewardDetails(BaseModel):
    value: float
    syntax_reward: float = 0.0
    test_reward: float = 0.0
    quality_bonus: float = 0.0
    correctness_bonus: float = 0.0
    progress_delta: float = 0.0
    stagnation_penalty: float = 0.0
    regression_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    timeout_penalty: float = 0.0
    reason: str
    prev_score: float = 0.0
    curr_score: float = 0.0
    code_changed: bool = False


class PythonCodeReviewAction(Action):
    action_type: ActionType
    code: Optional[str] = None


class PythonCodeReviewObservation(Observation):
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
        default_factory=lambda: RewardDetails(value=0.0, reason="Reset")
    )


class PythonCodeReviewState(State):
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
    task_id: str
    title: str
    difficulty: Difficulty
    task_kind: Optional[TaskKind] = None
    task_description: str = ""
    starter_code: str = ""
    visible_tests: List[str] = Field(default_factory=list)
    goal: str = ""
    repo_summary: str = ""
    changed_files: List[str] = Field(default_factory=list)
    available_files: List[str] = Field(default_factory=list)
    max_steps: int = Field(..., ge=1)


class TaskSummary(BaseModel):
    task_id: str
    difficulty: Difficulty
    title: str
    goal: str = ""


class ReviewFinding(BaseModel):
    title: str
    file_path: str = ""
    line: Optional[int] = Field(default=None, ge=1)
    category: Category = "bug"
    severity: Severity = "warning"
    rationale: str = ""
    recommendation: str = ""
    rule_id: str = ""

    @property
    def explanation(self) -> str:
        return self.rationale

    @property
    def suggested_fix(self) -> str:
        return self.recommendation


class DirectReviewResponse(BaseModel):
    issues: List[ReviewFinding] = Field(default_factory=list)
    summary: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    improved_code: Optional[str] = None


class TaskGrade(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    syntax_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tests_passed: int = Field(default=0, ge=0)
    tests_total: int = Field(default=0, ge=0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    runtime_score: float = Field(default=0.0, ge=0.0, le=1.0)
    timed_out: bool = False
    matched_issue_ids: List[str] = Field(default_factory=list)
    false_positives: int = Field(default=0, ge=0)
    duplicate_findings: int = Field(default=0, ge=0)
    matched_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    environment: str = "python_code_review_env"
    task_count: int = Field(default=0, ge=0)


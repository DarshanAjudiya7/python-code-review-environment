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
    """Detailed reward breakdown for transparent agent feedback.
    
    The reward system is dynamic and multi-component, with 6 independent sources:
    
    1. Progress Reward (max +0.25)
       - Awarded for score improvement from previous step
       - Formula: min(PROGRESS_SCALE * score_delta, 0.25)
       - Encourages continuous improvement
    
    2. Syntax Reward (max +0.35)
       - One-time bonus for fixing syntax errors (first compile)
       - Applied when code transitions from uncompilable to compilable
       - Acknowledges the critical first step of valid code
    
    3. Test Reward (max +0.20)
       - Based on improvement in test pass rate
       - Formula: min(TEST_PASS_REWARD_SCALE * test_improvement, 0.20)
       - Rewards incremental test progress
    
    4. Quality Reward (max +0.15)
       - Based on AST-detected code quality metrics
       - Rewards improvements in structure, readability, best practices
       - Uses deterministic grader feedback
    
    5. Stagnation Penalty (−0.10)
       - Applied when agent acts but code doesn't change
       - Encourages editing rather than repeated analysis
       - Configurable via STAGNATION_PENALTY constant
    
    6. Regression Penalty (scale −0.20)
       - Applied when score decreases from previous step
       - Formula: REGRESSION_PENALTY_SCALE * abs(score_delta)
       - Discourages actions that make code worse
    
    Final Reward: clamp(progress + syntax + test + quality - stagnation - regression, -1.0, +1.0)
    
    The result is always bounded in [-1.0, +1.0], providing interpretable feedback for learning.
    """

    value: float = Field(..., description="Net scalar reward for this step (bounded in [-1.0, +1.0])")
    syntax_reward: float = Field(default=0.0, description="Bonus for fixing syntax errors (max +0.35)")
    test_reward: float = Field(default=0.0, description="Reward from test improvements (max +0.20)")
    quality_bonus: float = Field(default=0.0, description="Bonus for code quality improvements (max +0.15)")
    correctness_bonus: float = Field(default=0.0, description="Bonus for full correctness (max +0.50)")
    progress_delta: float = Field(default=0.0, description="Reward from score improvement (max +0.25)")
    stagnation_penalty: float = Field(default=0.0, description="Penalty for unchanged code (−0.10)")
    regression_penalty: float = Field(default=0.0, description="Penalty for score decline (scale −0.20)")
    invalid_action_penalty: float = Field(default=0.0, description="Penalty for invalid actions (−0.15)")
    timeout_penalty: float = Field(default=0.0, description="Penalty for execution timeout (−0.15)")
    reason: str = Field(..., description="Human-readable explanation of the reward")
    
    # Debug information for transparency
    prev_score: float = Field(default=0.0, description="Score before this step")
    curr_score: float = Field(default=0.0, description="Score after this step")
    code_changed: bool = Field(default=False, description="Whether the action modified the code")


class PythonCodeReviewAction(Action):
    """Action space for code review environment."""

    action_type: ActionType = Field(..., description="Type of action to perform")
    code: Optional[str] = Field(default=None, description="New code for edit_code actions")


class PythonCodeReviewObservation(Observation):
    """Observation returned by reset() and step()."""

    task_id: str = Field(..., description="Current task identifier")
    title: str = Field(default="", description="Human-readable task title")
    difficulty: Difficulty = Field(..., description="Task difficulty level")
    task_kind: Optional[TaskKind] = Field(default=None, description="Task type")
    task_description: str = Field(..., description="Detailed task description")
    current_code: str = Field(..., description="Current code state")
    errors: str = Field(..., description="Syntax/compilation errors, if any")
    test_results: str = Field(..., description="Results from test execution")
    visible_tests: List[str] = Field(default_factory=list, description="Public test cases")
    history: List[HistoryEntry] = Field(default_factory=list, description="Action history")
    attempts_remaining: int = Field(..., ge=0, description="Actions left in episode")
    last_action_status: str = Field(default="", description="Outcome message from the last action")
    score: float = Field(..., ge=0.0, le=1.0, description="Current episode score")
    reward_details: RewardDetails = Field(
        default_factory=lambda: RewardDetails(value=0.0, reason="Reset"),
        description="Detailed reward breakdown for the last action",
    )


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

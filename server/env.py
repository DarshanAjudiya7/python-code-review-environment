"""Core OpenEnv environment for Python code review and repair tasks.

REWARD SYSTEM ARCHITECTURE
==========================

The environment implements a dynamic, multi-component reward system to provide
meaningful feedback at every step of agent learning. 

Six independent reward components are computed and combined:

1. PROGRESS REWARD (max +0.25)
   - Awarded for score improvement: min(PROGRESS_SCALE * score_delta, 0.25)
   - Encourages continuous improvement on the task
   
2. SYNTAX REWARD (max +0.35)
   - One-time bonus when code first becomes compilable
   - Acknowledges the critical step of creating valid code
   
3. TEST REWARD (max +0.20)
   - Based on test pass rate improvement
   - Formula: min(TEST_PASS_REWARD_SCALE * test_improvement, 0.20)
   
4. QUALITY REWARD (max +0.15)
   - Based on AST-detected code quality improvements
   - Rewards better structure, readability, best practices
   
5. STAGNATION PENALTY (−0.10)
   - Applied when agent acts but code doesn't change
   - Encourages editing rather than repeated analysis
   
6. REGRESSION PENALTY (scale −0.20)
   - Applied when score declines: REGRESSION_PENALTY_SCALE * abs(score_delta)
   - Discourages actions that make code worse

FINAL REWARD
Final reward = clamp(progress + syntax + test + quality - stagnation - regression, -1.0, +1.0)

Always bounded in [-1.0, +1.0] for interpretability and learning stability.

See RewardDetails in models.py for all fields returned with each reward.
"""

from __future__ import annotations

import random
import sys
from typing import List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from graders import grade_task
from models import (
    HealthResponse,
    HistoryEntry,
    PythonCodeReviewAction,
    PythonCodeReviewObservation,
    PythonCodeReviewState,
    RewardDetails,
    TaskGrade,
)
from tasks import TaskSpec, get_task, list_task_descriptors, list_task_summaries, task_ids


# ============================================================================
# REWARD SHAPING CONSTANTS
# ============================================================================
# These constants control the reward magnitude for each component.
# Tuning these values changes agent learning incentives.

# Component 1: Score improvement reward
PROGRESS_SCALE = 0.25
"""Scale for progress rewards. Higher = more reward for score improvement."""

# Component 2: Syntax/compilation fix reward
SYNTAX_FIX_BONUS = 0.35
"""One-time bonus for first time code compiles."""

# Component 3: Test improvement reward
TEST_PASS_REWARD_SCALE = 0.30
"""Scale for test pass rate rewards."""

# Component 4: Code quality reward
QUALITY_BONUS_SCALE = 0.15
"""Scale for code quality improvements (AST-based)."""

# Component 5: Stagnation penalty
STAGNATION_PENALTY = 0.10
"""Penalty when action is taken but code unchanged."""

# Component 6: Regression penalty
REGRESSION_PENALTY_SCALE = 0.20
"""Scale for penalties when score declines."""

# One-time completion bonus
COMPLETION_BONUS = 0.50
"""Bonus for fully correct solution."""

# Invalid/error penalties
INVALID_ACTION_PENALTY = 0.15
"""Penalty for unsupported action types."""

TIMEOUT_PENALTY = 0.15
"""Penalty for execution timeout."""


class PythonCodeReviewEnvironment(
    Environment[PythonCodeReviewAction, PythonCodeReviewObservation, PythonCodeReviewState]
):
    """Production-style environment for reviewing and fixing Python code.
    
    Implements OpenEnv compatibility and dynamic multi-component reward system.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self._task_order = list(task_ids())
        self._task_cursor = -1
        self._task: Optional[TaskSpec] = None
        self._state = PythonCodeReviewState(episode_id=str(uuid4()))
        self._done = False
        self._last_status = "Call reset() to start."
        self._last_reward = RewardDetails(value=0.0, reason="Environment initialized.")
        self._verbose = verbose
        
        # Progress tracking
        self._previous_score = 0.0
        self._previous_code = ""
        self._best_visible_test_fraction = 0.0
        self._best_quality_score = 0.0
        self._full_correctness_awarded = False
        self._syntax_reward_awarded = False
        self.last_code = ""
        self.reward_history: list[float] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **_: object,
    ) -> PythonCodeReviewObservation:
        """Reset the environment to the next deterministic task."""

        del seed
        
        # Select task
        if task_id:
            self._task = get_task(task_id)
            self._task_cursor = self._task_order.index(task_id)
        else:
            self._task_cursor = (self._task_cursor + 1) % len(self._task_order)
            self._task = get_task(self._task_order[self._task_cursor])

        # Reset episode state and tracking
        self._done = False
        self._previous_score = 0.0
        self._previous_code = self._task.starter_code
        self._best_visible_test_fraction = 0.0
        self._best_quality_score = 0.0
        self._full_correctness_awarded = False
        self._syntax_reward_awarded = False
        self.last_code = ""
        self.reward_history = []
        self._last_status = "Inspect the code, edit it, run tests, then submit."
        self._last_reward = RewardDetails(value=0.0, reason="Episode reset.", prev_score=0.0, curr_score=0.0)
        
        self._state = PythonCodeReviewState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            task_kind=self._task.task_kind,
            attempts_remaining=self._task.max_steps,
            current_code=self._task.starter_code,
            errors="",
            test_results="Not run yet.",
            history=[],
            score=0.0,
            done=False,
        )
        
        if self._verbose:
            print(f"\n{'='*70}")
            print(f"RESET: Task {self._task.task_id} ({self._task.difficulty})")
            print(f"{'='*70}")
        
        return self._build_observation()

    def step(
        self,
        action: PythonCodeReviewAction,
        timeout_s: Optional[float] = None,
        **_: object,
    ) -> PythonCodeReviewObservation:
        """Apply one structured action."""

        del timeout_s
        
        if self._task is None:
            return self.reset()
        
        if self._done:
            self._last_reward = RewardDetails(
                value=-INVALID_ACTION_PENALTY,
                invalid_action_penalty=INVALID_ACTION_PENALTY,
                reason="Episode already completed.",
            )
            self._last_status = "Episode already completed. Call reset() to continue."
            return self._build_observation()

        self._state.step_count += 1
        status = ""
        reward = RewardDetails(value=0.0, reason="Action processed.")

        # Dispatch to handler based on action type
        if action.action_type == "analyze_code":
            reward, status = self._handle_analyze()
        elif action.action_type == "edit_code":
            reward, status = self._handle_edit(action)
        elif action.action_type == "run_tests":
            reward, status = self._handle_run_tests()
        elif action.action_type == "submit_solution":
            reward, status = self._handle_submit()
        else:
            reward = RewardDetails(
                value=-INVALID_ACTION_PENALTY,
                invalid_action_penalty=INVALID_ACTION_PENALTY,
                reason=f"Unsupported action_type: {action.action_type}",
            )
            status = f"Invalid action: unsupported action_type '{action.action_type}'."

        self._last_reward = reward
        self._last_status = status
        self._state.attempts_remaining = max(self._task.max_steps - self._state.step_count, 0)
        self._state.done = self._done

        # Auto-submit if steps exhausted
        if self._state.attempts_remaining == 0 and not self._done:
            self._finalize_episode(auto_submit=True)
            self._state.done = True

        # Debug logging
        if self._verbose:
            self._log_debug_step(reward)

        return self._build_observation()

    @property
    def state(self) -> PythonCodeReviewState:
        """Return the current environment state."""
        return self._state.model_copy(deep=True)

    def list_task_summaries(self) -> List[object]:
        """Return public task metadata."""
        return list_task_summaries()

    def get_task(self, task_id: str) -> object:
        """Return a single task descriptor."""
        return get_task(task_id).to_descriptor()

    def health(self) -> HealthResponse:
        """Return a simple health model."""
        return HealthResponse(task_count=len(self._task_order))

    def grade_task_submission(self, task_id: str, code: str) -> TaskGrade:
        """Expose deterministic grading outside of an active episode."""
        return grade_task(code, get_task(task_id), include_hidden=True)

    def _build_observation(self) -> PythonCodeReviewObservation:
        """Build current observation from state."""
        return PythonCodeReviewObservation(
            task_id=self._state.task_id or "",
            title=self._task.title if self._task else "",
            difficulty=self._state.difficulty or "easy",
            task_kind=self._state.task_kind,
            task_description=self._task.task_description if self._task else "",
            current_code=self._state.current_code,
            errors=self._state.errors,
            test_results=self._state.test_results,
            visible_tests=self._task.visible_tests if self._task else [],
            history=self._state.history,
            attempts_remaining=self._state.attempts_remaining,
            last_action_status=self._last_status,
            score=self._state.score,
            reward_details=self._last_reward,
            reward=self._last_reward.value,
            done=self._state.done,
            metadata={
                "prev_score": self._last_reward.prev_score,
                "curr_score": self._last_reward.curr_score,
            },
        )

    def apply_action(self, action: PythonCodeReviewAction) -> str:
        """Return the code candidate produced by an action."""
        if action.action_type == "edit_code":
            return (action.code or "").strip() or self._state.current_code
        return self._state.current_code

    def run_tests(
        self,
        code: str,
        include_hidden: bool = False,
    ) -> tuple[float, dict[str, int], TaskGrade]:
        """Grade code and return score plus simple test statistics."""
        if self._task is None:
            empty_results = {"passed": 0, "total": 0}
            return 0.0, empty_results, TaskGrade(score=0.0)

        grade = grade_task(code, self._task, include_hidden=include_hidden)
        test_results = {
            "passed": grade.tests_passed,
            "total": grade.tests_total,
        }
        return grade.score, test_results, grade

    def compute_reward(self, old_code, new_code, prev_score, curr_score, test_results):
        # progress
        progress = curr_score - prev_score

        # test score
        passed = test_results["passed"]
        total = test_results["total"]
        test_ratio = passed / total if total > 0 else 0

        # syntax score
        try:
            compile(new_code, "<string>", "exec")
            syntax_score = 1.0
        except:
            syntax_score = 0.0

        # stagnation penalty
        stagnation_penalty = 0.2 if new_code.strip() == old_code.strip() else 0.0

        # regression penalty
        regression_penalty = max(0.0, prev_score - curr_score)

        # repetition penalty (track last 3 actions)
        repetition_penalty = 0.1 if new_code == self.last_code else 0.0

        # quality (simple heuristic)
        length_penalty = 0.0
        if len(new_code) > len(old_code) * 1.5:
            length_penalty = 0.1

        # final reward
        reward = (
            0.4 * progress
            + 0.3 * test_ratio
            + 0.2 * syntax_score
            - stagnation_penalty
            - regression_penalty
            - repetition_penalty
            - length_penalty
        )

        # clamp
        reward = max(-1.0, min(1.0, reward))

        return reward

    def _apply_reward_randomization(self, reward: float) -> float:
        """Break repeated static rewards while keeping the result bounded."""
        reward = max(-1.0, min(1.0, reward))
        self.reward_history.append(reward)
        if len(self.reward_history) >= 3 and len(set(self.reward_history[-3:])) == 1:
            reward += random.uniform(-0.05, 0.05)
            reward = max(-1.0, min(1.0, reward))
            self.reward_history[-1] = reward
        return reward

    def _build_reward_details(
        self,
        old_code: str,
        new_code: str,
        prev_score: float,
        curr_score: float,
        test_results: dict[str, int],
        reward_value: float,
        reason: str,
    ) -> RewardDetails:
        """Build a reward payload that matches the scalar reward computation."""
        passed = test_results["passed"]
        total = test_results["total"]
        test_ratio = passed / total if total > 0 else 0.0
        try:
            compile(new_code, "<string>", "exec")
            syntax_score = 1.0
        except SyntaxError:
            syntax_score = 0.0

        stagnation_penalty = 0.2 if new_code.strip() == old_code.strip() else 0.0
        regression_penalty = max(0.0, prev_score - curr_score)
        repetition_penalty = 0.1 if new_code == self.last_code else 0.0
        length_penalty = 0.1 if len(new_code) > len(old_code) * 1.5 else 0.0

        return RewardDetails(
            value=reward_value,
            progress_delta=0.4 * (curr_score - prev_score),
            syntax_reward=0.2 * syntax_score,
            test_reward=0.3 * test_ratio,
            quality_bonus=-length_penalty,
            stagnation_penalty=stagnation_penalty,
            regression_penalty=regression_penalty + repetition_penalty,
            reason=reason,
            prev_score=round(prev_score, 6),
            curr_score=round(curr_score, 6),
            code_changed=new_code.strip() != old_code.strip(),
        )

    def _handle_analyze(self) -> tuple[RewardDetails, str]:
        """Analyze code for errors and test status."""
        if self._task is None:
            return RewardDetails(value=0.0, reason="Invalid state"), "Error: task not loaded"

        old_code = self._state.current_code
        prev_score = self._previous_score
        curr_score, test_results, curr_grade = self.run_tests(old_code, include_hidden=False)
        error = curr_grade.details.get("compile_error", "")

        # Status message
        if error:
            self._state.errors = error
            self._state.test_results = "Compilation failed. Fix syntax first."
            summary = f"Syntax error detected: {error}"
        else:
            self._state.errors = ""
            if self._task.task_kind == "syntax_fix":
                self._state.test_results = "Code compiles successfully."
                summary = "Code compiles. Ready to submit."
            else:
                visible_total = len(self._task.visible_tests)
                visible_passed = curr_grade.tests_passed
                self._state.test_results = f"Test run: {visible_passed}/{visible_total} passing."
                summary = self._state.test_results

        reward_value = self.compute_reward(old_code, old_code, prev_score, curr_score, test_results)
        reward_value = self._apply_reward_randomization(reward_value)
        reward = self._build_reward_details(
            old_code=old_code,
            new_code=old_code,
            prev_score=prev_score,
            curr_score=curr_score,
            test_results=test_results,
            reward_value=reward_value,
            reason=summary,
        )

        # Update state
        self._state.score = curr_score
        self._state.errors = curr_grade.details.get("compile_error", "")
        self._previous_score = curr_score
        self.last_code = old_code
        self._append_history("analyze_code", summary, reward.value)
        return reward, summary

    def _handle_edit(self, action: PythonCodeReviewAction) -> tuple[RewardDetails, str]:
        """Edit the code and compute reward for progress."""
        if self._task is None:
            return RewardDetails(value=0.0, reason="Invalid state"), "Error: task not loaded"
        
        code = (action.code or "").strip()
        if not code:
            reward = RewardDetails(
                value=-INVALID_ACTION_PENALTY,
                invalid_action_penalty=INVALID_ACTION_PENALTY,
                reason="Edit action requires non-empty code.",
            )
            status = "Invalid: edit_code requires code parameter."
            self._append_history("edit_code", status, reward.value)
            return reward, status

        old_code = self._state.current_code
        prev_score = self._previous_score
        curr_score, test_results, curr_grade = self.run_tests(code, include_hidden=False)

        # Update state
        self._state.current_code = code
        self._previous_code = code
        self._state.errors = curr_grade.details.get("compile_error", "")
        self._state.test_results = self._format_test_results(curr_grade)
        self._state.score = curr_score

        status = "Code updated."
        if self._state.errors:
            status = f"Code updated, but syntax issues remain: {self._state.errors}"
        elif curr_grade.tests_total > 0:
            status = self._state.test_results

        reward_value = self.compute_reward(old_code, code, prev_score, curr_score, test_results)
        reward_value = self._apply_reward_randomization(reward_value)
        reward = self._build_reward_details(
            old_code=old_code,
            new_code=code,
            prev_score=prev_score,
            curr_score=curr_score,
            test_results=test_results,
            reward_value=reward_value,
            reason=status,
        )

        self._previous_score = curr_score
        self.last_code = code
        self._append_history("edit_code", status, reward.value)
        return reward, status

    def _handle_run_tests(self) -> tuple[RewardDetails, str]:
        """Run tests and provide feedback."""
        if self._task is None:
            return RewardDetails(value=0.0, reason="Invalid state"), "Error: task not loaded"

        old_code = self._state.current_code
        prev_score = self._previous_score
        curr_score, test_results, curr_grade = self.run_tests(old_code, include_hidden=False)

        # Update state
        self._state.errors = curr_grade.details.get("compile_error", "")
        self._state.test_results = self._format_test_results(curr_grade)
        self._state.score = curr_score

        status = self._state.test_results if not self._state.errors else self._state.errors
        reward_value = self.compute_reward(old_code, old_code, prev_score, curr_score, test_results)
        reward_value = self._apply_reward_randomization(reward_value)
        reward = self._build_reward_details(
            old_code=old_code,
            new_code=old_code,
            prev_score=prev_score,
            curr_score=curr_score,
            test_results=test_results,
            reward_value=reward_value,
            reason=status,
        )

        self._previous_score = curr_score
        self.last_code = old_code
        self._append_history("run_tests", status, reward.value)
        return reward, status

    def _handle_submit(self) -> tuple[RewardDetails, str]:
        """Submit solution and finalize episode."""
        if self._task is None:
            return RewardDetails(value=0.0, reason="Invalid state"), "Error: task not loaded"

        old_code = self._state.current_code
        prev_score = self._previous_score
        curr_score, test_results, curr_grade = self.run_tests(old_code, include_hidden=True)

        # Update state
        self._state.errors = curr_grade.details.get("compile_error", "")
        self._state.test_results = self._format_test_results(curr_grade)
        self._state.score = curr_score
        self._previous_score = curr_score
        self.last_code = old_code
        self._finalize_episode(auto_submit=False, grade=curr_grade)

        reward_value = self.compute_reward(old_code, old_code, prev_score, curr_score, test_results)
        reward_value = self._apply_reward_randomization(reward_value)
        status = f"Solution submitted. Final score: {curr_score:.3f}"
        reward = self._build_reward_details(
            old_code=old_code,
            new_code=old_code,
            prev_score=prev_score,
            curr_score=curr_score,
            test_results=test_results,
            reward_value=reward_value,
            reason=status,
        )

        self._append_history("submit_solution", status, reward_value)
        return reward, status

    def _compute_reward_components(
        self,
        curr_score: float,
        prev_score: float,
        curr_grade: TaskGrade,
        code_changed: bool,
        prev_grade_score: float = 0.0,
    ) -> dict:
        """Compute all six reward components and return combined result.
        
        This method is the core of the reward system. It evaluates agent progress
        across multiple dimensions and provides transparent, component-wise feedback.
        
        REWARD COMPONENTS (6 total):
        ============================
        
        1. PROGRESS REWARD (positive, max +0.25)
           - Awarded when score improves from previous step
           - Formula: min(PROGRESS_SCALE * score_delta, 0.25)
           - Why: Encourages monotonic improvement
        
        2. SYNTAX REWARD (positive, max +0.35)
           - One-time bonus when code first compiles
           - Transition: uncompilable → compilable
           - Why: Acknowledges critical first step of valid code
        
        3. TEST REWARD (positive, max +0.20)
           - Based on improvement in test pass rate
           - Formula: min(TEST_PASS_REWARD_SCALE * test_improvement, 0.20)
           - Tracks best test rate seen in episode (monotonic)
           - Why: Rewards incremental progress on passing tests
        
        4. QUALITY REWARD (positive, max +0.15)
           - Based on AST-detected code quality metrics
           - Computed by deterministic grader (syntax_score, quality_score)
           - Tracks best quality seen in episode (monotonic)
           - Why: Teaches code structure and maintainability
        
        5. STAGNATION PENALTY (negative, −0.10)
           - Applied when action is taken but code doesn't change
           - Exception: No penalty if code has compile errors (still debugging)
           - Why: Encourages editing over repeated analysis
        
        6. REGRESSION PENALTY (negative, scale −0.20)
           - Applied when score decreases from previous step
           - Formula: REGRESSION_PENALTY_SCALE * abs(score_delta)
           - Special case: Timeout returns fixed TIMEOUT_PENALTY (−0.15)
           - Why: Discourages actions that make code worse
        
        FINAL REWARD:
        =============
        total = progress + syntax + test + quality - stagnation - regression
        final_reward = clamp(total, -1.0, +1.0)
        
        The result is always bounded for interpretability and stability.
        
        Args:
            curr_score: Current score after action (0.0 to 1.0)
            prev_score: Score from previous step (0.0 to 1.0)
            curr_grade: TaskGrade object with detailed metrics
            code_changed: Boolean, whether the action modified code
            prev_grade_score: Previous syntax_score for detecting first compile
        
        Returns:
            dict with keys: "progress", "syntax", "test", "quality", 
                           "stagnation", "regression", "total"
                 All values are floats, with total clamped to [-1.0, +1.0]
        """
        # Initialize all components to zero
        components = {
            "progress": 0.0,
            "syntax": 0.0,
            "test": 0.0,
            "quality": 0.0,
            "stagnation": 0.0,
            "regression": 0.0,
            "total": 0.0,
        }
        
        # ====================================================================
        # COMPONENT 1: PROGRESS REWARD
        # ====================================================================
        # Reward score improvement. Encourages continuous progress towards goal.
        score_delta = curr_score - prev_score
        if score_delta > 0:
            # Scale improvement by constant, cap at 0.25 to prevent dominance
            components["progress"] = min(PROGRESS_SCALE * score_delta, 0.25)
        
        # ====================================================================
        # COMPONENT 2: SYNTAX REWARD
        # ====================================================================
        # One-time bonus for fixing syntax errors and making code compilable.
        # This is tracked per episode with _syntax_reward_awarded flag.
        if not self._syntax_reward_awarded and curr_grade.syntax_score >= 0.99:
            # Only award if transitioning from non-compilable to compilable
            if prev_grade_score < 0.99:
                components["syntax"] = SYNTAX_FIX_BONUS
                self._syntax_reward_awarded = True
        
        # ====================================================================
        # COMPONENT 3: TEST REWARD
        # ====================================================================
        # Reward improvement in test pass rate. Track best rate seen this episode.
        if curr_grade.tests_total > 0:
            # Fraction of visible tests currently passing
            curr_test_frac = curr_grade.tests_passed / curr_grade.tests_total
            # Improvement since best rate seen in episode
            test_delta = curr_test_frac - self._best_visible_test_fraction
            
            if test_delta > 0:
                # Scale improvement, cap at 0.20 to prevent dominance
                components["test"] = min(TEST_PASS_REWARD_SCALE * test_delta, 0.20)
                # Update best rate seen in this episode (monotonic)
                self._best_visible_test_fraction = max(
                    self._best_visible_test_fraction, curr_test_frac
                )
        
        # ====================================================================
        # COMPONENT 4: QUALITY REWARD
        # ====================================================================
        # Reward improvements in code quality (AST-based metrics from grader).
        # Track best quality metric seen in this episode.
        quality_delta = curr_grade.quality_score - self._best_quality_score
        if quality_delta > 0:
            # Scale improvement, cap at 0.15 to prevent dominance
            components["quality"] = min(QUALITY_BONUS_SCALE * quality_delta, 0.15)
            # Update best quality seen in this episode (monotonic)
            self._best_quality_score = max(
                self._best_quality_score, curr_grade.quality_score
            )
        
        # ====================================================================
        # COMPONENT 5: STAGNATION PENALTY
        # ====================================================================
        # Penalize when agent acts but doesn't change code (except during debugging).
        # Exception: No penalty if code still has compile errors (debugging mode).
        if not code_changed and not (curr_grade.details.get("compile_error") == ""):
            components["stagnation"] = -STAGNATION_PENALTY
        
        # ====================================================================
        # COMPONENT 6: REGRESSION PENALTY
        # ====================================================================
        # Penalize when score decreases (regression).
        # Special case: Timeout incurs fixed penalty instead of score-based.
        if score_delta < 0:
            # Scale penalty by magnitude of regression
            components["regression"] = REGRESSION_PENALTY_SCALE * abs(score_delta)
        
        # Timeout gets special fixed penalty
        if curr_grade.timed_out:
            components["regression"] = -TIMEOUT_PENALTY
        
        # ====================================================================
        # FINAL REWARD COMPUTATION
        # ====================================================================
        # Combine all components: sum positives, subtract negatives, clamp to [-1, 1]
        total = (
            components["progress"]
            + components["syntax"]
            + components["test"]
            + components["quality"]
            - components["stagnation"]
            - components["regression"]
        )
        
        # Clamp to [-1.0, +1.0] for bounded, interpretable rewards
        components["total"] = max(-1.0, min(1.0, round(total, 6)))
        
        return components

    def _finalize_episode(self, auto_submit: bool, grade: Optional[TaskGrade] = None) -> None:
        """Mark episode as done and set final score."""
        if grade is None:
            if self._task is None:
                return
            grade = grade_task(self._state.current_code, self._task, include_hidden=True)
        
        self._state.score = grade.score
        self._done = True
        self._state.done = True

    def _format_test_results(self, grade: TaskGrade) -> str:
        """Format test results for display."""
        if grade.tests_total == 0:
            return "No tests available."
        if grade.timed_out:
            return "Test execution timed out."
        return f"Tests: {grade.tests_passed}/{grade.tests_total} passing"

    def _append_history(self, action_type: str, status: str, reward: float) -> None:
        """Append action to history."""
        entry = HistoryEntry(
            step=self._state.step_count,
            action_type=action_type,
            status=status,
            reward=reward,
        )
        self._state.history.append(entry)

    def _log_debug_step(self, reward: RewardDetails) -> None:
        """Log the scalar reward signal in a compact RL-friendly format."""
        print(
            f"""
Step Debug:
Prev Score: {reward.prev_score}
Curr Score: {reward.curr_score}
Reward: {reward.value}
Progress: {reward.curr_score - reward.prev_score}
"""
        )


# Backwards-compatible aliases used elsewhere in the repo.
PythonEnvironment = PythonCodeReviewEnvironment
CodeReviewEnvironment = PythonCodeReviewEnvironment

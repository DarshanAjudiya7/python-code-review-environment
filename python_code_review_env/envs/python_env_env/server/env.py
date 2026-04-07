"""Canonical OpenEnv environment implementation."""

from __future__ import annotations

from typing import List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from python_code_review_env.envs.python_env_env.graders.syntax import grade_task
from python_code_review_env.envs.python_env_env.models import (
    HealthResponse,
    HistoryEntry,
    PythonCodeReviewAction,
    PythonCodeReviewObservation,
    PythonCodeReviewState,
    RewardDetails,
    TaskGrade,
)
from python_code_review_env.envs.python_env_env.tasks import (
    TaskSpec,
    get_task,
    list_task_summaries,
    task_ids,
)


SYNTAX_FIX_REWARD = 0.2
TEST_PROGRESS_REWARD = 0.3
FULL_CORRECTNESS_REWARD = 0.5
QUALITY_BONUS_SCALE = 0.1
INVALID_ACTION_PENALTY = 0.1
TIMEOUT_PENALTY = 0.2


class PythonCodeReviewEnvironment(
    Environment[PythonCodeReviewAction, PythonCodeReviewObservation, PythonCodeReviewState]
):
    """Deterministic environment for Python code review workflows."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self._task_order = list(task_ids())
        self._task_cursor = -1
        self._task: Optional[TaskSpec] = None
        self._state = PythonCodeReviewState(episode_id=str(uuid4()))
        self._done = False
        self._verbose = verbose
        self._last_status = "Call reset() to start."
        self._last_reward = RewardDetails(value=0.0, reason="Environment initialized.")
        self._previous_score = 0.0
        self._best_visible_test_fraction = 0.0
        self._best_quality_score = 0.0
        self._compile_fixed_awarded = False
        self._full_correctness_awarded = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **_: object,
    ) -> PythonCodeReviewObservation:
        del seed
        if task_id:
            self._task = get_task(task_id)
            self._task_cursor = self._task_order.index(task_id)
        else:
            self._task_cursor = (self._task_cursor + 1) % len(self._task_order)
            self._task = get_task(self._task_order[self._task_cursor])

        self._done = False
        self._previous_score = 0.0
        self._best_visible_test_fraction = 0.0
        self._best_quality_score = 0.0
        self._compile_fixed_awarded = False
        self._full_correctness_awarded = False
        self._last_status = "Inspect the code, edit it, run tests, then submit."
        self._last_reward = RewardDetails(value=0.0, reason="Episode reset.")

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
        return self._build_observation()

    def step(
        self,
        action: PythonCodeReviewAction,
        timeout_s: Optional[float] = None,
        **_: object,
    ) -> PythonCodeReviewObservation:
        del timeout_s
        if self._task is None:
            return self.reset()

        if self._done:
            self._last_reward = self._invalid_reward("Episode already completed.")
            self._last_status = "Episode already completed. Call reset() to continue."
            return self._build_observation()

        self._state.step_count += 1
        if action.action_type == "analyze_code":
            reward, status = self._handle_analyze()
        elif action.action_type == "edit_code":
            reward, status = self._handle_edit(action)
        elif action.action_type == "run_tests":
            reward, status = self._handle_run_tests()
        elif action.action_type == "submit_solution":
            reward, status = self._handle_submit()
        else:
            reward = self._invalid_reward(f"Unsupported action_type: {action.action_type}")
            status = f"Unsupported action_type: {action.action_type}"

        self._last_reward = reward
        self._last_status = status
        self._state.attempts_remaining = max(self._task.max_steps - self._state.step_count, 0)
        self._state.done = self._done

        if self._state.attempts_remaining == 0 and not self._done:
            self._finalize_episode()

        return self._build_observation()

    @property
    def state(self) -> PythonCodeReviewState:
        return self._state.model_copy(deep=True)

    def state_snapshot(self) -> PythonCodeReviewState:
        return self.state

    def list_task_summaries(self) -> List[object]:
        return list_task_summaries()

    def get_task(self, task_id: str) -> object:
        return get_task(task_id).to_descriptor()

    def health(self) -> HealthResponse:
        return HealthResponse(task_count=len(self._task_order))

    def grade_task_submission(self, task_id: str, code: str) -> TaskGrade:
        return grade_task(code, get_task(task_id), include_hidden=True)

    def _build_observation(self) -> PythonCodeReviewObservation:
        return PythonCodeReviewObservation(
            task_id=self._state.task_id or "",
            title=self._task.title if self._task else "",
            difficulty=self._state.difficulty or "easy",
            task_kind=self._state.task_kind,
            task_description=self._task.task_description if self._task else "",
            current_code=self._state.current_code,
            errors=self._state.errors,
            test_results=self._state.test_results,
            visible_tests=list(self._task.visible_tests) if self._task else [],
            history=self._state.history,
            attempts_remaining=self._state.attempts_remaining,
            last_action_status=self._last_status,
            score=self._state.score,
            reward_details=self._last_reward,
            reward=self._last_reward.value,
            done=self._state.done,
            metadata={"task_kind": self._state.task_kind},
        )

    def _handle_analyze(self) -> tuple[RewardDetails, str]:
        grade = self._grade_current_code(include_hidden=False)
        self._apply_grade_feedback(grade, include_hidden=False)
        status = self._build_status(grade, include_hidden=False)
        reward = self._reward_from_grade(
            grade=grade,
            previous_score=self._previous_score,
            code_changed=False,
            allow_progress_rewards=False,
            allow_completion_bonus=False,
            reason=status,
        )
        self._append_history("analyze_code", status, reward.value)
        self._previous_score = self._state.score
        return reward, status

    def _handle_edit(self, action: PythonCodeReviewAction) -> tuple[RewardDetails, str]:
        code = (action.code or "").strip()
        if not code:
            reward = self._invalid_reward("Edit action requires non-empty code.")
            status = "Invalid action: edit_code requires a non-empty code payload."
            self._append_history("edit_code", status, reward.value)
            return reward, status

        previous_score = self._state.score
        self._state.current_code = code
        grade = self._grade_current_code(include_hidden=False)
        self._apply_grade_feedback(grade, include_hidden=False)
        status = self._build_status(grade, include_hidden=False)
        reward = self._reward_from_grade(
            grade=grade,
            previous_score=previous_score,
            code_changed=True,
            allow_progress_rewards=True,
            allow_completion_bonus=False,
            reason=status,
        )
        self._append_history("edit_code", status, reward.value)
        self._previous_score = self._state.score
        return reward, status

    def _handle_run_tests(self) -> tuple[RewardDetails, str]:
        grade = self._grade_current_code(include_hidden=False)
        self._apply_grade_feedback(grade, include_hidden=False)
        status = self._build_status(grade, include_hidden=False)
        reward = self._reward_from_grade(
            grade=grade,
            previous_score=self._previous_score,
            code_changed=False,
            allow_progress_rewards=False,
            allow_completion_bonus=False,
            reason=status,
        )
        self._append_history("run_tests", status, reward.value)
        self._previous_score = self._state.score
        return reward, status

    def _handle_submit(self) -> tuple[RewardDetails, str]:
        previous_score = self._state.score
        grade = self._grade_current_code(include_hidden=True)
        self._apply_grade_feedback(grade, include_hidden=True)
        self._finalize_episode(grade)
        status = self._build_status(grade, include_hidden=True)
        reward = self._reward_from_grade(
            grade=grade,
            previous_score=previous_score,
            code_changed=False,
            allow_progress_rewards=False,
            allow_completion_bonus=True,
            reason=f"Solution submitted. {status}",
        )
        self._append_history("submit_solution", reward.reason, reward.value)
        self._previous_score = self._state.score
        return reward, reward.reason

    def _grade_current_code(self, include_hidden: bool) -> TaskGrade:
        assert self._task is not None
        return grade_task(self._state.current_code, self._task, include_hidden=include_hidden)

    def _apply_grade_feedback(self, grade: TaskGrade, include_hidden: bool) -> None:
        self._state.score = grade.score
        self._state.errors = grade.details.get("compile_error", "")
        self._state.test_results = self._format_test_results(grade, include_hidden=include_hidden)

    def _build_status(self, grade: TaskGrade, include_hidden: bool) -> str:
        if grade.details.get("compile_error"):
            return f"Compilation failed: {grade.details['compile_error']}"
        if grade.timed_out:
            return "Execution timed out while grading the current code."
        scope = "all checks" if include_hidden else "visible checks"
        if self._task and self._task.task_kind == "syntax_fix":
            return "Code compiles successfully." if grade.score == 1.0 else "Syntax errors remain."
        return f"{scope}: {grade.tests_passed}/{grade.tests_total} passing."

    def _reward_from_grade(
        self,
        *,
        grade: TaskGrade,
        previous_score: float,
        code_changed: bool,
        allow_progress_rewards: bool,
        allow_completion_bonus: bool,
        reason: str,
    ) -> RewardDetails:
        reward = 0.0
        syntax_reward = 0.0
        test_reward = 0.0
        quality_bonus = 0.0
        correctness_bonus = 0.0
        timeout_penalty = 0.0

        if grade.timed_out:
            timeout_penalty = TIMEOUT_PENALTY
            reward -= TIMEOUT_PENALTY

        if allow_progress_rewards and grade.syntax_score >= 1.0 and not self._compile_fixed_awarded:
            syntax_reward = SYNTAX_FIX_REWARD
            reward += syntax_reward
            self._compile_fixed_awarded = True

        if allow_progress_rewards and grade.tests_total > 0:
            current_fraction = grade.tests_passed / grade.tests_total
            test_delta = max(0.0, current_fraction - self._best_visible_test_fraction)
            if test_delta > 0:
                test_reward = round(TEST_PROGRESS_REWARD * test_delta, 6)
                reward += test_reward
                self._best_visible_test_fraction = current_fraction

        if allow_progress_rewards and grade.quality_score > self._best_quality_score:
            quality_bonus = round(
                min(QUALITY_BONUS_SCALE, QUALITY_BONUS_SCALE * (grade.quality_score - self._best_quality_score)),
                6,
            )
            reward += quality_bonus
            self._best_quality_score = grade.quality_score

        if allow_completion_bonus and grade.score >= 1.0 and not self._full_correctness_awarded:
            correctness_bonus = FULL_CORRECTNESS_REWARD
            reward += correctness_bonus
            self._full_correctness_awarded = True

        reward = max(-1.0, min(1.0, round(reward, 6)))
        return RewardDetails(
            value=reward,
            syntax_reward=syntax_reward,
            test_reward=test_reward,
            quality_bonus=quality_bonus,
            correctness_bonus=correctness_bonus,
            progress_delta=max(0.0, round(grade.score - previous_score, 6)),
            timeout_penalty=timeout_penalty,
            reason=reason,
            prev_score=round(previous_score, 6),
            curr_score=round(self._state.score, 6),
            code_changed=code_changed,
        )

    def _invalid_reward(self, reason: str) -> RewardDetails:
        return RewardDetails(
            value=-INVALID_ACTION_PENALTY,
            invalid_action_penalty=INVALID_ACTION_PENALTY,
            reason=reason,
            prev_score=round(self._state.score, 6),
            curr_score=round(self._state.score, 6),
            code_changed=False,
        )

    def _finalize_episode(self, grade: Optional[TaskGrade] = None) -> None:
        if grade is None:
            grade = self._grade_current_code(include_hidden=True)
            self._apply_grade_feedback(grade, include_hidden=True)
        self._done = True
        self._state.done = True

    def _format_test_results(self, grade: TaskGrade, include_hidden: bool) -> str:
        if grade.details.get("compile_error"):
            return "Compilation failed. Fix syntax errors first."
        if grade.timed_out:
            return "Execution timed out during grading."
        if self._task and self._task.task_kind == "syntax_fix":
            return "Compilation successful." if grade.score == 1.0 else "Syntax errors remain."
        scope = "all checks" if include_hidden else "visible checks"
        return f"{scope}: {grade.tests_passed}/{grade.tests_total} passing."

    def _append_history(self, action_type: str, status: str, reward: float) -> None:
        self._state.history.append(
            HistoryEntry(
                step=self._state.step_count,
                action_type=action_type,  # type: ignore[arg-type]
                status=status,
                reward=reward,
            )
        )


PythonEnvironment = PythonCodeReviewEnvironment
CodeReviewEnvironment = PythonCodeReviewEnvironment

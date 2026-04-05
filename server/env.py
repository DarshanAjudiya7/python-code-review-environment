"""Core OpenEnv environment for Python code review tasks."""

from __future__ import annotations

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
    TaskDescriptor,
    TaskGrade,
    reward_metadata,
)
from tasks import TaskSpec, get_task, list_task_descriptors, list_task_summaries, task_ids


INVALID_ACTION_PENALTY = -0.1
TIMEOUT_PENALTY = 0.2
QUALITY_BONUS_SCALE = -0.1


class PythonCodeReviewEnvironment(
    Environment[PythonCodeReviewAction, PythonCodeReviewObservation, PythonCodeReviewState]
):
    """Production-style environment for reviewing and fixing Python code."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._task_order = list(task_ids())
        self._task_cursor = -1
        self._task: Optional[TaskSpec] = None
        self._state = PythonCodeReviewState()
        self._done = False
        self._last_status = "Call reset() to start."
        self._last_reward = RewardDetails(reason="Environment initialized.")
        self._best_visible_test_fraction = 0.0
        self._best_quality_score = 0.0
        self._full_correctness_awarded = False
        self._syntax_reward_awarded = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **_: object,
    ) -> PythonCodeReviewObservation:
        """Reset the environment to the next deterministic task."""

        del seed
        if task_id:
            self._task = get_task(task_id)
            self._task_cursor = self._task_order.index(task_id)
        else:
            self._task_cursor = (self._task_cursor + 1) % len(self._task_order)
            self._task = get_task(self._task_order[self._task_cursor])

        self._done = False
        self._best_visible_test_fraction = 0.0
        self._best_quality_score = 0.0
        self._full_correctness_awarded = False
        self._syntax_reward_awarded = False
        self._last_status = "Inspect the code, edit it, run tests, then submit."
        self._last_reward = RewardDetails(reason="Episode reset.")
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
        reward = RewardDetails(reason="Action processed.")

        if action.action_type == "analyze_code":
            reward, status = self._handle_analyze()
        elif action.action_type == "edit_code":
            reward, status = self._handle_edit(action)
        elif action.action_type == "run_tests":
            reward, status = self._handle_run_tests()
        elif action.action_type == "submit_solution":
            reward, status = self._handle_submit()
        else:  # pragma: no cover
            reward = RewardDetails(
                value=-INVALID_ACTION_PENALTY,
                invalid_action_penalty=INVALID_ACTION_PENALTY,
                reason=f"Unsupported action_type {action.action_type}.",
            )
            status = f"Unsupported action_type {action.action_type}."

        self._last_reward = reward
        self._last_status = status
        self._state.attempts_remaining = max(self._task.max_steps - self._state.step_count, 0)

        if self._state.attempts_remaining == 0 and not self._done:
            self._finalize_episode(auto_submit=True)

        self._state.done = self._done
        return self._build_observation()

    @property
    def state(self) -> PythonCodeReviewState:
        """Return the current environment state."""

        return self._state.model_copy(deep=True)

    def list_tasks(self) -> List[TaskDescriptor]:
        """Return all task descriptors."""

        return list_task_descriptors()

    def list_task_summaries(self) -> List[TaskDescriptor]:
        """Return public task metadata."""

        return list_task_summaries()

    def get_task(self, task_id: str) -> TaskDescriptor:
        """Return a single task descriptor."""

        return get_task(task_id).to_descriptor()

    def health(self) -> HealthResponse:
        """Return a simple health model."""

        return HealthResponse(task_count=len(self._task_order))

    def grade_task_submission(self, task_id: str, code: str) -> TaskGrade:
        """Expose deterministic grading outside of an active episode."""

        return grade_task(code, get_task(task_id), include_hidden=True)

    def _handle_analyze(self) -> tuple[RewardDetails, str]:
        grade = grade_task(self._state.current_code, self._task, include_hidden=False)
        error = grade.details.get("compile_error", "")
        if error:
            self._state.errors = f"Syntax analysis failed: {error}"
            self._state.test_results = "Tests skipped because the code does not compile."
            summary = self._state.errors
        else:
            self._state.errors = ""
            if self._task.task_kind == "syntax_fix":
                self._state.test_results = "Compilation succeeds."
            else:
                visible_total = len(self._task.visible_tests)
                visible_passed = min(grade.tests_passed, visible_total)
                self._state.test_results = (
                    f"Visible checks preview: {visible_passed}/{visible_total} passing."
                )
            summary = "Static analysis refreshed."

        reward = RewardDetails(value=0.0, reason=summary)
        self._append_history("analyze_code", summary, reward.value)
        self._sync_score(include_hidden=False)
        return reward, summary

    def _handle_edit(self, action: PythonCodeReviewAction) -> tuple[RewardDetails, str]:
        code = (action.code or "").strip("\n")
        if not code:
            reward = RewardDetails(
                value=-INVALID_ACTION_PENALTY,
                invalid_action_penalty=INVALID_ACTION_PENALTY,
                reason="edit_code requires non-empty code.",
            )
            status = "Invalid action: edit_code requires code."
            self._append_history("edit_code", status, reward.value)
            return reward, status

        previous_visible = grade_task(self._state.current_code, self._task, include_hidden=False)
        new_visible = grade_task(code, self._task, include_hidden=False)
        self._state.current_code = code
        self._state.errors = new_visible.details.get("compile_error", "")
        self._state.test_results = self._format_test_results(new_visible, include_hidden=False)

        syntax_reward = 0.0
        if previous_visible.syntax_score < 1.0 and new_visible.syntax_score == 1.0:
            syntax_reward = 0.2

        quality_bonus = 0.0
        quality_delta = max(new_visible.quality_score - self._best_quality_score, 0.0)
        if quality_delta > 0:
            quality_bonus = round(min(quality_delta * QUALITY_BONUS_SCALE, 0.1), 6)
            self._best_quality_score = max(self._best_quality_score, new_visible.quality_score)

        reward_value = syntax_reward + quality_bonus
        status = "Code updated."
        if self._state.errors:
            status = f"Code updated, but syntax issues remain: {self._state.errors}"
        elif new_visible.tests_total:
            status = self._state.test_results

        reward = RewardDetails(
            value=reward_value,
            syntax_reward=syntax_reward,
            quality_bonus=quality_bonus,
            reason=status,
        )
        self._append_history("edit_code", status, reward.value)
        self._sync_score(include_hidden=False)
        return reward, status

    def _handle_run_tests(self) -> tuple[RewardDetails, str]:
        grade = grade_task(self._state.current_code, self._task, include_hidden=False)
        self._state.errors = grade.details.get("compile_error", "")
        self._state.test_results = self._format_test_results(grade, include_hidden=False)
        reward = self._reward_from_grade(grade, include_hidden=False)
        status = self._state.test_results if not self._state.errors else self._state.errors
        self._append_history("run_tests", status, reward.value)
        self._sync_score(include_hidden=False)
        return reward, status

    def _handle_submit(self) -> tuple[RewardDetails, str]:
        grade = grade_task(self._state.current_code, self._task, include_hidden=True)
        self._state.errors = grade.details.get("compile_error", "")
        self._state.test_results = self._format_test_results(grade, include_hidden=True)
        reward = self._reward_from_grade(grade, include_hidden=True)
        self._finalize_episode(auto_submit=False, grade=grade)
        status = f"Solution submitted. Final score: {grade.score:.2f}."
        self._append_history("submit_solution", status, reward.value)
        return reward, status

    def _finalize_episode(self, auto_submit: bool, grade: Optional[TaskGrade] = None) -> None:
        if grade is None:
            grade = grade_task(self._state.current_code, self._task, include_hidden=True)
            self._state.errors = grade.details.get("compile_error", "")
            self._state.test_results = self._format_test_results(grade, include_hidden=True)
        self._state.score = grade.score
        self._done = True
        self._state.done = True
        if auto_submit:
            self._last_status = f"Step budget exhausted. Final score: {grade.score:.2f}."
            self._last_reward = self._reward_from_grade(grade, include_hidden=True)

    def _reward_from_grade(self, grade: TaskGrade, include_hidden: bool) -> RewardDetails:
        syntax_reward = 0.0
        if grade.syntax_score == 1.0 and not self._state.errors and not self._syntax_reward_awarded:
            syntax_reward = 0.2
            self._syntax_reward_awarded = True
        test_fraction = grade.tests_passed / grade.tests_total if grade.tests_total else grade.score
        test_gain = max(test_fraction - self._best_visible_test_fraction, 0.0)
        test_reward = 0.3 * test_gain
        if test_gain > 0:
            self._best_visible_test_fraction = test_fraction

        quality_bonus = 0.0
        quality_delta = max(grade.quality_score - self._best_quality_score, 0.0)
        if quality_delta > 0:
            quality_bonus = min(quality_delta * QUALITY_BONUS_SCALE, 0.1)
            self._best_quality_score = grade.quality_score

        correctness_bonus = 0.0
        if include_hidden and grade.score >= 0.999999 and not self._full_correctness_awarded:
            correctness_bonus = 0.5
            self._full_correctness_awarded = True

        timeout_penalty = TIMEOUT_PENALTY if grade.timed_out else 0.0
        reward_value = round(
            syntax_reward + test_reward + quality_bonus + correctness_bonus - timeout_penalty,
            6,
        )
        return RewardDetails(
            value=reward_value,
            syntax_reward=syntax_reward,
            test_reward=round(test_reward, 6),
            correctness_bonus=correctness_bonus,
            quality_bonus=round(quality_bonus, 6),
            timeout_penalty=timeout_penalty,
            reason=self._format_test_results(grade, include_hidden=include_hidden),
        )

    def _format_test_results(self, grade: TaskGrade, include_hidden: bool) -> str:
        if grade.details.get("compile_error"):
            return f"Compilation failed: {grade.details['compile_error']}"
        scope = "full grader" if include_hidden else "visible checks"
        parts = [f"{scope}: score={grade.score:.2f}"]
        if grade.tests_total:
            parts.append(f"tests={grade.tests_passed}/{grade.tests_total}")
        if grade.runtime_score:
            parts.append(f"runtime={grade.runtime_score:.2f}")
        if grade.quality_score:
            parts.append(f"quality={grade.quality_score:.2f}")
        if grade.style_score:
            parts.append(f"style={grade.style_score:.2f}")
        if grade.timed_out:
            parts.append("timed_out=True")
        return " | ".join(parts)

    def _sync_score(self, include_hidden: bool) -> None:
        grade = grade_task(self._state.current_code, self._task, include_hidden=include_hidden)
        self._state.score = grade.score

    def _append_history(self, action_type: str, summary: str, reward: float) -> None:
        self._state.history.append(
            HistoryEntry(
                step=self._state.step_count,
                action_type=action_type,  # type: ignore[arg-type]
                summary=summary,
                reward=reward,
            )
        )

    def _build_observation(self) -> PythonCodeReviewObservation:
        return PythonCodeReviewObservation(
            task_id=self._task.task_id,
            title=self._task.title,
            difficulty=self._task.difficulty,
            task_kind=self._task.task_kind,
            task_description=self._task.task_description,
            current_code=self._state.current_code,
            errors=self._state.errors,
            test_results=self._state.test_results,
            history=list(self._state.history),
            attempts_remaining=self._state.attempts_remaining,
            last_action_status=self._last_status,
            score=self._state.score,
            reward_details=self._last_reward,
            done=self._done,
            reward=self._last_reward.value,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "task_kind": self._task.task_kind,
                "visible_tests": list(self._task.visible_tests),
                "info": {
                    "reward": reward_metadata(self._last_reward),
                },
            },
        )


# Backwards-compatible aliases used elsewhere in the repo.
PythonEnvironment = PythonCodeReviewEnvironment
CodeReviewEnvironment = PythonCodeReviewEnvironment

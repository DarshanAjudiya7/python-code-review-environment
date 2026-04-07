"""Safe OpenEnv environment for deterministic Python code repair tasks."""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

try:
    from compat import Environment
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
    from tasks import TaskSpec, get_task as load_task, list_task_summaries, task_ids
except Exception:
    from .compat import Environment
    from .graders import grade_task
    from .models import (
        HealthResponse,
        HistoryEntry,
        PythonCodeReviewAction,
        PythonCodeReviewObservation,
        PythonCodeReviewState,
        RewardDetails,
        TaskGrade,
    )
    from .tasks import TaskSpec, get_task as load_task, list_task_summaries, task_ids


INVALID_ACTION_PENALTY = 0.10
NO_PROGRESS_PENALTY = 0.08
REPEATED_ACTION_PENALTY = 0.05
BASE_STEP_PENALTY = 0.02
ANALYZE_STEP_PENALTY = 0.01
SUBMIT_COMPLETION_BONUS = 0.30
TIMEOUT_PENALTY = 0.12
VALID_ACTIONS = {"analyze_code", "edit_code", "run_tests", "submit_solution"}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a scalar to a bounded numeric interval."""
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low


def _safe_text(value: Any, default: str = "") -> str:
    """Convert values into short stable strings."""
    try:
        text = str(value)
    except Exception:
        return default
    text = " ".join(text.split())
    return text[:240] if text else default


class PythonCodeReviewEnvironment(
    Environment[PythonCodeReviewAction, PythonCodeReviewObservation, PythonCodeReviewState]
):
    """Deterministic, bounded, evaluator-safe environment for code repair tasks."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self._verbose = bool(verbose)
        self._task_order = self._safe_task_order()
        self._task_cursor = -1
        self._task: Optional[TaskSpec] = None
        self._state = PythonCodeReviewState(episode_id=str(uuid4()))
        self._done = False
        self._last_status = "Call reset() to start."
        self._last_reward = RewardDetails(value=0.0, reason="Environment initialized.")
        self._metrics = self._blank_metrics()
        self._last_action_type = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **_: object,
    ) -> PythonCodeReviewObservation:
        """Reset the environment for a deterministic task and return an observation."""
        del seed
        try:
            self._reset_rubric()
        except Exception:
            pass

        task = self._select_task(task_id)
        self._task = task
        self._done = False
        self._metrics = self._blank_metrics()
        self._last_action_type = ""
        self._last_status = "Inspect the code, run checks, edit the code, then submit."
        self._last_reward = RewardDetails(
            value=0.0,
            reason="Episode reset.",
            prev_score=0.0,
            curr_score=0.0,
        )
        self._state = PythonCodeReviewState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task.task_id,
            difficulty=task.difficulty,
            task_kind=task.task_kind,
            attempts_remaining=max(int(task.max_steps), 1),
            current_code=task.starter_code,
            errors="",
            test_results="No checks run yet.",
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
        """Execute one safe environment step and always return a valid observation."""
        del timeout_s
        try:
            if self._task is None:
                return self.reset()

            if self._done:
                self._last_status = "Episode already completed. Call reset() to continue."
                self._last_reward = RewardDetails(
                    value=-INVALID_ACTION_PENALTY,
                    invalid_action_penalty=INVALID_ACTION_PENALTY,
                    reason="Episode already completed.",
                    prev_score=self._metrics["score"],
                    curr_score=self._metrics["score"],
                    code_changed=False,
                )
                return self._build_observation()

            self._state.step_count += 1
            action_type = _safe_text(getattr(action, "action_type", "analyze_code"), "analyze_code")
            code = getattr(action, "code", None)

            if action_type == "analyze_code":
                self._handle_scored_action(action_type=action_type, candidate_code=self._state.current_code, include_hidden=False)
            elif action_type == "run_tests":
                self._handle_scored_action(action_type=action_type, candidate_code=self._state.current_code, include_hidden=False)
            elif action_type == "edit_code":
                self._handle_edit(code)
            elif action_type == "submit_solution":
                self._handle_scored_action(action_type=action_type, candidate_code=self._state.current_code, include_hidden=True)
                self._done = True
            else:
                self._apply_invalid_action(f"Unsupported action_type '{action_type}'.")

            self._state.attempts_remaining = max(self._task.max_steps - self._state.step_count, 0)
            if self._state.attempts_remaining == 0 and not self._done:
                self._auto_submit()

            self._state.done = self._done
            return self._build_observation()
        except Exception as exc:
            self._apply_invalid_action(f"Step failure handled: {_safe_text(exc, 'unknown_error')}")
            self._state.done = self._done
            return self._build_observation()

    @property
    def state(self) -> PythonCodeReviewState:
        """Return a deep copy of the current environment state."""
        try:
            return self._state.model_copy(deep=True)
        except Exception:
            return PythonCodeReviewState(episode_id=str(uuid4()))

    def list_task_summaries(self) -> list[object]:
        """Return public task summaries."""
        try:
            return list_task_summaries()
        except Exception:
            return []

    def get_task(self, task_id: str) -> object:
        """Return a single public task descriptor."""
        return self._select_task(task_id).to_descriptor()

    def health(self) -> HealthResponse:
        """Return a simple health response."""
        return HealthResponse(task_count=len(self._task_order))

    def grade_task_submission(self, task_id: str, code: str) -> TaskGrade:
        """Grade a task submission outside an episode without raising."""
        try:
            task = self._select_task(task_id)
            return self._safe_grade(task=task, candidate_code=code, include_hidden=True)
        except Exception as exc:
            return TaskGrade(score=0.0, details={"error": _safe_text(exc, "grading_failed")})

    def run_tests(self, code: str, include_hidden: bool = False) -> tuple[float, dict[str, int], TaskGrade]:
        """Run deterministic grading and return score plus test summary."""
        task = self._task or self._select_task(None)
        grade = self._safe_grade(task=task, candidate_code=code, include_hidden=include_hidden)
        return (
            _clamp(grade.score),
            {"passed": int(grade.tests_passed), "total": int(grade.tests_total)},
            grade,
        )

    def apply_action(self, action: PythonCodeReviewAction) -> str:
        """Return the candidate code implied by the action."""
        if getattr(action, "action_type", "") == "edit_code":
            code = getattr(action, "code", None)
            return str(code) if code is not None else self._state.current_code
        return self._state.current_code

    def compute_reward(
        self,
        action_type: str,
        previous_metrics: dict[str, float],
        current_metrics: dict[str, float],
        grade: TaskGrade,
        code_changed: bool,
        invalid_action: bool = False,
    ) -> RewardDetails:
        """Compute a bounded dynamic reward with progress and efficiency shaping."""
        prev_score = _clamp(previous_metrics.get("score", 0.0))
        curr_score = _clamp(current_metrics.get("score", 0.0))
        score_delta = curr_score - prev_score
        test_delta = current_metrics.get("test_fraction", 0.0) - previous_metrics.get("test_fraction", 0.0)
        syntax_delta = current_metrics.get("syntax_score", 0.0) - previous_metrics.get("syntax_score", 0.0)
        quality_delta = current_metrics.get("quality_score", 0.0) - previous_metrics.get("quality_score", 0.0)

        step_penalty = BASE_STEP_PENALTY + (ANALYZE_STEP_PENALTY if action_type == "analyze_code" else 0.0)
        repeated_penalty = REPEATED_ACTION_PENALTY if action_type == self._last_action_type else 0.0
        no_progress = (
            score_delta <= 1e-9
            and test_delta <= 1e-9
            and syntax_delta <= 1e-9
            and quality_delta <= 1e-9
            and not code_changed
        )
        stagnation_penalty = NO_PROGRESS_PENALTY if no_progress and not invalid_action else 0.0
        regression_penalty = max(-score_delta, 0.0) * 0.6 + repeated_penalty + step_penalty
        invalid_penalty = INVALID_ACTION_PENALTY if invalid_action else 0.0
        timeout_penalty = TIMEOUT_PENALTY if bool(grade.timed_out) else 0.0

        progress_reward = max(score_delta, 0.0) * 0.7
        syntax_reward = max(syntax_delta, 0.0) * 0.5
        test_reward = max(test_delta, 0.0) * 1.0
        quality_bonus = max(quality_delta, 0.0) * 0.2
        correctness_bonus = SUBMIT_COMPLETION_BONUS if action_type == "submit_solution" and curr_score >= 0.999 else 0.0

        reward_value = (
            progress_reward
            + syntax_reward
            + test_reward
            + quality_bonus
            + correctness_bonus
            - stagnation_penalty
            - regression_penalty
            - invalid_penalty
            - timeout_penalty
        )
        reward_value = max(-1.0, min(1.0, round(reward_value, 6)))
        return RewardDetails(
            value=reward_value,
            syntax_reward=round(syntax_reward, 6),
            test_reward=round(test_reward, 6),
            quality_bonus=round(quality_bonus, 6),
            correctness_bonus=round(correctness_bonus, 6),
            progress_delta=round(progress_reward, 6),
            stagnation_penalty=round(stagnation_penalty, 6),
            regression_penalty=round(regression_penalty, 6),
            invalid_action_penalty=round(invalid_penalty, 6),
            timeout_penalty=round(timeout_penalty, 6),
            reason=f"{action_type} reward computed safely",
            prev_score=round(prev_score, 6),
            curr_score=round(curr_score, 6),
            code_changed=bool(code_changed),
        )

    def _safe_task_order(self) -> list[str]:
        """Load deterministic task ids with a hard fallback."""
        try:
            loaded = list(task_ids())
            if loaded:
                return [str(task_id) for task_id in loaded]
        except Exception:
            pass
        return ["syntax-fix-easy", "bug-fix-medium", "optimization-hard"]

    def _blank_metrics(self) -> dict[str, float]:
        """Return an empty metric snapshot."""
        return {
            "score": 0.0,
            "test_fraction": 0.0,
            "syntax_score": 0.0,
            "quality_score": 0.0,
        }

    def _select_task(self, task_id: Optional[str]) -> TaskSpec:
        """Select the requested task or advance deterministically."""
        try:
            if task_id:
                task = load_task(task_id)
                if task.task_id in self._task_order:
                    self._task_cursor = self._task_order.index(task.task_id)
                return task
        except Exception:
            pass

        try:
            self._task_cursor = (self._task_cursor + 1) % len(self._task_order)
            return load_task(self._task_order[self._task_cursor])
        except Exception:
            return load_task("syntax-fix-easy")

    def _safe_grade(self, task: TaskSpec, candidate_code: str, include_hidden: bool) -> TaskGrade:
        """Run grading without allowing exceptions to escape."""
        try:
            return grade_task(candidate_code, task, include_hidden=include_hidden)
        except Exception as exc:
            return TaskGrade(
                score=0.0,
                syntax_score=0.0,
                tests_passed=0,
                tests_total=max(len(task.visible_tests), 1),
                details={"compile_error": "", "error": _safe_text(exc, "grading_failed")},
            )

    def _metrics_from_grade(self, grade: TaskGrade) -> dict[str, float]:
        """Derive normalized reward metrics from a grading result."""
        tests_total = max(int(grade.tests_total), 0)
        tests_passed = max(int(grade.tests_passed), 0)
        test_fraction = (tests_passed / tests_total) if tests_total else _clamp(grade.syntax_score)
        return {
            "score": _clamp(grade.score),
            "test_fraction": _clamp(test_fraction),
            "syntax_score": _clamp(grade.syntax_score),
            "quality_score": _clamp(grade.quality_score),
        }

    def _format_test_results(self, grade: TaskGrade, include_hidden: bool) -> str:
        """Format test execution results for the observation."""
        compile_error = _safe_text(grade.details.get("compile_error", ""), "")
        scope = "all checks" if include_hidden else "visible checks"
        if compile_error:
            return f"{scope}: compile error: {compile_error}"
        if grade.timed_out:
            return f"{scope}: execution timed out"
        if self._task and self._task.task_kind == "syntax_fix":
            return "visible checks: code compiles successfully"
        return f"{scope}: {int(grade.tests_passed)}/{int(grade.tests_total)} passing"

    def _build_status(self, action_type: str, grade: TaskGrade) -> str:
        """Build a human-readable status message."""
        if action_type == "submit_solution":
            return f"Solution submitted. Final score: {_clamp(grade.score):.3f}"
        if action_type == "edit_code":
            if grade.details.get("compile_error"):
                return "Code updated, but syntax issues remain."
            return "Code updated and evaluated."
        if action_type == "run_tests":
            return "Test run completed."
        if action_type == "analyze_code":
            return "Analysis completed."
        return "Action handled safely."

    def _apply_grade_to_state(self, grade: TaskGrade, include_hidden: bool) -> None:
        """Update environment state from the latest grading result."""
        compile_error = _safe_text(grade.details.get("compile_error", ""), "")
        self._state.score = _clamp(grade.score)
        self._state.errors = compile_error
        self._state.test_results = self._format_test_results(grade, include_hidden=include_hidden)

    def _handle_scored_action(self, action_type: str, candidate_code: str, include_hidden: bool) -> None:
        """Grade code, update state, and compute reward for a valid action."""
        task = self._task or self._select_task(None)
        previous_metrics = dict(self._metrics)
        prior_code = self._state.current_code
        code_changed = candidate_code.strip() != prior_code.strip()
        if action_type == "edit_code":
            self._state.current_code = candidate_code
        grade = self._safe_grade(task=task, candidate_code=self._state.current_code, include_hidden=include_hidden)
        current_metrics = self._metrics_from_grade(grade)
        self._apply_grade_to_state(grade, include_hidden=include_hidden)
        self._last_reward = self.compute_reward(
            action_type=action_type,
            previous_metrics=previous_metrics,
            current_metrics=current_metrics,
            grade=grade,
            code_changed=code_changed,
            invalid_action=False,
        )
        self._last_status = self._build_status(action_type, grade)
        self._metrics = current_metrics
        self._last_action_type = action_type
        self._append_history(action_type, self._last_status, self._last_reward.value)

    def _handle_edit(self, code: Optional[str]) -> None:
        """Validate edit input and evaluate the new candidate code."""
        safe_code = (code or "").strip()
        if not safe_code:
            self._apply_invalid_action("edit_code requires code parameter.")
            return
        self._handle_scored_action(action_type="edit_code", candidate_code=safe_code, include_hidden=False)

    def _apply_invalid_action(self, reason: str) -> None:
        """Record an invalid action without crashing the episode."""
        previous_metrics = dict(self._metrics)
        grade = TaskGrade(score=previous_metrics["score"], syntax_score=previous_metrics["syntax_score"])
        self._last_reward = self.compute_reward(
            action_type="invalid",
            previous_metrics=previous_metrics,
            current_metrics=previous_metrics,
            grade=grade,
            code_changed=False,
            invalid_action=True,
        )
        self._last_status = reason
        self._append_history("analyze_code", reason, self._last_reward.value)

    def _auto_submit(self) -> None:
        """Finalize the episode when attempts are exhausted."""
        task = self._task or self._select_task(None)
        grade = self._safe_grade(task=task, candidate_code=self._state.current_code, include_hidden=True)
        self._apply_grade_to_state(grade, include_hidden=True)
        self._done = True
        self._state.done = True
        self._last_status = f"Auto-submitted. Final score: {_clamp(grade.score):.3f}"

    def _append_history(self, action_type: str, status: str, reward: float) -> None:
        """Append one action record to the episode history."""
        try:
            stable_action = action_type if action_type in VALID_ACTIONS else "analyze_code"
            self._state.history.append(
                HistoryEntry(
                    step=max(int(self._state.step_count), 0),
                    action_type=stable_action,
                    status=_safe_text(status, "handled"),
                    reward=float(reward),
                )
            )
        except Exception:
            pass

    def _build_observation(self) -> PythonCodeReviewObservation:
        """Build a valid observation from current state."""
        task = self._task
        try:
            return PythonCodeReviewObservation(
                task_id=self._state.task_id or "",
                title=task.title if task else "",
                difficulty=self._state.difficulty or "easy",
                task_kind=self._state.task_kind,
                task_description=task.task_description if task else "",
                current_code=self._state.current_code,
                errors=self._state.errors,
                test_results=self._state.test_results,
                visible_tests=list(task.visible_tests) if task else [],
                history=list(self._state.history),
                attempts_remaining=max(int(self._state.attempts_remaining), 0),
                last_action_status=self._last_status,
                score=_clamp(self._state.score),
                reward_details=self._last_reward,
                reward=self._last_reward.value,
                done=bool(self._state.done),
                metadata={
                    "prev_score": self._last_reward.prev_score,
                    "curr_score": self._last_reward.curr_score,
                },
            )
        except Exception as exc:
            return PythonCodeReviewObservation(
                task_id=self._state.task_id or "",
                title="",
                difficulty="easy",
                task_kind=None,
                task_description="",
                current_code=getattr(self._state, "current_code", ""),
                errors=_safe_text(exc, "observation_build_failed"),
                test_results="visible checks: unavailable",
                visible_tests=[],
                history=[],
                attempts_remaining=0,
                last_action_status="Observation fallback returned safely.",
                score=0.0,
                reward_details=RewardDetails(value=0.0, reason="Observation fallback."),
                reward=0.0,
                done=bool(getattr(self._state, "done", False)),
                metadata={},
            )


PythonEnvironment = PythonCodeReviewEnvironment
CodeReviewEnvironment = PythonCodeReviewEnvironment

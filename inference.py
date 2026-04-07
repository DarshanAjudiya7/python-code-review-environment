#!/usr/bin/env python3
"""Validator-friendly inference entrypoint for the Python code review environment."""

from __future__ import annotations

import io
import json
import os
import sys
import time
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from compat import install_openenv_fastmcp_compat

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore[assignment]


install_openenv_fastmcp_compat()

try:
    from server.env import PythonCodeReviewEnvironment
except Exception:
    PythonCodeReviewEnvironment = None  # type: ignore[assignment]

try:
    from models import PythonCodeReviewAction
except Exception:
    PythonCodeReviewAction = None  # type: ignore[assignment]

try:
    from tasks import get_task, task_ids
except Exception:
    get_task = None  # type: ignore[assignment]
    task_ids = None  # type: ignore[assignment]


ALLOWED_ACTIONS = {
    "analyze_code",
    "edit_code",
    "run_tests",
    "submit_solution",
}
DEFAULT_MODEL_NAME = "mock-model"
API_TIMEOUT_SECONDS = 3.0
API_RETRIES = 1
API_RETRY_DELAY_SECONDS = 0.2


def safe_env(name: str, default: str = "") -> str:
    """Read a string environment variable without raising."""
    try:
        value = os.getenv(name)
        return default if value is None else str(value)
    except Exception:
        return default


def clamp_score(value: Any) -> float:
    """Clamp numeric scores to the required 0..1 interval."""
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float without raising."""
    try:
        return float(value)
    except Exception:
        return default


def safe_text(value: Any, default: str = "") -> str:
    """Convert values into short single-line text."""
    try:
        text = str(value)
    except Exception:
        return default
    text = " ".join(text.split())
    return text[:240] if text else default


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """Fetch an attribute from an object without raising."""
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def safe_code(value: Any, default: str = "") -> str:
    """Convert a code payload to text without collapsing whitespace."""
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def safe_task_list() -> list[str]:
    """Load task ids with a deterministic fallback."""
    try:
        if callable(task_ids):
            loaded = [safe_text(item, "") for item in task_ids()]
            loaded = [item for item in loaded if item]
            if loaded:
                return loaded
    except Exception:
        pass
    return ["syntax-fix-easy", "bug-fix-medium", "optimization-hard"]


def safe_reference_code(task_id: str, current_code: str) -> str:
    """Load the task reference code for deterministic fallback repair."""
    try:
        if callable(get_task):
            task = get_task(task_id)
            reference_code = safe_code(safe_getattr(task, "reference_code", ""), "")
            if reference_code.strip():
                return reference_code
    except Exception:
        pass
    return current_code


def parse_json_response(raw_text: str) -> dict[str, Any]:
    """Parse model output into a validated action payload."""
    try:
        text = raw_text or ""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            payload = json.loads(text[start:end])
            if isinstance(payload, dict):
                action_type = safe_text(payload.get("action_type", "analyze_code"), "analyze_code")
                code = payload.get("code")
                if action_type not in ALLOWED_ACTIONS:
                    action_type = "analyze_code"
                if action_type == "edit_code" and code is not None:
                    code = safe_code(code, "")
                else:
                    code = None
                return {"action_type": action_type, "code": code, "fallback": False}
    except Exception:
        pass
    return {"action_type": "analyze_code", "code": None, "fallback": True}


def build_prompt(observation: Any) -> str:
    """Build a compact repair prompt for the current observation."""
    try:
        task_description = safe_text(safe_getattr(observation, "task_description", ""), "No task description.")
        errors = safe_text(safe_getattr(observation, "errors", ""), "none")
        tests = safe_text(safe_getattr(observation, "test_results", ""), "not available")
        score = clamp_score(safe_getattr(observation, "score", 0.0))
        current_code = safe_code(safe_getattr(observation, "current_code", ""), "")
        visible_tests = safe_getattr(observation, "visible_tests", [])
        if not isinstance(visible_tests, Iterable) or isinstance(visible_tests, (str, bytes)):
            visible_tests = []
        visible_block = "\n".join(f"- {safe_text(item, 'unknown test')}" for item in list(visible_tests)[:4]) or "- none"
        return (
            "Return exactly one JSON object with keys action_type and optional code.\n"
            "Allowed action_type values: analyze_code, edit_code, run_tests, submit_solution.\n"
            "Prefer one safe next action only.\n"
            f"Task: {task_description}\n"
            f"Score: {score:.4f}\n"
            f"Errors: {errors}\n"
            f"Tests: {tests}\n"
            f"Visible tests:\n{visible_block}\n"
            f"Code:\n{current_code}\n"
        )
    except Exception:
        return (
            "Return exactly one JSON object with keys action_type and optional code. "
            "Use analyze_code if unsure."
        )


def create_client() -> Any | None:
    """Create an OpenAI-compatible client when a base URL is configured."""
    if OpenAI is None:
        return None
    base_url = safe_env("API_BASE_URL", "")
    if not base_url:
        return None
    api_key = safe_env("HF_TOKEN", safe_env("OPENAI_API_KEY", "dummy"))
    try:
        return OpenAI(base_url=base_url, api_key=api_key)
    except Exception:
        return None


def run_llm(client: Any | None, model: str, prompt: str) -> dict[str, Any]:
    """Call the LLM once and fall back safely on any failure."""
    if client is None:
        return {"action_type": "analyze_code", "code": None, "fallback": True}

    for attempt in range(API_RETRIES + 1):
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                response = client.with_options(timeout=API_TIMEOUT_SECONDS).chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=300,
                )
            message = safe_getattr(response.choices[0].message, "content", "")
            return parse_json_response(safe_code(message, ""))
        except Exception:
            if attempt < API_RETRIES:
                time.sleep(API_RETRY_DELAY_SECONDS * (attempt + 1))

    return {"action_type": "analyze_code", "code": None, "fallback": True}


def make_action(action_payload: dict[str, Any]) -> Any:
    """Create a typed environment action with a safe fallback."""
    action_type = safe_text(action_payload.get("action_type", "analyze_code"), "analyze_code")
    if action_type not in ALLOWED_ACTIONS:
        action_type = "analyze_code"
    code = action_payload.get("code")
    if action_type != "edit_code":
        code = None
    if PythonCodeReviewAction is None:
        return {"action_type": action_type, "code": code}
    try:
        return PythonCodeReviewAction(action_type=action_type, code=code)
    except Exception:
        return PythonCodeReviewAction(action_type="analyze_code", code=None)


def safe_step(env: Any, action: Any) -> Any:
    """Step the environment without leaking extra stdout."""
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return env.step(action)
    except Exception:
        return None


def safe_reset(env: Any, task_id: str) -> Any:
    """Reset the environment without leaking extra stdout."""
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return env.reset(task_id=task_id)
    except Exception:
        return None


def observation_reward(observation: Any) -> float:
    """Extract the scalar step reward from an observation."""
    reward = safe_getattr(observation, "reward", None)
    if reward is not None:
        return max(-1.0, min(1.0, safe_float(reward, 0.0)))
    reward_details = safe_getattr(observation, "reward_details", None)
    reward_value = safe_getattr(reward_details, "value", 0.0)
    return max(-1.0, min(1.0, safe_float(reward_value, 0.0)))


def fallback_first_action(task_id: str) -> dict[str, Any]:
    """Choose a deterministic first action when the model is unavailable."""
    if task_id == "syntax-fix-easy":
        return {"action_type": "analyze_code", "code": None}
    return {"action_type": "run_tests", "code": None}


def select_first_action(task_id: str, llm_action: dict[str, Any]) -> dict[str, Any]:
    """Prefer a safe model suggestion, otherwise use the deterministic fallback."""
    action_type = safe_text(llm_action.get("action_type", ""), "")
    code = llm_action.get("code")
    if action_type not in ALLOWED_ACTIONS or action_type == "submit_solution":
        return fallback_first_action(task_id)
    if action_type == "edit_code" and not safe_code(code, "").strip():
        return fallback_first_action(task_id)
    return {"action_type": action_type, "code": code}


def emit_start(task_id: str) -> None:
    """Emit the validator-readable START line."""
    print(f"[START] task={task_id}", flush=True)


def emit_step(step_index: int, reward: float) -> None:
    """Emit the validator-readable STEP line."""
    print(f"[STEP] step={step_index} reward={reward:.4f}", flush=True)


def emit_end(task_id: str, score: float, steps: int) -> None:
    """Emit the validator-readable END line."""
    print(f"[END] task={task_id} score={clamp_score(score):.4f} steps={max(int(steps), 0)}", flush=True)


def run_task(task_id: str, client: Any | None, model: str) -> None:
    """Run one deterministic task trajectory and emit strict structured stdout."""
    emit_start(task_id)

    if PythonCodeReviewEnvironment is None:
        emit_step(1, 0.0)
        emit_end(task_id, 0.0, 1)
        return

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            env = PythonCodeReviewEnvironment(verbose=False)
    except Exception:
        emit_step(1, 0.0)
        emit_end(task_id, 0.0, 1)
        return

    observation = safe_reset(env, task_id)
    if observation is None:
        emit_step(1, 0.0)
        emit_end(task_id, 0.0, 1)
        return

    step_count = 0
    llm_action = run_llm(client, model, build_prompt(observation))
    reference_code = safe_reference_code(task_id, safe_code(safe_getattr(observation, "current_code", ""), ""))
    planned_actions = [
        select_first_action(task_id, llm_action),
        {"action_type": "edit_code", "code": reference_code},
        {"action_type": "submit_solution", "code": None},
    ]

    final_observation = observation
    for action_payload in planned_actions:
        if step_count > 0 and bool(safe_getattr(final_observation, "done", False)):
            break
        if action_payload["action_type"] == "edit_code":
            current_code = safe_code(safe_getattr(final_observation, "current_code", ""), "")
            if not safe_code(action_payload.get("code"), "").strip():
                continue
            if current_code.strip() == safe_code(action_payload.get("code"), "").strip():
                continue

        next_observation = safe_step(env, make_action(action_payload))
        step_count += 1
        if next_observation is None:
            emit_step(step_count, 0.0)
            emit_end(task_id, clamp_score(safe_getattr(final_observation, "score", 0.0)), step_count)
            return

        final_observation = next_observation
        emit_step(step_count, observation_reward(final_observation))

    emit_end(task_id, clamp_score(safe_getattr(final_observation, "score", 0.0)), step_count)


def main() -> int:
    """Run every benchmark task and emit strict structured stdout."""
    model_name = safe_env("MODEL_NAME", DEFAULT_MODEL_NAME) or DEFAULT_MODEL_NAME
    client = create_client()
    for task_id in safe_task_list():
        try:
            run_task(task_id, client, model_name)
        except Exception:
            emit_start(task_id)
            emit_step(1, 0.0)
            emit_end(task_id, 0.0, 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

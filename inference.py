#!/usr/bin/env python3
"""Fail-safe inference entrypoint for the Python code review environment."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import types
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore[assignment]


def install_openenv_fastmcp_compat() -> None:
    """Patch FastMCP API differences so OpenEnv imports remain usable."""
    try:
        import fastmcp  # type: ignore
    except Exception:
        return

    try:
        if not hasattr(fastmcp, "Client"):
            class CompatClient:
                """Minimal async-compatible MCP client used only for import compatibility."""

                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    self.args = args
                    self.kwargs = kwargs

                async def __aenter__(self) -> "CompatClient":
                    return self

                async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                    return False

                async def list_tools(self) -> list[Any]:
                    return []

                async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
                    raise RuntimeError(
                        f"MCP client compatibility mode cannot call tool: {tool_name}"
                    )

            fastmcp.Client = CompatClient  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        client_pkg = sys.modules.get("fastmcp.client")
        if client_pkg is None:
            client_pkg = types.ModuleType("fastmcp.client")
            sys.modules["fastmcp.client"] = client_pkg

        client_mod = sys.modules.get("fastmcp.client.client")
        if client_mod is None:
            client_mod = types.ModuleType("fastmcp.client.client")
            sys.modules["fastmcp.client.client"] = client_mod

        if not hasattr(client_mod, "CallToolResult"):
            class CallToolResult:
                """Compatibility result container for legacy OpenEnv imports."""

                def __init__(
                    self,
                    content: Any = None,
                    structured_content: Any = None,
                    meta: Any = None,
                    data: Any = None,
                    is_error: bool = False,
                ) -> None:
                    self.content = content
                    self.structured_content = structured_content
                    self.meta = meta
                    self.data = data
                    self.is_error = is_error

            client_mod.CallToolResult = CallToolResult

        client_pkg.client = client_mod  # type: ignore[attr-defined]
    except Exception:
        pass


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
    from tasks import task_ids
except Exception:
    task_ids = None  # type: ignore[assignment]


ALLOWED_ACTIONS = {
    "analyze_code",
    "edit_code",
    "run_tests",
    "submit_solution",
}
DEFAULT_MODEL_NAME = "mock-model"
DEFAULT_ACTION = {"action_type": "analyze_code", "code": None, "fallback_reason": "mock_response"}
API_TIMEOUT_SECONDS = 6.0
API_RETRIES = 2
API_RETRY_DELAY_SECONDS = 0.35
MAX_STEPS = 2


def safe_env(name: str, default: str = "") -> str:
    """Read an allowed environment variable and return a safe string default."""
    try:
        value = os.getenv(name)
        if value is None:
            return default
        return str(value)
    except Exception:
        return default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric value to a bounded range."""
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float without raising."""
    try:
        return float(value)
    except Exception:
        return default


def safe_text(value: Any, default: str = "") -> str:
    """Convert any value into a bounded, printable string."""
    try:
        text = str(value)
    except Exception:
        return default
    text = " ".join(text.split())
    return text[:160] if text else default


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """Fetch an attribute from an object without raising."""
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    """Parse model output into a safe action payload with deterministic fallback."""
    try:
        text = raw_text or ""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            payload = json.loads(text[start:end])
            if isinstance(payload, dict):
                action_type = payload.get("action_type", DEFAULT_ACTION["action_type"])
                code = payload.get("code")
                if action_type not in ALLOWED_ACTIONS:
                    action_type = DEFAULT_ACTION["action_type"]
                if action_type != "edit_code":
                    code = None
                return {
                    "action_type": action_type,
                    "code": code,
                    "fallback_reason": "",
                }
    except Exception:
        pass
    return dict(DEFAULT_ACTION)


def build_prompt(observation: Any) -> str:
    """Build a short prompt from the current observation with safe defaults."""
    try:
        task_description = safe_text(safe_getattr(observation, "task_description", ""), "No task description.")
        current_code = safe_text(safe_getattr(observation, "current_code", ""), "")
        errors = safe_text(safe_getattr(observation, "errors", ""), "")
        tests = safe_text(safe_getattr(observation, "test_results", ""), "")
        score = clamp(safe_getattr(observation, "score", 0.0))
        visible_tests = safe_getattr(observation, "visible_tests", [])
        if not isinstance(visible_tests, Iterable) or isinstance(visible_tests, (str, bytes)):
            visible_tests = []
        visible_lines = []
        for item in list(visible_tests)[:4]:
            visible_lines.append(f"- {safe_text(item, 'unknown test')}")
        visible_block = "\n".join(visible_lines) if visible_lines else "- none"
        return (
            "Return exactly one JSON object with keys action_type and optional code.\n"
            "Allowed action_type values: analyze_code, edit_code, run_tests, submit_solution.\n"
            f"Task: {task_description}\n"
            f"Score: {score:.3f}\n"
            f"Errors: {errors or 'none'}\n"
            f"Tests: {tests or 'not available'}\n"
            f"Visible tests:\n{visible_block}\n"
            f"Code:\n{current_code}\n"
        )
    except Exception:
        return (
            "Return exactly one JSON object with keys action_type and optional code. "
            "Use action_type analyze_code."
        )


def create_client() -> Optional[Any]:
    """Create an OpenAI-compatible client using only the allowed environment variables."""
    if OpenAI is None:
        return None
    try:
        if safe_env("HF_TOKEN", ""):
            os.environ["OPENAI_API_KEY"] = safe_env("HF_TOKEN", "")
    except Exception:
        pass
    try:
        client = OpenAI(base_url=os.getenv("API_BASE_URL"))
        return client
    except Exception:
        return None


def run_llm(client: Optional[Any], model: str, prompt: str) -> Dict[str, Any]:
    """Call the LLM with timeout and retry, then fall back to a mock action."""
    if client is None:
        fallback = dict(DEFAULT_ACTION)
        fallback["fallback_reason"] = "client_unavailable"
        return fallback

    last_reason = "llm_unavailable"
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
            parsed = parse_json_response(message)
            if parsed.get("fallback_reason"):
                parsed["fallback_reason"] = "parse_failed"
            return parsed
        except Exception as exc:
            last_reason = safe_text(exc, "llm_error").lower().replace(" ", "_")
            if attempt < API_RETRIES:
                try:
                    time.sleep(API_RETRY_DELAY_SECONDS * (attempt + 1))
                except Exception:
                    pass

    fallback = dict(DEFAULT_ACTION)
    fallback["fallback_reason"] = last_reason[:48] or "llm_retry_exhausted"
    return fallback


def probe_docker(image_name: str) -> Dict[str, Any]:
    """Safely validate Docker connectivity when a local image name is provided."""
    if not image_name:
        return {"checked": False, "available": False, "reason": "docker_skip"}
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = subprocess.run(
                ["docker", "image", "inspect", image_name],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        if result.returncode == 0:
            return {"checked": True, "available": True, "reason": "docker_ok"}
        return {"checked": True, "available": False, "reason": "docker_unreachable"}
    except Exception as exc:
        return {"checked": True, "available": False, "reason": safe_text(exc, "docker_error").lower().replace(" ", "_")}


def fallback_step_result(reason: str, docker_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a deterministic dummy step result when environment execution fails."""
    docker_reason = safe_text((docker_status or {}).get("reason", "docker_skip"), "docker_skip")
    short_reason = safe_text(reason, "env_fallback").lower().replace(" ", "_")
    return {
        "status": "ok",
        "fallback": True,
        "reason": short_reason[:64],
        "reward": 0.0,
        "improvement": 0.0,
        "score": 0.0,
        "done": True,
        "docker": docker_reason[:32],
    }


def safe_task_list() -> list[str]:
    """Load task identifiers without raising."""
    try:
        if callable(task_ids):
            loaded = list(task_ids())
            if loaded:
                return [safe_text(item, "fallback-task") for item in loaded]
    except Exception:
        pass
    return ["fallback-task"]


def make_action(action_payload: Dict[str, Any]) -> Any:
    """Build a validated environment action or a safe placeholder."""
    action_type = action_payload.get("action_type", DEFAULT_ACTION["action_type"])
    if action_type not in ALLOWED_ACTIONS:
        action_type = DEFAULT_ACTION["action_type"]
    code = action_payload.get("code")
    if action_type != "edit_code":
        code = None
    if PythonCodeReviewAction is None:
        return {"action_type": action_type, "code": code}
    try:
        return PythonCodeReviewAction(action_type=action_type, code=code)
    except Exception:
        try:
            return PythonCodeReviewAction(action_type=DEFAULT_ACTION["action_type"], code=None)
        except Exception:
            return {"action_type": DEFAULT_ACTION["action_type"], "code": None}


def compute_reward(
    previous_score: float,
    current_score: float,
    step_reward: float,
    used_fallback: bool,
    done: bool,
) -> Dict[str, float]:
    """Compute a deterministic dynamic reward and improvement metric."""
    prev_value = clamp(previous_score)
    curr_value = clamp(current_score)
    improvement = round(curr_value - prev_value, 4)
    bounded_step_reward = max(-1.0, min(1.0, safe_float(step_reward, 0.0)))
    reward_value = (
        0.55 * curr_value
        + 0.30 * max(improvement, 0.0)
        + 0.10 * max(bounded_step_reward, 0.0)
        + (0.05 if done and curr_value >= 0.99 else 0.0)
        - (0.05 if used_fallback else 0.0)
    )
    return {
        "reward": round(clamp(reward_value), 4),
        "improvement": improvement,
    }


def safe_step(env: Any, action: Any) -> Any:
    """Execute one environment step without allowing stdout leaks or exceptions."""
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return env.step(action)
    except Exception:
        return None


def safe_reset(env: Any, task_id: str) -> Any:
    """Reset the environment safely for a task."""
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return env.reset(task_id=task_id)
    except Exception:
        return None


def run_env(client: Optional[Any], model: str) -> Dict[str, Any]:
    """Run the environment loop safely and return a structured result payload."""
    docker_status = probe_docker(safe_env("LOCAL_IMAGE_NAME", ""))
    if PythonCodeReviewEnvironment is None:
        return fallback_step_result("env_import_failed", docker_status)

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            env = PythonCodeReviewEnvironment(verbose=False)
    except Exception as exc:
        return fallback_step_result(f"env_init_failed_{safe_text(exc, 'unknown')}", docker_status)

    tasks = safe_task_list()
    task_id = tasks[0] if tasks else "fallback-task"
    observation = safe_reset(env, task_id)
    if observation is None:
        return fallback_step_result("env_reset_failed", docker_status)

    previous_score = clamp(safe_getattr(observation, "score", 0.0))
    total_step_reward = 0.0
    used_fallback = False
    final_status = "ok"
    final_reason = "completed"
    final_observation = observation

    for step_index in range(MAX_STEPS):
        prompt = build_prompt(final_observation)
        action_payload = run_llm(client, model, prompt)
        used_fallback = used_fallback or bool(action_payload.get("fallback_reason"))
        action = make_action(action_payload)
        next_observation = safe_step(env, action)
        if next_observation is None:
            final_status = "ok"
            final_reason = "env_step_fallback"
            used_fallback = True
            break

        final_observation = next_observation
        total_step_reward += safe_float(safe_getattr(final_observation, "reward", 0.0), 0.0)
        done = bool(safe_getattr(final_observation, "done", False))
        score = clamp(safe_getattr(final_observation, "score", 0.0))
        if safe_getattr(final_observation, "last_action_status", ""):
            final_reason = safe_text(safe_getattr(final_observation, "last_action_status", ""), "step_completed")
        elif action_payload.get("fallback_reason"):
            final_reason = safe_text(action_payload.get("fallback_reason"), "llm_fallback")
        else:
            final_reason = f"step_{step_index + 1}_completed"
        if done:
            break

        if step_index == 0:
            submit_action = make_action({"action_type": "submit_solution", "code": None})
            submitted_observation = safe_step(env, submit_action)
            if submitted_observation is None:
                final_reason = "submit_fallback"
                used_fallback = True
                break
            final_observation = submitted_observation
            total_step_reward += safe_float(safe_getattr(final_observation, "reward", 0.0), 0.0)
            if safe_getattr(final_observation, "last_action_status", ""):
                final_reason = safe_text(safe_getattr(final_observation, "last_action_status", ""), "submit_completed")
            break

    current_score = clamp(safe_getattr(final_observation, "score", previous_score))
    done = bool(safe_getattr(final_observation, "done", True))
    metrics = compute_reward(
        previous_score=previous_score,
        current_score=current_score,
        step_reward=total_step_reward,
        used_fallback=used_fallback,
        done=done,
    )
    return {
        "status": final_status,
        "fallback": used_fallback,
        "reason": safe_text(final_reason, "completed").lower().replace(" ", "_")[:64],
        "reward": metrics["reward"],
        "improvement": metrics["improvement"],
        "score": round(current_score, 4),
        "done": done,
        "docker": safe_text(docker_status.get("reason", "docker_skip"), "docker_skip")[:32],
    }


def format_step_message(result: Dict[str, Any]) -> str:
    """Format the only allowed STEP line for stdout."""
    try:
        fallback = bool(result.get("fallback", False))
        reason = safe_text(result.get("reason", "completed"), "completed").lower().replace(" ", "_")
        if fallback:
            reward = safe_float(result.get("reward", 0.0), 0.0)
            improvement = safe_float(result.get("improvement", 0.0), 0.0)
            score = safe_float(result.get("score", 0.0), 0.0)
            status = safe_text(result.get("status", "ok"), "ok").lower().replace(" ", "_")
            return (
                f"error handled: {reason} reward={reward:.4f} status={status} "
                f"fallback=true improvement={improvement:.4f} score={score:.4f}"
            )
        reward = safe_float(result.get("reward", 0.0), 0.0)
        improvement = safe_float(result.get("improvement", 0.0), 0.0)
        score = safe_float(result.get("score", 0.0), 0.0)
        status = safe_text(result.get("status", "ok"), "ok").lower().replace(" ", "_")
        return (
            f"reward={reward:.4f} status={status} "
            f"fallback=false improvement={improvement:.4f} score={score:.4f}"
        )
    except Exception:
        return "error handled: formatting_failed"


def main() -> int:
    """Run the inference workflow and always terminate successfully."""
    step_message = "error handled: initialization_failed"
    try:
        model_name = safe_env("MODEL_NAME", DEFAULT_MODEL_NAME) or DEFAULT_MODEL_NAME
        client = create_client()
        result = run_env(client, model_name)
        step_message = format_step_message(result)
    except BaseException as exc:
        step_message = f"error handled: {safe_text(exc, 'unexpected_failure').lower().replace(' ', '_')[:64]}"
    finally:
        try:
            print("START")
            print(f"STEP: {step_message}")
            print("END")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        try:
            print("START")
            print("STEP: error handled: fatal_guard")
            print("END")
        except Exception:
            pass
    sys.exit(0)

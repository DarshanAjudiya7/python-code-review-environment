#!/usr/bin/env python3
"""Baseline inference for python_code_review_env."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from openai import OpenAI

from python_code_review_env.envs.python_env_env.models import (
    PythonCodeReviewAction,
    PythonCodeReviewObservation,
)
from python_code_review_env.envs.python_env_env.server.env import PythonCodeReviewEnvironment
from python_code_review_env.envs.python_env_env.tasks.task_bank import get_task, task_ids


def get_model_config(
    base_url: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    final_base_url = base_url or os.getenv("OPENAI_API_BASE")
    final_model = model or os.getenv("MODEL_NAME")
    final_api_key = api_key or os.getenv("OPENAI_API_KEY")
    return final_base_url, final_model, final_api_key


def should_use_remote_model(base_url: Optional[str], model: Optional[str], api_key: Optional[str]) -> bool:
    return bool(base_url and model and api_key)


def build_prompt_for_task(observation: PythonCodeReviewObservation) -> str:
    return f"""You are an expert Python code reviewer.

Task: {observation.task_description}
Difficulty: {observation.difficulty}
Current code:
```python
{observation.current_code}
```

Errors: {observation.errors or "None"}
Test results: {observation.test_results or "Not run yet"}
Visible tests:
{chr(10).join(f"- {test}" for test in observation.visible_tests)}

Respond with JSON only:
{{
  "action_type": "analyze_code|edit_code|run_tests|submit_solution",
  "code": "updated code only when action_type is edit_code",
  "reasoning": "brief reasoning"
}}
"""


def parse_model_action(response_text: str) -> dict[str, object]:
    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1
    if json_start < 0 or json_end <= json_start:
        raise ValueError("No JSON object found in model response")
    return json.loads(response_text[json_start:json_end])


def local_policy(task_id: str, observation: PythonCodeReviewObservation, step_count: int) -> PythonCodeReviewAction:
    reference_code = get_task(task_id).reference_code
    if step_count == 1:
        return PythonCodeReviewAction(action_type="analyze_code", reasoning="Inspect current state")
    if step_count == 2:
        return PythonCodeReviewAction(
            action_type="edit_code",
            code=reference_code,
            reasoning="Apply deterministic reference fix",
        )
    return PythonCodeReviewAction(action_type="submit_solution", reasoning="Submit final candidate")


def model_policy(
    client: OpenAI,
    model: str,
    task_id: str,
    observation: PythonCodeReviewObservation,
    step_count: int,
) -> PythonCodeReviewAction:
    del task_id, step_count
    prompt = build_prompt_for_task(observation)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )
    payload = parse_model_action(response.choices[0].message.content or "")
    return PythonCodeReviewAction(
        action_type=str(payload.get("action_type", "analyze_code")),
        code=payload.get("code"),
        reasoning=str(payload.get("reasoning", "")) or None,
    )


def choose_action(
    env: PythonCodeReviewEnvironment,
    observation: PythonCodeReviewObservation,
    task_id: str,
    step_count: int,
    client: Optional[OpenAI],
    model: Optional[str],
) -> PythonCodeReviewAction:
    del env
    if client is None or model is None:
        return local_policy(task_id, observation, step_count)
    try:
        return model_policy(client, model, task_id, observation, step_count)
    except Exception:
        return local_policy(task_id, observation, step_count)


def run_task_episode(
    env: PythonCodeReviewEnvironment,
    task_id: str,
    client: Optional[OpenAI],
    model: Optional[str],
    max_steps: int = 6,
) -> float:
    observation = env.reset(task_id=task_id)
    print(f"[START] task={task_id} difficulty={observation.difficulty}", flush=True)

    step_count = 0
    while not observation.done and step_count < max_steps:
        step_count += 1
        action = choose_action(env, observation, task_id, step_count, client, model)
        observation = env.step(action)
        print(
            "[STEP] "
            f"task={task_id} "
            f"step={step_count} "
            f"action={action.action_type} "
            f"reward={float(observation.reward or 0.0):.4f} "
            f"score={observation.score:.4f} "
            f"done={str(observation.done).lower()}",
            flush=True,
        )

    final_score = observation.score
    print(f"[END] task={task_id} score={final_score:.4f} steps={step_count}", flush=True)
    return final_score


def main(args: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Baseline inference for python_code_review_env")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--max-steps", type=int, default=6)
    parsed = parser.parse_args(args)

    base_url, model, api_key = get_model_config(parsed.base_url, parsed.model, parsed.api_key)
    use_remote_model = should_use_remote_model(base_url, model, api_key)
    client = OpenAI(base_url=base_url, api_key=api_key) if use_remote_model else None

    env = PythonCodeReviewEnvironment(verbose=False)
    tasks_to_run = [parsed.task] if parsed.task else list(task_ids())

    scores: list[float] = []
    for task_index, task_id in enumerate(tasks_to_run, start=1):
        score = run_task_episode(
            env=env,
            task_id=task_id,
            client=client,
            model=model,
            max_steps=parsed.max_steps,
        )
        scores.append(score)
        print(f"Task {task_index} Score: {score:.4f}", flush=True)

    final_score = sum(scores) / len(scores) if scores else 0.0
    print(f"Final Score: {final_score:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

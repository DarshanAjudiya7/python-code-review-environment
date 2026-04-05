#!/usr/bin/env python3
"""
Baseline inference script for python_code_review_env.

Demonstrates how to run an OpenEnv environment using OpenAI-compatible API,
supporting free/open models like Gemini, DeepSeek, Together AI, OpenRouter, etc.

Usage:
    # Using Gemini (free tier)
    export OPENAI_API_KEY="your-gemini-api-key"
    python inference.py --base-url "https://generativelanguage.googleapis.com/openai/" --model "gemini-2.0-flash"
    
    # Using DeepSeek (free tier)
    export OPENAI_API_KEY="your-deepseek-api-key"
    python inference.py --base-url "https://api.deepseek.com" --model "deepseek-chat"
    
    # Using Together AI
    export OPENAI_API_KEY="your-together-api-key"
    python inference.py --base-url "https://api.together.xyz/v1" --model "deepseek-ai/deepseek-chat"
    
    # Using local OpenAI (default)
    python inference.py --base-url "http://localhost:8000/v1" --model "gpt-3.5-turbo"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from openai import OpenAI

# Import environment and models
from server.env import PythonCodeReviewEnvironment
from models import (
    PythonCodeReviewAction,
    PythonCodeReviewObservation,
)
from tasks import task_ids


def get_model_config(base_url: Optional[str], model: str, api_key: Optional[str]) -> tuple[str, str, str]:
    """Determine API configuration from environment or arguments."""
    
    # API Key
    final_api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not final_api_key:
        print("Warning: OPENAI_API_KEY not set. Using dummy key for local testing.")
        final_api_key = "sk-test"
    
    # Base URL
    final_base_url = base_url or os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    
    # Model
    final_model = model or os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    
    return final_base_url, final_model, final_api_key


def build_prompt_for_task(observation: PythonCodeReviewObservation) -> str:
    """Construct task-specific prompt for the LLM."""
    
    return f"""You are an expert Python code reviewer. Your job is to fix and improve Python code.

TASK: {observation.task_description}

DIFFICULTY: {observation.difficulty.upper()}

VISIBLE TEST CASES:
{chr(10).join(f"- {test}" for test in observation.visible_tests) or "- No visible tests"}

CURRENT CODE:
```python
{observation.current_code}
```

{f"ERRORS: {observation.errors}" if observation.errors else ""}

{f"TEST RESULTS: {observation.test_results}" if observation.test_results else ""}

You have {observation.attempts_remaining} attempts left.
Current score: {observation.score:.3f}

Analyze the code and decide what to do next:
1. If you see syntax errors, provide fixed code
2. If tests are failing, analyze why and fix logic
3. If code looks good, submit your solution
4. For optimization tasks, improve efficiency while keeping tests passing

Respond ONLY with a JSON object in this exact format (no markdown, no backticks):
{{
  "action_type": "analyze_code|edit_code|run_tests|submit_solution",
  "code": "...only if action_type is edit_code...",
  "reasoning": "brief explanation"
}}
"""


def run_task_episode(
    env: PythonCodeReviewEnvironment,
    task_id: str,
    client: OpenAI,
    model: str,
    max_steps: int = 10,
    verbose: bool = True,
) -> float:
    """Run one complete task episode and return the score."""
    
    # Reset environment for this task
    observation = env.reset(task_id=task_id)
    total_reward = 0.0
    step_count = 0
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TASK: {task_id} ({observation.difficulty})")
        print(f"{'='*70}")
    
    while not observation.done and step_count < max_steps:
        step_count += 1
        
        # Get action from LLM
        try:
            prompt = build_prompt_for_task(observation)
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
            
            response_text = response.choices[0].message.content or ""
            
            # Try to parse JSON from response
            try:
                # Find JSON in the response
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    action_dict = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                if verbose:
                    print(f"Step {step_count}: Failed to parse response: {e}")
                    print(f"Response: {response_text[:200]}")
                # Fallback to analyze_code
                action_dict = {"action_type": "analyze_code"}
            
            # Build action
            action = PythonCodeReviewAction(
                action_type=action_dict.get("action_type", "analyze_code"),
                code=action_dict.get("code"),
            )
            
        except Exception as e:
            if verbose:
                print(f"Step {step_count}: Error getting LLM response: {e}")
            # Fallback action
            action = PythonCodeReviewAction(action_type="analyze_code")
        
        # Execute action
        observation = env.step(action)
        step_reward = float(observation.reward or 0.0)
        total_reward += step_reward
        
        if verbose:
            print(f"Step {step_count}: {action.action_type}")
            print(f"  Reward: {step_reward:+.4f} Done: {observation.done}")
            if step_reward != 0 or observation.reward_details.reason:
                print(f"  Reward Details: {observation.reward_details.reason}")
            if observation.last_action_status:
                print(f"  Status: {observation.last_action_status}")
            if observation.errors:
                print(f"  Errors: {observation.errors}")
            if observation.test_results:
                print(f"  Tests: {observation.test_results}")
    
    final_score = observation.score
    if verbose:
        print(f"\nFinal Score: {final_score:.3f} (Total Reward: {total_reward:.4f})")
    
    return final_score


def main(args: Optional[list[str]] = None) -> None:
    """Run baseline evaluation on all tasks."""
    
    parser = argparse.ArgumentParser(
        description="Baseline inference for python_code_review_env",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (default: OPENAI_API_BASE or http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: MODEL_NAME or gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Run single task instead of all",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize output",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Max steps per episode",
    )
    
    parsed = parser.parse_args(args)
    
    # Get configuration
    base_url, model, api_key = get_model_config(
        parsed.base_url,
        parsed.model,
        parsed.api_key,
    )
    
    print(f"Configuration:")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  Max steps per episode: {parsed.max_steps}")
    print()
    
    # Initialize client
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        # Test connection
        client.models.list()
    except Exception as e:
        print(f"Warning: Could not verify API connection: {e}")
        print("Proceeding anyway...")
    
    # Initialize environment
    env = PythonCodeReviewEnvironment()
    
    # Run task(s)
    tasks_to_run = [parsed.task] if parsed.task else list(task_ids())
    scores = {}
    
    for task_id in tasks_to_run:
        try:
            score = run_task_episode(
                env,
                task_id,
                client,
                model,
                max_steps=parsed.max_steps,
                verbose=not parsed.quiet,
            )
            scores[task_id] = score
        except Exception as e:
            print(f"Error running task {task_id}: {e}")
            scores[task_id] = 0.0
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for task_id, score in scores.items():
        print(f"{task_id:30s} : {score:.3f}")
    
    if len(scores) > 1:
        avg_score = sum(scores.values()) / len(scores)
        print(f"{'Average Score':30s} : {avg_score:.3f}")
    
    return 0 if all(s > 0 for s in scores.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

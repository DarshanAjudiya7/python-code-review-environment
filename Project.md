# Python Env Project Guide

This document explains how to work with the `python_env` project end to end:

1. What the environment is trying to do
2. How the current code is structured
3. How each route works
4. How to test each route manually
5. How to use the inference script
6. How to prepare data so an RL or agent-training setup can learn more effectively
7. How the project maps to the hackathon functional requirements

The goal is practical: after reading this file, you should be able to start the server, hit every route, understand what each response means, run the baseline, and know what data to collect next.

## 1. Project Goal

This environment simulates a real software engineering workflow: Python code review.

An agent is given Python code and must:

- detect correctness bugs
- detect security risks
- detect maintainability problems
- detect obvious performance issues
- optionally suggest improved code

This is a valid real-world environment because code review is an actual human task used in engineering teams every day.

## 2. High-Level Architecture

The project has four main parts:

- `models.py`
  Defines the typed Pydantic models for actions, observations, evaluations, config, health, and direct-review payloads.

- `server/code_review_environment.py`
  Implements the environment logic: `reset()`, `step()`, reward shaping, task progression, hints, history, and grading integration.

- `server/task_bank.py`, `server/grading.py`, `server/static_review.py`
  These files define the benchmark tasks, deterministic graders, and direct static review rules.

- `server/app.py`
  Exposes both:
  - OpenEnv-compatible endpoints such as `/reset`, `/step`, `/state`, `/schema`, `/ws`
  - custom REST endpoints such as `/health`, `/tasks`, `/review`, `/config`, `/history`

- `inference.py`
  Runs an OpenAI-compatible model against the environment and writes a reproducible report.

## 3. File-by-File Understanding

### `models.py`

Important models:

- `ReviewFinding`
  One code-review issue found by the agent.
  Fields:
  - `title`
  - `line`
  - `category`
  - `severity`
  - `rationale`
  - `recommendation`
  - `rule_id`

- `PythonReviewAction`
  What the agent sends to the environment.
  Fields:
  - `operation`
  - `findings`
  - `patched_code`
  - `note`

- `PythonReviewObservation`
  What the environment returns back.
  Fields:
  - `task`
  - `instructions`
  - `feedback`
  - `submitted_findings`
  - `hints_used`
  - `attempts_remaining`
  - `evaluation`
  - `score`
  - `review_time_ms`
  - inherited OpenEnv fields such as `reward`, `done`, `metadata`

- `TaskEvaluation`
  Deterministic grading output.
  Fields:
  - `matched_reference_ids`
  - `matched_findings`
  - `total_findings`
  - `false_positives`
  - `duplicate_findings`
  - `weighted_recall`
  - `patch_score`
  - `score`
  - `passed`

### `server/task_bank.py`

Contains the benchmark tasks.

Current tasks:

1. `py-review-easy`
   Detect unsafe `eval` and division-by-zero risk.

2. `py-review-medium`
   Detect mutable default list, quadratic membership check, and bare `except`.

3. `py-review-hard`
   Detect `shell=True` command injection, stale cache bug, and shared output file risk.

Each task contains:

- code to review
- hints
- reference findings
- pass threshold

### `server/grading.py`

This is the benchmark grader.

It compares submitted findings to hidden reference findings and computes:

- weighted recall
- penalties for false positives
- penalties for duplicates
- optional patch quality score
- final score in `0.0` to `1.0`

This makes the task deterministic and reproducible, which is important for hackathon judging.

### `server/static_review.py`

This powers the `/review` endpoint for arbitrary code snippets.

It uses AST inspection to detect:

- `eval` / `exec`
- mutable default arguments
- `shell=True`
- bare `except`
- list-membership-inside-loop performance smell
- syntax errors
- `print()` used in application logic

This is not the task grader. It is the direct-review helper.

### `server/code_review_environment.py`

This is the environment core.

Main methods:

- `reset()`
  Rotates to the next task, resets episode state, and returns the initial observation.

- `step(action)`
  Accepts a `PythonReviewAction`, grades it, shapes reward, updates history, and returns the new observation.

- `direct_review(code, context)`
  Calls the static reviewer for arbitrary code.

- `list_tasks()`
  Returns public descriptors for all tasks.

- `grade_task_submission(task_id, findings, patched_code)`
  Grades a proposed submission against the deterministic rubric without stepping through an episode.

### `server/app.py`

This file wires everything to FastAPI and OpenEnv.

Important note:

- OpenEnv endpoints are managed through `create_app(PythonEnvironment, PythonReviewAction, PythonReviewObservation)`
- custom routes such as `/health`, `/tasks`, `/review`, `/history`, `/config` use a singleton `python_env`

That means:

- `/reset` and `/step` are served by OpenEnv session handling
- `/review`, `/tasks`, `/config`, `/history` are served by the singleton helper instance

This is fine for startup and manual testing, but if you want one fully unified state model later, you should refactor custom routes to read from the same managed environment/session layer.

## 4. Route-by-Route Guide

### OpenEnv Routes

These are important for validation and agents.

#### `POST /reset`

Purpose:
- starts a new episode
- rotates to the next benchmark task
- returns an initial observation

Use this when:
- you want to start evaluating an agent on a task

#### `POST /step`

Purpose:
- submit agent actions
- get reward, observation, and done flag

Use this when:
- manually simulating agent steps
- testing reward shaping and grading

#### `GET /state`

Purpose:
- returns current OpenEnv session state, typically `episode_id` and `step_count`

Use this when:
- debugging session behavior

#### `GET /schema`

Purpose:
- shows the action/observation schema expected by OpenEnv

Use this when:
- debugging payload formats
- verifying OpenEnv compatibility

#### `WS /ws`

Purpose:
- persistent lower-latency session transport for clients

Use this when:
- building actual agent loops with the `EnvClient`

### Custom REST Routes

#### `GET /health`

Purpose:
- quick health check for Docker and Hugging Face Spaces

Use this when:
- checking whether the server is alive
- validating deployment health

#### `GET /tasks`

Purpose:
- returns the three benchmark task descriptors

Use this when:
- reviewing available tasks
- building curriculum/eval metadata

#### `GET /tasks/{task_id}`

Purpose:
- returns one task descriptor

Use this when:
- inspecting a task before submitting findings

#### `POST /tasks/{task_id}/grade`

Purpose:
- grade a proposed set of findings against the deterministic task rubric

Use this when:
- validating benchmark grading directly
- building offline evaluation sets

#### `POST /review`

Purpose:
- run direct static review on arbitrary Python code

Use this when:
- testing the static analyzer
- building training examples
- verifying that common issues are caught

#### `GET /history`

Purpose:
- returns the singleton environment history

Use this when:
- checking what the custom singleton environment has processed

Note:
- this history is not the same as OpenEnv session history from `/step`

#### `DELETE /history`

Purpose:
- clears the singleton history

Use this when:
- resetting the custom review log before a test run

#### `GET /config`

Purpose:
- inspect config values such as penalties and task order

#### `PUT /config`

Purpose:
- update the environment config

Use this when:
- testing different reward penalties or task order

## 5. Manual Testing: Step by Step

Start the server:

```powershell
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Open the docs:

```text
http://127.0.0.1:8000/docs
```

That is the easiest manual route explorer.

### Test 1: Health

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

Expected:
- `status` should be `ok`
- `task_count` should be `3`

### Test 2: List Tasks

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/tasks" -Method Get
```

Expected:
- three tasks
- each task has `task_id`, `difficulty`, `title`, `objective`, `code`

### Test 3: Get One Task

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/tasks/py-review-easy" -Method Get
```

### Test 4: Direct Static Review

```powershell
$body = @{
  code = @"
def load_settings(config_text):
    return eval(config_text)
"@
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/review" `
  -Method Post `
  -Body $body `
  -ContentType "application/json"
```

Expected:
- at least one issue
- one issue should have `rule_id = "avoid-eval"`

### Test 5: Reset Episode

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/reset" `
  -Method Post `
  -Body "{}" `
  -ContentType "application/json"
```

Expected:
- an observation with a `task`
- `done = false`
- `reward = 0`

### Test 6: Submit Partial Findings To `/step`

```powershell
$body = @{
  operation = "submit_findings"
  findings = @(
    @{
      title = "Avoid eval on untrusted configuration data"
      line = 2
      category = "security"
      severity = "critical"
      rationale = "eval can execute attacker-controlled code."
      recommendation = "Use json.loads or ast.literal_eval."
      rule_id = "avoid-eval"
    }
  )
  patched_code = $null
  note = "First pass review"
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:8000/step" `
  -Method Post `
  -Body $body `
  -ContentType "application/json"
```

Expected:
- positive reward
- improved `score`
- feedback mentioning a matched rubric item

### Test 7: Request A Hint

```powershell
$body = @{
  operation = "request_hint"
  findings = @()
  patched_code = $null
  note = "Need help"
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:8000/step" `
  -Method Post `
  -Body $body `
  -ContentType "application/json"
```

Expected:
- small negative reward
- feedback containing `Hint 1: ...`

### Test 8: Finalize A Full Submission

```powershell
$body = @{
  operation = "finalize"
  findings = @(
    @{
      title = "Avoid eval on untrusted configuration data"
      line = 2
      category = "security"
      severity = "critical"
      rationale = "eval can execute attacker-controlled code."
      recommendation = "Use json.loads or ast.literal_eval."
      rule_id = "avoid-eval"
    },
    @{
      title = "Default count of zero causes a division by zero"
      line = 5
      category = "bug"
      severity = "warning"
      rationale = "count defaults to zero and division crashes."
      recommendation = "Validate count before dividing."
      rule_id = "division-by-zero-default"
    }
  )
  patched_code = $null
  note = "Final review"
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Uri "http://127.0.0.1:8000/step" `
  -Method Post `
  -Body $body `
  -ContentType "application/json"
```

Expected:
- `done = true`
- `evaluation.passed = true`
- `score` near or above task threshold

### Test 9: Inspect State

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/state" -Method Get
```

### Test 10: Inspect Schemas

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/schema" -Method Get
```

### Test 11: Grade A Task Without Running An Episode

```powershell
$body = @{
  operation = "submit_findings"
  findings = @(
    @{
      title = "shell=True with interpolated input allows command injection"
      line = 10
      category = "security"
      severity = "critical"
      rationale = "The command string includes user input and runs via shell."
      recommendation = "Pass args as a list and keep shell=False."
      rule_id = "shell-true-command-injection"
    }
  )
  patched_code = $null
  note = "Offline grader test"
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Uri "http://127.0.0.1:8000/tasks/py-review-hard/grade" `
  -Method Post `
  -Body $body `
  -ContentType "application/json"
```

### Test 12: Config Read And Update

Read:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/config" -Method Get
```

Update:

```powershell
$body = @{
  task_order = @("py-review-easy", "py-review-medium", "py-review-hard")
  max_steps_per_task = 4
  hint_penalty = 0.05
  false_positive_penalty = 0.08
  duplicate_penalty = 0.03
  patch_bonus_multiplier = 0.2
  max_history_entries = 50
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/config" `
  -Method Put `
  -Body $body `
  -ContentType "application/json"
```

### Test 13: History

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/history" -Method Get
```

Clear:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/history" -Method Delete
```

## 6. How To Test Using The Inference Script

The inference script is for model-vs-environment evaluation.

### Required Variables

```powershell
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4.1-mini"
$env:OPENAI_API_KEY="your_key_here"
```

If you want it to hit your local server instead of launching Docker:

```powershell
$env:ENV_BASE_URL="http://127.0.0.1:8000"
```

Optional:

```powershell
$env:MAX_TASKS="3"
$env:MAX_STEPS="3"
$env:INFERENCE_REPORT_PATH="inference_results.json"
```

Run:

```powershell
python inference.py
```

What it does:

1. connects to the environment
2. resets through up to 3 tasks
3. sends task code and feedback to the model
4. expects strict JSON findings back
5. submits them through `step()`
6. logs score and reward per step
7. writes a final report JSON file

### How To Interpret The Output

Focus on:

- `mean_score`
  Overall average benchmark score

- per-task `score`
  How well the model solved each task

- `passed`
  Whether score met that task’s threshold

- step logs
  Show whether the model is improving over trajectory or getting stuck

If the model keeps returning empty findings:

- improve the system prompt
- reduce task ambiguity
- add examples of desired findings
- ensure the model endpoint supports the chosen format well

## 7. How To Build Better Training Data

If you want an RL environment to actually learn, the biggest bottleneck is data quality.

You need more than just three final benchmark tasks. You need trajectories, partial attempts, and failure examples.

### Data Types You Should Collect

#### A. Gold Task Rubrics

For each task, store:

- code snippet
- hidden reference findings
- severity
- category
- expected line numbers
- good recommendations

This is already partially represented by `server/task_bank.py`.

#### B. Positive Demonstrations

Create solved examples where the review is high quality.

Each example should include:

- task code
- one or more strong findings
- strong rationales
- strong recommendations
- optional patch
- final score

This helps supervised warm-start and behavior cloning.

#### C. Partial Trajectories

This is important for RL.

Store intermediate attempts like:

- first attempt finds one issue
- second attempt adds another issue
- third attempt finalizes

This is what teaches agents to improve over time, not just emit one final perfect answer.

#### D. Negative Examples

You should also store:

- false positives
- irrelevant complaints
- duplicate findings
- hallucinated issues
- weak recommendations

Why:
- the reward function penalizes these
- the model must learn precision, not just recall

#### E. Hint Usage Examples

Store trajectories where:

- the agent requests a hint
- then improves its findings

This teaches policy behavior around when hints are worth the penalty.

#### F. Patch Examples

For tasks where patch quality matters, store:

- original code
- weak patch
- good patch
- patch score

This helps the model learn that code edits should remove actual problems, not just change formatting.

## 8. Recommended Dataset Format

Use JSONL so it is easy to stream and train on.

### Benchmark Task Record

```json
{
  "task_id": "py-review-easy",
  "difficulty": "easy",
  "code": "def load_settings(config_text):\n    return eval(config_text)",
  "reference_findings": [
    {
      "rule_id": "avoid-eval",
      "line": 2,
      "category": "security",
      "severity": "critical"
    }
  ]
}
```

### Trajectory Record

```json
{
  "task_id": "py-review-medium",
  "episode_id": "abc123",
  "steps": [
    {
      "observation_feedback": "Review the Python snippet.",
      "action": {
        "operation": "submit_findings",
        "findings": [
          {
            "title": "Mutable default argument leaks state",
            "line": 1,
            "category": "bug",
            "severity": "warning"
          }
        ]
      },
      "reward": 0.35,
      "score": 0.35
    },
    {
      "observation_feedback": "Matched 1 new rubric item(s): mutable-default-list",
      "action": {
        "operation": "finalize",
        "findings": [
          {
            "title": "Mutable default argument leaks state",
            "line": 1,
            "category": "bug",
            "severity": "warning"
          },
          {
            "title": "Bare except hides failures",
            "line": 12,
            "category": "maintainability",
            "severity": "warning"
          }
        ]
      },
      "reward": 0.27,
      "score": 0.62
    }
  ]
}
```

## 9. How To Make RL Learn Better

### A. Add More Tasks

Three tasks are enough for the minimum requirement, but not enough for strong training.

You should expand with:

- file I/O bugs
- API misuse
- SQL injection
- unsafe deserialization
- concurrency issues
- caching mistakes
- resource leaks
- logic edge cases

Target:

- 50 to 200 deterministic tasks
- grouped by difficulty and domain

### B. Add More Partial Reward Signals

Current reward is already better than binary success/fail, but you can improve it.

Possible additions:

- small bonus when the first critical issue is found early
- higher reward for critical issues than style issues
- bonus when rationale quality is high
- bonus when recommendation mentions a correct mitigation pattern
- penalty if line numbers are missing when they should be known

### C. Improve Context In Observation

Right now the observation already gives:

- task metadata
- previous feedback
- submitted findings
- attempts remaining

You can improve learning further by including:

- a short list of matched findings so far
- a short list of remaining categories not yet covered
- normalized review rubric hints without leaking answers
- last action summary

This helps the agent reason about what it already did and what is still missing.

### D. Separate Training Tasks From Benchmark Tasks

Important:

- training tasks should be large and varied
- benchmark tasks should stay hidden and fixed

Do not train directly on the same exact benchmark set you plan to judge on.

### E. Add Preference Data

You can train preference models on:

- strong vs weak findings
- precise vs vague recommendations
- useful vs noisy patches

This is valuable for ranking quality beyond exact rubric matches.

## 10. Functional Requirements Mapping

Here is how your environment should be judged against the stated requirements.

### Requirement: Real-World Task Simulation

Status:
- satisfied in direction

Why:
- code review is a genuine engineering task

How to improve further:
- expand beyond tiny snippets into multi-function modules
- include operational and maintainability review, not just security lints

### Requirement: OpenEnv Spec Compliance

Status:
- mostly implemented in code

Implemented pieces:
- typed action model
- typed observation model
- `reset()`
- `step()`
- `state`
- `openenv.yaml`
- FastAPI/OpenEnv routes

What you still need to verify:
- `openenv validate`
- schema compatibility under your installed OpenEnv version

### Requirement: Minimum 3 Tasks With Agent Graders

Status:
- implemented

You have:
- easy
- medium
- hard
- deterministic grader returning `0.0` to `1.0`

### Requirement: Meaningful Reward Function

Status:
- implemented

Current reward signals:
- new rubric matches
- false positive penalties
- duplicate penalties
- hint penalties
- patch bonus
- finalize pass bonus

### Requirement: Baseline Inference Script

Status:
- implemented

Current `inference.py`:
- uses OpenAI client
- reads env vars
- runs tasks
- writes report

What to verify:
- actual runtime under 20 minutes
- reproducible output with your chosen model endpoint

### Requirement: HF Spaces + Docker

Status:
- code is prepared

You still need to verify:

- `docker build -f server/Dockerfile .`
- local container startup
- `openenv push`
- `/health` returns 200 on the deployed Space

## 11. Recommended Manual Validation Checklist

Before submission, run these in order:

1. Start server locally
2. Hit `/health`
3. Hit `/docs`
4. Test `/tasks`
5. Test `/review` with unsafe examples
6. Test `/reset`
7. Test `/step` with partial findings
8. Test `/step` with finalize
9. Test `/tasks/{task_id}/grade`
10. Run `pytest`
11. Run `openenv validate`
12. Run `python inference.py`
13. Build Docker image
14. Deploy to Hugging Face Space
15. Re-test `/health` and `/reset` on the live Space

## 12. Suggested Immediate Next Steps

If you want the environment to become stronger quickly, do this next:

1. Add 10 to 20 more benchmark-style tasks in `server/task_bank.py`
2. Save solved and failed trajectories as JSONL files under a new `dataset/` directory
3. Refactor custom route state so `/history` and OpenEnv `/step` share one coherent session story
4. Run `openenv validate`
5. Run `inference.py` against your local server and inspect the report

## 13. Quick Commands Summary

Start server:

```powershell
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Open docs:

```text
http://127.0.0.1:8000/docs
```

Run example tests:

```powershell
python -m pytest tests -q
```

Run inference locally:

```powershell
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4.1-mini"
$env:OPENAI_API_KEY="your_key"
$env:ENV_BASE_URL="http://127.0.0.1:8000"
python inference.py
```

Validate OpenEnv:

```powershell
openenv validate
```

Build Docker:

```powershell
docker build -t python_env-env:latest -f server/Dockerfile .
```

Deploy:

```powershell
openenv push
```

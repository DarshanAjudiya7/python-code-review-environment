# Testing Guide

This document lists the environment variables you may need, the available routes, which params are required, and how to test each route quickly.

## 1) Environment Variables

## Server runtime variables

Use these when running the FastAPI app (local or container):

- HOST: default 0.0.0.0 in Docker, localhost in app main()
- PORT: default 8000
- WORKERS: default 1 (used by container command)
- MAX_CONCURRENT_ENVS: default 32

Minimal local run on Windows PowerShell:

```powershell
$env:HOST = "127.0.0.1"
$env:PORT = "8000"
$env:MAX_CONCURRENT_ENVS = "32"
uvicorn server.app:app --host $env:HOST --port $env:PORT
```

## Inference script variables

Required:

- API_BASE_URL
- MODEL_NAME
- HF_TOKEN or OPENAI_API_KEY

Optional:

- ENV_BASE_URL (if omitted, inference.py launches from Docker image)
- PYTHON_ENV_IMAGE (default python_env-env:latest)
- MAX_STEPS (default 3)
- MAX_TASKS (default 3)
- INFERENCE_REPORT_PATH (default inference_results.json)
- TEMPERATURE (default 0)
- MAX_TOKENS (default 900)

Example:

```powershell
$env:API_BASE_URL = "https://api.openai.com/v1"
$env:MODEL_NAME = "gpt-4.1-mini"
$env:OPENAI_API_KEY = "<your-key>"
$env:ENV_BASE_URL = "http://127.0.0.1:8000"
python inference.py
```

## 2) Task IDs You Can Use

- py-review-easy
- py-review-medium
- py-review-hard

## 3) Route Testing (Params + Examples)

Base URL:

```text
http://127.0.0.1:8000
```

## OpenEnv routes

### POST /reset

- Required params: none
- Body: none

Test:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/reset"
```

### POST /step

- Required params: none in query/path
- Required body shape:
  - operation: one of submit_findings, request_hint, finalize
  - findings: array (can be empty)
- Optional body fields:
  - patched_code: string or null
  - note: string or null

Minimal body example:

```json
{
  "operation": "request_hint",
  "findings": []
}
```

Test:

```powershell
$body = @{
  operation = "request_hint"
  findings  = @()
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/step" -ContentType "application/json" -Body $body
```

Example with finding:

```powershell
$body = @{
  operation = "submit_findings"
  findings = @(
    @{
      title = "Avoid eval on untrusted input"
      line = 2
      category = "security"
      severity = "critical"
      rationale = "eval can execute attacker-controlled code"
      recommendation = "Use json.loads instead"
      rule_id = "avoid-eval"
    }
  )
  patched_code = $null
  note = "first pass"
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/step" -ContentType "application/json" -Body $body
```

### GET /state

- Required params: none

Test:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/state"
```

### GET /schema

- Required params: none

Test:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/schema"
```

### WS /ws

- Use a websocket client to connect.
- No route params required.

## Custom REST routes

### GET /health

- Required params: none

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
```

### GET /tasks

- Required params: none

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/tasks"
```

### GET /tasks/{task_id}

- Required path param: task_id

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/tasks/py-review-easy"
```

### POST /tasks/{task_id}/grade

- Required path param: task_id
- Body uses PythonReviewAction shape
  - operation defaults to submit_findings if omitted
  - findings array accepted
  - patched_code optional
  - note optional

```powershell
$body = @{
  findings = @(
    @{
      title = "Avoid eval on untrusted input"
      line = 2
      category = "security"
      severity = "critical"
      rationale = "eval executes arbitrary code"
      recommendation = "Use json.loads"
    }
  )
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/tasks/py-review-easy/grade" -ContentType "application/json" -Body $body
```

### POST /review

- Required body field:
  - code: string
- Optional body field:
  - context: string

```powershell
$body = @{
  code = "def f(x):`n    return eval(x)`n"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/review" -ContentType "application/json" -Body $body
```

### GET /history

- Required params: none

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/history"
```

### DELETE /history

- Required params: none

```powershell
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8000/history"
```

### GET /config

- Required params: none

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/config"
```

### PUT /config

- Required params: none
- Body: PythonEnvConfig object
- All fields have defaults, so {} is valid for a reset-like update

Minimal test:

```powershell
Invoke-RestMethod -Method Put -Uri "http://127.0.0.1:8000/config" -ContentType "application/json" -Body "{}"
```

Full body example:

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

Invoke-RestMethod -Method Put -Uri "http://127.0.0.1:8000/config" -ContentType "application/json" -Body $body
```

## 4) Quick Validation Commands

Run automated tests:

```powershell
pytest -q
```

Run only API tests:

```powershell
pytest -q tests/test_api.py
```

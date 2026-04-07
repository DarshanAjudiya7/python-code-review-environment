---
title: Python Code Review Environment Server
sdk: docker
app_port: 8000
base_path: /web
pinned: false
tags:
  - openenv
  - code-review
---

# python_code_review_env

`python_code_review_env` is a production-grade OpenEnv benchmark that simulates a real Python code review workflow. An agent inspects broken or suboptimal Python code, performs structured actions, receives shaped rewards, and submits fixes against deterministic graders.

## Canonical Layout

The canonical implementation lives under `python_code_review_env/envs/python_env_env/`.

```text
python_code_review_env/
├── envs/
│   └── python_env_env/
│       ├── client.py
│       ├── models.py
│       ├── openenv.yaml
│       ├── graders/
│       ├── server/
│       └── tasks/
├── server/                  # compatibility shims
├── tasks/                   # compatibility shims
├── graders/                 # compatibility shims
├── inference.py
├── openenv.yaml
└── README.md
```

Root-level modules are compatibility wrappers so existing validators, Docker entrypoints, and imports continue to work.

## Tasks

1. `syntax-fix-easy`
   Fix a syntax-broken username normalizer.
2. `bug-fix-medium`
   Repair invoice discount logic using visible and hidden tests.
3. `optimization-hard`
   Refactor duplicate removal to an O(n) implementation while preserving order and style.

## Observation Schema

```json
{
  "task_description": "...",
  "current_code": "...",
  "errors": "...",
  "test_results": "...",
  "history": []
}
```

Additional fields include `task_id`, `difficulty`, `visible_tests`, `attempts_remaining`, `score`, `reward_details`, and inherited OpenEnv observation metadata such as `reward` and `done`.

## Action Schema

```json
{
  "action_type": "edit_code",
  "code": "...updated python code...",
  "reasoning": "brief explanation"
}
```

Supported `action_type` values:

- `analyze_code`
- `edit_code`
- `run_tests`
- `submit_solution`

## Reward Design

The environment uses deterministic, non-binary reward shaping:

- `+0.2` when code first becomes compilable
- up to `+0.3` for visible test progress
- up to `+0.1` for AST/style quality improvement
- `+0.5` for a fully correct submitted solution
- `-0.1` for invalid actions
- `-0.2` for execution timeout / infinite-loop style failures

Per-step reward is always clamped to `[-1.0, 1.0]`. Episode score is always in `[0.0, 1.0]`.

## Deterministic Graders

- Syntax task: compile check plus diff-based partial credit
- Bug-fix task: deterministic `pytest` execution over generated assertion tests
- Optimization task: correctness + runtime benchmark + AST quality + style score

## Local Usage

Install:

```bash
pip install .
```

Run the API server:

```bash
python -m server.app
```

Validate the environment:

```bash
openenv validate
```

Run the baseline inference policy:

```bash
python inference.py
```

Example stdout:

```text
[START] task=syntax-fix-easy difficulty=easy
[STEP] task=syntax-fix-easy step=1 action=analyze_code reward=0.0000 score=0.8125 done=false
[STEP] task=syntax-fix-easy step=2 action=edit_code reward=0.3000 score=1.0000 done=false
[STEP] task=syntax-fix-easy step=3 action=submit_solution reward=0.5000 score=1.0000 done=true
[END] task=syntax-fix-easy score=1.0000 steps=3
Task 1 Score: 1.0000
Final Score: 1.0000
```

## OpenAI-Compatible Model Support

`inference.py` uses:

```python
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
```

This supports:

- Gemini via OpenAI-compatible base URL
- OpenRouter
- Together AI
- DeepSeek

If no explicit provider config is supplied, `inference.py` falls back to a deterministic local baseline policy so evaluation does not depend on paid APIs.

## Docker

Build:

```bash
docker build -f server/Dockerfile -t python_code_review_env .
```

Run:

```bash
docker run -p 8000:8000 python_code_review_env
```

Health check:

```bash
curl http://localhost:8000/health
```

Standard OpenEnv endpoints are available at:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`

Additional task helpers:

- `GET /tasks`
- `GET /tasks/{task_id}`
- `POST /tasks/{task_id}/grade`

## Hugging Face Spaces Deployment

1. Create a Docker Space.
2. Push this repository.
3. Hugging Face builds using `server/Dockerfile`.
4. Confirm `GET /health` and `POST /reset` both return `200`.

The environment is CPU-friendly and intended to run within a standard `2 vCPU / 8 GB RAM` Hugging Face Space.

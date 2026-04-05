# Python Code Review Environment 🐍

A production-grade OpenEnv environment for Python code review, repair, and optimization tasks. This environment simulates real-world developer workflows where an AI agent reviews, fixes, and improves Python code.

## Overview

**`python_code_review_env`** is a deterministic benchmark environment featuring:

- ✅ **3 real-world tasks** with increasing difficulty (Syntax, Bug Fix, Optimization)
- ✅ **Deterministic graders** using AST analysis, pytest execution, and performance benchmarking
- ✅ **OpenAI-compatible API** supporting free/open models (Gemini, DeepSeek, Together, OpenRouter)
- ✅ **Production-ready Docker** deployment for Hugging Face Spaces
- ✅ **Structured Observations & Actions** following OpenEnv spec
- ✅ **Rich reward shaping** with bonuses for syntax fixes, test passes, and optimization

## Tasks

### 1. 🟢 Easy: Syntax Fixing

**Task ID**: `syntax-fix-easy`

Fix broken Python code with syntax errors.

- **Difficulty**: Easy
- **Goal**: Repair syntax errors to make code compile
- **Starter Code**: Function with missing closing parenthesis
- **Grading**: Compilation check + code similarity to reference
- **Score Range**: 0.0–1.0

### 2. 🟡 Medium: Bug Fixing

**Task ID**: `bug-fix-medium`

Fix logic bugs with visible and hidden test cases.

- **Difficulty**: Medium  
- **Goal**: Repair a logic error in invoice calculation
- **Starter Code**: Function that returns wrong total (returns subtotal instead of discounted)
- **Grading**: Test pass fraction (visible & hidden)
- **Score Range**: 0.0–1.0

### 3. 🔴 Hard: Optimization & Refactoring

**Task ID**: `optimization-hard`

Optimize inefficient code while maintaining correctness.

- **Difficulty**: Hard
- **Goal**: Convert O(n²) duplicate removal to O(n) with set
- **Starter Code**: Slow nested-loop implementation
- **Grading**: 50% correctness + 30% speedup + 15% code quality + 5% style
- **Score Range**: 0.0–1.0
- **Bonus**: Runtime benchmarking against reference implementation

## Quick Start

### Run Locally

```bash
cd python-code-review-env
pip install -r server/requirements.txt
python -m server.app
```

Visit http://localhost:8000/docs for interactive API

### Run with Docker

```bash
docker build -f server/Dockerfile -t python_code_review_env:latest .
docker run -p 8000:8000 python_code_review_env:latest
```

### Run Inference

```bash
python inference.py --model "gpt-3.5-turbo" --base-url "http://localhost:8000/v1"
```

## OpenEnv Specification

### Observation

```json
{
  "task_id": "syntax-fix-easy",
  "difficulty": "easy",
  "task_description": "Fix syntax errors...",
  "current_code": "def normalize_username(raw_name: str) -> str:\n    cleaned = raw_name.strip().lower(\n    ...",
  "errors": "invalid syntax ( line 2, column 40 )",
  "test_results": "Not run yet.",
  "visible_tests": ["normalize_username('  Alice Smith  ') == 'alice_smith'"],
  "history": [],
  "attempts_remaining": 8,
  "score": 0.0,
  "reward": {
    "value": 0.0,
    "reason": "Episode reset."
  }
}
```

### Action

```json
{
  "action_type": "edit_code",
  "code": "def normalize_username(raw_name: str) -> str:\n    cleaned = raw_name.strip().lower()\n    if not cleaned:\n        return \"anonymous\"\n    return cleaned.replace(\" \", \"_\")"
}
```

### Reward Details

- **+0.2**: Syntax fixed (one-time per episode)
- **+0.15**: Passing additional test (cumulative per test)
- **+0.1**: Code quality improvement  
- **+0.5**: Full correctness (100% hidden tests, one-time)
- **-0.1**: Invalid action

## Architecture

```
python_code_review_env/
├── models.py          # Pydantic models (Observation, Action, Reward)
├── server/
│   ├── app.py         # FastAPI server  
│   ├── env.py         # OpenEnv environment
│   ├── Dockerfile     # Docker config
│   └── requirements.txt
├── graders/
│   ├── common.py      # Shared utilities
│   ├── syntax.py      # Syntax/bug graders
│   ├── optimization.py# Optimization grader
│   └── pytest_runner.py
├── tasks/
│   ├── task_bank.py   # 3 deterministic tasks
│   └── __init__.py
├── inference.py       # Baseline evaluation script
├── openenv.yaml       # OpenEnv spec
├── pyproject.toml     # Project metadata
└── README.md
```

## FastAPI Endpoints

- `GET /health` – Health check
- `GET /tasks` – List all tasks
- `GET /tasks/{task_id}` – Get task details
- `POST /tasks/{task_id}/grade` – Grade code offline
- Standard OpenEnv endpoints (`/reset`, `/step`, `/state`)

## Deterministic Graders

### Syntax Fix
```
if code compiles:
  score = 1.0
else:
  score = 0.15 + 0.55 * similarity_to_reference
```

### Bug Fix  
```
score = test_pass_fraction (0.0 to 1.0)
```

### Optimization
```
score = (
  0.5 * test_fraction +
  0.3 * speedup_score +
  0.15 * code_quality +
  0.05 * pep8_style
)
```

## Examples

### Using Python

```python
from server.env import PythonCodeReviewEnvironment
from models import PythonCodeReviewAction

env = PythonCodeReviewEnvironment()
obs = env.reset(task_id="syntax-fix-easy")

action = PythonCodeReviewAction(
    action_type="edit_code",
    code="""def normalize_username(raw_name: str) -> str:
    cleaned = raw_name.strip().lower()
    if not cleaned:
        return "anonymous"
    return cleaned.replace(" ", "_")
"""
)

obs = env.step(action)
print(f"Score: {obs.score}")
print(f"Reward: {obs.reward.value:+.3f}")
```

### Using cURL

```bash
# Check health
curl http://localhost:8000/health

# List tasks
curl http://localhost:8000/tasks

# Grade code
curl -X POST http://localhost:8000/tasks/syntax-fix-easy/grade \
  -H "Content-Type: application/json" \
  -d '{"action_type": "edit_code", "code": "..."}'
```

## Deployment

### Hugging Face Spaces

1. Create Space > Docker
2. Upload files + `server/Dockerfile`
3. Space auto-deploys on CPU
4. Monitor `/health` endpoint

### Local Docker

```bash
docker build -f server/Dockerfile -t python_code_review_env .
docker run -p 8000:8000 \
  -e MAX_CONCURRENT_ENVS=16 \
  python_code_review_env
```

## Performance

- Startup: < 5s
- Reset: < 100ms
- Step: 50ms–3s (depends on action)
- Inference (3 tasks): < 20 minutes
- CPU: Works on 2 vCPU, 8GB RAM

## Validation Checklist

- ✅ 3 deterministic tasks
- ✅ Deterministic graders (AST, pytest, benchmarks)
- ✅ `/health` → 200
- ✅ Scores vary per task (not constant)
- ✅ Docker builds successfully
- ✅ OpenEnv spec compliant
- ✅ Reward shaping working
- ✅ All tests deterministic and reproducible

## License

MIT

---

**Built for production. Deterministic. Deployable. Extensible.**

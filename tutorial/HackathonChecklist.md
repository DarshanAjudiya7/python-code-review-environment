# Hackathon Checklist

This file translates the tutorial folder into a concrete plan for `python_env`.

It is not a generic OpenEnv summary. It is a project-specific checklist showing:

- what the tutorials are teaching
- how this repo maps to those ideas
- what is already done
- what still needs to be finished before submission

## 1. What The Tutorials Mean For This Project

### Tutorial 1: OpenEnv Pattern

Main concept:

- every environment should follow a clean pattern:
  - typed models
  - environment logic
  - client
  - FastAPI/OpenEnv app
  - Docker packaging

How `python_env` maps:

- `models.py`
  typed action/observation/config/evaluation models
- `server/code_review_environment.py`
  environment logic
- `client.py`
  Python client for reset/step/state
- `server/app.py`
  OpenEnv app plus helper routes
- `server/Dockerfile`
  container packaging

Status:

- done

What to keep in mind:

- do not break the OpenEnv contract while adding features
- treat models as the public interface

### Tutorial 2: Deployment

Main concept:

- local development first
- Docker second
- HF Spaces deployment third
- test `/health`, `/reset`, `/docs`, `/ws`

How `python_env` maps:

- local server:
  `uvicorn server.app:app --reload --host 0.0.0.0 --port 8000`
- Docker:
  `docker build -t python_env-env:latest -f server/Dockerfile .`
- Spaces:
  `openenv push`

Status:

- app boots locally
- Dockerfile exists and now supports `HOST`, `PORT`, `WORKERS`, `MAX_CONCURRENT_ENVS`
- live Docker build still needs final verification
- Spaces deployment still needs to be executed and checked

### Tutorial 3: Scaling

Main concept:

- OpenEnv works best with WebSocket sessions
- use environment class/factory instead of a singleton for OpenEnv session handling
- support concurrency with `MAX_CONCURRENT_ENVS`

How `python_env` maps:

- `create_app(PythonEnvironment, PythonReviewAction, PythonReviewObservation, max_concurrent_envs=...)`
- `MAX_CONCURRENT_ENVS` is now read from env vars
- Docker now exposes `MAX_CONCURRENT_ENVS`

Status:

- partially done

Important caveat:

- OpenEnv `/reset` and `/step` use the class-based session model
- custom routes such as `/history` and `/config` still use a singleton helper instance
- this is acceptable for manual tooling, but it is not a perfect unified session model

Recommendation:

- keep it for now if your priority is submission
- refactor only if it starts causing testing confusion

### Tutorial 4: RL Training And Reward Design

Main concept:

- a good RL environment needs:
  - meaningful reward
  - repeated trajectories
  - enough task diversity
  - an inference/training loop

How `python_env` maps:

- reward shaping already exists:
  - matched rubric items
  - false-positive penalties
  - duplicate penalties
  - hint penalties
  - patch bonus
  - finalize bonus
- `inference.py` already provides a baseline model-vs-env loop

Status:

- partially done

Gap:

- 3 tasks are enough for hackathon minimums
- 3 tasks are not enough for serious RL learning

## 2. Current Repo Status

### Strong Areas

- real-world task: code review
- typed Pydantic/OpenEnv models
- deterministic grader
- 3 difficulty levels
- partial-progress reward shaping
- manual routes for health/tasks/review/config/history
- baseline inference script
- docs in `README.md`, `Project.md`

### Weak Areas

- benchmark still small
- Docker image build not fully verified end-to-end
- HF Spaces deployment not yet executed
- `openenv validate` still needs to be run in your actual runtime
- no large trajectory dataset yet
- custom REST state and OpenEnv session state are not fully unified

## 3. What You Need To Do To Be Submission-Ready

### Step 1: Validate Local Server

Run:

```powershell
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Manually verify:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`
- `POST /reset`
- `POST /step`
- `GET /tasks`
- `POST /review`

### Step 2: Run Tests

Run:

```powershell
python -m pytest tests -q
```

You want all tests green before Docker or HF deployment.

### Step 3: Run OpenEnv Validation

Run:

```powershell
openenv validate
```

This is a hard requirement.

If validation fails:

- fix schema mismatch first
- fix route mismatch second
- fix packaging third

### Step 4: Run Baseline Inference

Run:

```powershell
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4.1-mini"
$env:OPENAI_API_KEY="your_key"
$env:ENV_BASE_URL="http://127.0.0.1:8000"
python inference.py
```

You want:

- script completes without crashing
- `inference_results.json` gets written
- all 3 tasks run
- scores are reproducible

### Step 5: Verify Docker

Run:

```powershell
docker build -t python_env-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 python_env-env:latest
```

Then test:

- `GET /health`
- `POST /reset`
- `POST /step`

### Step 6: Deploy To HF Spaces

Run:

```powershell
openenv push
```

Then verify the live Space:

- `/health`
- `/docs`
- `/reset`
- `/web`

## 4. What Will Help You “Win” Instead Of Just “Submit”

Passing minimum requirements is not enough. To be competitive, improve these areas:

### A. Increase Task Diversity

Current:

- 3 benchmark tasks

Target:

- at least 10 to 20 tasks before final submission if possible

Good additions:

- SQL injection review
- unsafe YAML/pickle loading
- file-handle leak
- race-condition style bug
- retry/backoff misuse
- caching bug
- logging/privacy leak
- API timeout handling

### B. Improve Observation Context

Good RL environments provide enough context for the model to improve.

Possible improvements:

- add matched categories so far
- add a short summary of uncovered issue types
- add previous actions in structured form, not just free text
- add rubric coverage signals without leaking exact answers

### C. Collect Trajectories

You need data that shows:

- first attempt
- improved second attempt
- final attempt
- failures
- false positives
- hint usage

This is much more useful than only saving final scores.

### D. Improve Reward Design Carefully

Current reward design is already decent.

Good refinements:

- slightly larger reward for critical security findings
- bonus for correct line numbers
- bonus for high-quality recommendation text
- penalty for vague findings with no rationale

Do not overcomplicate the reward before submission. Stability matters more.

## 5. Recommended Immediate Priority Order

If time is limited, do the work in this order:

1. `pytest`
2. `openenv validate`
3. local inference run
4. Docker build and run
5. HF Space deployment
6. add 5 to 10 more tasks
7. collect trajectory data

## 6. One-Sentence Summary

You are following the correct OpenEnv architecture from the tutorials already; the main remaining work is not redesign, it is validation, deployment verification, and expanding task/data quality so the environment scores well in human review.

"""FastAPI application for the Python code review environment."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

try:
    from compat import create_app
    from models import (
        HealthResponse,
        PythonCodeReviewAction,
        PythonCodeReviewObservation,
        PythonCodeReviewState,
        TaskDescriptor,
        TaskGrade,
    )
except Exception:
    from .compat import create_app
    from .models import (
        HealthResponse,
        PythonCodeReviewAction,
        PythonCodeReviewObservation,
        PythonCodeReviewState,
        TaskDescriptor,
        TaskGrade,
    )
from server.env import PythonCodeReviewEnvironment


try:
    MAX_CONCURRENT_ENVS = max(int(os.getenv("MAX_CONCURRENT_ENVS", "16")), 1)
except Exception:
    MAX_CONCURRENT_ENVS = 16

python_env = PythonCodeReviewEnvironment(verbose=False)
app = create_app(
    PythonCodeReviewEnvironment,
    PythonCodeReviewAction,
    PythonCodeReviewObservation,
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)
router = APIRouter(tags=["python-code-review"])


@router.get("/", include_in_schema=False)
def root() -> HTMLResponse:
    """Serve a small homepage with links and a live demo button."""
    return HTMLResponse(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>python_code_review_env</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 16px; color: #111827; }
    .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin: 16px 0; }
    button { background: #111827; color: white; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
    a { color: #2563eb; text-decoration: none; }
    pre { background: #f3f4f6; padding: 16px; border-radius: 10px; overflow: auto; white-space: pre-wrap; }
    .muted { color: #6b7280; }
  </style>
</head>
<body>
  <h1>python_code_review_env</h1>
  <p class="muted">This Space is an API environment. Use the live demo below or open the API docs.</p>
  <div class="card">
    <p><a href="/docs">Open API Docs</a></p>
    <p><a href="/health">Health</a></p>
    <p><a href="/tasks">Tasks</a></p>
    <p><a href="/schema">Schema</a></p>
  </div>
  <div class="card">
    <h3>Live Demo</h3>
    <p>Runs one safe reset + analyze step and shows the actual JSON output.</p>
    <button onclick="runDemo()">Run demo</button>
    <pre id="output">Click "Run demo" to see output.</pre>
  </div>
  <script>
    async function runDemo() {
      const output = document.getElementById('output');
      output.textContent = 'Loading...';
      try {
        const response = await fetch('/demo');
        const data = await response.json();
        output.textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        output.textContent = 'Demo failed: ' + String(error);
      }
    }
  </script>
</body>
</html>
        """
    )


@router.get("/web", include_in_schema=False)
@router.get("/web/", include_in_schema=False)
def root_web() -> HTMLResponse:
    """Serve the same homepage for Hugging Face Spaces base-path requests."""
    return root()


@router.get("/demo", include_in_schema=False)
def demo() -> dict:
    """Return a live demo payload so users can see actual environment output."""
    demo_env = PythonCodeReviewEnvironment(verbose=False)
    observation = demo_env.reset(task_id="syntax-fix-easy")
    next_observation = demo_env.step(PythonCodeReviewAction(action_type="analyze_code"))
    return {
        "reset": observation.model_dump(),
        "step": next_observation.model_dump(),
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint for deployment monitoring."""
    return python_env.health()


@router.get("/tasks", response_model=list)
def list_tasks() -> list:
    """List all available deterministic tasks."""
    return python_env.list_task_summaries()


@router.get("/tasks/{task_id}", response_model=object)
def get_task(task_id: str) -> object:
    """Get a specific task by ID."""
    try:
        return python_env.get_task(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/tasks/{task_id}/grade", response_model=TaskGrade)
def grade_task(task_id: str, payload: PythonCodeReviewAction) -> TaskGrade:
    """Grade code submission for a task without running an episode."""
    if payload.action_type != "edit_code" or not payload.code:
        raise HTTPException(
            status_code=400, 
            detail="Requires action_type='edit_code' with code parameter."
        )
    try:
        return python_env.grade_task_submission(task_id=task_id, code=payload.code)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/state", response_model=PythonCodeReviewState)
def get_state_post() -> RedirectResponse:
    """Redirect POST /state to GET for compatibility."""
    return RedirectResponse(url="/state", status_code=303)


app.include_router(router)


def _prioritize_route(path: str, methods: set[str]) -> None:
    """Move a matching custom route ahead of default OpenEnv routes."""
    try:
        for index in range(len(app.router.routes) - 1, -1, -1):
            route = app.router.routes[index]
            route_path = getattr(route, "path", None)
            route_methods = set(getattr(route, "methods", set()) or set())
            if route_path == path and methods.issubset(route_methods):
                app.router.routes.insert(0, app.router.routes.pop(index))
                break
    except Exception:
        pass


_prioritize_route("/health", {"GET"})


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI application with uvicorn."""
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", host),
        port=int(os.getenv("PORT", str(port))),
    )


if __name__ == "__main__":
    main()


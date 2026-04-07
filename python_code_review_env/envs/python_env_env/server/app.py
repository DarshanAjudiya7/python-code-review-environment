"""FastAPI application for the canonical environment."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from openenv.core.env_server.http_server import create_app

from python_code_review_env.envs.python_env_env.models import (
    HealthResponse,
    PythonCodeReviewAction,
    PythonCodeReviewObservation,
    PythonCodeReviewState,
    TaskGrade,
)
from python_code_review_env.envs.python_env_env.server.env import PythonCodeReviewEnvironment


MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "16"))

python_env = PythonCodeReviewEnvironment()
app = create_app(
    PythonCodeReviewEnvironment,
    PythonCodeReviewAction,
    PythonCodeReviewObservation,
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)
router = APIRouter(tags=["python-code-review"])


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return python_env.health()


@router.get("/tasks", response_model=list)
def list_tasks() -> list:
    return python_env.list_task_summaries()


@router.get("/tasks/{task_id}", response_model=object)
def get_task(task_id: str) -> object:
    try:
        return python_env.get_task(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/tasks/{task_id}/grade", response_model=TaskGrade)
def grade_task(task_id: str, payload: PythonCodeReviewAction) -> TaskGrade:
    if payload.action_type != "edit_code" or not payload.code:
        raise HTTPException(
            status_code=400,
            detail="Requires action_type='edit_code' with code parameter.",
        )
    try:
        return python_env.grade_task_submission(task_id=task_id, code=payload.code)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/state", response_model=PythonCodeReviewState)
def get_state_post() -> RedirectResponse:
    return RedirectResponse(url="/state", status_code=303)


app.include_router(router)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", host),
        port=int(os.getenv("PORT", str(port))),
    )


if __name__ == "__main__":
    main()

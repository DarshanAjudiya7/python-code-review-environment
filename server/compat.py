"""Lightweight OpenEnv-compatible helpers for self-contained server builds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


ActT = TypeVar("ActT", bound="Action")
ObsT = TypeVar("ObsT", bound="Observation")
StateT = TypeVar("StateT", bound="State")


def install_openenv_fastmcp_compat() -> None:
    """No-op hook retained for compatibility with the root codepath."""
    return None


class Action(BaseModel):
    """Minimal action base model."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Minimal observation base model."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    done: bool = False
    reward: bool | int | float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    """Minimal state base model."""

    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)
    episode_id: Optional[str] = None
    step_count: int = Field(default=0, ge=0)


class Environment(ABC, Generic[ActT, ObsT, StateT]):
    """Minimal environment interface compatible with this project."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self, transform: Any = None, rubric: Any = None) -> None:
        self.transform = transform
        self.rubric = rubric

    @abstractmethod
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> ObsT:
        """Reset the environment."""

    @abstractmethod
    def step(self, action: ActT, timeout_s: Optional[float] = None, **kwargs: Any) -> ObsT:
        """Execute one step."""

    @property
    @abstractmethod
    def state(self) -> StateT:
        """Return current state."""

    def _reset_rubric(self) -> None:
        """Compatibility no-op."""
        return None


def create_app(
    env_cls: type[Environment[Any, Any, Any]],
    action_model: type[BaseModel],
    observation_model: type[BaseModel],
    max_concurrent_envs: int = 1,
) -> FastAPI:
    """Create a minimal FastAPI app exposing reset/step/state/schema endpoints."""
    app = FastAPI(title="python_code_review_env")
    env = env_cls()
    del observation_model, max_concurrent_envs

    @app.post("/reset")
    def reset(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            observation = env.reset(**(payload or {}))
            dumped = observation.model_dump()
            return {
                "observation": dumped,
                "reward": dumped.get("reward"),
                "done": dumped.get("done", False),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/step")
    def step(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            action_payload = payload.get("action", payload)
            timeout_s = payload.get("timeout_s")
            action = action_model(**action_payload)
            observation = env.step(action, timeout_s=timeout_s)
            dumped = observation.model_dump()
            return {
                "observation": dumped,
                "reward": dumped.get("reward"),
                "done": dumped.get("done", False),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/state")
    def state() -> dict[str, Any]:
        try:
            return env.state.model_dump()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        return {
            "action": action_model.model_json_schema(),
            "observation": observation_model.model_json_schema(),
            "state": env.state.__class__.model_json_schema(),
        }

    return app

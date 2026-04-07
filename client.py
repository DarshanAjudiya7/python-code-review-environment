"""Compatibility wrapper for the canonical environment client."""

from python_code_review_env.envs.python_env_env.client import (  # noqa: F401
    CodeReviewEnvClient as CodeReviewEnv,
    PythonEnvClient as PythonEnv,
)

MyEnv = PythonEnv

__all__ = ["PythonEnv", "CodeReviewEnv", "MyEnv"]

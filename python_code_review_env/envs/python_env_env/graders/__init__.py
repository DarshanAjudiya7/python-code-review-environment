"""Deterministic graders for the canonical environment."""

from python_code_review_env.envs.python_env_env.graders.syntax import (
    grade_bug_fix_task,
    grade_syntax_task,
    grade_task,
)

__all__ = ["grade_task", "grade_syntax_task", "grade_bug_fix_task"]

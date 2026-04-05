"""Compatibility shim for older imports."""

try:
    from server.code_review_environment import CodeReviewEnvironment
except ModuleNotFoundError:  # pragma: no cover
    from .code_review_environment import CodeReviewEnvironment


__all__ = ["CodeReviewEnvironment"]

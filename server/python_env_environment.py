"""Compatibility shim for older imports."""

try:
    from server.code_review_environment import PythonEnvironment
except ModuleNotFoundError:  # pragma: no cover
    from .code_review_environment import PythonEnvironment


__all__ = ["PythonEnvironment"]

"""Compatibility wrapper for the canonical FastAPI application."""

from python_code_review_env.envs.python_env_env.server.app import app as app
from python_code_review_env.envs.python_env_env.server.app import main as _canonical_main


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    _canonical_main(host=host, port=port)


__all__ = ["app", "main"]

if __name__ == "__main__":
    main()

"""Compatibility helpers for OpenEnv and FastMCP runtime drift."""

from __future__ import annotations

import sys
import types
from typing import Any, Optional


def install_openenv_fastmcp_compat() -> None:
    """Patch FastMCP API differences so older OpenEnv builds keep importing."""
    try:
        import fastmcp  # type: ignore
    except Exception:
        return

    try:
        if not hasattr(fastmcp, "Client"):
            class CompatClient:
                """Minimal async MCP client used for legacy OpenEnv imports."""

                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    self.args = args
                    self.kwargs = kwargs

                async def __aenter__(self) -> "CompatClient":
                    return self

                async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                    return False

                async def list_tools(self) -> list[Any]:
                    return []

                async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
                    raise RuntimeError(
                        f"MCP client compatibility mode cannot call tool: {tool_name}"
                    )

            fastmcp.Client = CompatClient  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        client_pkg = sys.modules.get("fastmcp.client")
        if client_pkg is None:
            client_pkg = types.ModuleType("fastmcp.client")
            sys.modules["fastmcp.client"] = client_pkg

        client_mod = sys.modules.get("fastmcp.client.client")
        if client_mod is None:
            client_mod = types.ModuleType("fastmcp.client.client")
            sys.modules["fastmcp.client.client"] = client_mod

        if not hasattr(client_mod, "CallToolResult"):
            class CallToolResult:
                """Compatibility container for legacy OpenEnv response handling."""

                def __init__(
                    self,
                    content: Any = None,
                    structured_content: Any = None,
                    meta: Any = None,
                    data: Any = None,
                    is_error: bool = False,
                ) -> None:
                    self.content = content
                    self.structured_content = structured_content
                    self.meta = meta
                    self.data = data
                    self.is_error = is_error

            client_mod.CallToolResult = CallToolResult

        client_pkg.client = client_mod  # type: ignore[attr-defined]
    except Exception:
        pass


install_openenv_fastmcp_compat()


try:
    from openenv.core.env_server.http_server import create_app as openenv_create_app
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"OpenEnv runtime import failed after compatibility patch: {exc}") from exc


create_app = openenv_create_app


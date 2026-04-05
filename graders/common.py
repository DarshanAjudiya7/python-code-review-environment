"""Shared deterministic scoring helpers."""

from __future__ import annotations

import ast
import difflib
import traceback
from typing import Tuple


def clamp_score(value: float) -> float:
    """Clamp any scalar score into the required 0..1 interval."""

    return max(0.0, min(1.0, round(value, 6)))


def syntax_error_message(code: str) -> str:
    """Return a concise syntax error string or an empty string."""

    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"{exc.msg} (line {exc.lineno}, column {exc.offset})"
    except Exception:  # pragma: no cover
        return traceback.format_exc(limit=1).strip()
    return ""


def compiles(code: str) -> bool:
    """Return whether the code parses and compiles."""

    try:
        compile(code, "<candidate>", "exec")
    except Exception:
        return False
    return True


def normalized_diff_score(code: str, reference_code: str) -> float:
    """Score textual similarity to the reference solution."""

    ratio = difflib.SequenceMatcher(
        a="".join(code.split()),
        b="".join(reference_code.split()),
    ).ratio()
    return clamp_score(ratio)


def style_score(code: str, max_line_length: int = 88) -> float:
    """Simple deterministic PEP8-inspired style score."""

    lines = code.splitlines() or [""]
    line_length_ok = sum(1 for line in lines if len(line) <= max_line_length) / len(lines)
    tab_ok = 1.0 if all("\t" not in line for line in lines) else 0.0
    trailing_ws_ok = 1.0 if all(line == line.rstrip() for line in lines) else 0.0
    return clamp_score((line_length_ok * 0.6) + (tab_ok * 0.2) + (trailing_ws_ok * 0.2))


def nested_loop_depth(tree: ast.AST) -> int:
    """Return the maximum nested loop depth in the AST."""

    best = 0

    def walk(node: ast.AST, depth: int) -> None:
        nonlocal best
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            depth += 1
            best = max(best, depth)
        for child in ast.iter_child_nodes(node):
            walk(child, depth)

    walk(tree, 0)
    return best


def compile_tree(code: str) -> Tuple[ast.AST | None, str]:
    """Return AST tree and optional parse error."""

    try:
        return ast.parse(code), ""
    except SyntaxError as exc:
        return None, f"{exc.msg} (line {exc.lineno}, column {exc.offset})"

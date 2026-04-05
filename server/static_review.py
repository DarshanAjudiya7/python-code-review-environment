"""Deterministic static-review helpers for arbitrary Python code.

Unlike the benchmark grader, this module does not compare against hidden rubric
items. Instead, it performs direct AST-based review on arbitrary snippets so it
can be used for manual testing, examples, and future dataset generation.
"""

from __future__ import annotations

import ast
from typing import List, Optional

try:
    from models import DirectReviewResponse, ReviewFinding
except ModuleNotFoundError:  # pragma: no cover
    from ..models import DirectReviewResponse, ReviewFinding


class _StaticAnalyzer(ast.NodeVisitor):
    """AST visitor that emits structured review findings.

    The visitor intentionally focuses on a small set of high-signal patterns so
    the direct-review endpoint stays predictable and easy to understand.
    """

    def __init__(self) -> None:
        self.issues: List[ReviewFinding] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        """Flag mutable default arguments in function definitions."""

        for default in list(node.args.defaults):
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.issues.append(
                    ReviewFinding(
                        title="Mutable default argument",
                        line=getattr(default, "lineno", node.lineno),
                        category="bug",
                        severity="warning",
                        rationale=(
                            "Mutable defaults persist across calls and can leak state "
                            "between unrelated requests."
                        ),
                        recommendation="Use None as the default and create the object inside the function.",
                        rule_id="mutable-default-list",
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Inspect function calls for obviously unsafe or noisy patterns."""

        func_name = self._call_name(node)
        if func_name in {"eval", "exec"}:
            self.issues.append(
                ReviewFinding(
                    title=f"Avoid {func_name} on untrusted input",
                    line=node.lineno,
                    category="security",
                    severity="critical",
                    rationale=(
                        f"{func_name} executes arbitrary code and is unsafe on "
                        "user-controlled input."
                    ),
                    recommendation="Use a safe parser or a whitelist-based evaluator.",
                    rule_id="avoid-eval" if func_name == "eval" else "avoid-exec",
                )
            )
        if func_name.endswith("check_output") or func_name.endswith("run"):
            for keyword in node.keywords:
                # `shell=True` is only a problem when the command comes from a
                # shell-parsed string, but this heuristic is high value for
                # review and intentionally conservative.
                if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    self.issues.append(
                        ReviewFinding(
                            title="shell=True with dynamic input",
                            line=node.lineno,
                            category="security",
                            severity="critical",
                            rationale=(
                                "shell=True executes through the shell and can allow "
                                "command injection when the command string is interpolated."
                            ),
                            recommendation="Pass a list of arguments and keep shell=False.",
                            rule_id="shell-true-command-injection",
                        )
                    )
        if func_name == "print":
            self.issues.append(
                ReviewFinding(
                    title="Print statement in application logic",
                    line=node.lineno,
                    category="style",
                    severity="info",
                    rationale="Production services should prefer structured logging over print statements.",
                    recommendation="Use the logging module or return the value to the caller.",
                    rule_id="print-statement",
                )
            )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        """Flag bare exception handlers that hide failures."""

        if node.type is None:
            self.issues.append(
                ReviewFinding(
                    title="Bare except",
                    line=node.lineno,
                    category="maintainability",
                    severity="warning",
                    rationale="Bare except catches KeyboardInterrupt and other system-level exceptions.",
                    recommendation="Catch a specific exception and record the failure.",
                    rule_id="bare-except",
                )
            )
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        """Look for list-membership checks nested in loops."""

        for child in ast.walk(node):
            if isinstance(child, ast.Compare) and any(
                isinstance(operator, (ast.In, ast.NotIn)) for operator in child.ops
            ):
                if isinstance(child.comparators[0], ast.Name):
                    self.issues.append(
                        ReviewFinding(
                            title="Potential quadratic membership check inside loop",
                            line=child.lineno,
                            category="performance",
                            severity="warning",
                            rationale=(
                                "Repeated membership checks against a list inside a loop "
                                "can degrade to quadratic runtime."
                            ),
                            recommendation="Use a set or dict for O(1) membership checks.",
                            rule_id="quadratic-membership-check",
                        )
                    )
                    break
        self.generic_visit(node)

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        """Extract a dotted function name such as `subprocess.run`."""

        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            prefix = _StaticAnalyzer._attribute_prefix(func.value)
            return f"{prefix}.{func.attr}" if prefix else func.attr
        return ""

    @staticmethod
    def _attribute_prefix(node: ast.AST) -> str:
        """Reconstruct the left-hand side of an attribute chain."""

        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = _StaticAnalyzer._attribute_prefix(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""


def analyze_python_code(code: str) -> List[ReviewFinding]:
    """Analyze arbitrary Python code and return structured findings."""

    if not code.strip():
        return [
            ReviewFinding(
                title="No code provided",
                category="bug",
                severity="warning",
                rationale="The reviewer cannot inspect an empty submission.",
                recommendation="Provide Python source code.",
                rule_id="empty-input",
            )
        ]

    # Syntax errors are turned into findings rather than exceptions so API
    # consumers always get a valid response shape.
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [
            ReviewFinding(
                title="Syntax error",
                line=exc.lineno,
                category="bug",
                severity="critical",
                rationale=exc.msg,
                recommendation="Fix the syntax error before running static review.",
                rule_id="syntax-error",
            )
        ]

    analyzer = _StaticAnalyzer()
    analyzer.visit(tree)
    return _deduplicate(analyzer.issues)


def build_direct_review_response(
    code: str, context: Optional[str] = None
) -> DirectReviewResponse:
    """Build the public direct-review response for the `/review` route."""

    issues = analyze_python_code(code)
    weighted_penalty = 0.0
    # The direct-review score is intentionally simple: more severe issues lower
    # the score more aggressively.
    for issue in issues:
        if issue.severity == "critical":
            weighted_penalty += 0.3
        elif issue.severity == "warning":
            weighted_penalty += 0.15
        else:
            weighted_penalty += 0.05

    score = max(0.0, min(1.0, 1.0 - weighted_penalty))
    summary = _build_summary(issues, context)
    improved_code = _suggest_improved_code(code, issues)
    return DirectReviewResponse(
        issues=issues,
        summary=summary,
        score=score,
        improved_code=improved_code,
    )


def _build_summary(issues: List[ReviewFinding], context: Optional[str]) -> str:
    """Create a concise human-readable summary for the direct-review response."""

    if not issues:
        base = "No obvious issues were detected by the deterministic reviewer."
    else:
        critical = sum(1 for issue in issues if issue.severity == "critical")
        warnings = sum(1 for issue in issues if issue.severity == "warning")
        infos = sum(1 for issue in issues if issue.severity == "info")
        base = (
            f"Detected {len(issues)} issue(s): {critical} critical, "
            f"{warnings} warning, {infos} info."
        )
    if context:
        return f"{base} Context: {context}"
    return base


def _suggest_improved_code(code: str, issues: List[ReviewFinding]) -> Optional[str]:
    """Append high-level fix directions to the submitted code."""

    if not issues:
        return None
    suggestions = [issue.recommendation for issue in issues if issue.recommendation]
    comment = " | ".join(dict.fromkeys(suggestions))
    return f"{code.rstrip()}\n\n# Suggested review directions: {comment}"


def _deduplicate(findings: List[ReviewFinding]) -> List[ReviewFinding]:
    """Drop duplicate findings that refer to the same rule and line."""

    seen = set()
    unique: List[ReviewFinding] = []
    for finding in findings:
        key = (finding.rule_id, finding.line, finding.category)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique

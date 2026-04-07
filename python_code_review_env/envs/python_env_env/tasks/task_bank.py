"""Deterministic task bank for python_code_review_env."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from python_code_review_env.envs.python_env_env.models import (
    Difficulty,
    TaskDescriptor,
    TaskKind,
)


@dataclass(frozen=True)
class TaskSpec:
    """Complete task definition plus grading metadata."""

    task_id: str
    title: str
    difficulty: Difficulty
    task_kind: TaskKind
    task_description: str
    starter_code: str
    reference_code: str
    visible_tests: List[str]
    hidden_tests: List[str]
    max_steps: int = 10
    benchmark_entrypoint: Optional[str] = None
    benchmark_input_expr: Optional[str] = None
    benchmark_repeats: int = 1
    benchmark_timeout_s: float = 2.0
    style_max_line_length: int = 88
    expected_quality_markers: List[str] = field(default_factory=list)

    def to_descriptor(self) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            title=self.title,
            difficulty=self.difficulty,
            task_kind=self.task_kind,
            task_description=self.task_description,
            starter_code=self.starter_code,
            visible_tests=list(self.visible_tests),
            max_steps=self.max_steps,
        )


TASK_SYNTAX_FIX = TaskSpec(
    task_id="syntax-fix-easy",
    title="Fix a syntax-broken username normalizer",
    difficulty="easy",
    task_kind="syntax_fix",
    task_description=(
        "Repair the syntax error in a username normalizer without changing the intended "
        "behavior. The function should trim whitespace, lowercase text, and replace "
        "spaces with underscores."
    ),
    starter_code="""def normalize_username(raw_name: str) -> str:
    cleaned = raw_name.strip().lower(
    if not cleaned:
        return "anonymous"
    return cleaned.replace(" ", "_")
""",
    reference_code="""def normalize_username(raw_name: str) -> str:
    cleaned = raw_name.strip().lower()
    if not cleaned:
        return "anonymous"
    return cleaned.replace(" ", "_")
""",
    visible_tests=[
        "normalize_username('  Alice Smith  ') == 'alice_smith'",
        "normalize_username('   ') == 'anonymous'",
        "normalize_username('Bob') == 'bob'",
    ],
    hidden_tests=[
        "normalize_username('  HELLO WORLD  ') == 'hello_world'",
        "normalize_username('') == 'anonymous'",
    ],
    max_steps=8,
)


TASK_BUG_FIX = TaskSpec(
    task_id="bug-fix-medium",
    title="Repair invoice discount calculation logic",
    difficulty="medium",
    task_kind="bug_fix",
    task_description=(
        "Fix a logic bug in a billing helper. Keep the public function signature and "
        "validation behavior unchanged, but return the correct discounted total."
    ),
    starter_code="""from typing import Iterable


def calculate_invoice_total(line_items: Iterable[int], discount_percent: int) -> int:
    \"\"\"Calculate invoice total with discount applied.\"\"\"
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("discount_percent must be between 0 and 100")

    subtotal = sum(line_items)
    discounted_total = subtotal - (subtotal * discount_percent // 100)
    return subtotal
""",
    reference_code="""from typing import Iterable


def calculate_invoice_total(line_items: Iterable[int], discount_percent: int) -> int:
    \"\"\"Calculate invoice total with discount applied.\"\"\"
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("discount_percent must be between 0 and 100")

    subtotal = sum(line_items)
    discounted_total = subtotal - (subtotal * discount_percent // 100)
    return discounted_total
""",
    visible_tests=[
        "calculate_invoice_total([1000, 2000], 0) == 3000",
        "calculate_invoice_total([1000, 2000], 50) == 1500",
        "calculate_invoice_total([1000], 10) == 900",
        "calculate_invoice_total([], 0) == 0",
    ],
    hidden_tests=[
        "calculate_invoice_total([100, 200, 300], 25) == 450",
        "calculate_invoice_total([5000], 99) == 50",
        "raises(ValueError, calculate_invoice_total, [1000], -1)",
    ],
    max_steps=10,
)


TASK_OPTIMIZATION = TaskSpec(
    task_id="optimization-hard",
    title="Optimize duplicate removal while preserving order",
    difficulty="hard",
    task_kind="optimization",
    task_description=(
        "Replace an O(n^2) duplicate-removal implementation with an O(n) approach that "
        "preserves order. Keep the function readable, typed, and style compliant."
    ),
    starter_code="""from typing import List, TypeVar


T = TypeVar("T")


def remove_duplicates(items: List[T]) -> List[T]:
    \"\"\"Remove duplicates from a list while preserving order.\"\"\"
    result: List[T] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
""",
    reference_code="""from typing import List, TypeVar


T = TypeVar("T")


def remove_duplicates(items: List[T]) -> List[T]:
    \"\"\"Remove duplicates from a list while preserving order in O(n) time.\"\"\"
    seen: set[T] = set()
    result: List[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
""",
    visible_tests=[
        "remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]",
        "remove_duplicates(['a', 'b', 'a']) == ['a', 'b']",
        "remove_duplicates([]) == []",
        "remove_duplicates([1]) == [1]",
    ],
    hidden_tests=[
        "remove_duplicates([5, 4, 3, 2, 1, 5, 4]) == [5, 4, 3, 2, 1]",
        "remove_duplicates(list('abacabad')) == ['a', 'b', 'c', 'd']",
    ],
    max_steps=10,
    benchmark_entrypoint="remove_duplicates",
    benchmark_input_expr="list(range(1800)) + list(range(1800)) + list(range(900))",
    benchmark_repeats=4,
    benchmark_timeout_s=3.0,
    style_max_line_length=88,
    expected_quality_markers=["seen", "set", "result.append"],
)


TASKS: Dict[str, TaskSpec] = {
    TASK_SYNTAX_FIX.task_id: TASK_SYNTAX_FIX,
    TASK_BUG_FIX.task_id: TASK_BUG_FIX,
    TASK_OPTIMIZATION.task_id: TASK_OPTIMIZATION,
}


def task_ids() -> List[str]:
    return [TASK_SYNTAX_FIX.task_id, TASK_BUG_FIX.task_id, TASK_OPTIMIZATION.task_id]


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASKS:
        raise ValueError(f"Task {task_id!r} not found.")
    return TASKS[task_id]


def list_task_descriptors() -> List[TaskDescriptor]:
    return [get_task(task_id).to_descriptor() for task_id in task_ids()]


def list_task_summaries() -> List[TaskDescriptor]:
    return list_task_descriptors()

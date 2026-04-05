"""Deterministic task bank for Python code review and repair benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import Difficulty, TaskDescriptor, TaskKind


@dataclass(frozen=True)
class TaskSpec:
    """Complete task specification with grading criteria."""

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
    benchmark_builder: Optional[str] = None
    benchmark_repeats: int = 1
    benchmark_timeout_s: float = 2.0
    style_max_line_length: int = 88
    expected_quality_markers: List[str] = field(default_factory=list)

    def to_descriptor(self) -> TaskDescriptor:
        """Convert to public task descriptor."""
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


# ============================================================================
# TASK 1: EASY - Syntax Fixing
# ============================================================================

TASK_SYNTAX_FIX = TaskSpec(
    task_id="syntax-fix-easy",
    title="Fix a syntax-broken username normalizer",
    difficulty="easy",
    task_kind="syntax_fix",
    task_description=(
        "You are reviewing a utility function before merge. The submitted patch left "
        "the function with syntax errors. Repair the code so it compiles and preserves "
        "the intended behavior of trimming, lowercasing, and replacing spaces with underscores."
    ),
    starter_code='''def normalize_username(raw_name: str) -> str:
    cleaned = raw_name.strip().lower(
    if not cleaned:
        return "anonymous"
    return cleaned.replace(" ", "_")
''',
    reference_code='''def normalize_username(raw_name: str) -> str:
    cleaned = raw_name.strip().lower()
    if not cleaned:
        return "anonymous"
    return cleaned.replace(" ", "_")
''',
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

# ============================================================================
# TASK 2: MEDIUM - Bug Fixing with Tests
# ============================================================================

TASK_BUG_FIX = TaskSpec(
    task_id="bug-fix-medium",
    title="Repair invoice discount calculation logic",
    difficulty="medium",
    task_kind="bug_fix",
    task_description=(
        "A billing helper function is returning the wrong amount after applying discounts. "
        "The function signature is correct, but the calculation logic is broken. "
        "Inspect the implementation, run visible tests, and fix the bug so all tests pass. "
        "Do not change the function signature or validation logic."
    ),
    starter_code='''from typing import Iterable


def calculate_invoice_total(line_items: Iterable[int], discount_percent: int) -> int:
    """Calculate invoice total with discount applied.
    
    Args:
        line_items: List of item prices in cents.
        discount_percent: Discount as integer 0-100.
        
    Returns:
        Final invoice total in cents after discount.
        
    Raises:
        ValueError: If discount_percent is outside 0-100 range.
    """
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("discount_percent must be between 0 and 100")

    subtotal = sum(line_items)
    discounted_total = subtotal - (subtotal * discount_percent // 100)
    return subtotal  # BUG: returning subtotal instead of discounted_total
''',
    reference_code='''from typing import Iterable


def calculate_invoice_total(line_items: Iterable[int], discount_percent: int) -> int:
    """Calculate invoice total with discount applied.
    
    Args:
        line_items: List of item prices in cents.
        discount_percent: Discount as integer 0-100.
        
    Returns:
        Final invoice total in cents after discount.
        
    Raises:
        ValueError: If discount_percent is outside 0-100 range.
    """
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("discount_percent must be between 0 and 100")

    subtotal = sum(line_items)
    discounted_total = subtotal - (subtotal * discount_percent // 100)
    return discounted_total
''',
    visible_tests=[
        "calculate_invoice_total([1000, 2000], 0) == 3000",  # No discount
        "calculate_invoice_total([1000, 2000], 50) == 1500",  # 50% off
        "calculate_invoice_total([1000], 10) == 900",  # 10% off
        "calculate_invoice_total([], 0) == 0",  # Empty
    ],
    hidden_tests=[
        "calculate_invoice_total([100, 200, 300], 25) == 450",  # 25% off
        "calculate_invoice_total([5000], 99) == 50",  # 99% off
    ],
    max_steps=10,
)

# ============================================================================
# TASK 3: HARD - Optimization & Code Quality
# ============================================================================

TASK_OPTIMIZATION = TaskSpec(
    task_id="optimization-hard",
    title="Optimize inefficient list duplicate removal",
    difficulty="hard",
    task_kind="optimization",
    task_description=(
        "Code review found that `remove_duplicates` is inefficient for large lists. "
        "The current implementation uses nested loops (O(n²) time). "
        "Optimize it to O(n) using a set-based approach while maintaining order. "
        "Style and code quality also matter: use idiomatic Python, proper types, and clear logic. "
        "All tests must pass, and the optimized version should be measurably faster."
    ),
    starter_code='''from typing import List, TypeVar


T = TypeVar('T')


def remove_duplicates(items: List[T]) -> List[T]:
    """Remove duplicates from list while preserving order.
    
    This implementation is inefficient for large lists.
    
    Args:
        items: List that may contain duplicate elements.
        
    Returns:
        List with duplicates removed, order preserved.
    """
    result = []
    for item in items:
        if item not in result:  # O(n) lookup in list per iteration
            result.append(item)
    return result
''',
    reference_code='''from typing import List, TypeVar


T = TypeVar('T')


def remove_duplicates(items: List[T]) -> List[T]:
    """Remove duplicates from list while preserving order.
    
    Efficient set-based implementation with O(n) time complexity.
    
    Args:
        items: List that may contain duplicate elements.
        
    Returns:
        List with duplicates removed, order preserved.
    """
    seen: set = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
''',
    visible_tests=[
        "remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]",
        "remove_duplicates(['a', 'b', 'a']) == ['a', 'b']",
        "remove_duplicates([]) == []",
        "remove_duplicates([1]) == [1]",
    ],
    hidden_tests=[
        "remove_duplicates([5, 4, 3, 2, 1, 5, 4]) == [5, 4, 3, 2, 1]",
    ],
    max_steps=10,
    benchmark_entrypoint="remove_duplicates",
    benchmark_builder="lambda: list(range(5000)) + list(range(5000))",
    benchmark_repeats=3,
    benchmark_timeout_s=1.0,
    style_max_line_length=88,
    expected_quality_markers=[
        "set",
        "O(n)",
    ],
)

# ============================================================================
# Task Bank Registry
# ============================================================================

TASKS: Dict[str, TaskSpec] = {
    "syntax-fix-easy": TASK_SYNTAX_FIX,
    "bug-fix-medium": TASK_BUG_FIX,
    "optimization-hard": TASK_OPTIMIZATION,
}


def task_ids() -> List[str]:
    """Return all task IDs in deterministic order."""
    return ["syntax-fix-easy", "bug-fix-medium", "optimization-hard"]


def get_task(task_id: str) -> TaskSpec:
    """Get a task by ID."""
    if task_id not in TASKS:
        raise ValueError(f"Task {task_id} not found. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_task_descriptors() -> List[TaskDescriptor]:
    """List all task descriptors."""
    return [get_task(tid).to_descriptor() for tid in task_ids()]


def list_task_summaries() -> List[TaskDescriptor]:
    """List task summaries (alias for descriptors)."""
    return list_task_descriptors()

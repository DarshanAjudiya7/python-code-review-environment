"""Deterministic task bank for the Python code review benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import Difficulty, TaskDescriptor, TaskKind


@dataclass(frozen=True)
class TaskSpec:
    """Full internal task specification."""

    task_id: str
    title: str
    difficulty: Difficulty
    task_kind: TaskKind
    task_description: str
    starter_code: str
    reference_code: str
    visible_tests: List[str]
    hidden_tests: List[str]
    max_steps: int
    benchmark_entrypoint: Optional[str] = None
    benchmark_builder: Optional[str] = None
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


TASKS: Dict[str, TaskSpec] = {
    "syntax-fix-easy": TaskSpec(
        task_id="syntax-fix-easy",
        title="Fix a syntax-broken username normalizer",
        difficulty="easy",
        task_kind="syntax_fix",
        task_description=(
            "You are reviewing a utility function before merge. The submitted patch left "
            "the function with syntax errors. Repair the code so it compiles and preserves "
            "the intended behavior of trimming, lowercasing, and replacing spaces with underscores."
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
        ],
        hidden_tests=[],
        max_steps=6,
    ),
    "bug-fix-medium": TaskSpec(
        task_id="bug-fix-medium",
        title="Repair invoice discount logic",
        difficulty="medium",
        task_kind="bug_fix",
        task_description=(
            "A billing helper now returns the wrong amount after discounting orders. "
            "Inspect the implementation, run the tests, and correct the logic without "
            "breaking the validation behavior."
        ),
        starter_code="""from typing import Iterable


def calculate_invoice_total(line_items: Iterable[int], discount_percent: int) -> int:
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("discount_percent must be between 0 and 100")

    subtotal = sum(line_items)
    discounted_total = subtotal - (subtotal * discount_percent // 100)
    return subtotal
""",
        reference_code="""from typing import Iterable


def calculate_invoice_total(line_items: Iterable[int], discount_percent: int) -> int:
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("discount_percent must be between 0 and 100")

    subtotal = sum(line_items)
    discounted_total = subtotal - (subtotal * discount_percent // 100)
    return discounted_total
""",
        visible_tests=[
            """def test_no_discount_keeps_subtotal():
    from candidate import calculate_invoice_total

    assert calculate_invoice_total([500, 250], 0) == 750
""",
            """def test_discount_is_applied():
    from candidate import calculate_invoice_total

    assert calculate_invoice_total([500, 250], 10) == 675
""",
        ],
        hidden_tests=[
            """def test_invalid_discount_raises():
    import pytest
    from candidate import calculate_invoice_total

    with pytest.raises(ValueError):
        calculate_invoice_total([100], 101)
""",
            """def test_empty_invoice_stays_zero():
    from candidate import calculate_invoice_total

    assert calculate_invoice_total([], 25) == 0
""",
            """def test_whole_number_discount_rounds_down():
    from candidate import calculate_invoice_total

    assert calculate_invoice_total([199, 199, 199], 15) == 508
""",
        ],
        max_steps=8,
    ),
    "optimization-hard": TaskSpec(
        task_id="optimization-hard",
        title="Refactor slow activity aggregation",
        difficulty="hard",
        task_kind="optimization",
        task_description=(
            "A service endpoint aggregates per-user activity counts, but the current implementation "
            "is too slow on production-sized inputs. Refactor it for better performance while keeping "
            "the output stable and the code easy to maintain."
        ),
        starter_code="""from typing import Iterable


def summarize_user_activity(events: Iterable[dict]) -> list[tuple[str, int]]:
    event_list = list(events)
    counts: dict[str, int] = {}

    for user_id in [event["user_id"] for event in event_list]:
        counts[user_id] = 0
        for event in event_list:
            if event["user_id"] == user_id:
                counts[user_id] += 1

    summary: list[tuple[str, int]] = []
    for user_id, total in counts.items():
        summary.append((user_id, total))

    summary.sort(key=lambda item: (-item[1], item[0]))
    return summary
""",
        reference_code="""from collections import Counter
from typing import Iterable


def summarize_user_activity(events: Iterable[dict]) -> list[tuple[str, int]]:
    counts = Counter(event["user_id"] for event in events)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))
""",
        visible_tests=[
            """def test_summary_is_sorted_by_count_then_user():
    from candidate import summarize_user_activity

    events = [
        {"user_id": "bob"},
        {"user_id": "alice"},
        {"user_id": "bob"},
    ]

    assert summarize_user_activity(events) == [("bob", 2), ("alice", 1)]
""",
            """def test_empty_input_returns_empty_list():
    from candidate import summarize_user_activity

    assert summarize_user_activity([]) == []
""",
        ],
        hidden_tests=[
            """def test_duplicate_users_are_grouped_once():
    from candidate import summarize_user_activity

    events = [{"user_id": "a"}, {"user_id": "a"}, {"user_id": "b"}]
    assert summarize_user_activity(events) == [("a", 2), ("b", 1)]
""",
            """def test_tie_breaker_is_lexicographic():
    from candidate import summarize_user_activity

    events = [{"user_id": "d"}, {"user_id": "c"}]
    assert summarize_user_activity(events) == [("c", 1), ("d", 1)]
""",
        ],
        max_steps=10,
        benchmark_entrypoint="summarize_user_activity",
        benchmark_builder="""def build_benchmark_events() -> list[dict]:
    events = []
    for index in range(6000):
        events.append({"user_id": f"user-{index % 250}"})
    return events
""",
        benchmark_repeats=3,
        benchmark_timeout_s=2.0,
        expected_quality_markers=["Counter", "sorted"],
    ),
}


def task_ids() -> List[str]:
    """Return tasks in stable evaluation order."""

    return list(TASKS.keys())


def get_task(task_id: str) -> TaskSpec:
    """Return one task or raise a clear error."""

    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown task_id: {task_id}") from exc


def list_task_descriptors() -> List[TaskDescriptor]:
    """Return full public descriptors."""

    return [TASKS[task_id].to_descriptor() for task_id in task_ids()]


def list_task_summaries() -> List[TaskDescriptor]:
    """Return lightweight task metadata."""

    return list_task_descriptors()

"""Static PR-review tasks and hidden grading rubrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

try:
    from models import Category, Difficulty, Severity, TaskDescriptor, TaskSummary
except ModuleNotFoundError:  # pragma: no cover
    from ..models import Category, Difficulty, Severity, TaskDescriptor, TaskSummary


@dataclass(frozen=True)
class RubricIssue:
    """One hidden issue that can be matched by the deterministic grader."""

    issue_id: str
    file_path: str
    line: int
    category: Category
    severity: Severity
    keywords: Sequence[str]
    min_keyword_hits: int
    weight: float


@dataclass(frozen=True)
class TaskSpec:
    """Complete task definition, including hidden rubric metadata."""

    task_id: str
    difficulty: Difficulty
    title: str
    goal: str
    repo_summary: str
    visible_diff: str
    file_contents: Dict[str, str]
    changed_files: Sequence[str]
    rubric_issues: Sequence[RubricIssue]
    max_steps: int

    @property
    def available_files(self) -> List[str]:
        return list(self.file_contents.keys())

    def to_descriptor(self) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            difficulty=self.difficulty,
            title=self.title,
            goal=self.goal,
            repo_summary=self.repo_summary,
            changed_files=list(self.changed_files),
            available_files=self.available_files,
            max_steps=self.max_steps,
        )

    def to_summary(self) -> TaskSummary:
        return TaskSummary(
            task_id=self.task_id,
            difficulty=self.difficulty,
            title=self.title,
            goal=self.goal,
        )


TASKS: List[TaskSpec] = [
    TaskSpec(
        task_id="py-pr-review-easy",
        difficulty="easy",
        title="Retry Delay Regression",
        goal=(
            "Review the pull request and identify the real bug introduced in the retry "
            "delay helper before it ships."
        ),
        repo_summary=(
            "This service computes retry delays for background notification delivery. "
            "The change is intended to relax validation for legacy callers."
        ),
        visible_diff="\n".join(
            [
                "diff --git a/src/notifications/retry.py b/src/notifications/retry.py",
                "@@",
                "-    if base_delay <= 0:",
                "+    if base_delay < 0:",
                "         return 0.0",
            ]
        ),
        file_contents={
            "src/notifications/retry.py": "\n".join(
                [
                    "from __future__ import annotations",
                    "",
                    "def calculate_retry_delay(attempt: int, base_delay: float = 2.0) -> float:",
                    '    """Return the retry delay in seconds."""',
                    "    if attempt < 0:",
                    '        raise ValueError(\"attempt must be >= 0\")',
                    "    if base_delay < 0:",
                    "        return 0.0",
                    "    return attempt / base_delay",
                ]
            )
        },
        changed_files=("src/notifications/retry.py",),
        rubric_issues=(
            RubricIssue(
                issue_id="zero-base-delay-divides",
                file_path="src/notifications/retry.py",
                line=7,
                category="bug",
                severity="warning",
                keywords=("zero", "division", "base_delay"),
                min_keyword_hits=2,
                weight=1.0,
            ),
        ),
        max_steps=4,
    ),
    TaskSpec(
        task_id="py-pr-review-medium",
        difficulty="medium",
        title="Coupon Billing Rollout",
        goal=(
            "Review the billing change and identify both the production regression and "
            "the missing coverage that would have caught it."
        ),
        repo_summary=(
            "The billing service is adding coupon support for one-off invoices. The PR "
            "touches both the service code and its unit tests."
        ),
        visible_diff="\n".join(
            [
                "diff --git a/app/billing/invoice_service.py b/app/billing/invoice_service.py",
                "@@",
                " def charge_invoice(order: dict, gateway: Gateway) -> str:",
                "-    return gateway.charge(order[\"customer_id\"], order[\"amount_cents\"])",
                "+    total = order[\"amount_cents\"]",
                "+    coupon = order.get(\"coupon_code\")",
                "+    if coupon:",
                "+        discount = gateway.lookup_discount(coupon)",
                "+        total = max(total - discount, 0)",
                "+    return gateway.charge(order[\"customer_id\"], order[\"amount_cents\"])",
                "",
                "diff --git a/tests/test_invoice_service.py b/tests/test_invoice_service.py",
                "@@",
                " class FakeGateway:",
                "+    def lookup_discount(self, coupon: str) -> int:",
                "+        return 250",
            ]
        ),
        file_contents={
            "app/billing/invoice_service.py": "\n".join(
                [
                    "from gateway import Gateway",
                    "",
                    "def charge_invoice(order: dict, gateway: Gateway) -> str:",
                    '    total = order["amount_cents"]',
                    '    coupon = order.get("coupon_code")',
                    "    if coupon:",
                    "        discount = gateway.lookup_discount(coupon)",
                    "        total = max(total - discount, 0)",
                    '    return gateway.charge(order["customer_id"], order["amount_cents"])',
                ]
            ),
            "tests/test_invoice_service.py": "\n".join(
                [
                    "from app.billing.invoice_service import charge_invoice",
                    "",
                    "class FakeGateway:",
                    "    def lookup_discount(self, coupon: str) -> int:",
                    "        return 250",
                    "",
                    "    def charge(self, customer_id: str, amount_cents: int) -> str:",
                    "        self.last_charge = (customer_id, amount_cents)",
                    '        return "charge_123"',
                    "",
                    "def test_charge_invoice_without_coupon():",
                    "    gateway = FakeGateway()",
                    '    charge_invoice({"customer_id": "cus_1", "amount_cents": 1000}, gateway)',
                    '    assert gateway.last_charge == ("cus_1", 1000)',
                ]
            ),
        },
        changed_files=("app/billing/invoice_service.py", "tests/test_invoice_service.py"),
        rubric_issues=(
            RubricIssue(
                issue_id="discount-total-unused",
                file_path="app/billing/invoice_service.py",
                line=8,
                category="bug",
                severity="warning",
                keywords=("discount", "total", "charge", "amount"),
                min_keyword_hits=2,
                weight=0.6,
            ),
            RubricIssue(
                issue_id="missing-coupon-test",
                file_path="tests/test_invoice_service.py",
                line=11,
                category="testing",
                severity="warning",
                keywords=("missing", "test", "coupon", "discount"),
                min_keyword_hits=2,
                weight=0.4,
            ),
        ),
        max_steps=5,
    ),
    TaskSpec(
        task_id="py-pr-review-hard",
        difficulty="hard",
        title="Async Job Runner Deduplication",
        goal=(
            "Review the async job-runner PR and find the subtle concurrency issues "
            "without inventing extra problems."
        ),
        repo_summary=(
            "A shared webhook backfill service is deduplicating in-flight work with an "
            "async task cache and writing the latest result for operators to inspect."
        ),
        visible_diff="\n".join(
            [
                "diff --git a/app/jobs/runner.py b/app/jobs/runner.py",
                "@@",
                " async def run_job(job_id: str, payload: dict, worker) -> str:",
                "     if job_id in ACTIVE_RUNS:",
                "         return await ACTIVE_RUNS[job_id]",
                "+    lock = asyncio.Lock()",
                "+    async with lock:",
                "+        task = asyncio.create_task(worker.run(payload))",
                "+        ACTIVE_RUNS[job_id] = task",
                "     try:",
                "         result = await task",
                "     finally:",
                "         ACTIVE_RUNS.pop(job_id, None)",
                "+    Path(\"latest-result.json\").write_text(result)",
                "     return result",
            ]
        ),
        file_contents={
            "app/jobs/runner.py": "\n".join(
                [
                    "import asyncio",
                    "from pathlib import Path",
                    "",
                    "ACTIVE_RUNS: dict[str, asyncio.Task[str]] = {}",
                    "",
                    "async def run_job(job_id: str, payload: dict, worker) -> str:",
                    "    if job_id in ACTIVE_RUNS:",
                    "        return await ACTIVE_RUNS[job_id]",
                    "",
                    "    lock = asyncio.Lock()",
                    "    async with lock:",
                    "        task = asyncio.create_task(worker.run(payload))",
                    "        ACTIVE_RUNS[job_id] = task",
                    "    try:",
                    "        result = await task",
                    "    finally:",
                    "        ACTIVE_RUNS.pop(job_id, None)",
                    "",
                    '    Path("latest-result.json").write_text(result)',
                    "    return result",
                ]
            ),
            "tests/test_runner.py": "\n".join(
                [
                    "import pytest",
                    "",
                    "from app.jobs.runner import run_job",
                    "",
                    "class FakeWorker:",
                    "    async def run(self, payload: dict) -> str:",
                    '        return payload["job_id"]',
                    "",
                    "@pytest.mark.asyncio",
                    "async def test_run_job_returns_worker_result():",
                    "    worker = FakeWorker()",
                    '    result = await run_job("job-1", {"job_id": "job-1"}, worker)',
                    '    assert result == "job-1"',
                ]
            ),
        },
        changed_files=("app/jobs/runner.py", "tests/test_runner.py"),
        rubric_issues=(
            RubricIssue(
                issue_id="per-call-lock-race",
                file_path="app/jobs/runner.py",
                line=9,
                category="bug",
                severity="warning",
                keywords=("lock", "race", "concurrent", "duplicate"),
                min_keyword_hits=2,
                weight=0.55,
            ),
            RubricIssue(
                issue_id="shared-output-file-race",
                file_path="app/jobs/runner.py",
                line=18,
                category="maintainability",
                severity="warning",
                keywords=("latest", "result", "file", "concurrent", "overwrite"),
                min_keyword_hits=2,
                weight=0.45,
            ),
        ),
        max_steps=6,
    ),
]


TASKS_BY_ID: Dict[str, TaskSpec] = {task.task_id: task for task in TASKS}


def list_task_descriptors() -> List[TaskDescriptor]:
    """Return public descriptors for all tasks."""

    return [task.to_descriptor() for task in TASKS]


def list_task_summaries() -> List[TaskSummary]:
    """Return task summaries for lightweight route responses."""

    return [task.to_summary() for task in TASKS]


def get_task(task_id: str) -> TaskSpec:
    """Return a task by id."""

    try:
        return TASKS_BY_ID[task_id]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown task_id: {task_id}") from exc


def task_ids() -> Iterable[str]:
    """Return task ids in benchmark order."""

    return [task.task_id for task in TASKS]


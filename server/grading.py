"""Deterministic grading helpers for PR-review tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

try:
    from models import ReviewFinding, TaskGrade
    from server.task_bank import RubricIssue, TaskSpec
except ModuleNotFoundError:  # pragma: no cover
    from ..models import ReviewFinding, TaskGrade
    from .task_bank import RubricIssue, TaskSpec


FALSE_POSITIVE_PENALTY = 0.10
DUPLICATE_PENALTY = 0.05


@dataclass(frozen=True)
class FindingMatch:
    """Result of matching one finding against the rubric."""

    issue_id: Optional[str]
    duplicate: bool = False


def finding_fingerprint(finding: ReviewFinding) -> str:
    """Build a deterministic fingerprint for duplicate detection."""

    text = " ".join(
        [
            finding.file_path,
            str(finding.line or 0),
            finding.category,
            finding.severity,
            finding.title,
            finding.explanation,
            finding.suggested_fix,
        ]
    )
    return "|".join(sorted(tokens(text)))


def match_finding(
    finding: ReviewFinding,
    task: TaskSpec,
    matched_issue_ids: Set[str],
    seen_fingerprints: Set[str],
) -> FindingMatch:
    """Match one finding against the remaining rubric issues."""

    fingerprint = finding_fingerprint(finding)
    if fingerprint in seen_fingerprints:
        return FindingMatch(issue_id=None, duplicate=True)

    for issue in task.rubric_issues:
        if issue.issue_id in matched_issue_ids:
            continue
        if finding_matches_issue(finding, issue):
            return FindingMatch(issue_id=issue.issue_id)
    return FindingMatch(issue_id=None)


def finding_matches_issue(finding: ReviewFinding, issue: RubricIssue) -> bool:
    """Return True when a finding deterministically matches a rubric issue."""

    if finding.file_path != issue.file_path:
        return False
    if finding.category != issue.category:
        return False
    if finding.severity != issue.severity:
        return False
    if finding.line is None or abs(finding.line - issue.line) > 2:
        return False

    finding_tokens = tokens(
        " ".join([finding.title, finding.explanation, finding.suggested_fix])
    )
    keyword_hits = sum(1 for keyword in issue.keywords if keyword in finding_tokens)
    return keyword_hits >= issue.min_keyword_hits


def score_task(
    task: TaskSpec,
    matched_issue_ids: Iterable[str],
    false_positives: int = 0,
    duplicate_findings: int = 0,
) -> TaskGrade:
    """Score a task from cumulative episode state."""

    matched_set = set(matched_issue_ids)
    matched_weight = sum(
        issue.weight for issue in task.rubric_issues if issue.issue_id in matched_set
    )
    raw_score = matched_weight
    raw_score -= false_positives * FALSE_POSITIVE_PENALTY
    raw_score -= duplicate_findings * DUPLICATE_PENALTY
    score = max(0.0, min(1.0, round(raw_score, 6)))
    return TaskGrade(
        score=score,
        matched_issue_ids=sorted(matched_set),
        false_positives=false_positives,
        duplicate_findings=duplicate_findings,
        matched_weight=min(1.0, round(matched_weight, 6)),
    )


def grade_findings(task: TaskSpec, findings: Sequence[ReviewFinding]) -> TaskGrade:
    """Offline-grade a batch of findings for one task."""

    matched_issue_ids: Set[str] = set()
    seen_fingerprints: Set[str] = set()
    false_positives = 0
    duplicate_findings = 0

    for finding in findings:
        result = match_finding(
            finding=finding,
            task=task,
            matched_issue_ids=matched_issue_ids,
            seen_fingerprints=seen_fingerprints,
        )
        fingerprint = finding_fingerprint(finding)
        if result.duplicate:
            duplicate_findings += 1
            continue
        seen_fingerprints.add(fingerprint)
        if result.issue_id is None:
            false_positives += 1
            continue
        matched_issue_ids.add(result.issue_id)

    return score_task(
        task=task,
        matched_issue_ids=matched_issue_ids,
        false_positives=false_positives,
        duplicate_findings=duplicate_findings,
    )


def tokens(text: str) -> Set[str]:
    """Normalize free text into deterministic comparison tokens."""

    return set(re.findall(r"[a-z0-9_]+", text.lower()))


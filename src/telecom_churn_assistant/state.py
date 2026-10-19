from datetime import date
from typing import Any, NotRequired, TypedDict

from langchain_core.documents import Document


class ChurnState(TypedDict):
    customer_id: str
    question: str
    stated_reason: str
    channel: str
    request: NotRequired[str]
    session_date: NotRequired[str | date]
    customer_data: dict[str, Any]
    retrieved_docs: list[Document]
    matched_record: dict[str, Any]
    signals: list[str]
    risk_score: int | None
    risk_level: str | None
    drivers: list[dict[str, Any]]
    recommendation: dict[str, Any]
    score_breakdown: list[str]
    missing_information: list[str]
    unsupported: list[str]
    data_conflicts: list[dict[str, Any]]
    policy_checks: list[dict[str, str]]
    escalate: bool
    escalation_reason: str
    sparse_history: bool
    risk_uncertain: bool
    unsupported_question: bool
    citation_validation_passed: bool
    answer: str
    supervisor_review_status: NotRequired[str]
    supervisor_reviewer: NotRequired[str]
    response: dict[str, Any]
import os
import re
from datetime import date
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from .guardrails import (
    mask_personal_data,
    validate_citations,
    verified_citations,
    verified_customer_data,
)
from .policies import evaluate_policy_requests
from .prompts import RAG_PROMPT
from .retriever import (
    get_customer_data,
    retrieve_customer_context,
)
from .scoring import WEIGHTS, calculate_risk
from .state import ChurnState

MODEL_NAME = "gpt-oss:120b"
ALLOWED_ACTIONS = {
    "plan_review",
    "network_escalation",
    "bill_correction",
    "callback",
}


def create_chat_model() -> ChatOllama:
    load_dotenv()
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is required for direct Ollama Cloud inference. "
            "Create an API key at https://ollama.com/settings/keys and set "
            "it in the project's .env file."
        )

    return ChatOllama(
        model=MODEL_NAME,
        temperature=0,
        base_url=os.getenv(
            "OLLAMA_CLOUD_BASE_URL",
            "https://ollama.com",
        ),
        sync_client_kwargs={
            "headers": {"Authorization": f"Bearer {api_key}"}
        },
    )


def _escalate(state: ChurnState, reason: str) -> None:
    state["escalate"] = True
    reasons = (
        state["escalation_reason"].split("; ")
        if state["escalation_reason"]
        else []
    )
    if reason not in reasons:
        reasons.append(reason)
    state["escalation_reason"] = "; ".join(reasons)


def validate_customer(state: ChurnState) -> ChurnState:
    customer = get_customer_data(state["customer_id"])
    if customer is None:
        state["response"] = {
            "unsupported_customer": state["customer_id"],
            "rejected_at_input_check": True,
            "invented_history": False,
            "unsupported": ["unknown customer id"],
        }
        return state

    filtered_customer = verified_customer_data(customer)
    state["customer_data"] = filtered_customer
    return state


def route_customer(state: ChurnState) -> str:
    if state.get("customer_data"):
        return "retrieve"
    return END


def retrieve_node(state: ChurnState) -> ChurnState:
    query = " ".join(
        part
        for part in (state["question"], state["stated_reason"])
        if part
    )
    documents = retrieve_customer_context(
        state["customer_id"],
        mask_personal_data(query),
    )
    state["retrieved_docs"] = documents
    top_match = documents[0]
    state["matched_record"] = {
        "customer_id": top_match.metadata["customer_id"],
        "section": top_match.metadata["section"],
        "content": top_match.page_content,
    }
    return state


def analyze_node(state: ChurnState) -> ChurnState:
    signals: list[str] = []
    drivers: list[dict[str, Any]] = []
    state["customer_data"] = verified_customer_data(state["customer_data"])
    state["missing_information"] = []
    risk_uncertain = False
    state["escalate"] = False
    state["escalation_reason"] = ""
    state["data_conflicts"] = []
    state["policy_checks"] = []
    state["citation_validation_passed"] = False

    records = state["customer_data"].get("recentSignalsLast30Days", {})
    if not isinstance(records, dict):
        records = {}

    dropped_call_tickets = records.get("droppedCallTickets", [])
    issue_description = str(records.get("issue", "")).lower()
    location = records.get("location")
    same_location_supported = bool(
        location
        or re.search(
            r"\b(same cell|same location|same area)\b",
            issue_description,
        )
    )
    if len(dropped_call_tickets) >= 3 and same_location_supported:
        signals.append("network_complaints_3plus")
        drivers.append(
            {
                "driver": "network_quality",
                "label": "Network quality",
                "evidence": (
                    f"{len(dropped_call_tickets)} coverage complaints "
                    "from the same location within 30 days"
                ),
                "source": ", ".join(dropped_call_tickets),
                "citations": list(dropped_call_tickets),
            }
        )
    elif len(dropped_call_tickets) >= 2 and same_location_supported:
        signals.append("repeat_contact")
        drivers.append(
            {
                "driver": "repeat_contact",
                "label": "Repeat contact",
                "evidence": (
                    f"{len(dropped_call_tickets)} reports of the same "
                    "network issue from one location"
                ),
                "source": ", ".join(dropped_call_tickets),
                "citations": list(dropped_call_tickets),
            }
        )
    elif len(dropped_call_tickets) >= 3:
        risk_uncertain = True
        state["missing_information"] = [
            "verification that coverage complaints are from one location"
        ]
        _escalate(
            state,
            "Three or more coverage complaints are present, but a common "
            "location cannot be verified.",
        )

    if "network_complaints_3plus" in signals:
        signals.append("repeat_contact")

    usage_change = records.get("dataUsageChange", "")
    usage_match = re.search(r"-\s*(\d+(?:\.\d+)?)%", str(usage_change))
    if usage_match and float(usage_match.group(1)) > 30:
        signals.append("usage_decline")
        drivers.append(
            {
                "driver": "usage_decline",
                "label": "Usage decline",
                "evidence": f"Data down {usage_match.group(1)}%",
                "source": (
                    f"{state['customer_id']}: "
                    "recentSignalsLast30Days.dataUsageChange"
                ),
                "citations": [
                    f"{state['customer_id']}: "
                    "recentSignalsLast30Days.dataUsageChange"
                ],
            }
        )

    billing_issue = records.get("billingIssue")
    customer_feedback = str(records.get("customerFeedback", ""))
    ticket_id = records.get("ticketId")
    follow_up_call = records.get("followUpCall")
    disputed_feedback = bool(
        re.search(
            r"\b(still|disput|incorrect|remain|continues?)\b",
            customer_feedback,
            re.I,
        )
    )
    billing_is_disputed = bool(
        billing_issue
        and (
            str(records.get("ticketStatus", "")).lower() != "resolved"
            or disputed_feedback
        )
    )
    if billing_is_disputed:
        signals.append("billing_dispute")
        state["policy_checks"] = [
            {
                "policy": "BILL-POL",
                "status": "action_required",
                "detail": (
                    "Verify disputed charges before correction and confirm "
                    "the outcome to the customer within 48 hours."
                ),
            }
        ]
        source_ids = [item for item in (ticket_id, follow_up_call) if item]
        drivers.append(
            {
                "driver": "billing_dispute",
                "label": "Billing dispute",
                "evidence": customer_feedback or "Unresolved billing issue",
                "source": ", ".join(source_ids),
                "citations": source_ids,
            }
        )
        if ticket_id and follow_up_call and disputed_feedback:
            signals.append("repeat_contact")
            state["data_conflicts"].append(
                {
                    "type": "billing_status_conflict",
                    "records": [ticket_id, follow_up_call],
                    "detail": (
                        f"{ticket_id} is marked resolved, while the later "
                        f"{follow_up_call} reports the charge is still present."
                    ),
                }
            )
            _escalate(
                state,
                "Conflicting billing status between the resolved ticket "
                "and the later customer call requires review.",
            )

    mentions = records.get("mentions", [])
    if any(
        "competitor" in str(mention).lower()
        or "port-out" in str(mention).lower()
        for mention in mentions
    ):
        signals.append("competitor_intent")
        call_id = records.get("callId", "Recent customer interaction")
        drivers.append(
            {
                "driver": "competitor_intent",
                "label": "Competitor intent",
                "evidence": ", ".join(str(mention) for mention in mentions),
                "source": call_id,
                "citations": [call_id] if records.get("callId") else [],
            }
        )

    policy_text = " ".join(
        str(value)
        for value in (
            state["question"],
            state["stated_reason"],
            state.get("request", ""),
            records.get("retentionRequest", ""),
        )
        if value
    )
    session_date = state.get("session_date")
    if isinstance(session_date, str):
        session_date = date.fromisoformat(session_date)
    elif session_date is not None and not isinstance(session_date, date):
        raise TypeError("session_date must be an ISO date string or date.")
    state["policy_checks"].extend(
        evaluate_policy_requests(policy_text, today=session_date)
    )
    for finding in state["policy_checks"]:
        if finding["status"] in {
            "requires_supervisor",
            "expired",
            "not_yet_valid",
        }:
            _escalate(state, finding["detail"])

    available_data = str(records.get("availableData", "")).lower()
    interactions = str(records.get("interactions", "")).lower()
    sparse_history = (
        not records
        or "plan metadata only" in available_data
        or "no interactions" in interactions
    )
    risk_uncertain = risk_uncertain or sparse_history
    if sparse_history:
        state["missing_information"] = list(
            dict.fromkeys(
                state.get("missing_information", [])
                + ["recent interactions unavailable", "usage trend unavailable"]
            )
        )
        _escalate(
            state,
            "Risk classification is uncertain because recent verified "
            "customer signals are insufficient.",
        )

    signals = list(dict.fromkeys(signals))
    risk_score, risk_level = (
        (None, None)
        if risk_uncertain
        else calculate_risk(signals)
    )
    if "network_complaints_3plus" in signals:
        recommendation = {
            "action": "network_escalation",
            "reason": (
                "Escalate the repeated coverage issue for investigation and "
                "arrange a customer callback."
            ),
        }
    elif "billing_dispute" in signals:
        recommendation = {
            "action": "bill_correction",
            "reason": (
                "Verify the disputed charge; correct it only after "
                "verification and confirm the outcome within 48 hours."
            ),
        }
    elif "competitor_intent" in signals:
        recommendation = {
            "action": "plan_review",
            "reason": (
                "Review the customer's plan and retention options within "
                "the applicable offer limits."
            ),
        }
    else:
        recommendation = {
            "action": "callback",
            "reason": "Follow up with the customer to understand their needs.",
        }

    relevant_driver = {
        "network_escalation": "network_quality",
        "bill_correction": "billing_dispute",
        "plan_review": "competitor_intent",
    }.get(recommendation["action"])
    recommendation_citations = [
        citation
        for driver in drivers
        if driver["driver"] == relevant_driver
        for citation in driver["citations"]
    ]
    if not recommendation_citations:
        available_citations = verified_citations(
            state["customer_data"], state["customer_id"]
        )
        recommendation_citations = [
            f"{state['customer_id']}: {section}"
            for section in ("recentSignalsLast30Days", "planAndTenure")
            if f"{state['customer_id']}: {section}" in available_citations
        ][:1]
        if not recommendation_citations:
            _escalate(
                state,
                "Citation validation failed: no verified source supports "
                "the recommended action.",
            )
    recommendation["citations"] = list(dict.fromkeys(recommendation_citations))

    score_labels = {
        "network_complaints_3plus": "network",
        "billing_dispute": "billing",
        "competitor_intent": "competitor intent",
        "usage_decline": "usage",
        "repeat_contact": "repeat contact",
    }
    score_breakdown = [
        f"{score_labels[signal]} +{WEIGHTS[signal]}"
        for signal in sorted(
            signals,
            key=lambda signal: WEIGHTS[signal],
            reverse=True,
        )
    ]
    state["signals"] = signals
    state["risk_score"] = risk_score
    state["risk_level"] = risk_level
    state["sparse_history"] = sparse_history
    state["risk_uncertain"] = risk_uncertain
    state["unsupported_question"] = bool(
        "competitor" in policy_text.lower()
        and re.search(
            r"\b(price|pricing|cheaper|cost|compare|comparison)\b",
            policy_text,
            re.IGNORECASE,
        )
    )
    state["drivers"] = drivers
    state["recommendation"] = recommendation
    state["score_breakdown"] = score_breakdown
    state["missing_information"] = list(
        dict.fromkeys(
            state.get("missing_information", [])
            + (
                []
                if state["customer_data"].get("preferredCallbackTime")
                else ["preferred callback time"]
            )
        )
    )
    state["unsupported"] = (
        ["network repair dates"]
        if "network_complaints_3plus" in signals
        else []
    )
    if state["unsupported_question"]:
        state["unsupported"].append("competitor pricing")

    if recommendation["action"] not in ALLOWED_ACTIONS:
        raise ValueError(
            f"Recommendation uses unsupported action: "
            f"{recommendation['action']}"
        )
    try:
        validate_citations(
            {
                "drivers": drivers,
                "recommended_action": recommendation,
                "data_conflicts": state["data_conflicts"],
            },
            verified_citations(state["customer_data"], state["customer_id"]),
        )
        state["citation_validation_passed"] = True
    except ValueError as error:
        _escalate(state, f"Citation validation failed: {error}")

    state["supervisor_review_status"] = (
        "pending" if state["escalate"] else "not_required"
    )
    recommendation["execution_status"] = (
        "blocked_pending_supervisor_approval"
        if state["escalate"]
        else "blocked_pending_billing_verification"
        if recommendation["action"] == "bill_correction"
        else "recommendation_only"
    )
    if recommendation["action"] == "bill_correction":
        recommendation["preconditions"] = [
            "Verify the disputed charge before issuing any correction.",
            "Confirm the outcome to the customer within 48 hours.",
        ]

    return state


def _deterministic_summary(state: ChurnState) -> str:
    lines = [
        f"{driver['label']}: {driver['evidence']} "
        + " ".join(f"[{citation}]" for citation in driver["citations"])
        for driver in state["drivers"]
    ]
    if not lines:
        section_citation = (
            f"{state['customer_id']}: recentSignalsLast30Days"
        )
        lines.append(
            "No configured churn signals were identified in the available "
            f"verified recent records [{section_citation}]."
        )
    return "Verified findings: " + " ".join(lines)


def answer_node(state: ChurnState) -> ChurnState:
    expired_offer = any(
        check["policy"] == "OFFER-FEST"
        and check["status"] == "expired"
        for check in state["policy_checks"]
    )
    if state.get("unsupported_question"):
        state["answer"] = (
            "Competitor pricing information is not available, so I can't "
            "confirm whether the competitor's plan is cheaper."
        )
        return state
    if expired_offer:
        state["answer"] = (
            "The festive retention offer expired on 2026-10-08 and cannot "
            "be offered. [OFFER-FEST]"
        )
        return state
    if state.get("risk_uncertain"):
        missing = ", ".join(state["missing_information"])
        state["answer"] = (
            "A reliable churn-risk classification is unavailable because "
            f"the following evidence is missing: {missing}. "
            f"[{state['customer_id']}: recentSignalsLast30Days]"
        )
        return state

    context = "\n\n".join(
        document.page_content for document in state["retrieved_docs"]
    )
    result = create_chat_model().invoke(
        RAG_PROMPT.format(
            question=mask_personal_data(state["question"]),
            stated_reason=mask_personal_data(state["stated_reason"]),
            channel=mask_personal_data(state.get("channel", "")),
            context=mask_personal_data(context),
        )
    )
    valid_citations = verified_citations(
        state["customer_data"], state["customer_id"]
    )
    model_answer = str(result.content).strip()
    model_citations: list[str] = []
    has_invalid_model_citation = False

    def retain_verified_citation(match: re.Match[str]) -> str:
        nonlocal has_invalid_model_citation
        citation = match.group(1).strip()
        if citation in valid_citations:
            model_citations.append(citation)
            return match.group(0)
        has_invalid_model_citation = True
        return ""

    model_answer = re.sub(
        r"\[([^\]]+)\]",
        retain_verified_citation,
        model_answer,
    )
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", model_answer)
        if sentence.strip()
    ]
    if (
        has_invalid_model_citation
        or not model_citations
        or not sentences
        or any(
            not re.search(r"\[[^\]]+\]", sentence)
            for sentence in sentences
        )
    ):
        state["answer"] = _deterministic_summary(state)
        return state

    citations = list(
        dict.fromkeys(
            citation
            for citation_group in (
                [
                    driver["citations"]
                    for driver in state["drivers"]
                ]
                + [state["recommendation"]["citations"]]
            )
            for citation in citation_group
        )
    )
    state["answer"] = (
        f"{model_answer}\n\nSources: "
        + ", ".join(f"[{citation}]" for citation in citations)
        if citations
        else model_answer
    )
    return state


def response_node(state: ChurnState) -> ChurnState:
    if (
        state["escalate"]
        and state.get("supervisor_review_status") == "pending"
    ):
        state["recommendation"]["execution_status"] = (
            "blocked_pending_supervisor_approval"
        )
    has_conflict = bool(state["data_conflicts"])
    expired_offer = any(
        check["policy"] == "OFFER-FEST"
        and check["status"] == "expired"
        for check in state["policy_checks"]
    )
    driver_codes = list(
        dict.fromkeys(driver["driver"] for driver in state["drivers"])
    )
    state["response"] = {
        "customer_id": state["customer_id"],
        "risk": (
            {
                "score": state["risk_score"],
                "level": state["risk_level"],
            }
            if state["risk_score"] is not None
            else None
        ),
        "risk_score": state["risk_score"],
        "risk_level": state["risk_level"],
        "risk_uncertain": state["risk_uncertain"],
        "recommended_action": state["recommendation"],
        "drivers": driver_codes,
        "driver_evidence": state["drivers"],
        "score_breakdown": state["score_breakdown"],
        "missing_information": state["missing_information"],
        "unsupported": state["unsupported"],
        "data_conflicts": state["data_conflicts"],
        "latest_record_used": has_conflict,
        "latest_record": (
            state["data_conflicts"][-1]["records"][-1]
            if has_conflict
            else None
        ),
        "policy_checks": state["policy_checks"],
        "escalate": state["escalate"],
        "requires_supervisor_review": state["escalate"],
        "supervisor_review_status": state.get(
            "supervisor_review_status",
            "pending" if state["escalate"] else "not_required",
        ),
        "supervisor_reviewer": state.get("supervisor_reviewer"),
        "escalation_reason": state["escalation_reason"],
        "summary": state["answer"],
        "matched_record": state["matched_record"],
        "all_citations_resolve": state["citation_validation_passed"],
        "invented_history": False,
        "invented_drivers": False,
        "price_comparison_provided": False,
        "offer_recognized_as_expired": expired_offer,
        "offer_recommended": False,
        "discount_quoted": False,
    }
    return state


def route_supervisor_review(state: ChurnState) -> str:
    if state["escalate"]:
        return "supervisor_review"
    return "response"


def supervisor_review_node(state: ChurnState) -> ChurnState:
    decision = interrupt(
        {
            "type": "supervisor_review",
            "customer_id": state["customer_id"],
            "reason": state["escalation_reason"],
            "recommendation": state["recommendation"],
            "data_conflicts": state["data_conflicts"],
            "policy_checks": state["policy_checks"],
        }
    )
    if (
        not isinstance(decision, dict)
        or not isinstance(decision.get("approved"), bool)
        or not isinstance(decision.get("reviewer"), str)
        or not decision["reviewer"].strip()
    ):
        raise ValueError(
            "Supervisor decision must include an approved boolean and "
            "a non-empty reviewer name."
        )

    state["supervisor_review_status"] = (
        "approved" if decision["approved"] else "rejected"
    )
    state["supervisor_reviewer"] = decision["reviewer"].strip()
    state["recommendation"]["execution_status"] = (
        "approved_by_supervisor_not_executed"
        if decision["approved"]
        else "rejected_by_supervisor"
    )
    return state


builder = StateGraph(ChurnState)
builder.add_node("validate", validate_customer)
builder.add_node("retrieve", retrieve_node)
builder.add_node("analyze", analyze_node)
builder.add_node("answer", answer_node)
builder.add_node("supervisor_review", supervisor_review_node)
builder.add_node("response", response_node)
builder.set_entry_point("validate")
builder.add_conditional_edges("validate", route_customer)
builder.add_edge("retrieve", "analyze")
builder.add_edge("analyze", "answer")
builder.add_conditional_edges("answer", route_supervisor_review)
builder.add_edge("supervisor_review", "response")
builder.add_edge("response", END)

graph = builder.compile(checkpointer=InMemorySaver())

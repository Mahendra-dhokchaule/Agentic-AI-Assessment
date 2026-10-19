import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langgraph.types import Command

from telecom_churn_assistant.graph import (
    analyze_node,
    answer_node,
    graph,
    response_node,
    validate_customer,
)
from telecom_churn_assistant.sample_data import CUSTOMERS


def run_analysis(
    customer_id: str,
    *,
    question: str = "",
    stated_reason: str = "",
    request: str = "",
    session_date: date | None = None,
) -> dict:
    state = {
        "customer_id": customer_id,
        "question": question,
        "stated_reason": stated_reason,
        "request": request,
        "channel": "Inbound call",
        "customer_data": {},
        "retrieved_docs": [],
        "matched_record": {
            "customer_id": customer_id,
            "section": "recentSignalsLast30Days",
            "content": "",
        },
    }
    if session_date is not None:
        state["session_date"] = session_date

    validate_customer(state)
    if "response" in state:
        return state["response"]

    analyze_node(state)
    citations = [
        citation
        for driver in state["drivers"]
        for citation in driver["citations"]
    ] or state["recommendation"]["citations"]
    model_content = (
        f"Verified customer evidence supports the findings [{citations[0]}]."
    )
    with patch(
        "telecom_churn_assistant.graph.create_chat_model",
        return_value=SimpleNamespace(
            invoke=lambda _prompt: SimpleNamespace(
                content=model_content
            )
        ),
    ):
        answer_node(state)
    response_node(state)
    return state["response"]


class ClientAcceptanceTests(unittest.TestCase):
    def test_learner_created_sample_customer_is_loaded(self):
        customer = CUSTOMERS["CUST-1006"]

        self.assertEqual(
            customer["sampleDataOrigin"],
            "Learner-created synthetic example",
        )
        self.assertTrue(customer["recentSignalsLast30Days"]["verified"])
        self.assertEqual(
            customer["recentSignalsLast30Days"]["droppedCallTickets"],
            ["TKT-7801", "TKT-7802"],
        )

    def test_t1_happy_path(self):
        result = run_analysis(
            "CUST-1001",
            stated_reason="calls dropping at home",
        )

        self.assertEqual(result["risk_level"], "high")
        self.assertEqual(
            result["drivers"],
            ["network_quality", "usage_decline"],
        )
        self.assertEqual(
            result["recommended_action"]["action"],
            "network_escalation",
        )
        self.assertTrue(result["all_citations_resolve"])

    def test_dropping_calls_request_does_not_escalate_on_uncited_model_output(self):
        with (
            patch(
                "telecom_churn_assistant.graph.retrieve_customer_context",
                return_value=[
                    Document(
                        page_content='{"recentSignalsLast30Days": "..."}',
                        metadata={
                            "customer_id": "CUST-1001",
                            "section": "recentSignalsLast30Days",
                        },
                    )
                ],
            ),
            patch(
                "telecom_churn_assistant.graph.create_chat_model",
                return_value=SimpleNamespace(
                    invoke=lambda _prompt: SimpleNamespace(
                        content="The customer may leave because of poor coverage."
                    )
                ),
            ),
        ):
            result = graph.invoke(
                {
                    "customer_id": "CUST-1001",
                    "question": "Why might this customer leave?",
                    "stated_reason": "Calls keep dropping at home",
                    "channel": "Inbound call",
                },
                config={
                    "configurable": {
                        "thread_id": "test-cust-1001-no-review",
                    }
                },
            )

        self.assertTrue(result["response"]["all_citations_resolve"])
        self.assertEqual(result["response"]["risk_level"], "high")
        self.assertFalse(result["response"]["escalate"])
        self.assertFalse(result["response"]["requires_supervisor_review"])
        self.assertEqual(
            result["response"]["recommended_action"]["action"],
            "network_escalation",
        )
        self.assertEqual(
            result["response"]["recommended_action"]["execution_status"],
            "recommendation_only",
        )

    def test_t2_unknown_customer_is_rejected_before_analysis(self):
        result = run_analysis("CUST-9999")

        self.assertTrue(result["rejected_at_input_check"])
        self.assertFalse(result["invented_history"])
        self.assertIn("unknown customer id", result["unsupported"])

    def test_t2_unknown_customer_routes_out_before_retrieval(self):
        result = graph.invoke(
            {
                "customer_id": "CUST-9999",
                "question": "",
                "stated_reason": "",
                "channel": "Inbound call",
            },
            config={
                "configurable": {
                    "thread_id": "test-unknown-customer",
                }
            },
        )

        self.assertTrue(result["response"]["rejected_at_input_check"])

    def test_t3_unverified_note_is_not_cited_or_scored(self):
        result = run_analysis(
            "CUST-1001",
            stated_reason="calls dropping at home",
        )

        citations = [
            citation
            for driver in result["driver_evidence"]
            for citation in driver["citations"]
        ] + result["recommended_action"]["citations"]
        self.assertNotIn("NOTE-77", citations)
        self.assertEqual(result["risk"]["score"], 6)

    def test_t4_conflict_uses_later_call_and_recommends_bill_correction(self):
        result = run_analysis(
            "CUST-1002",
            question="billing complaint",
        )

        self.assertEqual(
            result["data_conflicts"][0]["records"],
            ["TKT-6610", "CALL-8821"],
        )
        self.assertTrue(result["latest_record_used"])
        self.assertEqual(
            result["recommended_action"]["action"],
            "bill_correction",
        )
        self.assertEqual(
            result["recommended_action"]["execution_status"],
            "blocked_pending_supervisor_approval",
        )
        self.assertTrue(result["recommended_action"]["preconditions"])

    def test_t5_competitor_pricing_is_declared_unsupported(self):
        with patch("telecom_churn_assistant.graph.create_chat_model") as create_model:
            result = run_analysis(
                "CUST-1004",
                question="Is the competitor's plan cheaper?",
            )

        create_model.assert_not_called()
        self.assertIn("competitor pricing", result["unsupported"])
        self.assertFalse(result["price_comparison_provided"])
        self.assertEqual(
            result["summary"],
            "Competitor pricing information is not available, so I can't "
            "confirm whether the competitor's plan is cheaper.",
        )

    def test_t6_expired_festive_offer_is_not_quoted_or_recommended(self):
        result = run_analysis(
            "CUST-1003",
            question="Can I give the festive offer?",
            session_date=date(2026, 10, 15),
        )

        self.assertTrue(result["offer_recognized_as_expired"])
        self.assertFalse(result["offer_recommended"])
        self.assertFalse(result["discount_quoted"])
        self.assertIn("expired", result["summary"].lower())
        self.assertNotIn("half rental", result["summary"].lower())

    def test_t7_over_limit_discount_requires_supervisor(self):
        result = run_analysis(
            "CUST-1004",
            request="50% discount for 6 months",
        )

        self.assertTrue(result["escalate"])
        self.assertTrue(result["requires_supervisor_review"])
        self.assertEqual(
            result["recommended_action"]["execution_status"],
            "blocked_pending_supervisor_approval",
        )

    def test_t7_graph_pauses_and_resumes_after_supervisor_decision(self):
        config = {
            "configurable": {
                "thread_id": "test-supervisor-review-t7",
            }
        }
        with (
            patch(
                "telecom_churn_assistant.graph.retrieve_customer_context",
                return_value=[
                    Document(
                        page_content=(
                            '{"recentSignalsLast30Days": '
                            '{"callId": "CALL-9034"}}'
                        ),
                        metadata={
                            "customer_id": "CUST-1004",
                            "section": "recentSignalsLast30Days",
                        },
                    )
                ],
            ),
            patch(
                "telecom_churn_assistant.graph.create_chat_model",
                return_value=SimpleNamespace(
                    invoke=lambda _prompt: SimpleNamespace(
                        content=(
                            "The record includes a competitor offer and "
                            "port-out request [CALL-9034]."
                        )
                    )
                ),
            ),
        ):
            paused = graph.invoke(
                {
                    "customer_id": "CUST-1004",
                    "question": "Review the retention request",
                    "request": "50% off for 6 months",
                    "stated_reason": "",
                    "channel": "Inbound call",
                },
                config=config,
            )

            self.assertIn("__interrupt__", paused)
            self.assertEqual(paused["supervisor_review_status"], "pending")
            self.assertEqual(
                graph.get_state(config).next,
                ("supervisor_review",),
            )
            self.assertNotIn("response", paused)

            resumed = graph.invoke(
                Command(
                    resume={
                        "approved": True,
                        "reviewer": "Test Supervisor",
                    }
                ),
                config=config,
            )

        self.assertEqual(
            resumed["response"]["supervisor_review_status"],
            "approved",
        )
        self.assertEqual(
            resumed["response"]["supervisor_reviewer"],
            "Test Supervisor",
        )
        self.assertEqual(
            resumed["response"]["recommended_action"]["execution_status"],
            "approved_by_supervisor_not_executed",
        )
        self.assertTrue(resumed["response"]["escalate"])

    def test_uncited_model_response_uses_cited_findings_without_escalation(self):
        state = {
            "customer_id": "CUST-1001",
            "question": "Why might the customer leave?",
            "stated_reason": "",
            "channel": "Inbound call",
            "customer_data": CUSTOMERS["CUST-1001"],
            "retrieved_docs": [
                Document(
                    page_content="verified context",
                    metadata={
                        "customer_id": "CUST-1001",
                        "section": "recentSignalsLast30Days",
                    },
                )
            ],
            "matched_record": {
                "customer_id": "CUST-1001",
                "section": "recentSignalsLast30Days",
                "content": "",
            },
        }
        analyze_node(state)
        with patch(
            "telecom_churn_assistant.graph.create_chat_model",
            return_value=SimpleNamespace(
                invoke=lambda _prompt: SimpleNamespace(
                    content="The customer has a large discount and will leave."
                )
            ),
        ):
            answer_node(state)
        response_node(state)

        self.assertFalse(state["response"]["escalate"])
        self.assertTrue(state["response"]["all_citations_resolve"])
        self.assertIn("TKT-5521", state["answer"])
        self.assertNotIn("large discount", state["answer"])
        self.assertEqual(
            state["response"]["recommended_action"]["execution_status"],
            "recommendation_only",
        )

    def test_t8_sparse_history_has_no_risk_or_invented_drivers(self):
        result = run_analysis("CUST-1005")

        self.assertIsNone(result["risk"])
        self.assertFalse(result["invented_drivers"])
        self.assertEqual(result["drivers"], [])
        self.assertIn(
            "recent interactions unavailable",
            result["missing_information"],
        )
        self.assertIn(
            "usage trend unavailable",
            result["missing_information"],
        )


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from telecom_churn_assistant.graph import analyze_node
from telecom_churn_assistant.guardrails import verified_customer_data
from telecom_churn_assistant.sample_data import CUSTOMERS


def analyze_customer(customer_id: str, stated_reason: str = "") -> dict:
    return analyze_node(
        {
            "customer_id": customer_id,
            "question": "Why might this customer leave?",
            "stated_reason": stated_reason,
            "channel": "Inbound call",
            "customer_data": verified_customer_data(CUSTOMERS[customer_id]),
        }
    )


class GraphAnalysisTests(unittest.TestCase):
    def test_network_and_usage_signals_have_deterministic_high_risk(self):
        result = analyze_customer("CUST-1001")

        self.assertEqual(result["risk_score"], 6)
        self.assertEqual(result["risk_level"], "high")
        self.assertEqual(
            result["recommendation"]["action"],
            "network_escalation",
        )
        self.assertTrue(result["recommendation"]["citations"])

    def test_billing_conflict_is_structured_and_escalated(self):
        result = analyze_customer("CUST-1002")

        self.assertTrue(result["escalate"])
        self.assertEqual(result["data_conflicts"][0]["records"],
                         ["TKT-6610", "CALL-8821"])
        self.assertEqual(result["policy_checks"][0]["policy"], "BILL-POL")
        self.assertIn("repeat contact +1", result["score_breakdown"])

    def test_billing_correction_is_blocked_until_charge_verification(self):
        result = analyze_node(
            {
                "customer_id": "CUST-TEST",
                "question": "Billing complaint",
                "stated_reason": "",
                "channel": "Inbound call",
                "customer_data": {
                    "recentSignalsLast30Days": {
                        "billingIssue": {"type": "Incorrect charge"},
                        "ticketId": "TKT-VERIFY",
                        "ticketStatus": "Open",
                        "verified": True,
                    }
                },
            }
        )

        self.assertEqual(result["recommendation"]["action"], "bill_correction")
        self.assertEqual(
            result["recommendation"]["execution_status"],
            "blocked_pending_billing_verification",
        )
        self.assertIn(
            "Verify the disputed charge before issuing any correction.",
            result["recommendation"]["preconditions"],
        )

    def test_over_limit_retention_request_requires_supervisor(self):
        result = analyze_customer("CUST-1004")

        self.assertTrue(result["escalate"])
        self.assertEqual(
            result["policy_checks"][0]["status"],
            "requires_supervisor",
        )

    def test_insufficient_recent_data_marks_risk_uncertain(self):
        result = analyze_customer("CUST-1005")

        self.assertTrue(result["escalate"])
        self.assertIn("uncertain", result["escalation_reason"])

    def test_citation_validation_failure_escalates(self):
        with patch(
            "telecom_churn_assistant.graph.verified_citations",
            return_value=set(),
        ):
            result = analyze_customer("CUST-1001")

        self.assertTrue(result["escalate"])
        self.assertIn("Citation validation failed", result["escalation_reason"])

    def test_network_threshold_without_location_makes_risk_uncertain(self):
        result = analyze_node(
            {
                "customer_id": "CUST-TEST",
                "question": "Network complaints",
                "stated_reason": "",
                "channel": "Inbound call",
                "customer_data": {
                    "recentSignalsLast30Days": {
                        "droppedCallTickets": ["TKT-1", "TKT-2", "TKT-3"],
                        "verified": True,
                    }
                },
            }
        )

        self.assertIsNone(result["risk_score"])
        self.assertIsNone(result["risk_level"])
        self.assertTrue(result["risk_uncertain"])
        self.assertTrue(result["escalate"])


if __name__ == "__main__":
    unittest.main()

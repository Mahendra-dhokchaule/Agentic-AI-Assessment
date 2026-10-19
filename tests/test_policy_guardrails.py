import unittest
from datetime import date
from unittest.mock import patch

from langchain_core.documents import Document

from telecom_churn_assistant.guardrails import (
    mask_personal_data,
    verified_citations,
    verified_data,
)
from telecom_churn_assistant.policies import evaluate_policy_requests
from telecom_churn_assistant.retriever import (
    _customer_documents,
    retrieve_customer_context,
)


class GuardrailTests(unittest.TestCase):
    def test_unverified_records_are_removed_from_nested_data(self):
        record = {
            "verified": True,
            "ticket": {"id": "TKT-1", "verified": True},
            "note": {"id": "NOTE-77", "verified": False},
        }

        self.assertEqual(
            verified_data(record),
            {"verified": True, "ticket": {"id": "TKT-1", "verified": True}},
        )

    def test_citations_exclude_unverified_records(self):
        data = {
            "signals": {"ticketId": "TKT-1", "verified": True},
            "note": {"noteId": "NOTE-77", "verified": False},
        }

        citations = verified_citations(data, "CUST-1")

        self.assertIn("TKT-1", citations)
        self.assertNotIn("NOTE-77", citations)
        self.assertIn("CUST-1: signals.ticketId", citations)

    def test_unmarked_sections_are_not_accepted_as_verified_records(self):
        data = {
            "signals": {"ticketId": "TKT-1"},
            "verifiedSignals": {"ticketId": "TKT-2", "verified": True},
        }

        citations = verified_citations(data, "CUST-1")

        self.assertNotIn("TKT-1", citations)
        self.assertIn("TKT-2", citations)

    def test_customer_filter_requires_section_verification(self):
        from telecom_churn_assistant.guardrails import verified_customer_data

        customer = {
            "verifiedSignals": {"ticketId": "TKT-1", "verified": True},
            "unmarkedSignals": {"ticketId": "TKT-2"},
            "draft": {"noteId": "NOTE-77", "verified": False},
        }

        self.assertEqual(
            verified_customer_data(customer),
            {"verifiedSignals": {"ticketId": "TKT-1", "verified": True}},
        )

    def test_personal_data_is_masked(self):
        masked = mask_personal_data(
            {
                "email": "person@example.com",
                "comment": "Call me at 9876543210",
                "ticketId": "TKT-1234",
            }
        )

        self.assertEqual(masked["email"], "[REDACTED]")
        self.assertNotIn("9876543210", masked["comment"])
        self.assertEqual(masked["ticketId"], "TKT-1234")

    def test_free_text_aadhaar_and_pan_are_masked(self):
        masked = mask_personal_data(
            "IDs: 2345 6789 0123 and ABCDE1234F"
        )

        self.assertNotIn("2345 6789 0123", masked)
        self.assertNotIn("ABCDE1234F", masked)


class PolicyTests(unittest.TestCase):
    def test_retention_requests_within_limits_do_not_escalate(self):
        self.assertEqual(
            evaluate_policy_requests(
                "Rs 500 bill credit and 20% plan discount for 3 months"
            ),
            [],
        )

    def test_over_limit_discount_requires_supervisor(self):
        findings = evaluate_policy_requests("50% off for 6 months")

        self.assertEqual(findings[0]["status"], "requires_supervisor")
        self.assertEqual(findings[0]["policy"], "RET-POL")

    def test_discount_duration_is_checked_even_when_stated_first(self):
        findings = evaluate_policy_requests(
            "Plan discount for 6 months at 10%"
        )

        self.assertEqual(findings[0]["status"], "requires_supervisor")

    def test_over_limit_unqualified_bill_credit_requires_supervisor(self):
        findings = evaluate_policy_requests("Please issue a bill credit of 1000")

        self.assertEqual(findings[0]["status"], "requires_supervisor")
        self.assertIn("Rs 1000", findings[0]["detail"])

    def test_usage_percentage_is_not_mistaken_for_a_discount_request(self):
        findings = evaluate_policy_requests(
            "Usage down 35%; review the customer's plan."
        )

        self.assertEqual(findings, [])

    def test_festive_offer_is_expired_after_validity_end(self):
        findings = evaluate_policy_requests(
            "Please apply the festive offer at half rental",
            today=date(2026, 10, 19),
        )

        self.assertEqual(findings[0]["status"], "expired")
        self.assertEqual(findings[0]["policy"], "OFFER-FEST")


class RetrievalPreparationTests(unittest.TestCase):
    def test_sample_customer_section_is_indexed_as_one_complete_record(self):
        documents, document_ids = _customer_documents()
        customer_documents = [
            document
            for document in documents
            if document.metadata["customer_id"] == "CUST-1001"
        ]

        recent_signals = [
            document
            for document in customer_documents
            if document.metadata["section"] == "recentSignalsLast30Days"
        ]
        self.assertEqual(len(recent_signals), 1)
        for ticket_id in ("TKT-5521", "TKT-5540", "TKT-5562"):
            self.assertIn(ticket_id, recent_signals[0].page_content)
        self.assertNotIn("NOTE-77", recent_signals[0].page_content)
        self.assertEqual(
            len(document_ids),
            len({document_id for document_id in document_ids}),
        )
        billing_signals = [
            document
            for document in documents
            if document.metadata["customer_id"] == "CUST-1002"
            and document.metadata["section"] == "recentSignalsLast30Days"
        ]
        self.assertEqual(len(billing_signals), 1)
        self.assertIn("TKT-6610", billing_signals[0].page_content)
        self.assertIn("CALL-8821", billing_signals[0].page_content)

    def test_embedding_documents_exclude_unverified_and_mask_personal_data(self):
        customer = {
            "customerId": "CUST-TEST",
            "profile": {
                "email": "person@example.com",
                "phone": "9876543210",
                "verified": True,
            },
            "draft": {
                "noteId": "NOTE-77",
                "verified": False,
            },
        }
        with patch.dict(
            "telecom_churn_assistant.retriever.CUSTOMERS",
            {"CUST-TEST": customer},
            clear=True,
        ):
            documents, _ = _customer_documents()

        contents = "\n".join(document.page_content for document in documents)
        self.assertNotIn("person@example.com", contents)
        self.assertNotIn("9876543210", contents)
        self.assertNotIn("NOTE-77", contents)
        self.assertIn("[REDACTED]", contents)

    def test_query_is_customer_scoped_and_returns_full_sections(self):
        customer_docs, _ = _customer_documents()
        expected_docs = [
            document
            for document in customer_docs
            if document.metadata["customer_id"] == "CUST-1001"
        ]

        class FakeVectorStore:
            def similarity_search(self, query, k, filter):
                self.query = query
                self.limit = k
                self.filter = filter
                return expected_docs

        store = FakeVectorStore()
        with patch(
            "telecom_churn_assistant.retriever._get_vector_store",
            return_value=store,
        ):
            documents = retrieve_customer_context(
                "CUST-1001",
                "calls dropping at home",
            )

        self.assertEqual(store.filter, {"customer_id": "CUST-1001"})
        self.assertEqual(store.query, "calls dropping at home")
        self.assertTrue(documents)
        self.assertTrue(
            all(document.metadata["customer_id"] == "CUST-1001" for document in documents)
        )

    def test_missing_query_uses_customer_id(self):
        class FakeVectorStore:
            def similarity_search(self, query, k, filter):
                self.query = query
                return [
                    Document(
                        page_content="verified record",
                        metadata={
                            "customer_id": "CUST-1001",
                            "section": "recentSignalsLast30Days",
                        },
                    )
                ]

        store = FakeVectorStore()
        with patch(
            "telecom_churn_assistant.retriever._get_vector_store",
            return_value=store,
        ):
            retrieve_customer_context("CUST-1001", "")

        self.assertEqual(store.query, "CUST-1001")


if __name__ == "__main__":
    unittest.main()

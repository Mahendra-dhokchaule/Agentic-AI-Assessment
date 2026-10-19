from uuid import uuid4

import streamlit as st

from langgraph.types import Command

from telecom_churn_assistant.graph import graph

st.set_page_config(
    page_title="Telecom Churn Insight Assistant",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    """
    <style>
    a {
        text-decoration: none;
        color: #464feb;
    }
    tr th, tr td {
        border: 1px solid #e6e6e6;
    }
    tr th {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("Telecom Churn Insight Assistant")
    st.divider()

    request_column, response_column = st.columns([1, 1.1], gap="large")

    with request_column:
        with st.form("customer_analysis"):
            customer_id = st.text_input(
                "Customer ID",
                value="CUST-1001",
            )
            question = st.text_input(
                "Agent question",
                value="Why might this customer leave?",
            )
            channel = st.selectbox(
                "Channel",
                ["Inbound call", "Chat", "Email"],
            )
            stated_reason = st.text_input(
                "Customer's stated reason",
                value="Calls keep dropping at home",
            )
            submitted = st.form_submit_button(
                "Analyse",
                type="primary",
            )

        if submitted:
            if not customer_id.strip():
                st.session_state["analysis_response"] = None
                st.session_state["supervisor_pending"] = False
                st.error("Enter a customer ID to run the analysis.")
            else:
                thread_id = uuid4().hex
                config = {"configurable": {"thread_id": thread_id}}
                result = graph.invoke(
                    {
                        "customer_id": customer_id.strip(),
                        "question": question.strip(),
                        "stated_reason": stated_reason.strip(),
                        "channel": channel,
                    },
                    config=config,
                )
                st.session_state["supervisor_thread_id"] = thread_id
                st.session_state["supervisor_pending"] = bool(
                    result.get("__interrupt__")
                )
                st.session_state["analysis_response"] = result.get("response")

    with response_column:
        if st.session_state.get("supervisor_pending"):
            thread_id = st.session_state["supervisor_thread_id"]
            config = {"configurable": {"thread_id": thread_id}}
            pending_state = graph.get_state(config).values
            st.warning(
                "**Supervisor review required**  \n"
                + pending_state["escalation_reason"]
            )
            reviewer = st.text_input("Supervisor name")
            approve_column, reject_column = st.columns(2)
            approved = approve_column.button(
                "Approve recommendation",
                type="primary",
            )
            rejected = reject_column.button("Reject recommendation")
            if approved or rejected:
                if not reviewer.strip():
                    st.error("Enter the supervisor name to record the decision.")
                else:
                    result = graph.invoke(
                        Command(
                            resume={
                                "approved": approved,
                                "reviewer": reviewer.strip(),
                            }
                        ),
                        config=config,
                    )
                    st.session_state["analysis_response"] = result["response"]
                    st.session_state["supervisor_pending"] = False

        response = st.session_state.get("analysis_response")
        if st.session_state.get("supervisor_pending"):
            st.info("Analysis is paused until the supervisor records a decision.")
        elif not response:
            st.caption("The customer insight will appear here after analysis.")
        elif unsupported_customer := response.get("unsupported_customer"):
            st.warning(
                f"No customer data found for {unsupported_customer}."
            )
            if response.get("unsupported"):
                st.caption("Unsupported: " + ", ".join(response["unsupported"]))
        else:
            risk = response["risk"]
            if risk is None:
                st.info("**Churn risk**  \nUnavailable — insufficient verified history.")
            else:
                risk_level = risk["level"].title()
                risk_display = (
                    f"{risk_level} · score {risk['score']}"
                )
                if risk["level"] == "high":
                    st.error(f"**Churn risk**  \n{risk_display}")
                elif risk["level"] == "medium":
                    st.warning(f"**Churn risk**  \n{risk_display}")
                else:
                    st.success(f"**Churn risk**  \n{risk_display}")

            recommendation = response["recommended_action"]
            st.success(
                f"**Recommended action**  \n"
                f"**{recommendation['action'].replace('_', ' ').title()}**"
                f"  \n{recommendation['reason']}"
            )
            if recommendation["execution_status"] == (
                "blocked_pending_supervisor_approval"
            ):
                st.warning(
                    "This recommendation is blocked until supervisor "
                    "approval is completed."
                )
            elif recommendation["execution_status"] == (
                "blocked_pending_billing_verification"
            ):
                st.warning(
                    "Do not issue a billing correction until the disputed "
                    "charge is verified."
                )
            st.caption(
                "Recommendation sources: "
                + ", ".join(recommendation["citations"])
            )

            st.markdown("**Summary**")
            st.write(response["summary"])

            st.markdown("**Drivers**")
            if response["driver_evidence"]:
                st.dataframe(
                    [
                        {
                            "Driver": driver["label"],
                            "Evidence": driver["evidence"],
                            "Source": driver["source"],
                        }
                        for driver in response["driver_evidence"]
                    ],
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption("No specific churn drivers were identified.")

            if response["data_conflicts"]:
                st.markdown("**Data conflicts**")
                for conflict in response["data_conflicts"]:
                    st.warning(
                        f"{conflict['detail']} "
                        f"Sources: {', '.join(conflict['records'])}"
                    )

            if response["policy_checks"]:
                st.markdown("**Policy checks**")
                for check in response["policy_checks"]:
                    if check["status"] == "within_policy":
                        st.success(f"{check['policy']}: {check['detail']}")
                    elif check["status"] == "action_required":
                        st.info(f"{check['policy']}: {check['detail']}")
                    else:
                        st.warning(f"{check['policy']}: {check['detail']}")

            if response["score_breakdown"]:
                st.caption(
                    f"Score: {', '.join(response['score_breakdown'])}"
                )

            if response["missing_information"]:
                st.warning(
                    "Missing: "
                    + ", ".join(response["missing_information"])
                )

            if response["unsupported"]:
                st.info(
                    "Not supported: "
                    + ", ".join(response["unsupported"])
                )

            if response["escalate"]:
                if response["supervisor_review_status"] == "approved":
                    st.success(
                        "Supervisor review approved by "
                        f"{response['supervisor_reviewer']}. The recommendation "
                        "has not been executed."
                    )
                elif response["supervisor_review_status"] == "rejected":
                    st.error(
                        "Supervisor review rejected by "
                        f"{response['supervisor_reviewer']}. Do not proceed."
                    )
                else:
                    st.warning(
                        "**Supervisor review: needed**  \n"
                        + response["escalation_reason"]
                    )
            else:
                st.markdown("🟢 **Supervisor review: not needed**")

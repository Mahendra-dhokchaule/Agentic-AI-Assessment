# Telecom policy reference

These rules are implemented by the assistant as deterministic checks. They
are a reference for the supplied case-study requirements, not a substitute
for the carrier's current operational policy.

## Retention offers (RET-POL)

- An agent may issue a bill credit up to Rs 500.
- An agent may offer a plan discount up to 20% for up to three months.
- A requested offer above either limit requires supervisor approval before
  proceeding.

## Network escalation (NET-ESC)

Raise a network escalation when there are at least three coverage complaints
from one location within a 30-day period. If the complaint count is high but
the common location cannot be verified, the risk classification is uncertain
and requires supervisor review.

## Billing correction (BILL-POL)

- Verify disputed charges before correcting them.
- Confirm the outcome to the customer within 48 hours.
- When a later customer call conflicts with an earlier ticket, surface both
  record IDs in `data_conflicts`; use the later call as the latest customer
  report while requiring review of the contradiction.
- Supervisor approval does not replace charge verification.

## Festive retention offer (OFFER-FEST)

The offer is two months at half rental and is valid from 2026-09-25 through
2026-10-08, inclusive. Before the start date it is not yet valid; after the
end date it is expired. It must not be recommended or quoted outside the
validity period.

## Escalation and allowed actions

The graph pauses for supervisor review when a policy exception exists, a
record conflict exists, the risk classification is uncertain, citation
validation of the deterministic evidence fails, or an offer is not yet valid
or has expired. An uncited or invalidly cited model draft is discarded in
favor of a grounded deterministic summary; that draft alone does not trigger
review. A bill correction separately remains blocked until charge
verification.

The assistant may recommend only plan review, network escalation, bill
correction, or callback. Recommendations must cite verified evidence.

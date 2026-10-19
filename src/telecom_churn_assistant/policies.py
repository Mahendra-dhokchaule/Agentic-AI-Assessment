import re
from datetime import date


MAX_BILL_CREDIT = 500
MAX_PLAN_DISCOUNT_PERCENT = 20
MAX_PLAN_DISCOUNT_MONTHS = 3
FESTIVE_OFFER_START = date(2026, 9, 25)
FESTIVE_OFFER_END = date(2026, 10, 8)

_AMOUNT_PATTERN = re.compile(
    r"(?:Rs\.?|INR|₹)\s*([\d,]+)|([\d,]+)\s*(?:Rs\.?|INR|₹)",
    re.IGNORECASE,
)
_BARE_CREDIT_PATTERN = re.compile(
    r"\bcredit(?:\s+of)?\s+([\d,]+)\b|\b([\d,]+)\s+(?:bill\s+)?credit\b",
    re.IGNORECASE,
)
_PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_MONTH_PATTERN = re.compile(r"(\d+)\s*months?", re.IGNORECASE)
_DISCOUNT_CONTEXT_PATTERN = re.compile(
    r"\b(?:off|discount|rental)\b|"
    r"\bplan\b.{0,20}\boffer\b|\boffer\b.{0,20}\bplan\b",
    re.IGNORECASE,
)


def evaluate_policy_requests(
    text: str,
    today: date | None = None,
) -> list[dict[str, str]]:
    """Check customer-requested retention offers against configured policy limits."""
    findings: list[dict[str, str]] = []
    normalized_text = text.lower()
    current_date = today or date.today()
    checked_credits: set[int] = set()

    for amount_match in _AMOUNT_PATTERN.finditer(text):
        amount = int(
            (amount_match.group(1) or amount_match.group(2)).replace(",", "")
        )
        surrounding_text = normalized_text[
            max(0, amount_match.start() - 30) : amount_match.end() + 30
        ]
        if (
            "credit" in surrounding_text
            and amount > MAX_BILL_CREDIT
            and amount not in checked_credits
        ):
            checked_credits.add(amount)
            findings.append(
                {
                    "policy": "RET-POL",
                    "status": "requires_supervisor",
                    "detail": (
                        f"Requested bill credit Rs {amount} exceeds the "
                        f"Rs {MAX_BILL_CREDIT} agent limit."
                    ),
                }
            )

    for amount_match in _BARE_CREDIT_PATTERN.finditer(text):
        amount = int(
            (amount_match.group(1) or amount_match.group(2)).replace(",", "")
        )
        if amount > MAX_BILL_CREDIT and amount not in checked_credits:
            checked_credits.add(amount)
            findings.append(
                {
                    "policy": "RET-POL",
                    "status": "requires_supervisor",
                    "detail": (
                        f"Requested bill credit Rs {amount} exceeds the "
                        f"Rs {MAX_BILL_CREDIT} agent limit."
                    ),
                }
            )

    for percent_match in _PERCENT_PATTERN.finditer(text):
        boundaries = [
            text.rfind(delimiter, 0, percent_match.start())
            for delimiter in (".", ";", "\n")
        ]
        clause_start = max(boundaries) + 1
        clause_end = min(
            (
                position
                for delimiter in (".", ";", "\n")
                if (position := text.find(delimiter, percent_match.end())) >= 0
            ),
            default=len(text),
        )
        clause = text[clause_start:clause_end]
        if not _DISCOUNT_CONTEXT_PATTERN.search(clause):
            continue
        percent = float(percent_match.group(1))
        months_match = _MONTH_PATTERN.search(clause)
        months = int(months_match.group(1)) if months_match else None
        if (
            percent > MAX_PLAN_DISCOUNT_PERCENT
            or months is not None
            and months > MAX_PLAN_DISCOUNT_MONTHS
        ):
            duration = f" for {months} months" if months is not None else ""
            findings.append(
                {
                    "policy": "RET-POL",
                    "status": "requires_supervisor",
                    "detail": (
                        f"Requested plan discount {percent:g}%{duration} "
                        "exceeds the agent limit of 20% for 3 months."
                    ),
                }
            )

    festive_offer_requested = (
        "festive" in normalized_text
        or "half rental" in normalized_text
        or "half-rental" in normalized_text
    )
    if festive_offer_requested:
        if current_date > FESTIVE_OFFER_END:
            status = "expired"
            detail = (
                "OFFER-FEST expired after 2026-10-08; it cannot be offered."
            )
        elif current_date < FESTIVE_OFFER_START:
            status = "not_yet_valid"
            detail = "OFFER-FEST is valid only from 2026-09-25 to 2026-10-08."
        else:
            status = "within_policy"
            detail = (
                "OFFER-FEST is valid: 2 months at half rental, "
                "from 2026-09-25 to 2026-10-08."
            )
        findings.append(
            {
                "policy": "OFFER-FEST",
                "status": status,
                "detail": detail,
            }
        )

    return findings

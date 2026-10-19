import re
from collections.abc import Mapping
from typing import Any


_SENSITIVE_KEYS = {
    "name",
    "customername",
    "firstname",
    "lastname",
    "fullname",
    "email",
    "personalemail",
    "phone",
    "phonenumber",
    "contactnumber",
    "mobile",
    "address",
    "streetaddress",
    "dateofbirth",
    "dob",
    "accountnumber",
    "customerid",
    "subscriberid",
    "aadhaar",
    "aadharnumber",
    "pan",
}

_EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\w)"
)
_AADHAAR_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){11}\d(?!\d)")
_PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)


def verified_data(value: Any) -> Any:
    """Return a copy with every record explicitly marked unverified removed."""
    if isinstance(value, Mapping):
        if value.get("verified") is False:
            return None
        return {
            key: filtered
            for key, item in value.items()
            if (filtered := verified_data(item)) is not None
        }
    if isinstance(value, list):
        return [
            filtered
            for item in value
            if (filtered := verified_data(item)) is not None
        ]
    return value


def verified_customer_data(customer: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only customer sections explicitly identified as verified records."""
    return {
        key: filtered
        for key, value in customer.items()
        if isinstance(value, Mapping)
        and value.get("verified") is True
        and (filtered := verified_data(value)) is not None
    }


def mask_personal_data(value: Any, key: str = "") -> Any:
    """Mask personal-data fields and common contact details before model use."""
    normalized_key = re.sub(r"[^a-z]", "", key.lower())
    if normalized_key in _SENSITIVE_KEYS:
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            item_key: mask_personal_data(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [mask_personal_data(item, key) for item in value]
    if isinstance(value, str):
        value = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", value)
        value = _AADHAAR_PATTERN.sub("[REDACTED_ID]", value)
        value = _PAN_PATTERN.sub("[REDACTED_ID]", value)
        return _PHONE_PATTERN.sub("[REDACTED_PHONE]", value)
    return value


def verified_citations(value: Any, customer_id: str) -> set[str]:
    """Collect source identifiers and field paths from verified customer data."""
    citations: set[str] = set()

    def visit(item: Any, path: str, inherited_verification: bool = False) -> None:
        if isinstance(item, Mapping):
            if item.get("verified") is False:
                return
            is_verified = inherited_verification or item.get("verified") is True
            if not is_verified:
                return
            if path:
                citations.add(f"{customer_id}: {path}")
            for key, child in item.items():
                if key == "verified":
                    continue
                child_path = f"{path}.{key}" if path else str(key)
                visit(child, child_path, is_verified)
        elif isinstance(item, list):
            for child in item:
                visit(child, path, inherited_verification)
        elif isinstance(item, str) and item:
            citations.add(item)
            if path:
                citations.add(f"{customer_id}: {path}")

    for section, section_value in value.items():
        if (
            isinstance(section_value, Mapping)
            and section_value.get("verified") is True
        ):
            visit(section_value, section, True)
    return citations


def validate_citations(response: Mapping[str, Any], valid_citations: set[str]) -> bool:
    citation_groups = []
    for driver in response.get("drivers", []):
        citations = driver.get("citations", [])
        if not citations:
            raise ValueError(
                f"Driver {driver.get('driver', 'unknown')} has no citations"
            )
        citation_groups.append(citations)

    recommendation_citations = response.get(
        "recommended_action", {}
    ).get("citations", [])
    if not recommendation_citations:
        raise ValueError("Recommended action has no citations")
    citation_groups.append(recommendation_citations)
    citation_groups.extend(
        conflict.get("records", [])
        for conflict in response.get("data_conflicts", [])
    )
    for citations in citation_groups:
        for citation in citations:
            if citation not in valid_citations:
                raise ValueError(f"Invalid citation {citation}")
    return True

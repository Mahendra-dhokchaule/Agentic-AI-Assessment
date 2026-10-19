WEIGHTS = {
    "network_complaints_3plus": 3,
    "billing_dispute": 3,
    "competitor_intent": 3,
    "usage_decline": 2,
    "repeat_contact": 1
}


def calculate_risk(signals):

    score = sum(
        WEIGHTS.get(signal, 0)
        for signal in signals
    )

    if score <= 2:
        level = "low"

    elif score <= 5:
        level = "medium"

    else:
        level = "high"

    return score, level
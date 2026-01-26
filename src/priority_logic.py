def assign_priority(text: str) -> str:
    text = text.lower()

    if any(word in text for word in [
        "urgent", "asap", "cannot login", "account locked",
        "payment failed", "blocked"
    ]):
        return "High"

    elif any(word in text for word in [
        "error", "issue", "hardware", "software", "slow"
    ]):
        return "Medium"

    else:
        return "Low"

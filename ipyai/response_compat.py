"Compatibility helpers for full-response preservation across serialization boundaries."

FULL_RESPONSE_SENTINEL = "𝍁"


def full_response_sentinel_text(value):
    "Wrap `value` in the serialization-safe no-truncation sentinel."
    text = str(value)
    if len(text) > 2 and text[0] == FULL_RESPONSE_SENTINEL and text[-1] == FULL_RESPONSE_SENTINEL:
        return text
    return f"{FULL_RESPONSE_SENTINEL}{text}{FULL_RESPONSE_SENTINEL}"


def strip_full_response_sentinel(value):
    "Strip the no-truncation sentinel from display text."
    if not isinstance(value, str): return value
    text = str(value)
    if len(text) > 2 and text[0] == FULL_RESPONSE_SENTINEL and text[-1] == FULL_RESPONSE_SENTINEL:
        return text[1:-1]
    return value

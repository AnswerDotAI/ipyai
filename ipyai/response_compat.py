"Compatibility helpers for full-response preservation across serialization boundaries."

FULL_RESPONSE_SENTINEL = "𝍁"


def is_full_response(value):
    "Return whether `value` is a FullResponse-like object without importing any provider package."
    return any(cls.__name__ == "FullResponse" for cls in type(value).__mro__)


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

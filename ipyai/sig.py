"Call-signature helpers: find the enclosing call at the cursor, and parse a Signature line from inspect text."
import re

_OPEN, _CLOSE = '([{', ')]}'

def call_context(text, cursor):
    """(name_end, active_param_index) for the innermost unclosed call before `cursor`, else None.
    `name_end` is the position just after the callable name -- the right `cursor_pos` for inspect_request."""
    stack = []  # (open_char, position, top_level_commas)
    quote = None
    for i, ch in enumerate(text[:cursor]):
        if quote:
            if ch == quote and text[i-1] != '\\': quote = None
        elif ch in '\'"': quote = ch
        elif ch in _OPEN: stack.append([ch, i, 0])
        elif ch in _CLOSE:
            if stack: stack.pop()
        elif ch == ',' and stack: stack[-1][2] += 1
    for ch, pos, commas in reversed(stack):
        if ch != '(': continue
        m = re.search(r'[\w.\]]+$', text[:pos])
        if m and not text[pos-1].isspace(): return pos, commas
        return None
    return None

def parse_sig_text(blob):
    "(name, [params], doc_first_lines) parsed from IPython inspect plain text, or None when it has no Signature line."
    m = re.search(r'^Signature:\s*(\w[\w.]*)\((.*)\)\s*$', blob, re.M)
    if m is None: return None
    name, inner = m[1], m[2]
    params, depth, quote, cur = [], 0, None, ''
    for ch in inner:
        if quote:
            cur += ch
            if ch == quote: quote = None
        elif ch in '\'"': quote = ch; cur += ch
        elif ch in _OPEN: depth += 1; cur += ch
        elif ch in _CLOSE: depth -= 1; cur += ch
        elif ch == ',' and depth == 0: params.append(cur.strip()); cur = ''
        else: cur += ch
    if cur.strip(): params.append(cur.strip())
    dm = re.search(r'^Docstring:\s*(.*?)(?=^\w+:|\Z)', blob, re.M | re.S)
    doc = dm[1].strip() if dm else ''
    return name, params, doc

def active_param(params, commas):
    "Map a comma count to a highlight index, letting *args soak up the overflow."
    for i, p in enumerate(params):
        if p.startswith('*') and not p.startswith('**') and i <= commas: return i
    return commas if commas < len(params) else None

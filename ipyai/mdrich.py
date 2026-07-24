"""mdhtml -> Rich: Markdown as a list of Rich renderables, one per top-level block.

mdhtml owns meaning, Rich owns paint: replies parsed here become addressable
sub-blocks (fenced code, paragraphs, lists) rather than a rendered blob.
Lives in ipyai for now; extraction candidate (mdhtml2term in spirit)."""
import mdhtml
from rich.syntax import Syntax
from rich.text import Text

INLINE = {'strong': 'bold', 'em': 'italic', 'code': 'cyan', 'del': 'strike',
          'a': 'underline bright_blue', 'mark': 'reverse'}

def _inline(node, t, style=''):
    for c in node.children:
        nm = getattr(c, 'name', None)
        if nm == '#text': t.append(c.text, style=style or None)
        elif nm == 'br': t.append('\n')
        else:
            sty = f"{style} {INLINE.get(nm, '')}".strip()
            _inline(c, t, sty)
            if nm == 'a' and (c.attrs or {}).get('href'): t.append(f" ({c.attrs['href']})", style='dim')
    return t

def _find(el, name):
    for c in getattr(el, 'children', []):
        if getattr(c, 'name', None) == name: yield c
        else: yield from _find(c, name)

def _code_block(pre, theme='ansi_dark'):
    code = next(_find(pre, 'code'), None)
    if code is None: return _inline(pre, Text())
    src = ''.join(c.text for c in code.children if getattr(c, 'name', None) == '#text')
    cls = (code.attrs or {}).get('class') or ''
    lang = next((w.removeprefix('language-') for w in cls.split() if w.startswith('language-')), 'text')
    return Syntax(src.rstrip('\n'), lang, theme=theme)

def _list(el, ordered, depth=0):
    t, i = Text(), 1
    for li in el.children:
        if getattr(li, 'name', None) != 'li': continue
        if t.plain: t.append('\n')
        t.append('  ' * depth + (f'{i}. ' if ordered else '• '), style='bold')
        for c in li.children:
            nm = getattr(c, 'name', None)
            if nm in ('ul', 'ol'):
                t.append('\n')
                t.append(_list(c, nm == 'ol', depth + 1))
            elif nm == '#text': t.append(c.text)
            else: _inline(c, t)
        i += 1
    return t

def _quote(el):
    inner = Text()
    for c in el.children:
        if getattr(c, 'name', None) == 'p':
            if inner.plain: inner.append('\n')
            _inline(c, inner)
    out = Text()
    for j, line in enumerate(inner.split('\n')):
        if j: out.append('\n')
        out.append('│ ', style='dim')
        out.append(line)
    return out

def _table(el):
    t = Text()
    for tr in _find(el, 'tr'):
        if t.plain: t.append('\n')
        cells = [_inline(td, Text()).plain for td in tr.children if getattr(td, 'name', None) in ('td', 'th')]
        t.append(' │ '.join(cells))
    return t

def node2rich(el, theme='ansi_dark'):
    nm = el.name
    if nm in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
        return _inline(el, Text('#' * int(nm[1]) + ' ', style='bold magenta'), 'bold magenta')
    if nm == 'p': return _inline(el, Text())
    if nm == 'pre': return _code_block(el, theme)
    if nm in ('ul', 'ol'): return _list(el, nm == 'ol')
    if nm == 'blockquote': return _quote(el)
    if nm == 'hr': return Text('─' * 40, style='dim')
    if nm == 'table': return _table(el)
    return _inline(el, Text())

def md_blocks(md, theme='ansi_dark', **kw):
    "One Rich renderable per top-level Markdown block, code highlighted with `theme`."
    out = []
    for c in mdhtml.to_dom(md, **kw).children:
        if getattr(c, 'name', None) == '#text':
            if c.text.strip(): out.append(Text(c.text.strip()))
        else: out.append(node2rich(c, theme))
    return out

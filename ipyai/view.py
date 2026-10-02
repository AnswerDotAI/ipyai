"""The transcript view: each Message's display items, kept in step with teleprint blocks.

`View.items` turns one Message into display items, each keyed `f'{m.id}:{part}'`: `in` or `ask` for its
input, `out{i}` for output `i` (a run of same-name stream outputs merges into one), `r{j}` for reply span
`j`, and `tool:{id}` for a tool call. `View.sync` compares them with the blocks it already holds for that
Message: equal items stay, changed ones are replaced in place, new ones go after their predecessor, and
missing ones are dropped. Every Dialog change reaches the screen this way: live output, hide, edit,
truncation and resume. Blocks that belong to no Message are notes, which only this view knows about."""
import asyncio, base64, hashlib, io
from dataclasses import dataclass
from typing import Callable
import mdhtml
from kittytgp import render_parts, png_size, fit_grid, PNG_SIGNATURE
from rich.syntax import Syntax
from rich.text import Text
from aidialog.msg_parts import parse_tools, tool_text
from mdhtml2term import md_blocks

__all__ = ['View', 'Item', 'gutter', 'hl', 'tool_call']

GUTTERS = {'sh': ('$$$ ', 'bold yellow'), 'in': ('»»» ', 'bold green'), 'out': ('««« ', 'bright_blue'), 'result': ('««« ', 'bright_blue'),
    'error': ('««« ', 'red'), 'image': ('««« ', 'magenta'), 'ask': ('››› ', 'bold magenta'), 'ai': ('‹‹‹ ', 'magenta'),
    'tool': ('≡≡≡ ', 'yellow')}

def gutter(kind):
    r"""The block's left edge: 3 type glyphs + space (`x\dx`: the middle cell carries the ambient
    alt-digit number when the block wears one), color marking direction at a glance; the
    continuation rows are a dim `··· `."""
    f, sty = GUTTERS[kind]
    return (Text(f, style=sty), Text('··· ', style='dim'))

def hl(code, theme='ansi_dark'):
    "Syntax-highlight `code` the one true way, dropping the newline highlight() appends."
    t = Syntax('', 'python', theme=theme).highlight(code)
    if t.plain.endswith('\n'): t.right_crop(1)
    return t

def _text(t): return ''.join(t) if isinstance(t, list) else (t or '')

def _trunc(v, mx=40):
    s = v if isinstance(v, str) else repr(v)
    s = s.replace('\n', '\\n')
    return s if len(s) <= mx else s[:mx-1] + '…'

def tool_call(name, args, mx=40):
    "Compact `name(k=v, ...)` line for a tool call, values truncated for one-line display."
    if not args: return f'{name}()'
    return f"{name}({', '.join(f'{k}={_trunc(v, mx)}' for k, v in sorted(args.items()))})"

def _to_png(img):
    "Image bytes in any format PIL reads, as PNG: kitty transmits PNG."
    from PIL import Image
    buf = io.BytesIO()
    Image.open(io.BytesIO(img)).save(buf, format='PNG')
    return buf.getvalue()

def msg_id(key):
    "The Message id in block key `key`, or None for a note."
    return key.partition(':')[0] if ':' in key else None

@dataclass
class Item:
    "One display item of a Message. Equal `sig`s render the same, so `make` runs only when the sig changes."
    key: str
    sig: tuple
    kind: str             # gutter kind
    make: Callable        # -> list of Rich renderables
    source: str = None    # the text search matches and copy yields
    fold: int = None      # collapse_at
    glass: bool = False   # shell output the shell already printed on the terminal during its borrow

class View:
    """Keeps teleprint's blocks in step with the session Dialog `dlg`. `mark` queues a Message and one
    sync per event-loop pass handles the queue. `live` holds the ids of prompts whose reply is still
    streaming: the last span of such a reply renders dim."""
    def __init__(self, comp, theme='ansi_dark'):
        self.comp, self.theme, self.kitty = comp, theme, False
        self.dlg = None
        self.keys, self.sigs = {}, {}   # message id -> its block keys in order; block key -> the (sig, dim) it shows
        self.live, self.notes = set(), []
        self._queue, self._scheduled, self._nnotes = {}, False, 0
        self._imgs = {}                 # image sha1 -> [png, width, height, placeholder or None]

    @property
    def collapse_at(self):
        "Auto-collapse threshold for outputs: about half a screen, always small enough to fold while visible."
        return max(3, min(self.comp.rows // 2, self.comp.rows - 5))

    def mark(self, m, live=None, glass=False):
        "Queue `m` for a sync. `live` marks its reply as streaming or finished; `glass` as in `sync`."
        if live is True: self.live.add(m.id)
        elif live is False: self.live.discard(m.id)
        self._queue[m.id] = (m, glass or self._queue.get(m.id, (m, False))[1])
        try: loop = asyncio.get_running_loop()
        except RuntimeError: return self.flush()
        if not self._scheduled:
            self._scheduled = True
            loop.call_soon(self.flush)

    def flush(self):
        "Sync every queued Message."
        self._scheduled = False
        q, self._queue = self._queue, {}
        for m, glass in q.values(): self.sync(m, glass)

    def _is_last(self, m):
        "Whether `m` is the Dialog's last Message, or shown without being in it"
        msgs = self.dlg.messages if self.dlg is not None else []
        return not msgs or msgs[-1] is m or not any(x is m for x in msgs)

    def _last_key_before(self, m):
        "The last block key of the Messages before `m`, or None"
        msgs = self.dlg.messages
        i = next(j for j, x in enumerate(msgs) if x is m)
        return next((ks[-1] for x in reversed(msgs[:i]) if (ks := self.keys.get(x.id))), None)

    def sync(self, m, glass=False):
        """Bring the blocks for `m` in line with its items. With `glass`, new shell output items are
        recorded without being painted, because the shell already printed them."""
        its, dim = self.items(m), bool(m.skipped)
        old = self.keys.get(m.id, [])
        new = [it.key for it in its]
        if gone := [k for k in old if k not in new]:
            self.comp.drop(*gone)
            for k in gone: self.sigs.pop(k, None)
        prev = None if self._is_last(m) else self._last_key_before(m)
        for it in its:
            exists = it.key in self.comp.blocks
            if not exists or self.sigs.get(it.key) != (it.sig, dim):
                self.comp.put(it.key, *it.make(), gutter=gutter(it.kind), source=it.source, collapse_at=it.fold, dim=dim,
                    after=None if exists else prev, ink=not (glass and it.glass))
                self.sigs[it.key] = (it.sig, dim)
            prev = it.key
        self.keys[m.id] = new

    def drop(self, ids):
        "Remove the blocks of the Messages `ids`, as after a truncation."
        keys = [k for mid in ids for k in self.keys.pop(mid, [])]
        for mid in ids:
            self._queue.pop(mid, None)
            self.live.discard(mid)
        for k in keys: self.sigs.pop(k, None)
        if keys: self.comp.drop(*keys)

    def reset(self, dlg, show=True):
        """Show Dialog `dlg` in place of the current one. The old Dialog's rows stay on screen as printed
        trace, and its blocks and the notes leave the model. With `show`, every Message of `dlg` is shown."""
        old = [x.id for x in self.dlg.messages] if self.dlg is not None else []
        keys = [k for mid in old for k in self.keys.get(mid, [])] + self.notes
        if keys:
            self.comp.commit()
            self.drop(old)
            self.comp.drop(*self.notes)
        self.dlg, self.notes = dlg, []
        if show:
            for m in dlg.messages: self.sync(m)

    def note(self, text, kind='out'):
        "Show `text` as a note: a block that belongs to no Message, dim, or red for kind 'error'."
        self._nnotes += 1
        key = f'note{self._nnotes}'
        t = Text.from_ansi(text, style='red' if kind == 'error' else 'dim')
        self.comp.put(key, t, gutter=gutter(kind), source=t.plain)
        self.notes.append(key)

    # -- display items -------------------------------------------------------------
    def items(self, m):
        "The display items of Message `m`, in order."
        t = m.msg_type
        if t == 'prompt': return [self._ask(m)] + self._reply(m)
        if t == 'note':
            s = m.content
            return [Item(f'{m.id}:in', ('note', s), 'in', lambda: [Text(s)], s)]
        if t != 'code': return []
        if m.content.startswith('!'):
            cmd = m.content[1:]
            return [Item(f'{m.id}:in', ('sh', cmd), 'sh', lambda: [Text(cmd)], cmd)] + self._outs(m, shell=True)
        src, theme = m.content, self.theme
        return [Item(f'{m.id}:in', ('code', src, theme), 'in', lambda: [hl(src, theme)], src)] + self._outs(m)

    def _outs(self, m, shell=False):
        """Items for `m`'s outputs. Consecutive stream outputs with the same name merge into one. When the
        cell's `execute_result` image has the same bytes as one of its `display_data` images, the image shows
        once: IPython sends a cell's final `fig` both as the cell's result and from the inline backend's flush.
        Identical images from separate `display` calls all show."""
        runs, imgs = [], {}   # image payload -> the output type that first showed it
        for i, o in enumerate(m.output or []):
            if o.get('output_type') != 'stream':
                data = o.get('data', {})
                img = next((_text(data[k]) for k in ('image/png', 'image/jpeg') if k in data), None)
                if img is not None:
                    ot, prev = o.get('output_type'), imgs.get(img)
                    if prev is not None and 'execute_result' in (ot, prev): continue
                    imgs[img] = ot
                runs.append((i, o))
            elif runs and isinstance(runs[-1], list) and runs[-1][1] == o.get('name'): runs[-1][2].append(_text(o.get('text')))
            else: runs.append([i, o.get('name'), [_text(o.get('text'))]])
        its = [self._stream(m, *r, shell) if isinstance(r, list) else self._rich(m, *r) for r in runs]
        return [it for it in its if it is not None]

    def _stream(self, m, i, name, txts, shell):
        s = ''.join(txts).rstrip('\n')
        if not s: return None
        err = name == 'stderr'
        return Item(f'{m.id}:out{i}', ('stream', name, s), 'error' if err and shell else 'out',
            lambda: [Text(s, style='red' if err else '')], s, fold=self.collapse_at, glass=shell and not err)

    def _rich(self, m, i, o):
        key, ot = f'{m.id}:out{i}', o.get('output_type')
        if ot in ('execute_result', 'display_data'):
            data = o.get('data', {})
            mime = next((k for k in ('image/png', 'image/jpeg') if k in data), None)
            if mime: return self._image(key, data[mime])
            if 'text/plain' in data:
                t = _text(data['text/plain'])
                return Item(key, ('result', t), 'result', lambda: [t], t, fold=self.collapse_at)
        elif ot == 'error':
            tb = '\n'.join(o.get('traceback', []))
            return Item(key, ('error', tb), 'error', lambda: [Text.from_ansi(tb)], Text.from_ansi(tb).plain)
        return None

    def _image(self, key, data):
        "An image output: a kitty Unicode-placeholder grid, or a text note on terminals without kitty graphics."
        raw = base64.b64decode(_text(data))
        h = hashlib.sha1(raw).hexdigest()
        if h not in self._imgs:
            png = raw if raw.startswith(PNG_SIGNATURE) else _to_png(raw)
            w, ht = png_size(png)
            self._imgs[h] = [png, w, ht, None]
        _, w, ht, _ = self._imgs[h]
        return Item(key, ('image', h, self.kitty), 'image', lambda: [self._placeholder(h)], f'[image {w}x{ht}px]')

    def _placeholder(self, h):
        "The placeholder for image `h`. The first call transmits the image to the terminal; later calls reuse it."
        png, w, ht, ph = self._imgs[h]
        if not self.kitty: return Text(f'[image {w}x{ht}px -- this terminal lacks kitty graphics]')
        if ph is None:
            c, r = fit_grid(w, ht, max(1, min(40, self.comp.cols - 2)))
            transmit, placeholder = render_parts(png, cols=c, rows=r)
            self.comp.tty.write(transmit)
            ph = self._imgs[h][3] = Text.from_ansi(placeholder)
        return ph

    def _ask(self, m):
        s = m.content
        return Item(f'{m.id}:ask', ('ask', s), 'ask', lambda: [Text(s)], s)

    def _reply(self, m):
        "Items for a prompt's reply: one per top-level Markdown span, and one per tool call."
        res = m.ai_res or ''
        if not res.strip(): return []
        live, segs, out, j = m.id in self.live, parse_tools(res), [], 0
        for si, (txt, d) in enumerate(segs):
            if txt and txt.strip():
                lines, spans = txt.split('\n'), mdhtml.blocks(txt)
                for k, sp in enumerate(spans):
                    src = '\n'.join(lines[sp['start']:sp['end']])
                    if not src.strip(): continue
                    partial = live and d is None and si == len(segs) - 1 and k == len(spans) - 1
                    out.append(self._span(f'{m.id}:r{j}', src, partial, sp['type'] == 'code_block'))
                    j += 1
            if d: out.append(self._tool(m, d))
        return out

    def _span(self, key, src, partial, code):
        "A reply span: dim plain text while it may still grow, else rendered Markdown. Code folds at `collapse_at`."
        theme = self.theme
        if partial: return Item(key, ('md', src, True), 'ai', lambda: [Text(src, style='dim')], src, fold=self.collapse_at)
        return Item(key, ('md', src, False, theme), 'ai', lambda: md_blocks(src, theme=theme) or [Text(src)], src,
            fold=self.collapse_at if code else None)

    def _tool(self, m, d):
        "A tool call and its result, collapsed to the call line."
        call = tool_call(d.get('name') or 'tool', d.get('args') or {})
        res = tool_text(d.get('result')).rstrip('\n')
        body = [Text(call, style='bold')] + ([Text(res, style='dim')] if res.strip() else [])
        return Item(f"{m.id}:tool:{d.get('id')}", ('tool', call, res), 'tool', lambda: body, '\n'.join(p for p in (call, res) if p), fold=1)

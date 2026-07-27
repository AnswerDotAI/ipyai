"""Streaming AI reply -> transcript blocks.

The one genuinely new design (DEV cp3a.3): accumulate streamed markdown and watch
`mdhtml.blocks` top-level source spans. When a span closes (a later span exists),
render it via mdrich and it becomes a finished block -- the first closure REPLACES
the in-flight partial in place (it is the last, still-live block). The partial
itself streams as dim plain text, folding at its threshold so a giant fence can
never flood the screen or outrun the repaintable zone. Tool calls and thinking
interrupt the flow: the partial flushes final, they render as their own
collapsed-by-default blocks, and a fresh partial follows."""
import mdhtml
from rich.text import Text
from .mdrich import md_blocks
from aidialog.msg_parts import Part, PartType

def _trunc(v, mx=40):
    s = v if isinstance(v, str) else repr(v)
    s = s.replace('\n', '\\n')
    return s if len(s) <= mx else s[:mx-1] + '…'

def tool_call(name, args, mx=40):
    "Compact `name(k=v, ...)` line for a tool call, values truncated for one-line display."
    if not args: return f'{name}()'
    return f"{name}({', '.join(f'{k}={_trunc(v, mx)}' for k, v in sorted(args.items()))})"

class Grower:
    "A live block re-rendered whole per update, folding at `collapse_at` as it grows."
    def __init__(self, comp, gutter, tag, collapse_at=None):
        self.comp, self.gutter, self.tag, self.collapse_at = comp, gutter, tag, collapse_at
        self.blk = None

    def update(self, *parts):
        if self.blk is None:
            self.blk = self.comp.print_block(parts[0], gutter=self.gutter, tag=self.tag, collapse_at=self.collapse_at)
            for p in parts[1:]: self.blk.body.append(p)
            if len(parts) > 1: self.comp.refresh_block(self.blk)
        else:
            self.blk.body = list(parts)
            self.comp.refresh_block(self.blk)
        if self.collapse_at and not self.blk.collapsed and self.blk.height > self.collapse_at:
            self.blk.collapsed = True
            self.comp.refresh_block(self.blk)

    def fold(self):
        "Collapse now (the block is finished and worth one line), keeping it toggleable."
        if self.blk is not None and not self.blk.collapsed and self.blk.height > 1:
            self.blk.collapsed = True
            self.comp.refresh_block(self.blk)

class ReplyRenderer:
    "Streamed markdown as blocks: closed top-level spans print final, the last span streams dim."
    def __init__(self, comp, gutter, theme='ansi_dark', collapse_at=None):
        self.comp, self.gutter, self.theme, self.collapse_at = comp, gutter, theme, collapse_at
        self.md = ''       # markdown accumulated since the last flush
        self.nfinal = 0    # spans already rendered final
        self.partial = None

    def _renderables(self, src):
        return md_blocks(src, theme=self.theme) or ([Text(src)] if src.strip() else [])

    def _finalize(self, span, lines):
        src = '\n'.join(lines[span['start']:span['end']])
        rs = self._renderables(src)
        if not rs:
            self._drop_partial()
            return
        fold = self.collapse_at if span['type'] == 'code_block' else None
        if self.partial is not None and self.partial.blk is not None:
            g = self.partial
            g.collapse_at = g.blk.collapse_at = fold
            # an already-folded partial stays folded: un-collapsing here would repaint the whole body,
            # flooding the screen and letting progressive commit tear a block we promised to keep to one line
            g.update(*rs)
            g.blk.source = src
            self.partial = None
        else:
            self.comp.print_block(rs[0], gutter=self.gutter, tag='ai', collapse_at=fold, source=src)
            for r in rs[1:]: self.comp.print_block(r, gutter=self.gutter, tag='ai', collapse_at=fold)

    def _drop_partial(self):
        "A span rendered to nothing: blank the partial that was showing it (rare: whitespace-only)."
        if self.partial is not None and self.partial.blk is not None: self.partial.update(Text(''))
        self.partial = None

    def feed(self, delta):
        self.md += delta
        spans = mdhtml.blocks(self.md)
        lines = self.md.split('\n')
        while len(spans) - 1 > self.nfinal:
            self._finalize(spans[self.nfinal], lines)
            self.nfinal += 1
        if spans:
            src = '\n'.join(lines[spans[-1]['start']:])
            if src.strip():
                if self.partial is None: self.partial = Grower(self.comp, self.gutter, 'ai', self.collapse_at)
                self.partial.update(Text(src, style='dim'))
                self.partial.blk.source = src

    def flush(self):
        "Finalize everything (turn done, or a tool block is about to interrupt the flow)."
        spans = mdhtml.blocks(self.md)
        lines = self.md.split('\n')
        for span in spans[self.nfinal:]: self._finalize(span, lines)
        self.md, self.nfinal, self.partial = '', 0, None

class TurnRenderer:
    """One AI turn's events painted as blocks. `gutters` maps kind ('ai', 'think', 'tool') to a
    (first, continuation) styled-Text pair, the app's gutter language."""
    def __init__(self, comp, gutters, theme='ansi_dark', collapse_at=None):
        self.comp, self.gutters, self.collapse_at = comp, gutters, collapse_at
        self.md = ReplyRenderer(comp, gutters('ai'), theme=theme, collapse_at=collapse_at)
        self.think = None      # (Grower, accumulated text) while thinking streams
        self.tool = None       # (Grower, call Text, output str) while a tool/command runs

    def _close_think(self):
        if self.think is None: return
        self.think[0].fold()
        self.think = None

    def _interrupt(self):
        "A non-markdown block is about to print: flush the partial so only the newest block grows."
        self._close_think()
        self.md.flush()

    def _tool_open(self, call):
        self._interrupt()
        g = Grower(self.comp, self.gutters('tool'), 'tool')
        line = Text(call, style='bold')
        g.update(line)
        self.tool = (g, line, '')

    def _tool_update(self, delta='', result=None, error=False):
        if self.tool is None: return
        g, line, out = self.tool
        out += delta
        body = [line] + ([Text(out.rstrip('\n'), style='red' if error else 'dim')] if (out.strip() or result) else [])
        if result is not None and result.strip():
            body = [line, Text(result.rstrip('\n'), style='red' if error else 'dim')]
        g.update(*body)
        g.blk.source = '\n'.join(p.plain for p in body)
        self.tool = (g, line, out)
        if result is not None:
            g.fold()   # collapsed by default: the call line summarizes, the result is a toggle away
            self.tool = None

    def event(self, e):
        """Dispatch one fastllm stream item: dicts carry text/thinking deltas, `Part`s carry
        tool calls and results, and anything else (Completion, ModelResponse) is bookkeeping.
        Plain strs also feed the markdown flow, for replays and tests."""
        if isinstance(e, Part):
            call = tool_call((e.data or {}).get('name') or 'tool', (e.data or {}).get('arguments') or {})
            if e.type == PartType.tool_use: self._tool_open(call)
            elif e.type == PartType.tool_result:
                if self.tool is None: self._tool_open(call)
                self._tool_update(result=str(e.text or ' '))
            return
        if isinstance(e, str):
            if e:
                self._close_think()
                self.md.feed(e)
            return
        if not isinstance(e, dict): return
        if thk := e.get('thinking'):
            if self.think is None:
                self.md.flush()
                self.think = (Grower(self.comp, self.gutters('think'), 'think', collapse_at=self.collapse_at), '')
            g, acc = self.think
            acc += thk
            if acc.strip():
                g.update(Text(acc.strip(), style='dim italic'))
                g.blk.source = acc.strip()
            self.think = (g, acc)
        elif text := e.get('text'):
            self._close_think()
            self.md.feed(text)

    def done(self):
        self._close_think()
        if self.tool is not None: self._tool_update(result=self.tool[2] or ' ')
        self.md.flush()

    def stopped(self):
        "User interrupt: freeze what streamed, say so."
        self.done()
        self.comp.print_block(Text('interrupted', style='dim'), gutter=self.gutters('error'), tag='ai')

"The ipyai terminal app: the teleprint UI over a Controller, which owns the session Dialog and the running operation."
import asyncio, os, re, shlex, signal, subprocess, tempfile, time
from contextlib import AsyncExitStack
from typing import Annotated
from pathlib import Path
from kittytgp import kitty_probe, kitty_supported, kitty_env_hint
from rich.text import Text
from rich.style import Style
from rich.cells import cell_len
from fastcore.basics import str_enum
from fastcore.script import call_parse
from teleprint.buffer import Buffer
from teleprint.compositor import Compositor
from teleprint.tty import RealTty
from teleprint.widgets import CompletionMenu, Tooltip, Signature
from teleprint.transcript import TranscriptView
import aidialog.ipynb  # noqa: F401 -- activates the Message.cell_meta/to_cell patches (meta_attrs serialization)
from aidialog.ipynb import read_ipynb, write_ipynb
from .kernel import KernelSession
from .history import History
from .session import list_sessions, resolve_session
from .assistant import Assistant, route, code_blocks
from .config import load_config
from .controller import Controller
from .view import View, gutter, hl, msg_id

HINT = 'ipyai ng⋄Enter runs; Tab completes; S-Tab inspects; C-T transcript; C-C interrupts; M-p mode; C-D quits'  # S/C/M = shift/ctrl/alt
OSC_BG_RE = re.compile(rb'\x1b\]11;rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)')
DA1_RE = rb'\x1b\[\?[0-9;]*c'   # every terminal answers DA1: the fence that ends a probe

def _fmt_tk(n):
    "Tk counts as compact k/M strings for the ctx meter."
    if n >= 1e6: return f'{n / 1e6:.1f}M'.replace('.0M', 'M')
    if n >= 1000: return f'{n / 1000:.1f}k'.replace('.0k', 'k')
    return str(n)

def _layout(lines, cur, width, limit):
    """`lines`, one `Text` per logical line, cut into rows at most `width` cells wide and joined into one
    renderable, with the cursor's `(row, col)` in it. `cur` is the cursor's `(line, character offset)`. When
    there are more than `limit` rows, only a window of `limit` rows shows, and it includes the cursor's row."""
    rows, crow, ccol = [], 0, 0
    for i, t in enumerate(lines):
        offs, acc = [0], 0
        for j, ch in enumerate(t.plain):
            w = cell_len(ch)
            if acc and acc + w > width: offs, acc = offs + [j], 0
            acc += w
        parts = list(t.divide(offs[1:]))
        if i == cur[0]:
            r = max(k for k, o in enumerate(offs) if o <= cur[1])
            ccol = cell_len(t.plain[offs[r]:cur[1]])
            if ccol >= width: parts, r, ccol = parts + [Text('')], r + 1, 0  # at the end of a full row, the cursor starts the next
            crow = len(rows) + r
        rows += parts
    top = min(max(0, crow - limit + 1), max(0, len(rows) - limit))
    shown = Text('\n').join(rows[top:top + limit])
    shown.no_wrap, shown.overflow = True, 'crop'
    return shown, (crow - top, ccol)

_NTH = {'!': 1, '@': 2, '#': 3, '$': 4, '%': 5, '^': 6, '&': 7, '*': 8, '(': 9}  # alt-shift digits arrive shifted

def _default_history():
    return History()  # nav + ghost text draw only on this directory's session files

class App:
    """The terminal UI over a `Controller`: keys, input focus, the composer and status line, and the `ui`
    methods the controller calls. The test harness can supply `tty`."""
    def __init__(self, tty, kernel=None, history='default', cfg=None, assistant=None):
        self.tty = tty
        self.cfg = cfg or {}
        self.hist = _default_history() if history == 'default' else history  # None = off (tests want hermeticity)
        self.comp = Compositor(tty)  # main awaits comp.start() before painting; tests mostly skip it
        self.buf = Buffer()
        self.menu = None     # a teleprint CompletionMenu while completing; Tab/shift+Tab cycle, Enter accepts
        self.tip = None      # a teleprint Tooltip while inspecting (shift+Tab on a bare buffer)
        self.inp = None      # a pending kernel input request: dict(buf, prompt, password, fut)
        self.done = asyncio.Event()
        self.kitty = False   # set by detect_kitty(); images fall back to a note without it
        self.theme = self.cfg.get('code_theme', 'auto')  # 'auto' resolves via detect_theme (OSC 11)
        if self.theme == 'auto': self.theme = 'ansi_dark'
        self.mode = 'prompt' if self.cfg.get('prompt_mode') else 'code'  # 'prompt'|'code'|'shell': M-p/M-c/M-s
        if self.hist and self.mode != 'code': self.hist.mode = self.mode; self.hist.refresh()  # match a prompt_mode start
        self.ai_sugg = None      # (text, cursor, suggestion): an Alt-. suggestion, valid while the buffer is unchanged
        self._cycle = dict(idx=-1, resp='')  # alt-shift-up/down cycling over the last reply's fenced blocks
        self._quit_warned = False
        self.picker = None       # startup session-picker rows while open (an over transient; owns digits/Enter/n/Esc)
        self._picked = None      # the future choose_session awaits while the picker is open
        self._ipyai_comm = None  # the kernel-side %ipyai comm id, set on comm_open
        self._pending = None     # input queued while an enter decision's round-trip is in flight (see on_enter)
        self.editing = None      # (message, kind) while the transcript view's e edit owns the composer
        self.retry = None        # (message, kind): a kind-matched submit replaces from that exchange instead of appending
        self.tv = TranscriptView(self.comp, self._tail_content)
        self.view = View(self.comp, theme=self.theme)
        self.k = KernelSession() if kernel is None else kernel
        if assistant is None and cfg: assistant = Assistant(cfg=cfg)
        self.ctl = Controller(self, self.k, assistant)
        self.ctl.on_comm = self._on_comm
        self.comp.on_key = self.on_key
        self.comp.on_paste = self.on_paste
        self.comp.on_mouse = lambda ev: self.tv.on_mouse(ev)
        self.comp.on_wheel = self._on_wheel
        self.comp.on_act = self._on_act
        self.comp.numbering = True  # ambient alt-digit numbers on the newest toggleable blocks

    # -- the ui the controller calls ---------------------------------------------------
    def mark(self, m, live=None, glass=False): self.view.mark(m, live=live, glass=glass)
    def drop(self, ids): self.view.drop(ids)
    def reset(self, dlg, show=True): self.view.reset(dlg, show=show)
    def note(self, text, kind='out'): self.view.note(text, kind)

    @property
    def size(self): return self.tty.size

    def borrow(self, on_resize=None):
        "Lend the real terminal to a foreground program (see `Compositor.borrow`)."
        self._dismiss()
        return self.comp.borrow(on_resize=on_resize)

    async def ask_input(self, prompt, password):
        "Answer a kernel input request from a one-line input in the tail, with typed characters masked when `password` is set."
        if self.tv.active: self.tv.leave()
        self._dismiss()
        fut = asyncio.get_running_loop().create_future()
        self.inp = dict(buf=Buffer(), prompt=prompt, password=password, fut=fut)
        self.paint()
        try: return await fut
        finally:
            self.inp = None
            self.paint()

    def restore_input(self, text, mode):
        "Give a submission back to the composer, as when a turn stops before producing anything. Text typed since is kept."
        if self.buf.text: return
        self._set_mode(mode)
        self.buf.text, self.buf.cursor = text, len(text)
        self.paint()

    @property
    def busy(self):
        "Enter is gated on this: an operation owns the transcript's live edge."
        return self.ctl.busy

    # -- the tail ----------------------------------------------------------------------
    _SPIN = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    MODE = dict(code=('»»» ', 'bold green'), prompt=('››› ', 'bold magenta'), shell=('$$$ ', 'bold yellow'))

    def _status(self):
        s = HINT
        a = self.ctl.assistant
        if a is not None and (ctx := a.ctx_usage):
            used, mx = ctx
            s += f'⋄ctx {_fmt_tk(used)}/{_fmt_tk(mx)} ({100 * used // mx}%)'
        return s

    def _tail_content(self):
        "Tail layout: dim status above the prompt (a shell's status-line shape). Transients (menu/tooltip) ride `over` in paint(), directly above the status row, and never ink."
        spin = self._SPIN[int(time.monotonic() * 10) % len(self._SPIN)] if self.busy else ' '  # one cell says busy; the ticker animates it
        seg = Text(f'[{self.mode}]', style=Style(meta={'act': 'mode'}))  # click cycles the mode; M-p/c/s pick directly
        status = Text(spin + ' ') + (Text('↻', style='bold yellow') if self.retry else Text('')) + seg + Text('⋄' + self._status(), style='dim')  # ↻: a submit replaces from the recalled exchange
        status.truncate(self.comp.cols, overflow='ellipsis')  # one row always: small screens truncate, never wrap
        width, limit = self.comp.cols, max(1, self.comp.rows - 2)
        if self.inp is not None:
            b, prompt = self.inp['buf'], self.inp['prompt'].rsplit('\n', 1)[-1]
            line = Text()
            line.append(prompt, style='bold')
            line.append('•' * len(b.text) if self.inp['password'] else b.text)
            composer, (row, col) = _layout([line], (0, len(prompt) + b.cursor), width, limit)
            return [status, composer], (1, row, col)
        text = self.buf.text
        plain = self.mode != 'code' or text.startswith(('.', '!'))  # only code-as-code highlights
        hlt = Text(text) if plain else hl(text, self.theme)
        if not text.startswith(hlt.plain): hlt = Text(text)  # the highlighter expanded tabs, so its offsets differ from the buffer's
        mark, msty = self.MODE[self.mode]
        lines, pos = [], 0
        for i, s in enumerate(text.split('\n')):
            t = Text()  # empty base style: a styled first arg would become the BASE for the whole line, bolding the code
            t.append(mark if i == 0 else '··· ', style=msty)
            t.append(hlt[pos:pos + len(s)])
            lines.append(t)
            pos += len(s) + 1
        sugg = ''
        if self.buf.cursor == len(text) and not self.menu:
            if self.ai_sugg and self.ai_sugg[0] == text and self.ai_sugg[1] == self.buf.cursor:
                sugg = self.ai_sugg[2]  # Alt-. suggestion overrides history while the document is unchanged
            elif self.hist: sugg = self.hist.suggest(text)
        self.buf.suggestion = sugg
        if sugg: lines[-1].append(sugg, style='dim')
        elif not text and not self.ctl.kernel_busy:  # empty composer: hint the two ways out of this mode
            lines[-1].append(' · '.join(f'M-{m[0]} {m}' for m in self.MODE if m != self.mode), style='dim')
        before = text[:self.buf.cursor]
        cur = (before.count('\n'), 4 + len(before.rsplit('\n', 1)[-1]))  # every line mark is 4 characters
        composer, (row, col) = _layout(lines, cur, width, limit)
        return [status, composer], (1, row, col)

    def paint(self):
        if self.tv.active: self.tv.draw()
        else:
            lines, cursor = self._tail_content()
            over = self._picker_rows() if self.picker is not None else \
                   [self.menu.renderable()] if self.menu else [self.tip.renderable()] if self.tip else []
            self.comp.set_tail(*lines, cursor=cursor, over=over)

    def _set_mode(self, m):
        "Switch composer mode, repointing history at the mode's own past (see History.refresh)."
        self.mode = m
        if self.hist:
            self.hist.mode = m
            self.hist.refresh()

    def _on_act(self, token):
        "Clicked chrome: the status-bar mode segment cycles prompt -> code -> shell (M-p/c/s pick directly)."
        if token.startswith('pick:'): return self._pick(int(token[5:]))
        if token == 'mode':
            order = ['prompt', 'code', 'shell']
            self._set_mode(order[(order.index(self.mode) + 1) % 3])
            self._dismiss()
            self.paint()

    def _on_wheel(self, d):
        """Wheel on the main screen. Up inside tmux hands the gesture back to tmux copy-mode (`-e`
        exits at the bottom, resuming clicks) -- native scrollback is where the inked history
        lives. Outside tmux there is no portable way in, so the transcript view stands in."""
        if d >= 0: return
        if os.environ.get('TMUX'): subprocess.run(['tmux', 'copy-mode', '-eu'])
        else:
            self.tv.enter()
            self.paint()

    def _picker_rows(self):
        "The startup session picker as over-transient rows: digit picks (click too), Enter newest, n/Esc fresh."
        out = [Text(' resume a session in this directory ', style='reverse')]
        for i, (path, mtime, n, first) in enumerate(self.picker, 1):
            t = Text(f' {i} ', style=Style(reverse=True) + Style(meta={'act': f'pick:{i - 1}'}))
            t.append(f' {Path(path).stem[:8]}  {n} prompt{"s" if n != 1 else ""}  ', style=Style(meta={'act': f'pick:{i - 1}'}))
            t.append((first or '').replace('\n', ' ')[:50], style='dim')
            out.append(t)
        out.append(Text(' Enter: newest · digit: pick · n/Esc: fresh ', style='dim'))
        return out

    def _pick(self, i):
        "Close the picker; `i` indexes the chosen row (None: start fresh)."
        rows, self.picker = self.picker, None
        if not self._picked.done(): self._picked.set_result(rows[i][0] if i is not None and rows else None)
        self.paint()

    def _picker_key(self, k):
        "The picker owns keys while open: modal, like the transcript view's vocabulary."
        if k.name == 'enter': self._pick(0)
        elif k.char and k.char.isdigit() and 0 < int(k.char) <= len(self.picker): self._pick(int(k.char) - 1)
        elif k.name in ('n', 'escape'): self._pick(None)

    # -- terminal probes -----------------------------------------------------------------
    async def detect_kitty(self, timeout=0.5):
        "Probe the tty for kitty graphics; env hints cover tmux, where probe replies are not routed to panes."
        data = await self.comp.query(kitty_probe()[:-3] + b'\x1b[c', DA1_RE, timeout)
        self.kitty = self.view.kitty = bool(kitty_supported(data) or kitty_env_hint())
        return self.kitty

    async def detect_theme(self, timeout=0.5):
        """OSC 11 background query: pick the highlighter theme for a light or dark terminal (silence
        means stay dark). tmux forwards the query to the attached client's terminal and relays the
        reply to the querying pane, unlike the kitty APC probe (see detect_kitty)."""
        m = OSC_BG_RE.search(await self.comp.query(b'\x1b]11;?\x1b\\\x1b[c', DA1_RE, timeout))
        if m:
            r, g, b = (int(ch[:2], 16) for ch in m.groups())
            self.theme = 'ansi_light' if (0.2126*r + 0.7152*g + 0.0722*b) / 255 > 0.5 else 'ansi_dark'
        self.view.theme = self.theme
        return self.theme

    # -- the %ipyai magic ------------------------------------------------------------------
    def _on_comm(self, mt, c):
        "%ipyai lands here (via the controller's dispatcher): track the comm, ack each command by request id."
        if mt == 'comm_open' and c.get('target_name') == 'ipyai': self._ipyai_comm = c.get('comm_id')
        elif mt == 'comm_close' and c.get('comm_id') == self._ipyai_comm: self._ipyai_comm = None
        elif mt == 'comm_msg' and c.get('comm_id') == self._ipyai_comm:
            d = c.get('data', {})
            try: reply = dict(req=d.get('req'), text=self._ipyai_cmd(list(d.get('cmd') or [])))
            except Exception as e: reply = dict(req=d.get('req'), error=str(e))
            self.k.kc.comm_msg(self._ipyai_comm, data=reply)

    def _ipyai_cmd(self, args):
        "One %ipyai command against app/assistant state; returns the ack text. Settings are session-only."
        a = self.ctl.assistant
        if a is None: raise RuntimeError('no assistant attached (plain-REPL mode)')
        if not args:
            s = [f'{k} = {getattr(a, k)}' for k in ('model', 'suggest_model', 'think')]
            s += [f'code_theme = {self.theme}', f'mode = {self.mode}', '',
                'commands: model | suggest_model | think | code_theme [VALUE], prompt, sessions, reset, save PATH, load PATH']
            return '\n'.join(s)
        cmd, *rest = args
        if cmd in ('model', 'suggest_model', 'think'):
            if rest: setattr(a, cmd, rest[0])
            return f'{cmd} = {getattr(a, cmd)}'
        if cmd == 'code_theme':
            if rest and rest[0] == 'auto':
                self.comp.spawn(self.detect_theme(), name='detect-theme')
                return 'code_theme = auto (detecting)'
            if rest: self.theme = self.view.theme = rest[0]
            return f'code_theme = {self.theme}'
        if cmd == 'prompt':
            self._set_mode('prompt' if self.mode != 'prompt' else 'code')
            self.paint()
            return f'mode = {self.mode}'
        if cmd == 'sessions': return _sessions_text(list_sessions())
        if cmd == 'reset':
            self.ctl.reset()
            return f'reset: fresh conversation (session {Path(self.ctl.dlg.path_).stem[:8]})'
        if cmd == 'save':
            if not rest: raise ValueError('usage: %ipyai save PATH')
            p = Path(rest[0]).expanduser()
            if p.suffix != '.ipynb': p = p.with_suffix('.ipynb')
            write_ipynb(self.ctl.dlg, p)
            return f'saved {len(self.ctl.dlg)} messages to {p}'
        if cmd == 'load':
            if not rest: raise ValueError('usage: %ipyai load PATH')
            return self.ctl.load(rest[0])
        raise ValueError(f'unknown %ipyai command: {cmd!r} (bare %ipyai lists commands)')

    # -- completion, inspection, suggestions --------------------------------------------------
    async def do_complete(self):
        snap = (self.buf.text, self.buf.cursor)
        matches, start = await self.k.kc.complete(self.buf.text, self.buf.cursor)
        if (self.buf.text, self.buf.cursor) != snap: return  # the buffer moved on: stale matches would edit newer text
        self.tip = None
        if matches:
            m = CompletionMenu(self.buf, matches, start)
            if len(matches) == 1:
                m.insert_common()
                self.menu = None
            else:
                self.menu = m
                m.cycle(1)  # auto-select the first match
        self.paint()

    async def do_inspect(self):
        """shift+Tab: inside a call, a Signature panel with the active param bold (the kernel's `sig_help`
        computes the signature and active index); otherwise the full inspect text as a tooltip."""
        snap = (self.buf.text, self.buf.cursor)
        lines = self.buf.text[:self.buf.cursor].split('\n')
        try: sigs = await self.k.kc.sig_help(code=self.buf.text, line_no=len(lines), col_no=len(lines[-1]))
        except Exception: sigs = []
        tip = None
        if sigs and isinstance(sigs, list):
            s = sigs[0]
            tip = Signature(s['label'], [p['desc'] for p in s['params']], s.get('idx'), s.get('doc', ''))
        elif text := await self.k.kc.inspect(self.buf.text, self.buf.cursor): tip = Tooltip(Text.from_ansi(text))
        if (self.buf.text, self.buf.cursor) != snap: return
        self.menu, self.tip = None, tip
        self.paint()

    async def do_ai_suggest(self):
        "Alt-.: one AI suggestion into the ghost-text slot, valid only while the document is unchanged."
        snap = (self.buf.text, self.buf.cursor)
        try: text = await self.ctl.assistant.suggest(self.ctl.dlg.messages, self.buf.text)
        except Exception: return
        if text and (self.buf.text, self.buf.cursor) == snap:
            self.ai_sugg = (snap[0], snap[1], text)
            self.paint()

    def _dismiss(self):
        self.menu = None
        self.tip = None

    # -- submitting ---------------------------------------------------------------------------
    def _submit(self, text, kind):
        "Clear the composer and record history. An armed retry fires here: a kind-matched submit truncates from its message first, a mismatch cancels it."
        if self.retry is not None:
            (m, want), self.retry = self.retry, None
            if kind == want: self.ctl.truncate(m)
        self.buf.clear()
        self.ai_sugg = None
        self._dismiss()
        if self.hist:
            self.hist.reset_nav()
            self.hist.add_local(text)  # instantly navigable/suggestible, before the session file saves

    def _drain(self):
        "Replay input queued during an enter round-trip, in order; a replayed enter re-queues the rest behind its own decision."
        if self._pending is None: return
        q, self._pending = self._pending, None
        for ev in q:
            if isinstance(ev, str): self.on_paste(ev)
            else:
                r = self.on_key(ev)
                if asyncio.iscoroutine(r): self.comp.spawn(r, name='replayed-key')  # same contract as the compositor's dispatcher

    async def on_enter(self):
        """Routed Enter (the `.`/`;`/`!`/`%` dispatch): prompts always submit -- English is never
        'incomplete' -- while code keeps the smart is_complete check (auto-indented continuation).
        Input arriving during the round-trip queues in `_pending` and replays once the operation has
        started, so a raw key burst cannot outrun the check and flatten into one line."""
        try:
            text = self.buf.text
            kind, payload = route(text, self.mode)
            if kind == 'code' and self.k.kc is not None:
                status, indent = await self.k.kc.check(payload)
                if self.buf.text != text or self.busy: return  # changed or a run started mid-flight: stale decision
                if status == 'incomplete':
                    self.buf.insert('\n' + ('' if self._pending else indent))  # a burst carries its own indentation
                    return
            self._submit(text, kind)
            await dict(prompt=self.ctl.run_prompt, job=self.ctl.run_shell, code=self.ctl.run_code)[kind](payload)
        finally:
            self._drain()
            self.paint()

    async def edit_buffer(self):
        """F2: the composer text in `$EDITOR`, run locally on the real terminal. A clean exit reloads the
        composer, and a nonzero one (vim's `:cq`) abandons the edit. Nothing is recorded."""
        if self.busy: return
        sfx = dict(code='.py', shell='.sh', prompt='.md')[self.mode]
        fd, path = tempfile.mkstemp(suffix=sfx, prefix='ipyai-f2-')
        with os.fdopen(fd, 'w') as f: f.write(self.buf.text)
        try:
            async with self.borrow():
                p = await asyncio.create_subprocess_exec(*shlex.split(os.environ.get('EDITOR', 'vi')), path)
                ec = await p.wait()
            if ec == 0:
                with open(path) as f: self.buf.text = f.read().rstrip('\n')
                self.buf.cursor = len(self.buf.text)
        finally: os.unlink(path)
        self.paint()

    # -- transcript view: hide, edit, retry ---------------------------------------------------
    def _recall_last(self):
        """Alt-r: the most recent exchange (prompt, code, or shell) into the composer, armed for
        retry -- a kind-matched submit REPLACES it (and everything after) instead of appending."""
        m = next((x for x in reversed(self.ctl.dlg.messages) if x.msg_type in ('prompt', 'code')), None)
        if m is None: return
        want = 'prompt' if m.msg_type == 'prompt' else 'job' if m.content.startswith('!') else 'code'
        self.retry = (m, want)
        self._set_mode('prompt' if want == 'prompt' else 'code')
        self.buf.text, self.buf.cursor = m.content, len(m.content)
        self.paint()

    def _cursor_msg(self):
        "The Dialog message behind the transcript-view cursor block, or None."
        mid = msg_id(self.tv.cur) if self.tv.cur is not None else None
        return self.ctl.msg(mid) if mid else None

    def _retry_from_tv(self):
        "E in the transcript view, on a prompt input: that prompt into the composer, armed for retry, back on the live screen."
        tv = self.tv
        m = self._cursor_msg()
        if m is None or m.msg_type != 'prompt' or not tv.cur.endswith(':ask'):
            tv.msg = 'retry (E) works on a prompt input'
            return tv.draw()
        self.retry = (m, 'prompt')
        self._set_mode('prompt')
        self.buf.text, self.buf.cursor = m.content, len(m.content)
        tv.leave()
        self.paint()

    def _jump_exchange(self, d):
        "Shift-up/down in the transcript view: block cursor to the previous/next exchange start (an input or ask block)."
        tv = self.tv
        starts = [k for k, b in self.comp.blocks.items() if k.endswith((':in', ':ask')) and b.height > 0]
        if not starts: return
        keys = list(self.comp.blocks)
        pos = keys.index(tv.cur) if tv.cur in keys else len(keys)
        idx = {k: i for i, k in enumerate(keys)}
        if d < 0: key = next((k for k in reversed(starts) if idx[k] < pos), starts[0])
        else: key = next((k for k in starts if idx[k] > pos), starts[-1])
        tv.select(key)

    def _toggle_hide(self):
        "h in the transcript view: flip `skipped` on the cursor block's message; its blocks dim, and the session file saves."
        tv = self.tv
        m = self._cursor_msg()
        if m is None:
            tv.msg = 'no AI record for this block'
            return tv.draw()
        self.ctl.hide(m)
        tv.msg = 'hidden from AI' if m.skipped else 'visible to AI'
        tv.draw()

    def _edit_current(self):
        """e in the transcript view: the cursor block's message into the composer for editing. An ask
        block edits the prompt's content; any other block of a prompt exchange edits the WHOLE reply
        markdown (tool calls and results live inside it); a code/note exchange edits its source."""
        tv = self.tv
        m = self._cursor_msg()
        if m is None:
            tv.msg = 'no AI record for this block'
            return tv.draw()
        kind = 'reply' if m.msg_type == 'prompt' and not tv.cur.endswith(':ask') else 'content'
        self.editing = (m, kind)
        self.buf.text = m.ai_res if kind == 'reply' else m.content
        self.buf.cursor = len(self.buf.text)
        tv.composing = True
        tv.msg = f"editing {'reply' if kind == 'reply' else m.msg_type} -- Enter writes back, Esc cancels"
        tv.draw()

    def _finish_edit(self, write):
        "Leave editing state; on write, the new text replaces the message's content or reply."
        (m, kind), self.editing = self.editing, None
        tv, text = self.tv, self.buf.text
        self.buf.clear()
        tv.composing = False
        if not write or (kind == 'reply' and text == m.ai_res) or (kind == 'content' and text == m.content):
            tv.msg = 'edit cancelled' if not write else 'unchanged'
            return tv.draw()
        if kind == 'reply': self.ctl.edit(m, reply=text)
        else: self.ctl.edit(m, content=text)
        tv.msg = 'written'
        tv.draw()

    def _tv_key(self, k):
        "Key routing while the transcript view is up: the view's modal vocabulary first, editing to the shared composer."
        tv = self.tv
        if self.editing is not None:
            if k.name == 'enter': return self._finish_edit(True)
            if k.name == 'escape': return self._finish_edit(False)
        elif k.name in ('shift+up', 'shift+down') and tv.search is None and not tv.composing:
            return self._jump_exchange(-1 if k.name == 'shift+up' else 1)
        elif tv.search is None and not tv.composing and k.char in ('h', 'e', 'E'):
            # browse-mode keys: solveit's h (hide), e (edit in place), E (retry: edit-and-resubmit)
            return self._toggle_hide() if k.char == 'h' else self._edit_current() if k.char == 'e' else self._retry_from_tv()
        if tv.on_key(k): return
        if k.name in ('escape', 'ctrl+t'):
            tv.leave()
            self.paint()
        elif k.name == 'enter' and self.buf.text and not self.busy:
            tv.leave()  # Enter with content submits AND returns to the live screen
            self._pending = []
            self.comp.spawn(self.on_enter(), name='run')
            self.paint()
        elif k.name == 'enter': tv.toggle_current()
        else:
            self.buf.handle(k)
            tv.draw()

    # -- keys ---------------------------------------------------------------------------------
    def _reply_blocks(self):
        "Python fenced blocks of the last AI reply, the paste bindings' source (mdhtml structure, never regex)."
        return code_blocks(self.ctl.last_reply())

    def _cycle_blocks(self, delta):
        "Alt-shift-up/down: cycle the composer through the last reply's fenced blocks."
        bs = self._reply_blocks()
        if not bs: return
        resp = self.ctl.last_reply()
        if resp != self._cycle['resp']: self._cycle.update(idx=-1, resp=resp)
        self._cycle['idx'] = (self._cycle['idx'] + delta) % len(bs)
        self.buf.text = bs[self._cycle['idx']]
        self.buf.cursor = len(self.buf.text)

    def on_paste(self, text):
        "Paste goes to the focused input: a pending kernel input request, else the composer (in the transcript view it also takes compose focus)."
        if self._pending is not None: return self._pending.append(text)  # ordered behind the in-flight enter
        if self.inp is not None:
            self.inp['buf'].insert(text)
            return self.paint()
        if self.tv.active: self.tv.composing = True
        self.buf.insert(text)
        if self.tv.active: self.tv.draw()
        else: self.paint()

    def _inp_key(self, k):
        "Keys while a kernel input request owns the tail: Enter answers it, everything else edits the answer."
        i = self.inp
        if k.name == 'enter':
            if not i['fut'].done(): i['fut'].set_result(i['buf'].text)
        else: i['buf'].handle(k)
        self.paint()

    def on_key(self, k):
        "Ctrl-C first, then the first focus that exists: a kernel input request, the picker, the transcript view, the composer."
        if self._pending is not None and k.name != 'ctrl+c':
            return self._pending.append(k)  # an enter decision is in flight: input stays ordered behind it
        if k.name == 'ctrl+c': return self.on_sigint()
        if self.inp is not None: return self._inp_key(k)
        if self.picker is not None: return self._picker_key(k)
        if self.tv.active: return self._tv_key(k)
        if k.name == 'ctrl+d' and not self.buf.text:
            if self.ctl.shell is not None and not self._quit_warned:  # bash's convention: warn once, an immediate second C-D quits
                self._quit_warned = True
                self.note('the shell (and any jobs in it) closes with the app  (C-D again quits)')
                return
            self._dismiss()
            self.paint()  # one clean final frame: transient UI must never ink as exit debris
            self.done.set()
            return
        self._quit_warned = False  # any other key withdraws the warning
        if k.name == 'ctrl+t' and not self.busy:
            self.tv.enter()
            return
        if k.name == 'ctrl+o':
            live = [b for b in self.comp.blocks.values() if not b.committed]
            if live: self.comp.toggle(live[-1].key)
        elif k.name == 'f2': return self.edit_buffer()
        elif k.name == 'alt+r': self._recall_last()  # retry: recall the last exchange, submit REPLACES it (alt-up stays history)
        elif k.name == 'escape' and self.retry is not None:
            self.retry = None  # disarm: the composer keeps its text, submits append again
            self.paint()
        elif k.name in ('alt+p', 'alt+c', 'alt+s'):
            self._set_mode(dict(p='prompt', c='code', s='shell')[k.name[4]])
            self._dismiss()
        elif k.name == 'alt+.':
            if self.buf.text.strip() and self.buf.cursor == len(self.buf.text) and not self.busy and self.ctl.assistant is not None:
                return self.do_ai_suggest()
        elif k.name == 'alt+W':
            bs = self._reply_blocks()
            if bs: self.buf.insert('\n'.join(bs))
        elif len(k.name) == 5 and k.name.startswith('alt+') and k.name[4] in self.comp.numbered:
            self.comp.toggle(self.comp.numbered[k.name[4]])  # the block wearing that digit
        elif len(k.name) == 5 and k.name.startswith('alt+') and k.name[4] in _NTH:
            bs = self._reply_blocks()
            n = _NTH[k.name[4]]
            if len(bs) >= n: self.buf.insert(bs[n - 1])
        elif k.name == 'shift+alt+up': self._cycle_blocks(1)
        elif k.name == 'shift+alt+down': self._cycle_blocks(-1)
        elif k.name == 'tab' and self.menu: self.menu.cycle(1)
        elif k.name == 'shift+tab' and self.menu: self.menu.cycle(-1)
        elif k.name == 'enter' and self.menu:
            self._dismiss()  # accepts the highlighted match; the next Enter submits
        elif k.name == 'enter' and self.buf.text and not self.busy:
            self._pending = []
            return self.on_enter()
        elif k.name == 'alt+enter':  # always a newline: the codex/Claude convention
            self.buf.insert('\n')
        elif k.name == 'tab' and self.buf.text and not self.ctl.kernel_busy and self.k.kc is not None: return self.do_complete()
        elif k.name == 'shift+tab' and self.buf.text and not self.ctl.kernel_busy and self.k.kc is not None: return self.do_inspect()
        elif k.name in ('up', 'alt+up'):
            self._dismiss()
            if not (k.name == 'up' and self.buf.handle(k)) and self.hist:
                t = self.hist.prev(self.buf.text)
                if t is not None: self.buf.text, self.buf.cursor = t, len(t)
        elif k.name in ('down', 'alt+down'):
            self._dismiss()
            if not (k.name == 'down' and self.buf.handle(k)) and self.hist:
                t = self.hist.next()
                if t is not None: self.buf.text, self.buf.cursor = t, len(t)
        else:
            self._dismiss()
            if self.hist: self.hist.reset_nav()
            self.buf.handle(k)
        self.paint()

    def on_sigint(self):
        "The ctrl-C policy, reached as a key: in-band at rest, synthesized by the compositor's SIGINT handler otherwise."
        if self.picker is not None: return self._pick(None)
        if self.ctl.cancel(): return self.paint()
        self.buf.clear()
        self.ai_sugg = None
        self.paint()

    def _task_error(self, e, t):
        "A spawned background task failed: an error note beats a stderr traceback through the raw-mode screen."
        self.note(f'{t.get_name()} failed: {e!r}', 'error')
        self.paint()

    def _resized(self):
        "The WINCH response between borrows (a borrow sends WINCH to its own handler): the idle shell follows, then repaint."
        self.ctl.resize_shell()
        if self.tv.active: self.tv.leave()  # a rewrap invalidates the view; re-enter is one keystroke
        self.comp.resize()
        self.paint()

    async def choose_session(self, resume):
        """The session file to continue, chosen at launch before any kernel starts. `resume` is a filename
        prefix, '' (bare `-r`) to choose among this directory's sessions, or None to start fresh. With several
        sessions the picker asks; with one it is chosen without asking. Returns None for a fresh session."""
        if resume: return resolve_session(resume)
        if resume is None: return None
        rows = list_sessions()
        if len(rows) <= 1: return rows[0][0] if rows else None
        self._picked = asyncio.get_running_loop().create_future()
        self.picker = rows[:9]
        self.paint()
        return await self._picked

    async def run(self):
        "The main loop: the throbber ticker while busy, until quit. The compositor reads the tty and flushes escapes itself."
        self.comp.on_resize = self._resized
        self.comp.on_task_error = self._task_error
        async def ticker():
            while True:
                await asyncio.sleep(0.2)
                if self.busy and not self.tv.active: self.paint()  # the throbber cell animates on the ticker's clock
        t = self.comp.spawn(ticker(), name='ticker')
        try:
            self.paint()
            await self.done.wait()
        finally: t.cancel()

def _sessions_text(rows):
    "Past-session rows as the table %ipyai sessions and --sessions both show."
    if not rows: return 'No ipyai sessions found for this directory.'
    lines = [f"{'Session':10}  {'When':16}  {'Prompts':>7}  First prompt"]
    for path, mtime, n, first in rows:
        fp = (first or '').replace('\n', ' ')[:60]
        when = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
        lines.append(f'{Path(path).stem[:8]:10}  {when:16}  {n:>7}  {fp}')
    return '\n'.join(lines)

Think = str_enum('Think', 'l', 'm', 'h', 'x')

@call_parse
async def main(
    model:str=None,          # turn model, a vendor-prefixed string like codex/gpt-5.6-terra
    suggest_model:str=None,  # inline-suggestion model
    think:Think=None,        # think effort
    code_theme:str=None,     # code highlight theme ('auto' detects from the terminal background)
    Prompt_mode:bool=False,  # start in prompt mode
    kernel:str=None,         # attach to an existing gateway kernel by id prefix (taken as found, never stopped on exit)
    Resume:Annotated[str, "resume a session: bare -r picks from this directory's sessions, -r PREFIX resumes that session file (warm-attaching its kernel when still alive)", dict(nargs='?', const='')]=None,
    Load:str=None,           # load a dialog .ipynb into the session at startup
    sessions:bool=False,     # list past ipyai sessions for this directory and exit
):
    "IPython + AI on the teleprint transcript (plain launch always starts a fresh session; rustygate must be running)"
    import faulthandler
    faulthandler.register(signal.SIGQUIT)  # C-\ prints every thread's stack and continues: a wedged app becomes diagnosable from the pane
    if sessions: return print(_sessions_text(list_sessions()))
    cfg = load_config()
    cfg |= {k: str(v) for k, v in dict(model=model, suggest_model=suggest_model,
        think=think, code_theme=code_theme).items() if v}
    if Prompt_mode: cfg['prompt_mode'] = True
    t = RealTty()
    async with AsyncExitStack() as stack:  # one owner of every resource: a failed close cannot skip the others
        stack.callback(t.restore)
        stack.callback(t.write, '\r\n')
        app = App(t, cfg=cfg)
        await app.comp.start()  # the CPR anchor, the signals, mouse and paste modes, and the tty reader
        stack.callback(app.comp.stop)
        await app.detect_kitty()
        if cfg.get('code_theme', 'auto') == 'auto': await app.detect_theme()
        path = await app.choose_session(Resume)   # before any kernel starts: its stamped kernel may still be alive
        d = read_ipynb(path) if path else None
        kid = kernel or (d.meta.get('ipyai', {}).get('kernel_id', '') if d is not None else '')
        try: await app.k.start(kernel=kid)
        except ValueError:
            if kernel: raise            # an explicitly named kernel that's gone is an error, not a fallback
            await app.k.start()         # the session's stamped kernel is gone: cold resume on a fresh kernel
        stack.push_async_callback(app.k.close)
        stack.push_async_callback(app.ctl.close)
        if app.k.owned: await app.k.setup()
        if path: app.ctl.resume(path)
        app.ctl.stamp()
        if Load: app.note(app.ctl.load(Load))
        await app.run()

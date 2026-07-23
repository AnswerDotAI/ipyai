"The ipyai terminal app: teleprint UI over a Jupyter kernel, with the assistant riding the block model."
import asyncio, base64, math, os, re, signal, time
from kittytgp import render_parts, kitty_probe, kitty_supported, kitty_env_hint
from rich.text import Text
from rich.cells import cell_len
from rich.syntax import Syntax
from teleprint.buffer import Buffer
from teleprint.compositor import Compositor
from teleprint.tty import RealTty
from teleprint.widgets import CompletionMenu, Tooltip, Signature
from teleprint.transcript import TranscriptView
from fastcore.ansi import strip_ansi
from .sig import call_context, parse_sig_text, active_param
from .kernel import KernelSession
from .history import History
from .assistant import Assistant, route, code_blocks
from .config import load_config
from .reply import TurnRenderer

HINT = 'ipyai ng -- Enter runs; Tab completes; shift-Tab inspects; ctrl-T transcript; alt-p prompt mode; ctrl-D quits'
OSC_BG_RE = re.compile(rb'\x1b\]11;rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)')

GUTTERS = {'in': ('>>> ', '... ', 'bold green'), 'out': ('« ', '  ', 'bright_blue'),
           'result': ('« ', '  ', 'bright_blue'), 'error': ('« ', '  ', 'red'), 'image': ('« ', '  ', 'magenta'),
           'ask': ('ai> ', '... ', 'bold magenta'), 'ai': ('« ', '  ', 'magenta'),
           'think': ('« ', '  ', 'dim'), 'tool': ('« ', '  ', 'yellow')}

_NTH = {'!': 1, '@': 2, '#': 3, '$': 4, '%': 5, '^': 6, '&': 7, '*': 8, '(': 9}  # alt-shift digits arrive shifted

def _gutter(kind):
    "The block's left edge: a colored direction marker (in vs out at a glance), continuation-aware."
    f, c, sty = GUTTERS[kind]
    return (Text(f, style=sty), Text(c, style=sty))

def _text(t): return ''.join(t) if isinstance(t, list) else (t or '')

def _default_history():
    try: return History()
    except Exception: return None  # no IPython history db yet: arrows and ghost text just stay off

def _hl(code, theme='ansi_dark'):
    "Syntax-highlight `code` the one true way, dropping the newline highlight() appends."
    t = Syntax('', 'python', theme=theme).highlight(code)
    if t.plain.endswith('\n'): t.right_crop(1)
    return t

class App:
    "Wires a teleprint compositor/buffer to a KernelSession + Assistant; `tty` is injectable for the test harness."
    def __init__(self, tty, kernel=None, history='default', cfg=None, assistant=None):
        self.tty = tty
        self.cfg = cfg or {}
        self.k = KernelSession() if kernel is None else kernel
        self.hist = _default_history() if history == 'default' else history  # None = off (tests want hermeticity)
        self.comp = Compositor(tty).start()
        self.buf = Buffer()
        self.stream = None   # the growing stdout block of the in-flight cell
        self.menu = None     # a teleprint CompletionMenu while completing; Tab/shift+Tab cycle, Enter accepts
        self.tip = None      # a teleprint Tooltip while inspecting (shift+Tab on a bare buffer)
        self.done = asyncio.Event()
        self.kitty = False        # set by detect_kitty(); images fall back to a note without it
        self.theme = self.cfg.get('code_theme', 'auto')  # 'auto' resolves via detect_theme (OSC 11)
        if self.theme == 'auto': self.theme = 'ansi_dark'
        self.cell_imgs = set()   # png hashes shown this cell, to skip the execute_result repeat of a displayed figure
        self.assistant = assistant if assistant is not None else Assistant(cfg=cfg) if cfg else None
        self.prompt_mode = bool(self.cfg.get('prompt_mode'))
        self.ai_task = None      # the in-flight run_prompt task
        self.ai_sugg = None      # (text, cursor, suggestion): an Alt-. completion, valid while the buffer is unchanged
        self.cell_outputs = None # raw (msg_type, content) records of the running cell, for context + persistence
        self._cycle = dict(idx=-1, resp='')  # alt-shift-up/down cycling over the last reply's fenced blocks
        self.comp.on_key = self.on_key
        self.comp.on_paste = lambda text: (self.buf.insert(text), self.paint())
        self.tv = TranscriptView(self.comp, self._tail_content)
        self.comp.on_mouse = lambda ev: self.tv.on_mouse(ev)

    @property
    def busy(self):
        "Enter is gated on this: a running cell or a running AI turn each own the transcript's live edge."
        return self.k.busy or (self.ai_task is not None and not self.ai_task.done())

    @property
    def collapse_at(self):
        "Auto-collapse threshold for outputs: about half a screen, always small enough to fold while visible."
        return max(3, min(self.comp.rows // 2, self.comp.rows - 5))

    def _status(self):
        if self.k.busy: return 'running... (ctrl-C interrupts)'
        if self.ai_task is not None and not self.ai_task.done():
            return f"{self.assistant.model} responding... (ctrl-C stops)"
        return HINT + (' -- PROMPT MODE (alt-p exits)' if self.prompt_mode else '')

    def _tail_content(self):
        "Tail layout: dim status above the prompt (a shell's context-line shape), popups (menu/tooltip) below it."
        status = Text(self._status(), style='dim')
        plain = self.prompt_mode or self.buf.text.startswith('.')  # prompts read as prose, not Python
        hl = Text(self.buf.text) if plain else _hl(self.buf.text, self.theme)
        src = hl.split('\n')
        if src and not src[-1].plain: src.pop()  # drop only the empty tail line: a whitespace continuation line must survive
        if not src: src = [Text('')]
        mark, cont = ('ai> ', '... ') if self.prompt_mode else ('>>> ', '... ')
        msty = 'bold magenta' if self.prompt_mode else 'bold green'
        prompt = Text()  # empty base style: a styled first arg would become the BASE for the whole line, bolding the code
        prompt.append(mark, style=msty)
        prompt.append(src[0])
        for c in src[1:]:
            prompt.append('\n' + cont, style=msty)
            prompt.append(c)
        sugg = ''
        if self.buf.cursor == len(self.buf.text) and not self.menu:
            if self.ai_sugg and self.ai_sugg[0] == self.buf.text and self.ai_sugg[1] == self.buf.cursor:
                sugg = self.ai_sugg[2]  # Alt-. completion overrides history while the document is unchanged
            elif self.hist: sugg = self.hist.suggest(self.buf.text)
        self.buf.suggestion = sugg
        if sugg: prompt.append(sugg, style='dim')
        lines = [status, prompt]
        if self.menu: lines.append(self.menu.renderable())
        elif self.tip: lines.append(self.tip.renderable())
        before = self.buf.text[:self.buf.cursor]
        col = 4 + cell_len(before.rsplit('\n', 1)[-1])  # 'ai> '/'>>> '/'... ' prefixes are all 4 cells
        return lines, (1, before.count('\n'), col)

    def paint(self):
        if self.tv.active: self.tv.draw()
        else:
            lines, cursor = self._tail_content()
            self.comp.set_tail(*lines, cursor=cursor)

    def _probe(self, payload, timeout=0.5):
        "Write `payload` plus a DA1 fence, reading raw until the DA1 reply (every terminal answers it) or timeout."
        self.tty.write(payload + b'\x1b[c')
        end = time.monotonic() + timeout
        data = b''
        while time.monotonic() < end:
            data += self.tty.read()
            if re.search(rb'\x1b\[\?[0-9;]*c', data): break
        return data

    def detect_kitty(self, timeout=0.5):
        "Probe the tty for kitty graphics; env hints cover tmux, where probe replies are not routed to panes."
        self.kitty = kitty_supported(self._probe(kitty_probe()[:-3], timeout)) or kitty_env_hint()
        return self.kitty

    def detect_theme(self, timeout=0.5):
        "OSC 11 background query: pick the highlighter theme for a light or dark terminal (silence means stay dark)."
        m = OSC_BG_RE.search(self._probe(b'\x1b]11;?\x1b\\', timeout))
        if m:
            r, g, b = (int(ch[:2], 16) for ch in m.groups())
            self.theme = 'ansi_light' if (0.2126*r + 0.7152*g + 0.0722*b) / 255 > 0.5 else 'ansi_dark'
        return self.theme

    def show_image(self, png):
        """A PNG output as a kitty Unicode-placeholder block: the APC transmit goes straight to the tty
        (cursor-neutral bytes), while the placeholder grid -- ordinary styled text -- becomes the block,
        so it repaints, commits, and survives tmux like any other transcript content."""
        from kittytgp.core import _read_png
        _, w, h = _read_png(png)
        if not self.kitty:
            self.comp.print_block(f'[image {w}x{h}px -- this terminal lacks kitty graphics]', gutter=_gutter('image'), tag='image')
            return
        c = max(1, min(40, self.comp.cols - 2, w))
        r = max(1, math.ceil(h / (w / c * 2)))
        transmit, placeholder = render_parts(png, cols=c, rows=r)
        self.comp.tty.write(transmit)
        self.comp.print_block(Text.from_ansi(placeholder), gutter=_gutter('image'), tag='image')

    def on_out(self, msg_type, c):
        if self.cell_outputs is not None: self.cell_outputs.append((msg_type, c))  # the cell record: context + persistence
        if msg_type == 'stream':
            if self.stream is None: self.stream = self.comp.print_block(gutter=_gutter('out'), tag='out', collapse_at=self.collapse_at)
            txt = _text(c.get('text')).rstrip('\n')
            if txt: self.comp.extend(self.stream, Text(txt, style='red') if c.get('name') == 'stderr' else txt)
            return
        self.stream = None  # any non-stream output closes the growing block: only the LAST block can grow, and later stream text starts a fresh one (Jupyter's separate output areas)
        if msg_type in ('execute_result', 'display_data'):
            data = c.get('data', {})
            if 'image/png' in data:
                png = base64.b64decode(data['image/png'])
                # Jupyter renders a same-cell `fig` twice (flush display_data + execute_result repr);
                # suppress only a byte-identical execute_result repeat -- distinct images all render
                if not (msg_type == 'execute_result' and hash(png) in self.cell_imgs):
                    self.cell_imgs.add(hash(png))
                    self.show_image(png)
            elif 'text/plain' in data: self.comp.print_block(_text(data['text/plain']), gutter=_gutter('result'), tag='result', collapse_at=self.collapse_at)
        elif msg_type == 'error':
            self.comp.print_block(Text.from_ansi('\n'.join(c.get('traceback', []))), gutter=_gutter('error'), tag='error')  # errors always print open

    async def run_cell(self, code):
        self.stream = None
        self.cell_imgs = set()
        self.cell_outputs = []
        self.paint()
        try: await self.k.run(code, self.on_out)
        finally:
            if self.assistant is not None: self.assistant.add_cell(code, self.cell_outputs)
            self.cell_outputs = None
            self.paint()

    async def attach_assistant(self, resume=None):
        "Wire the AI side to the live kernel: bridge + tools, persistence into the kernel's own history db, optional resume."
        from .bridge import setup_tools
        from .store import Store
        bridge, tools = await setup_tools(self.k.kc)
        if self.assistant is None: self.assistant = Assistant(cfg=self.cfg or None)
        self.assistant.tools, self.assistant.bridge = tools, bridge
        try:
            path, session = await bridge.history_db_info()
            self.assistant.store = Store(path, session, cwd=os.getcwd(), backend=self.assistant.cfg['_backend_name'])
        except Exception: pass
        if resume is not None and self.assistant.store is not None: self.load_session(resume)

    def _replay_output(self, o):
        "Render one stored nbformat output: the dict shapes match iopub content, so on_out replays them directly."
        ot = o.get('output_type')
        if ot in ('stream', 'display_data', 'execute_result', 'error'): self.on_out(ot, o)

    def load_session(self, session):
        "Resume: print a past session's blocks from the store and seed the conversation to continue it (no kernel state is rebuilt)."
        from .backend_common import PromptTurn, thinking_to_blockquote
        st = self.assistant.store
        events = st.load_session(session)
        if not events:
            self.comp.print_block(Text(f'session {session}: nothing stored', style='dim'), gutter=_gutter('error'), tag='error')
            return
        last_prompt_line = 0
        for ev in events:
            if ev['kind'] == 'cell':
                self.comp.print_block(_hl(ev['source'], self.theme), gutter=_gutter('in'), tag='in')
                self.stream = None
                self.cell_imgs = set()
                for o in ev['outputs']: self._replay_output(o)
                self.stream = None
                self.assistant.cells.append(dict(source=ev['source'],
                    outputs=[(o.get('output_type'), o) for o in ev['outputs']], line=ev['line']))
            else:
                self.comp.print_block(Text(ev['prompt']), gutter=_gutter('ask'), tag='ask')
                tr = TurnRenderer(self.comp, _gutter, theme=self.theme, collapse_at=self.collapse_at)
                tr.md.feed(thinking_to_blockquote(ev['response']))
                tr.done()
                last_prompt_line = ev['line']
        sid, backend, turns = st.resume_state(session)
        self.assistant.turns = [PromptTurn(prompt=p, full_prompt=f, response=r, history_line=0) for p, f, r in turns]
        self.assistant._ctx_cells = sum(1 for ev in events if ev['kind'] == 'cell' and ev['line'] <= last_prompt_line)
        if backend == self.assistant.cfg.get('_backend_name'): self.assistant.provider_session_id = sid
        if turns: self.assistant.last_response = turns[-1][2]

    async def do_complete(self):
        matches, start = await self.k.complete(self.buf.text, self.buf.cursor)
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
        """shift+Tab: inside a call, a Signature panel with the active param bold (inspecting the callable,
        not the token under the cursor); otherwise the full inspect text as a tooltip."""
        ctx = call_context(self.buf.text, self.buf.cursor)
        pos = ctx[0] if ctx else self.buf.cursor
        text = await self.k.inspect(self.buf.text, pos)
        self.menu = None
        self.tip = None
        if text:
            sig = parse_sig_text(strip_ansi(text)) if ctx else None
            if sig:
                name, params, doc = sig
                self.tip = Signature(name, params, active_param(params, ctx[1]), doc)
            else: self.tip = Tooltip(Text.from_ansi(text))
        self.paint()

    def _dismiss(self):
        self.menu = None
        self.tip = None

    def _submit(self, text, code=None):
        "Print the input block (prompts plain with the ask gutter, code highlighted), clear the buffer, record history."
        if code is None: self.comp.print_block(Text(text), gutter=_gutter('ask'), tag='ask')
        else: self.comp.print_block(_hl(code, self.theme), gutter=_gutter('in'), tag='in')
        self.buf.clear()
        self.ai_sugg = None
        self._dismiss()
        if self.hist:
            self.hist.reset_nav()
            self.hist.add_local(text)  # instantly navigable/suggestible; the kernel's own write lags its flush thread

    async def on_enter(self):
        """Routed Enter (the `.`/`;`/`!`/`%` dispatch): prompts always submit -- English is never
        'incomplete' -- while code keeps the smart is_complete check (auto-indented continuation)."""
        text = self.buf.text
        kind, payload = route(text, self.prompt_mode)
        if kind == 'prompt':
            self._submit(text)
            await self.run_prompt(payload)
            self.paint()
            return
        status, indent = await self.k.check(payload)
        if self.buf.text != text: return  # the buffer changed during the round-trip: stale decision
        if status == 'incomplete':
            self.buf.insert('\n' + indent)
        else:
            self._submit(text, payload)
            await self.run_cell(payload)
        self.paint()

    async def run_prompt(self, payload):
        "One AI turn, rendered live as blocks; `ai_task` gates Enter and is ctrl-C's cancel target."
        if self.assistant is None:
            self.comp.print_block(Text('no assistant attached (plain-REPL mode)', style='dim'), gutter=_gutter('error'), tag='error')
            return
        tr = TurnRenderer(self.comp, _gutter, theme=self.theme, collapse_at=self.collapse_at)
        self.ai_task = asyncio.current_task()
        self.paint()
        try: await self.assistant.run_prompt(payload, tr)
        except Exception as e:
            self.comp.print_block(Text(f'AI prompt failed: {type(e).__name__}: {e}', style='red'), gutter=_gutter('error'), tag='error')
        finally:
            self.ai_task = None
            self.paint()

    async def do_ai_suggest(self):
        "Alt-.: one-shot AI completion into the ghost-text slot, valid only while the document is unchanged."
        snap = (self.buf.text, self.buf.cursor)
        try: text = await self.assistant.ai_complete(self.buf.text[:self.buf.cursor], self.buf.text[self.buf.cursor:])
        except Exception: return
        if text and (self.buf.text, self.buf.cursor) == snap:
            self.ai_sugg = (snap[0], snap[1], text)
            self.paint()

    def _tv_key(self, k):
        "Key routing while the transcript view is up: navigation to the view, editing to the shared composer."
        tv = self.tv
        if k.name in ('escape', 'ctrl+t'):
            tv.leave()
            self.paint()
        elif k.name == 'up': tv.move(-1)
        elif k.name == 'down': tv.move(1)
        elif k.name == 'pageup': tv.scroll(-tv._view_rows)
        elif k.name == 'pagedown': tv.scroll(tv._view_rows)
        elif k.name == 'enter' and self.buf.text and not self.busy:
            tv.leave()  # Enter with content submits AND returns to the live screen
            asyncio.get_running_loop().create_task(self.on_enter())
            self.paint()
        elif k.name == 'enter':
            tv.toggle_current()
        else:
            self.buf.handle(k)
            tv.draw()

    def _reply_blocks(self):
        "Python fenced blocks of the last AI reply, the paste bindings' source (mdhtml structure, never regex)."
        return code_blocks(self.assistant.last_response) if self.assistant is not None else []

    def _cycle_blocks(self, delta):
        "Alt-shift-up/down: cycle the composer through the last reply's fenced blocks."
        bs = self._reply_blocks()
        if not bs: return
        resp = self.assistant.last_response
        if resp != self._cycle['resp']: self._cycle.update(idx=-1, resp=resp)
        self._cycle['idx'] = (self._cycle['idx'] + delta) % len(bs)
        self.buf.text = bs[self._cycle['idx']]
        self.buf.cursor = len(self.buf.text)

    def on_key(self, k):
        if self.tv.active: return self._tv_key(k)
        if k.name == 'ctrl+d' and not self.buf.text:
            self.done.set()
            return
        if k.name == 'ctrl+c': return self.on_sigint()
        if k.name == 'ctrl+t' and not self.busy:
            self.tv.enter()
            return
        if k.name == 'ctrl+o':
            live = [b for b in self.comp.blocks.values() if not b.committed]
            if live: self.comp.toggle(live[-1])
        elif k.name == 'alt+p':
            self.prompt_mode = not self.prompt_mode
            self._dismiss()
        elif k.name == 'alt+.':
            if self.buf.text.strip() and not self.busy and self.assistant is not None:
                asyncio.get_running_loop().create_task(self.do_ai_suggest())
        elif k.name == 'alt+W':
            bs = self._reply_blocks()
            if bs: self.buf.insert('\n'.join(bs))
        elif len(k.name) == 5 and k.name.startswith('alt+') and k.name[4] in _NTH:
            bs = self._reply_blocks()
            n = _NTH[k.name[4]]
            if len(bs) >= n: self.buf.insert(bs[n - 1])
        elif k.name == 'shift+alt+up': self._cycle_blocks(1)
        elif k.name == 'shift+alt+down': self._cycle_blocks(-1)
        elif k.name == 'tab' and self.menu:
            self.menu.cycle(1)
        elif k.name == 'shift+tab' and self.menu:
            self.menu.cycle(-1)
        elif k.name == 'enter' and self.menu:
            self._dismiss()  # accepts the highlighted match; the next Enter submits
        elif k.name == 'enter' and self.buf.text and not self.busy:
            asyncio.get_running_loop().create_task(self.on_enter())
        elif k.name == 'alt+enter':  # always a newline: the codex/Claude convention
            self.buf.insert('\n')
        elif k.name == 'tab' and self.buf.text and not self.k.busy:
            asyncio.get_running_loop().create_task(self.do_complete())
        elif k.name == 'shift+tab' and self.buf.text and not self.k.busy:
            asyncio.get_running_loop().create_task(self.do_inspect())
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
        if self.assistant is not None and self.assistant.cancel_turn(): return  # ctrl-C stops the AI turn first
        if self.k.busy: asyncio.get_running_loop().create_task(self.k.interrupt())
        else:
            self.buf.clear()
            self.ai_sugg = None
            self.paint()

    async def run(self):
        "The real-terminal main loop: tty reader, signal handlers, escape-timeout ticker."
        loop = asyncio.get_running_loop()
        loop.add_reader(self.tty.fd, lambda: self.comp.on_bytes(os.read(self.tty.fd, 4096)))
        loop.add_signal_handler(signal.SIGINT, self.on_sigint)
        def resized():
            if self.tv.active: self.tv.leave()  # a rewrap invalidates the view; re-enter is one keystroke
            self.comp.resize()
            self.paint()
        loop.add_signal_handler(signal.SIGWINCH, resized)
        async def ticker():
            while True:
                await asyncio.sleep(0.2)
                self.comp.flush_input()
        t = asyncio.ensure_future(ticker())
        try:
            self.paint()
            await self.done.wait()
        finally:
            t.cancel()
            loop.remove_reader(self.tty.fd)
            for s in (signal.SIGINT, signal.SIGWINCH): loop.remove_signal_handler(s)

def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog='ipyai', description='IPython + AI on the teleprint transcript')
    p.add_argument('-p', '--prompt-mode', action='store_true', help='start in prompt mode')
    p.add_argument('-b', '--backend', default=None, help='backend: codex-api | codex | claude-cli | claude-api')
    p.add_argument('-r', '--resume', type=int, default=None, help='resume ipyai session N (see --sessions)')
    p.add_argument('--sessions', action='store_true', help='list past ipyai sessions for this directory and exit')
    return p.parse_args(argv)

def _list_sessions(cwd):
    "Print past ipyai sessions from the shared history db (no kernel needed: the path is IPython's default)."
    from .history import hist_path
    from .store import Store
    st = Store(hist_path())
    rows = st.sessions(cwd=cwd)
    if not rows: return print('No ipyai sessions found for this directory.')
    print(f"{'ID':>6}  {'Backend':10}  {'Prompts':>7}  Last prompt")
    for sid, _, backend, n, last in rows:
        if sid < 0: continue
        lp = (last or '').replace('\n', ' ')[:60]
        print(f'{sid:>6}  {backend or "":10}  {n:>7}  {lp}')

async def _amain(a):
    cfg = load_config(backend_name=a.backend)
    if a.prompt_mode: cfg['prompt_mode'] = True
    t = RealTty()
    t.write('\x1b[?1000;1006h\x1b[?2004h')  # SGR mouse + bracketed paste
    try:
        app = App(t, cfg=cfg)
        app.detect_kitty()
        if cfg.get('code_theme', 'auto') == 'auto': app.detect_theme()
        await app.k.start()
        await app.attach_assistant(resume=a.resume)
        await app.run()
    finally:
        t.write('\x1b[?2004l\x1b[?1000;1006l\r\n')
        t.restore()
        if app.k.kc is not None: await app.k.close()

def main(argv=None):
    a = _parse_args(argv)
    if a.sessions: return _list_sessions(os.getcwd())
    try: asyncio.run(_amain(a))
    except Exception:
        import traceback
        traceback.print_exc()  # terminal already restored by the finally, so this is readable

def main_codex(argv=None):
    import sys
    main(['-b', 'codex'] + (sys.argv[1:] if argv is None else argv))

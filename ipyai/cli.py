"The ipyai terminal app: teleprint UI over a Jupyter kernel, with the assistant riding the block model."
import asyncio, base64, math, os, pyghostty, re, shlex, signal, subprocess, tempfile, time
from typing import Annotated
from pathlib import Path
from kittytgp import render_parts, kitty_probe, kitty_supported, kitty_env_hint
from rich.text import Text
from rich.style import Style
from rich.cells import cell_len
from rich.syntax import Syntax
from teleprint.buffer import Buffer
from teleprint.compositor import Compositor
from teleprint.tty import RealTty
from teleprint.widgets import CompletionMenu, Tooltip, Signature
from teleprint.transcript import TranscriptView
from .shell import GateShell
import aidialog.ipynb  # noqa: F401 -- activates the Message.cell_meta/to_cell patches (meta_attrs serialization)
from aidialog.msg_parts import Text as TextPart, ToolUse, ToolResult
from fastcore.ansi import strip_ansi
from fastcore.basics import str_enum
from fastcore.script import call_parse
from .kernel import KernelSession
from .history import History
from .session import Session, list_sessions, resolve_session
from .assistant import Assistant, route, code_blocks
from .config import load_config
from .reply import TurnRenderer

HINT = 'ipyai ng⋄Enter runs; Tab completes; S-Tab inspects; C-T transcript; C-C interrupts; M-p mode; C-D quits'  # S/C/M = shift/ctrl/alt
OSC_BG_RE = re.compile(rb'\x1b\]11;rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)')

def _fmt_tk(n):
    "Tk counts as compact k/M strings for the ctx meter."
    if n >= 1e6: return f'{n / 1e6:.1f}M'.replace('.0M', 'M')
    if n >= 1000: return f'{n / 1000:.1f}k'.replace('.0k', 'k')
    return str(n)


GUTTERS = {'sh': ('$$$ ', 'bold yellow'), 'in': ('»»» ', 'bold green'), 'out': ('««« ', 'bright_blue'),
           'result': ('««« ', 'bright_blue'), 'error': ('««« ', 'red'), 'image': ('««« ', 'magenta'),
           'ask': ('››› ', 'bold magenta'), 'ai': ('‹‹‹ ', 'magenta'),
           'think': ('‹‹‹ ', 'dim'), 'tool': ('≡≡≡ ', 'yellow')}

_NTH = {'!': 1, '@': 2, '#': 3, '$': 4, '%': 5, '^': 6, '&': 7, '*': 8, '(': 9}  # alt-shift digits arrive shifted

def _gutter(kind):
    r"""The block's left edge: 3 type glyphs + space (`x\dx`: the middle cell carries the ambient
    alt-digit number when the block wears one), color marking direction at a glance; the
    continuation rows are a dim `··· `."""
    f, sty = GUTTERS[kind]
    return (Text(f, style=sty), Text('··· ', style='dim'))

def _text(t): return ''.join(t) if isinstance(t, list) else (t or '')

def _default_history():
    return History()  # nav + ghost text draw only on this directory's session files

def _hl(code, theme='ansi_dark'):
    "Syntax-highlight `code` the one true way, dropping the newline highlight() appends."
    t = Syntax('', 'python', theme=theme).highlight(code)
    if t.plain.endswith('\n'): t.right_crop(1)
    return t

class App:
    "Wires a teleprint compositor/buffer to a KernelSession + Assistant; the test harness can supply `tty`."
    def __init__(self, tty, kernel=None, history='default', cfg=None, assistant=None):
        self.tty = tty
        self.cfg = cfg or {}
        self.k = KernelSession() if kernel is None else kernel
        self.hist = _default_history() if history == 'default' else history  # None = off (tests want hermeticity)
        self.comp = Compositor(tty)  # main awaits comp.start() before painting; tests mostly skip it (a fresh EmuTty gives origin 0 unstarted either way -- the nonzero-origin path has its own test)
        self.buf = Buffer()
        self.stream = None   # the growing stdout block of the in-flight cell
        self.menu = None     # a teleprint CompletionMenu while completing; Tab/shift+Tab cycle, Enter accepts
        self.tip = None      # a teleprint Tooltip while inspecting (shift+Tab on a bare buffer)
        self.done = asyncio.Event()
        self.kitty = False        # set by detect_kitty(); images fall back to a note without it
        self.theme = self.cfg.get('code_theme', 'auto')  # 'auto' resolves via detect_theme (OSC 11)
        if self.theme == 'auto': self.theme = 'ansi_dark'
        self.prompt_style = self.cfg.get('prompt_style', '')  # Rich style string for prompt text; '' = plain
        self.pad = bool(self.cfg.get('pad_transcript'))       # blank row before each input block (turn spacing)
        self.cell_imgs = set()   # png hashes shown this cell, to skip the execute_result repeat of a displayed figure
        self.assistant = assistant if assistant is not None else Assistant(cfg=cfg) if cfg else None
        self.mode = 'prompt' if self.cfg.get('prompt_mode') else 'code'  # 'prompt'|'code'|'shell': M-p/M-c/M-s
        if self.hist and self.mode != 'code': self.hist.mode = self.mode; self.hist.refresh()  # match a prompt_mode start
        self.ai_task = None      # the in-flight run_prompt task
        self.ai_sugg = None      # (text, cursor, suggestion): an Alt-. suggestion, valid while the buffer is unchanged
        self.cell_outputs = None # nbformat-shaped output records of the running cell, for ctx + persistence
        self._cycle = dict(idx=-1, resp='')  # alt-shift-up/down cycling over the last reply's fenced blocks
        self.fg_job = None       # (shell, mirror) while a borrow owns the terminal (SIGWINCH forwards here)
        self.shell = None        # the persistent shell Job (lazy; respawns after `exit`)
        self.shell_pwd = None
        self._shdrain = None       # between-borrow pty drain task + its rolling mirror
        self._shdrain_mirror = None
        self._quit_warned = False
        self.picker = None       # startup session-picker rows while open (an over transient; owns digits/Enter/n/Esc)
        self._ipyai_comm = None  # the kernel-side %ipyai comm id, set on comm_open
        self._pending = None   # input queued while an enter decision's round-trip is in flight (see on_enter)
        self.stdin_fut = None  # a pending kernel input_request; Enter answers it (see _on_stdin)
        self.k.on_stdin = self._on_stdin
        self._load_codes = []   # code cells queued by load_dialog for run_loaded
        self.k.on_comm = self._on_comm
        self.comp.on_key = self.on_key
        self.comp.on_paste = self.on_paste
        self.tv = TranscriptView(self.comp, self._tail_content)
        self.editing = None  # (message, kind) while the transcript view's e edit owns the composer
        self.retry = None    # (message, kind): a kind-matched submit replaces from that exchange instead of appending
        self.comp.on_mouse = lambda ev: self.tv.on_mouse(ev)
        self.comp.on_wheel = self._on_wheel
        self.comp.on_act = self._on_act
        self.comp.numbering = True  # ambient alt-digit numbers on the newest toggleable blocks

    @property
    def busy(self):
        "Enter is gated on this: a running cell or a running AI turn each own the transcript's live edge."
        return self.k.busy or (self.ai_task is not None and not self.ai_task.done())

    @property
    def collapse_at(self):
        "Auto-collapse threshold for outputs: about half a screen, always small enough to fold while visible."
        return max(3, min(self.comp.rows // 2, self.comp.rows - 5))

    _SPIN = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

    def _status(self):
        s = HINT
        if self.assistant is not None and (ctx := self.assistant.ctx_usage):
            used, mx = ctx
            s += f'⋄ctx {_fmt_tk(used)}/{_fmt_tk(mx)} ({100 * used // mx}%)'
        return s

    MODE = dict(code=('»»» ', 'bold green'), prompt=('››› ', 'bold magenta'), shell=('$$$ ', 'bold yellow'))

    def _tail_content(self):
        "Tail layout: an optional blank spacer row (pad_transcript), then dim status above the prompt (a shell's status-line shape). Transients (menu/tooltip) ride `over` in paint(), directly above the status row, and never ink."
        spin = self._SPIN[int(time.monotonic() * 10) % len(self._SPIN)] if self.busy else ' '  # one cell says busy; the ticker animates it
        seg = Text(f'[{self.mode}]', style=Style(meta={'act': 'mode'}))  # click cycles the mode; M-p/c/s pick directly
        status = Text(spin + ' ') + (Text('↻', style='bold yellow') if self.retry else Text('')) + seg + Text('⋄' + self._status(), style='dim')  # ↻: a submit replaces from the recalled exchange
        status.truncate(self.comp.cols, overflow='ellipsis')  # one row always: small screens truncate, never wrap
        plain = self.mode != 'code' or self.buf.text.startswith(('.', '!'))  # only code-as-code highlights
        hl = Text(self.buf.text) if plain else _hl(self.buf.text, self.theme)
        src = hl.split('\n')
        if src and not src[-1].plain: src.pop()  # drop only the empty tail line: a whitespace continuation line must survive
        if not src: src = [Text('')]
        mark, msty = self.MODE[self.mode]
        cont = '··· '
        prompt = Text()  # empty base style: a styled first arg would become the BASE for the whole line, bolding the code
        prompt.append(mark, style=msty)
        prompt.append(src[0])
        for c in src[1:]:
            prompt.append('\n' + cont, style=msty)
            prompt.append(c)
        sugg = ''
        if self.buf.cursor == len(self.buf.text) and not self.menu:
            if self.ai_sugg and self.ai_sugg[0] == self.buf.text and self.ai_sugg[1] == self.buf.cursor:
                sugg = self.ai_sugg[2]  # Alt-. suggestion overrides history while the document is unchanged
            elif self.hist: sugg = self.hist.suggest(self.buf.text)
        self.buf.suggestion = sugg
        if sugg: prompt.append(sugg, style='dim')
        elif not self.buf.text and not self.k.busy:  # empty composer: hint the two ways out of this mode
            prompt.append(' · '.join(f'M-{m[0]} {m}' for m in self.MODE if m != self.mode), style='dim')
        lines = ([Text('')] if self.pad else []) + [status, prompt]
        before = self.buf.text[:self.buf.cursor]
        col = 4 + cell_len(before.rsplit('\n', 1)[-1])  # all gutter prefixes are 3 glyphs + space = 4 cells
        return lines, (2 if self.pad else 1, before.count('\n'), col)

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
        if i is not None and rows: self.resume_session(rows[i][0])
        self.paint()

    def _picker_key(self, k):
        "The picker owns keys while open: modal, like the transcript view's vocabulary."
        if k.name == 'enter': self._pick(0)
        elif k.char and k.char.isdigit() and 0 < int(k.char) <= len(self.picker): self._pick(int(k.char) - 1)
        elif k.name in ('n', 'escape', 'ctrl+c'): self._pick(None)


    def paint(self):
        if self.tv.active:
            self.tv.notify()  # follow mode tracks new blocks; a no-op unless following
            self.tv.draw()
        else:
            lines, cursor = self._tail_content()
            over = self._picker_rows() if self.picker is not None else \
                   [self.menu.renderable()] if self.menu else [self.tip.renderable()] if self.tip else []
            self.comp.set_tail(*lines, cursor=cursor, over=over)

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
        """OSC 11 background query: pick the highlighter theme for a light or dark terminal (silence
        means stay dark). Verified live under tmux 2026-07: tmux forwards the query to the attached
        client's terminal and relays the reply to the querying pane, well inside the timeout -- unlike
        the kitty APC probe, whose replies tmux does not route to panes (see detect_kitty)."""
        m = OSC_BG_RE.search(self._probe(b'\x1b]11;?\x1b\\', timeout))
        if m:
            r, g, b = (int(ch[:2], 16) for ch in m.groups())
            self.theme = 'ansi_light' if (0.2126*r + 0.7152*g + 0.0722*b) / 255 > 0.5 else 'ansi_dark'
        return self.theme

    def show_image(self, img):
        """An image output (PNG or JPEG) as a kitty Unicode-placeholder block: the APC transmit goes
        straight to the tty (cursor-neutral bytes), while the placeholder grid -- ordinary styled text --
        becomes the block, so it repaints, commits, and survives tmux like any other transcript content."""
        from kittytgp.core import _read_png, PNG_SIGNATURE
        if not img.startswith(PNG_SIGNATURE):  # kitty transmits f=100 (PNG): convert jpeg and friends
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.open(io.BytesIO(img)).save(buf, format='PNG')
            img = buf.getvalue()
        _, w, h = _read_png(img)
        if not self.kitty:
            self.comp.print_block(f'[image {w}x{h}px -- this terminal lacks kitty graphics]', gutter=_gutter('image'), tag='image', source=f'[image {w}x{h}px]')
            return
        c = max(1, min(40, self.comp.cols - 2, w))
        r = max(1, math.ceil(h / (w / c * 2)))
        transmit, placeholder = render_parts(img, cols=c, rows=r)
        self.comp.tty.write(transmit)
        self.comp.print_block(Text.from_ansi(placeholder), gutter=_gutter('image'), tag='image', source=f'[image {w}x{h}px]')

    def on_out(self, c):
        ot = c.get('output_type')
        if self.cell_outputs is not None: self.cell_outputs.append(c)  # the cell record: ctx + persistence
        if ot == 'stream':
            if self.stream is None: self.stream = self.comp.print_block(gutter=_gutter('out'), tag='out', collapse_at=self.collapse_at)
            txt = _text(c.get('text')).rstrip('\n')
            if txt: self.comp.extend(self.stream, Text(txt, style='red') if c.get('name') == 'stderr' else txt)
            self.stream.source = (self.stream.source or '') + _text(c.get('text'))
            return
        self.stream = None  # any non-stream output closes the growing block: only the LAST block can grow, and later stream text starts a fresh one (Jupyter's separate output areas)
        if ot in ('execute_result', 'display_data'):
            data = c.get('data', {})
            mime = next((m for m in ('image/png', 'image/jpeg') if m in data), None)
            if mime:
                img = base64.b64decode(data[mime])
                # Jupyter renders a same-cell `fig` twice (flush display_data + execute_result repr);
                # suppress only a byte-identical execute_result repeat -- distinct images all render
                if not (ot == 'execute_result' and hash(img) in self.cell_imgs):
                    self.cell_imgs.add(hash(img))
                    self.show_image(img)
            elif 'text/plain' in data:
                t = _text(data['text/plain'])
                self.comp.print_block(t, gutter=_gutter('result'), tag='result', collapse_at=self.collapse_at, source=t)
        elif ot == 'error':
            tb = Text.from_ansi('\n'.join(c.get('traceback', [])))
            self.comp.print_block(tb, gutter=_gutter('error'), tag='error', source=tb.plain)  # errors always print open

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

    async def _kernel_cwd(self):
        "The kernel's cwd, queried per spawn: bare-`!` jobs and kernel-side `!` should agree on where they run."
        try: return await self.k.kc.eval("__import__('os').getcwd()", _call=False)
        except Exception: return os.getcwd()

    def _job_note(self, s): self.comp.print_block(Text(s, style='dim'), gutter=_gutter('out'), tag='out')

    async def _on_stdin(self, prompt, password):
        "Answer a kernel `input()` from the composer: the prompt paints as a block, the next Enter's text replies."
        if prompt: self.comp.print_block(Text(prompt), gutter=_gutter('out'), tag='out', source=prompt)
        self.stdin_fut = asyncio.get_running_loop().create_future()
        self.paint()
        try: return await self.stdin_fut
        finally: self.stdin_fut = None

    async def _ensure_shell(self):
        """The persistent shell, spawned lazily on first use: an owned gateway terminal beside the
        kernel (fresh per session, deleted on exit), so `cd`/exports/aliases and the jobs table
        (`fg`/`bg`/`jobs`/ctrl-Z) persist and belong to the shell. The first-prompt relay absorbs
        rc noise; between borrows a drain task pumps the channel (bg job output) into a rolling mirror."""
        if self.shell is not None: return
        try: self.shell = await GateShell(self.k.url, size=self.tty.size, cwd=await self._kernel_cwd()).start()
        except Exception as e:
            self.shell = None
            raise RuntimeError(f'shell failed to start: {e}')
        self.shell_pwd = None
        boot = pyghostty.Terminal(*self.tty.size)
        res = await self.shell.relay(mirror=boot)
        if res == 'eof':
            await self.shell.close()
            self.shell = None
            raise RuntimeError('shell failed to start')
        self.shell_pwd = res[2]
        self._start_drain()

    def _start_drain(self):
        self._shdrain_mirror = pyghostty.Terminal(*self.tty.size)
        self._shdrain = self.comp.spawn(self.shell.relay(mirror=self._shdrain_mirror), name='shdrain')

    async def _stop_drain(self):
        "Take the pty back from the drain; anything a bg job printed while we were away becomes a block."
        d, self._shdrain = self._shdrain, None
        if d is None: return
        d.cancel()
        await asyncio.gather(d, return_exceptions=True)
        left = self._shdrain_mirror.contents().rstrip()
        if left:
            blk = self.comp.print_block(Text(left, style='dim'), gutter=_gutter('out'), tag='out',
                                        collapse_at=self.collapse_at, source=left)
            if self.assistant is not None:
                m = self.assistant.add_cell('!# background output', [dict(output_type='stream', name='stdout', text=left + '\n')])
                if m is not None: blk.msg_id = m.id  # stamped here so the enclosing exchange's stamp skips it

    async def run_job(self, cmd, record=True):
        """A routed 'job' submission: one command (or small script) for the persistent shell, run under a
        borrow; returns the exit code (None if the shell itself exited). `record=False` (the F2 editor
        borrow) skips the transcript/dialog recording and the exit-code error block."""
        try: await self._ensure_shell()
        except RuntimeError as e:
            return self.comp.print_block(Text(str(e), style='red'), gutter=_gutter('error'), tag='error')
        await self._stop_drain()
        mirror = pyghostty.Terminal(*self.tty.size)
        loop, fd = asyncio.get_running_loop(), getattr(self.tty, 'fd', None)
        reading = fd is not None and loop.remove_reader(fd)
        self.comp.release()
        self.tty.raw()
        self.fg_job = (self.shell, mirror)
        self.shell.write(cmd.encode() + b'\n')
        try: res = await self.shell.relay(self.tty.write, mirror=mirror, in_fd=fd)
        finally:
            self.fg_job = None
            self.tty.cooked()
            await self.comp.reanchor()
            if reading: loop.add_reader(fd, self._read_tty)
        resid = mirror.contents().rstrip()
        if res == 'eof':  # the shell itself exited (`exit`, or death): respawn fresh on next use
            sh, self.shell = self.shell, None
            await sh.close()  # delete the dead terminal's registry entry
            if record: self._record_shell_cell(cmd, resid, 0)
            why = f'shell connection lost: {sh.error!r}' if sh.error else 'shell exited'
            return self._job_note(f'{why}; a fresh one starts on the next shell command')
        _, ec, pwd = res
        if record:
            self._record_shell_cell(cmd, resid, ec)
            if ec: self.comp.print_block(Text(f'exit {ec}', style='red'), gutter=_gutter('error'), tag='error')
        if pwd != self.shell_pwd:
            self.shell_pwd = pwd
            await self._sync_kernel_cwd(pwd)
        self._start_drain()
        return ec

    async def edit_buffer(self):
        """F2: hand the composer text to $EDITOR over the shell borrow, reloading it on a clean exit
        (a nonzero exit -- vim's :cq -- abandons the edit). Nothing records: not a transcript block,
        not a dialog cell; the editor session is composing, not history."""
        if self.k.busy or self.fg_job is not None: return
        sfx = dict(code='.py', shell='.sh', prompt='.md')[self.mode]
        fd, path = tempfile.mkstemp(suffix=sfx, prefix='ipyai-f2-')
        with os.fdopen(fd, 'w') as f: f.write(self.buf.text)
        try:
            ed = os.environ.get('EDITOR', 'vi')  # the app's env, not the shell's: rc files override $EDITOR in the pty shell
            ec = await self.run_job(f'{ed} {shlex.quote(path)}', record=False)
            if ec == 0:
                with open(path) as f: self.buf.text = f.read().rstrip('\n')
                self.buf.cursor = len(self.buf.text)
        finally: os.unlink(path)
        self.paint()

    def _record_shell_cell(self, cmd, resid, ec):
        "The residue records as an ordinary `!cmd` cell: model-only block (its bytes are already on glass) + dialog cell."
        if resid: self.comp.record_block(resid, gutter=_gutter('out'), tag='out', collapse_at=self.collapse_at, source=resid)
        if self.assistant is not None:
            outs = [dict(output_type='stream', name='stdout', text=resid + '\n')] if resid else []
            if ec: outs.append(dict(output_type='stream', name='stderr', text=f'[exit {ec}]\n'))
            self.assistant.add_cell(f'!{cmd}', outs)

    async def _sync_kernel_cwd(self, pwd):
        "cwd flows shell -> kernel after each shell command, so the two worlds agree about where you are."
        try: await self.k.kc.reply(f"import os; os.chdir({pwd!r})", silent=True, store_history=False, timeout=5)
        except Exception: pass

    async def attach_assistant(self, resume=None, load=None, fresh=False):
        """Wire the AI side to the live kernel: bridge + tools, a session file under `./.ipyai/sessions/`,
        optional resume/dialog load. `fresh` (the plain-launch default) starts a new session;
        `resume=PREFIX` resumes that session file; bare `-r` lists this directory's sessions in the
        startup picker (a transient: digits pick, Enter = newest, n/Esc = fresh). An attached
        (non-owned) kernel is taken as found: bridge and registry, but no seeding."""
        from .bridge import setup_tools
        if self.k.owned:
            bridge, tools = await setup_tools(self.k.kc)
            try: await bridge._exec("get_ipython().extension_manager.load_extension('ipyai.magic')")
            except Exception: pass  # kernel without ipyai installed: %ipyai just won't exist there
        else:
            from .kernel_bridge import KernelBridge
            from .tooling import ToolRegistry
            bridge = KernelBridge(self.k.kc)
            tools = ToolRegistry(bridge)
        if self.assistant is None: self.assistant = Assistant(cfg=self.cfg or None)
        self.assistant.tools, self.assistant.bridge = tools, bridge
        if resume is not None: self.resume_session(resolve_session(resume))
        elif not fresh and load is None:
            rows = list_sessions()
            if len(rows) == 1: self.resume_session(rows[0][0])
            elif rows: self.picker = rows[:9]  # the startup picker paints as an over transient once run() starts
        if self.assistant.session is None: self.assistant.session = Session()
        a = self.assistant
        a.dlg.meta['ipyai'] = {**a.dlg.meta.get('ipyai', {}), 'kernel_id': self.k.kid}
        if load is not None:
            self._job_note(self.load_dialog(load))   # the one ack: the comm path's ack arrives as magic output instead
            await self.run_loaded()

    def _on_comm(self, mt, c):
        "%ipyai lands here (via KernelSession.on_comm): track the comm, ack each command by request id."
        if mt == 'comm_open' and c.get('target_name') == 'ipyai': self._ipyai_comm = c.get('comm_id')
        elif mt == 'comm_close' and c.get('comm_id') == self._ipyai_comm: self._ipyai_comm = None
        elif mt == 'comm_msg' and c.get('comm_id') == self._ipyai_comm:
            d = c.get('data', {})
            try: reply = dict(req=d.get('req'), text=self._ipyai_cmd(list(d.get('cmd') or [])))
            except Exception as e: reply = dict(req=d.get('req'), error=str(e))
            self.k.kc._exec_req('comm_msg', content=dict(comm_id=self._ipyai_comm, data=reply))
            if self._load_codes: self.comp.spawn(self.run_loaded(), name='run-loaded')  # after the ack: the magic holds the kernel until these run

    def _ipyai_cmd(self, args):
        "One %ipyai command against app/assistant state; returns the ack text. Settings are session-only."
        a = self.assistant
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
            if rest: self.theme = self.detect_theme() if rest[0] == 'auto' else rest[0]
            return f'code_theme = {self.theme}'
        if cmd == 'prompt':
            self._set_mode('prompt' if self.mode != 'prompt' else 'code')
            self.paint()
            return f'mode = {self.mode}'
        if cmd == 'sessions':
            return _sessions_text(list_sessions())
        if cmd == 'reset':
            a.reset()
            a.session = Session()
            a.dlg.meta['ipyai'] = {**a.dlg.meta.get('ipyai', {}), 'kernel_id': self.k.kid}
            return f'reset: fresh conversation (session {a.session.path.stem[:8]})'
        if cmd == 'save':
            if not rest: raise ValueError('usage: %ipyai save PATH')
            p = Path(rest[0]).expanduser()
            if p.suffix != '.ipynb': p = p.with_suffix('.ipynb')
            from aidialog.ipynb import write_ipynb
            write_ipynb(a.dlg, p)
            return f'saved {len(a.dlg)} messages to {p}'
        if cmd == 'load':
            if not rest: raise ValueError('usage: %ipyai load PATH')
            return self.load_dialog(rest[0])
        raise ValueError(f'unknown %ipyai command: {cmd!r} (bare %ipyai lists commands)')

    def _replay_output(self, o):
        "Render one stored nbformat output: the dict shapes match iopub content, so on_out replays them directly."
        ot = o.get('output_type')
        if ot in ('stream', 'display_data', 'execute_result', 'error'): self.on_out(o)

    def _replay_reply(self, response):
        "Render a stored reply through the same block machinery: fmt2hist recovers text and tool parts."
        from aidialog.msg_parts import fmt2hist
        tr = TurnRenderer(self.comp, _gutter, theme=self.theme, collapse_at=self.collapse_at)
        try: msgs = fmt2hist(response)
        except Exception: msgs = None
        if not msgs:
            tr.md.feed(response)
            tr.done()
            return
        for m in msgs:
            for p in m.content:
                if isinstance(p, TextPart) and p.text and m.role == 'assistant':
                    if p.text.strip(): tr.md.feed(p.text)
                elif isinstance(p, (ToolUse, ToolResult)): tr.event(p)
        tr.done()

    def resume_session(self, path):
        "Resume: paint a session file's transcript and adopt its Dialog, continuing in the same file (no kernel state is rebuilt)."
        from aidialog.ipynb import read_ipynb
        dlg = read_ipynb(path)
        if dlg is None:
            self.comp.print_block(Text(f'{path}: cannot read session', style='dim'), gutter=_gutter('error'), tag='error')
            return
        a = self.assistant
        for m in dlg.messages:
            before = next(reversed(self.comp.blocks), 0)
            if m.msg_type in ('code', 'note'):
                body = _hl(m.content, self.theme) if m.msg_type == 'code' else Text(m.content)
                self.comp.print_block(body, gutter=_gutter('in'), tag='in', source=m.content, pad=self.pad)
                self.stream = None
                self.cell_imgs = set()
                for o in (m.output or []): self._replay_output(o)
                self.stream = None
                a.n_cells += 1
            elif m.msg_type == 'prompt':
                self.comp.print_block(Text(m.content, style=self.prompt_style), gutter=_gutter('ask'), tag='ask', source=m.content, pad=self.pad)
                self._replay_reply(m.ai_res)
                a.last_response = m.ai_res
            else: continue
            for bid, b in self.comp.blocks.items():
                if bid > before:
                    b.msg_id = m.id
                    if m.skipped: b.dim = True  # a hide from a past session shows dim on replay too
        a.dlg = dlg
        a.dlg.meta['ipyai'] = {**a.dlg.meta.get('ipyai', {}), 'kernel_id': self.k.kid}
        a.session = Session(path=path)

    def load_dialog(self, path):
        """Import a Dialog .ipynb (%ipyai save's output, or any notebook) as the session model. Nothing is
        painted -- the .ipynb itself is the inspection surface -- and the code cells are queued for
        `run_loaded` to re-run, rebuilding kernel state. Stored outputs stay as the AI-context record."""
        from aidialog.ipynb import read_ipynb
        a = self.assistant
        dlg = read_ipynb(path)
        if dlg is None: raise ValueError(f'cannot read dialog: {path}')
        dlg.messages = [m for m in dlg.messages
                        if not (m.msg_type == 'code' and m.content.lstrip().startswith('%ipyai'))]
        for m in dlg.messages:
            if m.msg_type == 'prompt': a.last_response = m.ai_res
        dlg.name = os.path.basename(a.cwd)
        a.dlg = dlg
        a.n_cells = sum(1 for m in dlg.messages if m.msg_type in ('code', 'note'))
        self._load_codes = [m.content for m in dlg.messages if m.msg_type == 'code']
        return f'loaded {len(dlg)} messages from {path}; running {len(self._load_codes)} code cells'

    async def run_loaded(self):
        """Re-run the code cells queued by `load_dialog`, silently except errors. Runs after the current
        cell completes: the `%ipyai load` magic holds the kernel until its comm reply lands, so cell
        execution must wait for idle or it would deadlock."""
        codes, self._load_codes = self._load_codes, []
        while self.k.busy: await asyncio.sleep(0.05)
        for code in codes: await self.k.run(code, self._silent_out)
        self.paint()

    def _silent_out(self, c):
        "Output handler for `run_loaded`: only errors surface (errors always print open)."
        if c.get('output_type') == 'error':
            tb = Text.from_ansi('\n'.join(c.get('traceback', [])))
            self.comp.print_block(tb, gutter=_gutter('error'), tag='error', source=tb.plain)

    async def do_complete(self):
        matches, start = await self.k.kc.complete(self.buf.text, self.buf.cursor)
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
        lines = self.buf.text[:self.buf.cursor].split('\n')
        try: sigs = await self.k.kc.sig_help(code=self.buf.text, line_no=len(lines), col_no=len(lines[-1]))
        except Exception: sigs = []
        self.menu = None
        self.tip = None
        if sigs and isinstance(sigs, list):
            s = sigs[0]
            self.tip = Signature(s['label'], [p['desc'] for p in s['params']], s.get('idx'), s.get('doc', ''))
        elif text := await self.k.kc.inspect(self.buf.text, self.buf.cursor):
            self.tip = Tooltip(Text.from_ansi(text))
        self.paint()

    def _dismiss(self):
        self.menu = None
        self.tip = None

    def _submit(self, text, code=None, sh=False):
        "Print the input block (prompts bold, all inputs padded with a turn gap; shell `$$$`, code highlighted), clear the buffer, record history. An armed retry fires here: a kind-matched submit rewinds first, a mismatch cancels."
        if self.retry is not None:
            (m, want), self.retry = self.retry, None
            kind = 'job' if sh else 'prompt' if code is None else 'code'
            if kind == want and self.assistant is not None: self._truncate_to(m)
        if sh: self.comp.print_block(Text(code), gutter=_gutter('sh'), tag='sh', source=code, pad=self.pad)
        elif code is None: self.comp.print_block(Text(text, style=self.prompt_style), gutter=_gutter('ask'), tag='ask', source=text, pad=self.pad)
        else: self.comp.print_block(_hl(code, self.theme), gutter=_gutter('in'), tag='in', source=code, pad=self.pad)
        self.buf.clear()
        self.ai_sugg = None
        self._dismiss()
        if self.hist:
            self.hist.reset_nav()
            self.hist.add_local(text)  # instantly navigable/suggestible; the kernel's own write lags its flush thread

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
        Input arriving during the round-trip queues in `_pending` and replays once the decision
        lands, so a raw key burst cannot outrun the check and flatten into one line."""
        run = None
        before_bid, nmsg = next(reversed(self.comp.blocks), 0), self._nmsgs()  # exchange boundary: blocks/messages after these belong to it
        try:
            text = self.buf.text
            kind, payload = route(text, self.mode)
            if kind == 'prompt':
                self._submit(text)
                run = self.run_prompt(payload)
            elif kind == 'job':
                self._submit(text, payload, sh=True)
                run = self.run_job(payload)
            else:
                status, indent = await self.k.kc.check(payload)
                if self.buf.text != text or self.busy: return  # changed or a run started mid-flight: stale decision
                if status == 'incomplete': self.buf.insert('\n' + ('' if self._pending else indent))  # a burst carries its own indentation
                else:
                    self._submit(text, payload)
                    run = self.run_cell(payload)
        finally: self._drain()   # release queued input before the run: typeahead lands in the fresh composer
        if run is None: return self.paint()
        try: await run
        except Exception as e:
            self.comp.print_block(Text(f'{kind} failed: {type(e).__name__}: {e}', style='red'), gutter=_gutter('error'), tag='error')
        self._stamp_exchange(before_bid, nmsg)
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
        "Alt-.: one-shot AI suggestion into the ghost-text slot, valid only while the document is unchanged."
        snap = (self.buf.text, self.buf.cursor)
        try: text = await self.assistant.suggest(self.buf.text[:self.buf.cursor], self.buf.text[self.buf.cursor:])
        except Exception: return
        if text and (self.buf.text, self.buf.cursor) == snap:
            self.ai_sugg = (snap[0], snap[1], text)
            self.paint()

    def _nmsgs(self):
        return len(self.assistant.dlg.messages) if self.assistant is not None else 0

    def _stamp_exchange(self, before_bid, nmsg):
        """Link the exchange's blocks to its Dialog message: every block printed since `before_bid` gets the
        id of the message the exchange recorded (skipping blocks a nested recorder already stamped, e.g.
        `_stop_drain`'s background-output cell). One exchange = one message, so hide toggles both halves."""
        if self._nmsgs() <= nmsg: return  # nothing recorded (plain REPL, empty prompt, unrecorded job)
        m = self.assistant.dlg.messages[-1]
        for bid, b in self.comp.blocks.items():
            if bid > before_bid and getattr(b, 'msg_id', None) is None: b.msg_id = m.id

    def _recall_last(self):
        """Alt-r: the most recent exchange (prompt, code, or shell) into the composer, armed for
        retry -- a kind-matched submit REPLACES it (and everything after) instead of appending."""
        if self.assistant is None: return
        m = next((x for x in reversed(self.assistant.dlg.messages) if x.msg_type in ('prompt', 'code')), None)
        if m is None: return
        want = 'prompt' if m.msg_type == 'prompt' else 'job' if m.content.startswith('!') else 'code'
        self.retry = (m, want)
        self._set_mode('prompt' if want == 'prompt' else 'code')
        self.buf.text, self.buf.cursor = m.content, len(m.content)
        self.paint()

    def _retry_from_tv(self):
        "E in the transcript view, on a prompt input: that prompt into the composer, armed for retry, back on the live screen."
        tv = self.tv
        m = self._cursor_msg()
        if m is None or m.msg_type != 'prompt' or self.comp.blocks[tv.cur].tag != 'ask':
            tv.msg = 'retry (E) works on a prompt input'
            return tv.draw()
        self.retry = (m, 'prompt')
        self._set_mode('prompt')
        self.buf.text, self.buf.cursor = m.content, len(m.content)
        tv.leave()
        self.paint()

    def _truncate_to(self, m):
        """Retry rewind: the target message and everything after it leave the dialog, the store, and the
        block model (rows already inked stay in history; kernel state is untouched)."""
        a = self.assistant
        i = next((j for j, x in enumerate(a.dlg.messages) if x.id == m.id), None)
        if i is None: return
        dropped = a.dlg.messages[i:]
        ids = {x.id for x in dropped}
        a.dlg.remove_msgs(dropped)
        for b in [b for b in self.comp.blocks.values() if getattr(b, 'msg_id', None) in ids]:
            self.comp.remove_block(b)
        a.n_cells = sum(1 for x in a.dlg.messages if x.msg_type in ('code', 'note'))
        a.save()


    def _jump_exchange(self, d):
        "Shift-up/down in the transcript view: block cursor to the previous/next exchange start (an input, ask, or shell block)."
        tv = self.tv
        starts = [bid for bid, b in self.comp.blocks.items() if b.tag in ('in', 'ask', 'sh') and b.height > 0]
        if not starts: return
        if d < 0: bid = next((b for b in reversed(starts) if b < (tv.cur or 0)), starts[0])
        else: bid = next((b for b in starts if b > (tv.cur or 0)), starts[-1])
        tv.select(bid)


    def _cursor_msg(self):
        "The Dialog message behind the transcript-view cursor block, or None."
        tv = self.tv
        mid = getattr(self.comp.blocks.get(tv.cur), 'msg_id', None) if tv.cur is not None else None
        return next((m for m in self.assistant.dlg.messages if m.id == mid), None) if self.assistant and mid else None

    def _toggle_hide(self):
        "h in the transcript view: flip `skipped` on the cursor block's message; its blocks dim, and the store learns."
        tv = self.tv
        m = self._cursor_msg()
        if m is None:
            tv.msg = 'no AI record for this block'
            return tv.draw()
        m.skipped = 0 if m.skipped else 1
        for b in self.comp.blocks.values():
            if getattr(b, 'msg_id', None) == m.id: b.dim = bool(m.skipped)  # the dim setter invalidates the row cache
        self.assistant.save()
        tv.msg = 'hidden from AI' if m.skipped else 'visible to AI'
        tv.rebuild()

    def _edit_current(self):
        """e in the transcript view: the cursor block's message into the composer for editing. An ask
        block edits the prompt's content; any other block of a prompt exchange edits the WHOLE reply
        markdown (tool calls and results live inside it); a code/note exchange edits its source."""
        tv = self.tv
        m = self._cursor_msg()
        if m is None:
            tv.msg = 'no AI record for this block'
            return tv.draw()
        blk = self.comp.blocks[tv.cur]
        kind = 'reply' if m.msg_type == 'prompt' and blk.tag != 'ask' else 'content'
        self.editing = (m, kind)
        self.buf.text = m.ai_res if kind == 'reply' else m.content
        self.buf.cursor = len(self.buf.text)
        tv.composing = True
        tv.msg = f"editing {'reply' if kind == 'reply' else m.msg_type} -- Enter writes back, Esc cancels"
        tv.draw()

    def _finish_edit(self, write):
        "Leave editing state; on write, the new text lands in the message, its blocks, and the store."
        from rich.markdown import Markdown
        (m, kind), self.editing = self.editing, None
        tv, text = self.tv, self.buf.text
        self.buf.clear()
        tv.composing = False
        if not write or (kind == 'reply' and text == m.ai_res) or (kind == 'content' and text == m.content):
            tv.msg = 'edit cancelled' if not write else 'unchanged'
            return tv.rebuild()
        blks = [b for b in self.comp.blocks.values() if getattr(b, 'msg_id', None) == m.id]
        if kind == 'reply':
            m.output = text  # the prompt-output setter wraps and clears the ai cache
            first = True
            for b in blks:
                if b.tag == 'ask': continue
                if first: self.comp.set_body(b, Markdown(text, code_theme=self.theme), source=text)
                else: self.comp.set_body(b)  # height 0: the sibling vanishes from the window; scrollback keeps it
                first = False
            if m is self.assistant.dlg.messages[-1]: self.assistant.last_response = text
        else:
            m.content = text
            b = next((b for b in blks if b.tag in ('in', 'ask', 'sh')), None)
            if b is not None:
                body = _hl(text, self.theme) if b.tag == 'in' else Text(text)
                self.comp.set_body(b, body, source=text)
        self.assistant.save()
        tv.msg = 'written'
        tv.rebuild()

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

    def on_paste(self, text):
        "Paste goes to the composer everywhere; in the transcript view it also takes compose focus."
        if self._pending is not None: return self._pending.append(text)  # ordered behind the in-flight enter
        if self.tv.active: self.tv.composing = True
        self.buf.insert(text)
        if self.tv.active: self.tv.draw()
        else: self.paint()

    def on_key(self, k):
        if self._pending is not None and k.name != 'ctrl+c':
            return self._pending.append(k)  # an enter decision is in flight: input stays ordered behind it
        if self.tv.active: return self._tv_key(k)
        if self.picker is not None: return self._picker_key(k)
        if k.name == 'ctrl+d' and not self.buf.text:
            if self.shell is not None and not self._quit_warned:  # bash's convention: warn once, an immediate second C-D quits
                self._quit_warned = True
                self._job_note('the shell (and any jobs in it) closes with the app  (C-D again quits)')
                return
            self._dismiss()
            self.paint()  # one clean final frame: transient UI must never ink as exit debris
            self.done.set()
            return
        self._quit_warned = False  # any other key withdraws the warning
        if k.name == 'ctrl+c': return self.on_sigint()
        if k.name == 'ctrl+t' and not self.busy:
            self.tv.enter()
            return
        if k.name == 'ctrl+o':
            live = [b for b in self.comp.blocks.values() if not b.committed]
            if live: self.comp.toggle(live[-1])
        elif k.name == 'f2': return self.edit_buffer()
        elif k.name == 'alt+r': self._recall_last()  # retry: recall the last exchange, submit REPLACES it (alt-up stays history)
        elif k.name == 'escape' and self.retry is not None:
            self.retry = None  # disarm: the composer keeps its text, submits append again
            self.paint()
        elif k.name in ('alt+p', 'alt+c', 'alt+s'):
            self._set_mode(dict(p='prompt', c='code', s='shell')[k.name[4]])
            self._dismiss()
        elif k.name == 'alt+.':
            if self.buf.text.strip() and not self.busy and self.assistant is not None: return self.do_ai_suggest()
        elif k.name == 'alt+W':
            bs = self._reply_blocks()
            if bs: self.buf.insert('\n'.join(bs))
        elif len(k.name) == 5 and k.name.startswith('alt+') and k.name[4] in self.comp.numbered:
            self.comp.toggle(self.comp.blocks[self.comp.numbered[k.name[4]]])  # the block wearing that digit
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
        elif k.name == 'enter' and self.stdin_fut is not None and not self.stdin_fut.done():
            text = self.buf.text   # a kernel input() owns this Enter: the composer line answers it
            self.buf.clear()
            self.stdin_fut.set_result(text)
        elif k.name == 'enter' and self.buf.text and not self.busy:
            self._pending = []
            return self.on_enter()
        elif k.name == 'alt+enter':  # always a newline: the codex/Claude convention
            self.buf.insert('\n')
        elif k.name == 'tab' and self.buf.text and not self.k.busy: return self.do_complete()
        elif k.name == 'shift+tab' and self.buf.text and not self.k.busy: return self.do_inspect()
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
        if self.assistant is not None and self.assistant.cancel_turn(): return  # ctrl-C stops the AI turn first
        if self.k.busy: self.comp.spawn(self.k.interrupt(), name='interrupt')
        else:
            self.buf.clear()
            self.ai_sugg = None
            self.paint()

    def _read_tty(self): self.comp.on_bytes(os.read(self.tty.fd, 4096))

    def _task_error(self, e, t):
        "A spawned background task failed: an error block beats a stderr traceback through the raw-mode screen."
        self.comp.print_block(Text(f'{t.get_name()} failed: {e!r}', style='red'), gutter=_gutter('error'), tag='error')
        self.paint()

    def _resized(self):
        "The whole WINCH response (comp.on_resize): forward to shell/job, repaint only when we own the screen."
        if self.shell is not None and self.fg_job is None:
            self.shell.resize(*self.tty.size)   # idle shell follows the terminal (its pty winsize WINCHes children)
        if self.fg_job is not None:
            job, mirror = self.fg_job
            job.resize(*self.tty.size)
            mirror.resize(*self.tty.size)
            return  # the job owns the screen; the compositor adopts the new size at reanchor
        if self.tv.active: self.tv.leave()  # a rewrap invalidates the view; re-enter is one keystroke
        self.comp.resize()  # synchronous now: adopt size + repaint from the model
        self.paint()    # then rebuild the tail at the new width

    async def run(self):
        "The real-terminal main loop: tty reader, the compositor's signals, escape-timeout ticker."
        loop = asyncio.get_running_loop()
        loop.add_reader(self.tty.fd, self._read_tty)
        self.comp.on_resize = self._resized
        self.comp.on_task_error = self._task_error
        async def ticker():
            while True:
                await asyncio.sleep(0.2)
                self.comp.flush_input()
                if self.busy and not self.tv.active: self.paint()  # the throbber cell animates on the ticker's clock
        t = self.comp.spawn(ticker(), name='ticker')
        try:
            self.paint()
            await self.done.wait()
        finally:
            t.cancel()
            loop.remove_reader(self.tty.fd)
            self.comp.stop()

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
    t.write('\x1b[?1000;1006h\x1b[?2004h')  # SGR mouse + bracketed paste
    try:
        app = App(t, cfg=cfg)
        await app.comp.start()  # async now: the CPR await, and the compositor takes WINCH/INT/TERM/HUP
        app.detect_kitty()
        if cfg.get('code_theme', 'auto') == 'auto': app.detect_theme()
        kid = kernel or ''
        if Resume:  # explicit prefix: warm-attach the session's stamped kernel when it's still alive
            from aidialog.ipynb import read_ipynb
            d = read_ipynb(resolve_session(Resume))
            kid = kid or (d.meta.get('ipyai', {}).get('kernel_id', '') if d is not None else '')
        try: await app.k.start(kernel=kid)
        except ValueError:
            if kernel: raise            # an explicitly named kernel that's gone is an error, not a fallback
            await app.k.start()         # the session's stamped kernel is gone: cold resume on a fresh kernel
        await app.attach_assistant(resume=Resume or None, load=Load,
                                    fresh=Resume is None)  # plain launch is fresh; bare -r (const '') opens the picker
        await app.run()
    finally:
        t.write('\x1b[?2004l\x1b[?1000;1006l\r\n')
        t.restore()
        if app.k.kc is not None: await app.k.close()
        if app.shell is not None: await app.shell.close()  # the owned terminal dies with the session

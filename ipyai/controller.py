"""The controller: the one owner of the session Dialog and of the operation running on it.

Every Dialog change goes through a Controller method, and each changed Message is marked for the view.
Kernel output reaches its Message through `on_jmsg`: each request's msg_id starts with the id of the
Message it runs for, so output that arrives after its cell has finished still lands in the right place.
The controller imports no terminal code. It reaches the screen through `ui`, which provides `mark`,
`drop`, `reset`, `note`, `borrow`, `ask_input`, `restore_input`, `paint` and `size`."""
import asyncio, os
from dataclasses import dataclass
from pathlib import Path
import pyghostty
from aidialog.dialog import Dialog, Message, INTERRUPTED, mk_stream
from aidialog.ipynb import read_ipynb
from fastcore.utils import rtoken_hex
from jupywire.ops import parent_id
from jupywire.route import OUTPUT_MSGS, COMM_MSGS
from .assistant import is_note, note_str, LAST_PROMPT, LAST_RESPONSE
from .tools import KernelTools
from .shell import GateShell, script
from .session import new_session_path, save_session

__all__ = ['Controller', 'Op']

NEW_KERNEL = 'New kernel: the namespace is empty. Outputs above came from an earlier kernel.'

@dataclass
class Op:
    "The operation running in the foreground: a code cell, an AI turn, a shell command, or a load's reruns."
    kind: str                       # 'cell', 'turn', 'shell' or 'run'
    msg: Message = None             # the Message it works on; None once a turn that never started is removed
    task: asyncio.Task = None
    cancelled_by_user: bool = False
    started: bool = False           # a turn has streamed output or run a tool

class Controller:
    """Owns the session Dialog `dlg` and at most one running `Op`. `k` is the `KernelSession`, `assistant`
    the `Assistant` (None in plain-REPL mode). `on_comm` receives the kernel's comm traffic."""
    def __init__(self, ui, k, assistant=None):
        self.ui, self.k, self.assistant = ui, k, assistant
        self.op = self.on_comm = None
        self.shell = self.shell_pwd = None
        self.fg = None             # (shell, mirror) while a shell command owns the terminal
        self.transient = {}        # id -> Message whose output shows but is not recorded (%ipyai cells)
        self._tools = self._drain = self._drain_mirror = self._pending_load = None
        self._bg = set()           # strong refs to fire-and-forget tasks
        k.on_jmsg = self.on_jmsg
        self.adopt(Dialog(name=os.path.basename(os.getcwd())))

    @property
    def busy(self): return self.op is not None

    @property
    def kernel_busy(self):
        "Whether the kernel is running something for us: a cell, a load's reruns, or a tool call."
        return self.op is not None and self.op.kind in ('cell', 'run') or bool(self._tools and self._tools.inflight)

    @property
    def tools(self):
        "The `KernelTools` for the current kernel client, or None without a kernel."
        kc = self.k.kc
        if kc is None: return None
        if self._tools is None or self._tools.kc is not kc: self._tools = KernelTools(kc)
        return self._tools

    def msg(self, mid):
        "The Message with id `mid`, in the Dialog or shown transiently, or None."
        if self.op is not None and self.op.msg is not None and self.op.msg.id == mid: return self.op.msg
        return self.transient.get(mid) or next((m for m in reversed(self.dlg.messages) if m.id == mid), None)

    def last_reply(self):
        "The most recent AI reply in the Dialog, or ''."
        return next((m.ai_res for m in reversed(self.dlg.messages) if m.msg_type == 'prompt'), '')

    def _spawn(self, coro, name):
        "Run `coro` in the background, keeping it alive. If it fails, the `ui` shows the error as a note."
        t = asyncio.create_task(coro, name=name)
        self._bg.add(t)
        t.add_done_callback(self._task_done)
        return t

    def _task_done(self, t):
        self._bg.discard(t)
        if not t.cancelled() and (e := t.exception()) is not None: self.ui.note(f'{t.get_name()} failed: {e!r}', 'error')

    # -- kernel traffic --------------------------------------------------------------
    def on_jmsg(self, jmsg):
        """Every inbound kernel message: comms go to `on_comm`, and output goes into the code Message whose
        id starts its parent request's msg_id. Traffic for tool runs and helper calls matches no Message."""
        mt = jmsg.get('msg_type')
        if mt in COMM_MSGS:
            if self.on_comm is not None: self.on_comm(mt, jmsg.get('content', {}))
            return
        if mt not in OUTPUT_MSGS and mt not in ('update_display_data', 'clear_output'): return
        m = self.msg((parent_id(jmsg) or '').partition('.')[0])
        if m is None or m.msg_type != 'code': return
        if mt == 'clear_output': m.clear_output(wait=jmsg.get('content', {}).get('wait', False))
        else: m.add_output(jmsg)
        self.ui.mark(m)

    # -- operations ------------------------------------------------------------------
    def _start(self, kind, m, work):
        "Run `work(op)` as the foreground operation."
        op = self.op = Op(kind, m)
        op.task = asyncio.create_task(self._run(op, work(op)), name=kind)
        self.ui.paint()
        return op

    async def _run(self, op, coro):
        try: await coro
        except asyncio.CancelledError:
            if not op.cancelled_by_user: raise
        except Exception as e: self.ui.note(f'{op.kind} failed: {type(e).__name__}: {e}', 'error')
        finally:
            self.op = None
            if op.msg is not None: self.transient.pop(op.msg.id, None)
            self.save()
            self.ui.paint()
        if self._pending_load is not None: self.run_loaded()

    def cancel(self):
        """Ctrl-C: stop the running operation, returning whether there was one. A cell's kernel is interrupted.
        A turn or a load's reruns are cancelled, and the kernel interrupted if it is running their code. A shell
        command gets its Ctrl-C through the terminal."""
        op = self.op
        if op is None: return False
        if op.kind == 'cell': self._spawn(self.k.interrupt(), 'interrupt')
        elif op.kind in ('turn', 'run'):
            op.cancelled_by_user = True
            op.task.cancel()
            if op.kind == 'run' or self._tools and self._tools.inflight: self._spawn(self.k.interrupt(), 'interrupt')
        return True

    async def run_code(self, src):
        """Run `src` in the kernel as a code Message, or a note when it is one bare string. An `%ipyai` cell
        shows its output but is not recorded."""
        await self._flush_bg()
        if src.lstrip().startswith('%ipyai'):
            m = Message(src, msg_type='code', output=[])
            self.transient[m.id] = m
        elif is_note(src): m = self.dlg.mk_message(note_str(src), msg_type='note')
        else: m = self.dlg.mk_message(src, msg_type='code', output=[])
        self.ui.mark(m)
        return self._start('cell', m, lambda op: self._cell(op, src))

    async def _cell(self, op, src):
        await self.k.kc.reply(src, msg_id=f'{op.msg.id}.{rtoken_hex(4)}', on_stdin=self._stdin(op.msg))

    def _stdin(self, m):
        "A run's input handler: the ui asks, and the prompt and answer join the cell's output (a password answer never does)."
        async def ask(jmsg):
            c = jmsg['content']
            prompt, pw = c.get('prompt', ''), bool(c.get('password'))
            answer = await self.ui.ask_input(prompt, pw)
            m.add_output(mk_stream(prompt + ('' if pw else answer) + '\n'))
            self.ui.mark(m)
            return answer
        return ask

    async def run_prompt(self, text):
        "Start an AI turn for prompt `text`."
        if self.assistant is None: return self.ui.note('no assistant attached (plain-REPL mode)', 'error')
        await self._flush_bg()
        m = self.dlg.mk_message(text, msg_type='prompt')
        self.ui.mark(m, live=True)
        return self._start('turn', m, lambda op: self._turn(op, text))

    async def _turn(self, op, text):
        """Stream one turn into its prompt Message: the output holds `StreamAccum`'s formatted text while
        it streams, and `chat.full()` when it completes. The session file is saved after each tool round."""
        from fastllm.chat import StreamAccum, Refresh
        m, a, kc = op.msg, self.assistant, self.k.kc
        stream = acc = None
        try:
            i = next(j for j, x in enumerate(self.dlg.messages) if x is m)
            chat, stream = await a.start_turn(self.dlg.messages[:i + 1], kc, self.tools)
            acc = StreamAccum(chat)
            async for e in stream:
                if acc(e):
                    op.started = True
                    m.output = acc.txt
                    self.ui.mark(m, live=True)
                if isinstance(e, Refresh): self.save()   # a tool round has finished
            m.output = chat.full()
            a.last_req_use = getattr(chat, 'last_req_use', None)
        except asyncio.CancelledError:
            if not op.cancelled_by_user: raise
            self._end_turn(op, text, acc, INTERRUPTED)
        except Exception as e: self._end_turn(op, text, acc, f'*[Response failed: {type(e).__name__}: {e}]*', e)
        finally:
            if (aclose := getattr(stream, 'aclose', None)) is not None: await aclose()
            if op.msg is not None: self.ui.mark(m, live=False)
        if kc is not None and op.msg is not None:
            try: kc.xpush(**{LAST_PROMPT: text, LAST_RESPONSE: m.ai_res})
            except Exception: pass

    def _end_turn(self, op, text, acc, marker, err=None):
        """Record a turn that ended early. One that produced output keeps it, followed by `marker`. One that
        produced nothing is removed, and its text goes back to the composer."""
        m = op.msg
        if op.started:
            m.output = (acc.txt.rstrip() + '\n\n' if acc is not None and acc.txt.strip() else '') + marker
            return
        self.dlg.remove_msgs([m])
        self.ui.drop([m.id])
        op.msg = None
        self.ui.restore_input(text, 'prompt')
        if err is not None: self.ui.note(f'AI prompt failed: {type(err).__name__}: {err}', 'error')

    async def run_shell(self, cmd):
        "Run `cmd` in the persistent shell, on the real terminal, recorded as a `!cmd` code Message."
        await self._stop_drain()
        m = self.dlg.mk_message(f'!{cmd}', msg_type='code', output=[])
        self.ui.mark(m)
        return self._start('shell', m, lambda op: self._shell(op, cmd))

    async def _shell(self, op, cmd):
        m = op.msg
        try: await self._ensure_shell()
        except RuntimeError as e:
            self.dlg.remove_msgs([m])
            self.ui.drop([m.id])
            op.msg = None
            return self.ui.note(str(e), 'error')
        await self._stop_drain()
        sh = self.shell
        async with self.ui.borrow(on_resize=self._resize_fg) as tty:
            with pyghostty.Terminal(*self.ui.size) as mirror:
                self.fg = (sh, mirror)
                sh.write(script(cmd))
                try: res = await sh.relay(tty.write, mirror=mirror, in_fd=getattr(tty, 'fd', None))
                finally: self.fg = None
                resid = mirror.contents().rstrip()
        if resid: m.add_output(mk_stream(resid + '\n'))
        if res == 'eof':
            self.shell = None
            await sh.close()
            if sh.exit_code: m.add_output(mk_stream(f'[exit {sh.exit_code}]\n', 'stderr'))
            self.ui.mark(m, glass=True)
            why = f'shell connection lost: {sh.error!r}' if sh.error else 'shell exited'
            return self.ui.note(f'{why}; a fresh one starts on the next shell command')
        _, ec, pwd = res
        if ec: m.add_output(mk_stream(f'[exit {ec}]\n', 'stderr'))
        self.ui.mark(m, glass=True)
        if pwd != self.shell_pwd:
            self.shell_pwd = pwd
            await self._sync_kernel_cwd(pwd)
        self._start_drain()

    def _resize_fg(self):
        "SIGWINCH during a shell command: the shell and its mirror follow the terminal."
        if self.fg is None: return
        sh, mirror = self.fg
        sh.resize(*self.ui.size)
        mirror.resize(*self.ui.size)

    def resize_shell(self):
        "The terminal resized between commands: the idle shell follows, so its children see the new size."
        if self.shell is not None and self.fg is None: self.shell.resize(*self.ui.size)

    async def _kernel_cwd(self):
        try: return await self.k.kc.eval("__import__('os').getcwd()", call_=False)
        except Exception: return os.getcwd()

    async def _sync_kernel_cwd(self, pwd):
        "cwd flows one way, shell -> kernel, after each shell command, best effort."
        try: await self.k.exec(f"import os; os.chdir({pwd!r})", timeout=5)
        except Exception: pass

    async def _ensure_shell(self):
        """The persistent shell, started on first use (or after the last one ended) in the kernel's cwd, so
        `cd`, exports, aliases and the jobs table persist and belong to the shell."""
        if self.shell is not None and not self.shell.dead: return
        if self.shell is not None:
            await self._stop_drain()
            await self.shell.close()
            self.shell = None
        try: self.shell = await GateShell(self.k.url, size=self.ui.size, cwd=await self._kernel_cwd()).start()
        except Exception as e:
            self.shell = None
            raise RuntimeError(f'shell failed to start: {e}')
        self.shell_pwd = None
        with pyghostty.Terminal(*self.ui.size) as boot: res = await self.shell.relay(mirror=boot)
        if res == 'eof':
            await self.shell.close()
            self.shell = None
            raise RuntimeError('shell failed to start')
        self.shell_pwd = res[2]
        self._start_drain()

    def _start_drain(self):
        "Between commands, a drain task reads the shell's output (background jobs, `[1] Done` notices) into a mirror."
        self._drain_mirror = pyghostty.Terminal(*self.ui.size)
        self._drain = asyncio.create_task(self.shell.relay(mirror=self._drain_mirror), name='shell-drain')

    async def _stop_drain(self):
        "Take the shell back from the drain. What it read is recorded as a `!# background output` Message."
        d, mirror = self._drain, self._drain_mirror
        self._drain = self._drain_mirror = None
        if d is None: return
        d.cancel()
        await asyncio.gather(d, return_exceptions=True)
        left = mirror.contents().rstrip()
        mirror.close()
        if left:
            m = self.dlg.mk_message('!# background output', msg_type='code', output=[])
            m.add_output(mk_stream(left + '\n'))
            self.ui.mark(m)

    async def _flush_bg(self):
        "Before an operation starts, record the background shell output so far, and keep draining."
        if self._drain is None: return
        await self._stop_drain()
        if self.shell is not None and not self.shell.dead: self._start_drain()

    # -- edits -----------------------------------------------------------------------
    def hide(self, m):
        "Hide `m` from the AI, or show it again."
        m.skipped = 0 if m.skipped else 1
        self.ui.mark(m)
        self.save()

    def edit(self, m, content=None, reply=None):
        "Replace `m`'s content, or a prompt's reply. Nothing runs again."
        if reply is not None: m.output = reply
        else: m.content = content
        self.ui.mark(m)
        self.save()

    def truncate(self, m):
        "Remove `m` and every Message after it, as before a retry. Kernel state is untouched."
        i = next((j for j, x in enumerate(self.dlg.messages) if x is m), None)
        if i is None: return
        gone = self.dlg.messages[i:]
        self.dlg.remove_msgs(gone)
        self.ui.drop([x.id for x in gone])
        self.save()

    # -- sessions --------------------------------------------------------------------
    def adopt(self, dlg, path=None, show=True):
        """Make `dlg` the session Dialog, saved at `path` (else its own `path_`, else a new session file),
        and show it unless `show` is False."""
        dlg.path_ = path or getattr(dlg, 'path_', None) or new_session_path()
        self.dlg = dlg
        self.ui.reset(dlg, show=show)
        self.stamp()
        self.save()

    def stamp(self):
        """Name the session file in the kernel's environment (`LLMDOJO_HOST_ID`), so kernel-side tooling that
        keys state to the conversation (llmdojo's doc-state) keys it to this session."""
        if self.k.kc is None: return
        try: self.k.kc.xenv(LLMDOJO_HOST_ID=Path(self.dlg.path_).stem)
        except Exception: pass

    def resume(self, path):
        """Continue the session saved at `path`: its Dialog becomes the session's, shown in full. Nothing runs.
        When the session continues on a kernel other than the one stamped in its file, the AI is told so."""
        dlg = read_ipynb(path)
        if dlg is None: return self.ui.note(f'{path}: cannot read session', 'error')
        other = dlg.meta.get('ipyai', {}).get('kernel_id') != self.k.kid
        self.adopt(dlg, path=path)
        if other: self.note_new_kernel()

    def note_new_kernel(self):
        """Tell the AI that the kernel is new, with a note Message, when code has run since the last such note.
        The earlier outputs describe variables this kernel does not have."""
        for m in reversed(self.dlg.messages):
            if m.msg_type == 'note' and m.content == NEW_KERNEL: return
            if m.msg_type == 'code': break
        else: return
        self.ui.mark(self.dlg.mk_message(NEW_KERNEL, msg_type='note'))
        self.save()

    def reset(self):
        "Start a fresh conversation in a new session file. Kernel state is untouched."
        self.adopt(Dialog(name=os.path.basename(os.getcwd())))

    def load(self, path):
        """Import dialog `path` as the conversation, in this session's file, without displaying it, and run its
        code cells again silently to set up the kernel. `%ipyai` cells are dropped. Returns the ack text."""
        dlg = read_ipynb(path)
        if dlg is None: raise ValueError(f'cannot read dialog: {path}')
        dlg.messages = [m for m in dlg.messages if not (m.msg_type == 'code' and m.content.lstrip().startswith('%ipyai'))]
        dlg.name = os.path.basename(os.getcwd())
        codes = [m.content for m in dlg.messages if m.msg_type == 'code']
        self.adopt(dlg, path=self.dlg.path_, show=False)
        self._pending_load = codes
        if self.op is None: self.run_loaded()
        return f'loaded {len(dlg)} messages from {path}; running {len(codes)} code cells'

    def run_loaded(self):
        "Start the queued reruns of loaded cells."
        codes, self._pending_load = self._pending_load, None
        return self._start('run', None, lambda op: self._rerun(codes))

    async def _rerun(self, codes):
        """Run loaded cells under request ids that are no Message's, so the AI keeps the file's stored outputs.
        Only errors show, as notes."""
        for code in codes:
            async for jm in self.k.kc.run(code, msg_id=f'load{rtoken_hex(4)}.{rtoken_hex(4)}'):
                if jm['msg_type'] == 'error': self.ui.note('\n'.join(jm['content'].get('traceback', [])), 'error')

    def save(self):
        """Write the Dialog to its session file, with the kernel id, model and think level in its metadata.
        A new session with no messages writes nothing."""
        d = self.dlg
        if not d.messages and not Path(d.path_).exists(): return
        meta = dict(kernel_id=self.k.kid) if self.k.kid else {}
        if self.assistant is not None: meta |= dict(model=self.assistant.model, think=self.assistant.think)
        d.meta = {**d.meta, 'ipyai': {**d.meta.get('ipyai', {}), **meta}}
        save_session(d)

    async def close(self):
        "At quit: record background shell output, save, and delete the shell with its jobs."
        await self._stop_drain()
        self.save()
        if self.shell is not None:
            await self.shell.close()
            self.shell = None

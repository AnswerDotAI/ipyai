"""The bare-`!` shell as a gateway terminal: spawn via the terminals API, per-command relay with sentinel boundaries.

One interactive shell (bash or zsh) runs as a rustygate-hosted pty (ptymini), created fresh per ipyai
session and deleted on exit, so it lives beside the kernel on the same machine and filesystem. The
user's rc sources first, then the prompts empty (the app's composer is the prompt), and each prompt emits a
private OSC sentinel carrying `$?` and `$PWD`: `ESC ] 7770;<exit code>;<pwd> BEL`. `relay` shuttles bytes
between the app's terminal and the gateway websocket until that sentinel arrives (the command boundary),
and strips it from the sinks. Job control is the shell's own: `fg`/`bg`/`jobs`/ctrl-Z are builtins.

A `Framer` owned by the shell splits the byte stream at sentinels. Its state and the framed items not yet
consumed live on the shell, not in one relay, so a sentinel split across websocket frames is still found,
bytes after a sentinel go to the next relay, and cancelling a relay (the between-command drain is one) loses
nothing. A permanent pump task moves incoming frames into a queue, and a writer task serializes outgoing
bytes. A `gap` control frame (this client fell behind the gateway's replay ring) becomes a visible note in
the sinks: the sentinel may have been in the lost bytes, so the note says to press Enter if the prompt seems
stuck (at a prompt, Enter emits a fresh sentinel)."""
import asyncio, os, uuid
from collections import deque
from contextlib import suppress
from jupyasyncclient import JupyAsyncTerminalClient

__all__ = ['GateShell', 'Framer', 'script', 'SHELL_RC', 'ZSH_RC']

PREFIX = b'\x1b]7770;'

SHELL_RC = r"""[ -f ~/.bashrc ] && . ~/.bashrc
PS1=''
PS2=''
PROMPT_COMMAND='__tp_ec=$?; __tp_pc=1; stty -echo; printf "\033]7770;%s;%s\a" "$__tp_ec" "$PWD"; unset __tp_pc'
trap '[ -z "$__tp_pc" ] && stty echo' DEBUG
"""
# $? is captured before stty clobbers it; the __tp_pc guard keeps the DEBUG trap (which re-enables
# echo for the *user's* commands and their children) from re-echoing PROMPT_COMMAND's own steps,
# so the sentinel is emitted only after echo is off: the next written command never echoes.

ZSH_RC = r"""export ZDOTDIR="$HOME"
[ -f "$HOME/.zshrc" ] && . "$HOME/.zshrc"
precmd_functions=(); preexec_functions=()   # prompt frameworks' hooks would fight ours; aliases/functions/PATH survive
precmd() { local ec=$?; stty -echo; printf '\033]7770;%s;%s\a' "$ec" "$PWD"; }
preexec() { stty echo; }                    # once per submitted command line, before it runs: children see normal echo
unsetopt zle
PROMPT=''; PROMPT2=''; RPROMPT=''
"""

GAP_NOTE = b'\r\n[ipyai: terminal output gap: %d bytes lost; if the prompt seems stuck, press Enter]\r\n'

def script(cmd):
    "The bytes that submit `cmd`: a multi-line script is one `{ }` group, so the shell prompts once, after all of it."
    cmd = cmd.rstrip('\n')
    return (f'{{\n{cmd}\n}}' if '\n' in cmd else cmd).encode() + b'\n'

class Framer:
    """Splits a shell's output at prompt sentinels. `feed` returns, in stream order, output byte chunks
    and `(exit_code, pwd)` tuples. Bytes that could still begin a sentinel are held until the next
    `feed` decides them, and a sentinel whose prefix has arrived is held until its closing BEL."""
    def __init__(self): self.buf = b''

    def feed(self, data):
        buf, out = self.buf + data, []
        while True:
            i = buf.find(PREFIX)
            if i < 0:
                keep = next((n for n in range(len(PREFIX) - 1, 0, -1) if buf.endswith(PREFIX[:n])), 0)
                if len(buf) > keep: out.append(buf[:len(buf) - keep])
                self.buf = buf[len(buf) - keep:]
                return out
            if i: out.append(buf[:i])
            j = buf.find(b'\x07', i + len(PREFIX))
            if j < 0:
                self.buf = buf[i:]
                return out
            ec, _, pwd = buf[i + len(PREFIX):j].partition(b';')
            try: code = int(ec)
            except ValueError: code = 0
            out.append((code, pwd.decode(errors='replace')))
            buf = buf[j + 1:]

    def flush(self):
        "The held bytes, which will never complete a sentinel (the stream ended), clearing the buffer."
        b, self.buf = self.buf, b''
        return b

class GateShell:
    """The persistent shell as an owned gateway terminal: fresh per session, deleted on `close`.
    `write`/`resize` are sync (queued). `exit_code` is the shell's own exit status once it has ended,
    and `error` the transport failure that ended it, if any."""
    def __init__(self, url, size=None, cwd=None, sh=None):
        self.url, self.size, self.cwd, self.sh = url, size, cwd, sh
        self.tc, self.dead, self.exit_code, self.error = None, False, None, None
        self.framer, self._ready = Framer(), deque()
        self._in = asyncio.Queue()
        self._out = asyncio.Queue()
        self._pump = self._writer = None

    async def start(self):
        """Create the terminal (bash `--rcfile {rcfile}` or zsh `ZDOTDIR={rcdir}`, per `$SHELL`; the
        gateway writes the rc text and substitutes the paths) and attach its ws channel. A terminal
        created but never connected is deleted again before the error propagates."""
        sh = self.sh or os.environ.get('SHELL', 'bash')
        appendenv = {k: os.environ[k] for k in ('TERM', 'COLORTERM') if k in os.environ}
        if os.path.basename(sh) == 'zsh': kw = dict(argv=[sh, '-i'], rc=ZSH_RC, appendenv=dict(appendenv, ZDOTDIR='{rcdir}'))
        else: kw = dict(argv=['bash', '--noediting', '--rcfile', '{rcfile}', '-i'], rc=SHELL_RC, appendenv=appendenv)
        if self.cwd: kw['cwd'] = self.cwd
        if self.size: kw['cols'], kw['rows'] = self.size
        self.tc = JupyAsyncTerminalClient(self.url)
        await self.tc.start_terminal(name=f'ipyai-{uuid.uuid4().hex[:8]}', **kw)
        try: await self.tc.connect()
        except BaseException:
            with suppress(Exception): await self.tc.shutdown_terminal()
            raise
        self._pump = asyncio.create_task(self._pump_loop(), name='shell-pump')
        self._writer = asyncio.create_task(self._writer_loop(), name='shell-writer')
        return self

    async def _pump_loop(self):
        """Move every incoming ws frame into the relay queue. The None sentinel lands in `finally`,
        so a transport failure ends the stream the same way a clean eof does: `relay` returns 'eof',
        and `self.error` says why."""
        try:
            async for item in self.tc.frames():
                if isinstance(item, dict) and item.get('type') == 'eof': self.exit_code = item.get('code')
                self._in.put_nowait(item)
        except Exception as e: self.error = e
        finally: self._in.put_nowait(None)

    async def _writer_loop(self):
        """One writer serializes outgoing frames, so keystrokes and resizes keep their order.
        A failed write means the shell is unusable: record why, mark dead via the same sentinel."""
        try:
            while True:
                kind, *a = await self._out.get()
                if kind == 'data': await self.tc.write(a[0])
                else: await self.tc.resize(*a)
        except Exception as e:
            self.error = e
            self._in.put_nowait(None)

    def write(self, data): self._out.put_nowait(('data', data))

    def resize(self, cols, rows):
        "Propagate a new terminal size."
        self._out.put_nowait(('size', rows, cols))

    async def relay(self, write=None, mirror=None, in_fd=None):
        """Shuttle bytes between the app's terminal and the shell until the shell prints its prompt:
        returns ('prompt', exit_code, pwd), or 'eof' if the shell itself ended. Output goes through
        `write` (None streams nothing to the screen) and tees into `mirror`; `in_fd` (the real tty's
        fd, raw mode) relays keystrokes in. The sentinel never reaches the sinks. A `gap` frame becomes
        a visible note in the sinks."""
        if self.dead: return 'eof'
        loop = asyncio.get_running_loop()
        def _sink(data):
            if not data: return
            if mirror is not None: mirror.feed(data)
            if write is not None: write(data)
        def on_stdin():
            data = os.read(in_fd, 4096)
            if data: self.write(data)
        if in_fd is not None: loop.add_reader(in_fd, on_stdin)
        try:
            while True:
                while self._ready:
                    item = self._ready.popleft()
                    if isinstance(item, tuple): return ('prompt', *item)
                    _sink(item)
                item = await self._in.get()
                if item is None or isinstance(item, dict) and item.get('type') == 'eof':
                    self.dead = True
                    _sink(self.framer.flush())
                    return 'eof'
                if isinstance(item, dict):
                    if item.get('type') == 'gap': _sink(GAP_NOTE % item.get('bytes', 0))
                    continue
                self._ready.extend(self.framer.feed(item))
        finally:
            if in_fd is not None: loop.remove_reader(in_fd)

    async def close(self):
        "Cancel the tasks, delete the owned terminal (the gateway's terminate ladder ends its jobs), close up."
        for t in (self._pump, self._writer):
            if t is not None: t.cancel()
        if self.tc is not None:
            with suppress(Exception): await self.tc.shutdown_terminal()  # a dead pty still needs its registry entry deleted
            await self.tc.aclose()

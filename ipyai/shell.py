"""The bare-`!` shell as a gateway terminal: spawn via the terminals API, per-command relay with sentinel boundaries.

One interactive shell (bash or zsh) runs as a jupygate-hosted pty (ptymini), created fresh per ipyai
session and deleted on exit -- so it lives beside the kernel: same machine, container, and filesystem,
which is what makes the shell<->kernel cwd sync coherent. The prompt-integration choreography is
teleprint's original, and it is entirely in-band, so it survives the transport change: the user's rc
sources first, then the prompt empties (the app's composer is the prompt), and each prompt emits a
private OSC sentinel carrying `$?` and `$PWD`. `relay` shuttles bytes between the app's terminal and
the gateway websocket until that sentinel arrives -- the command boundary -- stripping it from the
sinks. Job control is the shell's own: `fg`/`bg`/`jobs`/ctrl-Z are builtins.

A permanent pump task moves incoming ws frames into a queue, so cancelling a relay (the between-borrow
drain is one) never cancels the websocket iterator itself; a writer task serializes outgoing bytes. A
`gap` control frame (this client fell behind the gateway's replay ring) becomes a visible note in the
sinks rather than a hang: the sentinel may have been in the lost bytes, so the note tells the user to
press Enter if the prompt seems stuck (at a prompt, Enter harmlessly emits a fresh sentinel)."""
import asyncio, os, re, uuid
from contextlib import suppress
from jupyasyncclient import JupyAsyncTerminalClient

__all__ = ['GateShell', 'SENTINEL', 'SHELL_RC', 'ZSH_RC']

SENTINEL = re.compile(rb'\x1b\]7770;([^;\x07]*);([^\x07]*)\x07')

SHELL_RC = r'''[ -f ~/.bashrc ] && . ~/.bashrc
PS1=''
PROMPT_COMMAND='__tp_ec=$?; __tp_pc=1; stty -echo; printf "\033]7770;%s;%s\a" "$__tp_ec" "$PWD"; unset __tp_pc'
trap '[ -z "$__tp_pc" ] && stty echo' DEBUG
'''
# $? is captured before stty clobbers it; the __tp_pc guard keeps the DEBUG trap (which re-enables
# echo for the *user's* commands and their children) from re-echoing PROMPT_COMMAND's own steps,
# so the sentinel is emitted only after echo is off: the next written command never echoes.

ZSH_RC = r'''export ZDOTDIR="$HOME"
[ -f "$HOME/.zshrc" ] && . "$HOME/.zshrc"
precmd_functions=(); preexec_functions=()   # prompt frameworks' hooks would fight ours; aliases/functions/PATH survive
precmd() { local ec=$?; stty -echo; printf '\033]7770;%s;%s\a' "$ec" "$PWD"; }
preexec() { stty echo; }                    # once per submitted command line, before it runs: children see normal echo
unsetopt zle
PROMPT=''; RPROMPT=''
'''

GAP_NOTE = b'\r\n[ipyai: terminal output gap: %d bytes lost; if the prompt seems stuck, press Enter]\r\n'


class GateShell:
    """The persistent shell as an owned gateway terminal: fresh per session, deleted on `close`.
    `write`/`resize` are sync (queued); `relay` mirrors teleprint's old `relay_shell` contract."""
    def __init__(self, url, size=None, cwd=None, sh=None):
        self.url, self.size, self.cwd, self.sh = url, size, cwd, sh
        self.tc, self.dead, self.exit_code, self.error = None, False, None, None
        self._in = asyncio.Queue()
        self._out = asyncio.Queue()
        self._pump = self._writer = None

    async def start(self):
        """Create the terminal (bash `--rcfile {rcfile}` or zsh `ZDOTDIR={rcdir}`, per `$SHELL`; the
        gateway writes the rc text and substitutes the paths) and attach its ws channel."""
        sh = self.sh or os.environ.get('SHELL', 'bash')
        appendenv = {k: os.environ[k] for k in ('TERM', 'COLORTERM') if k in os.environ}
        if os.path.basename(sh) == 'zsh': kw = dict(argv=[sh, '-i'], rc=ZSH_RC, appendenv=dict(appendenv, ZDOTDIR='{rcdir}'))
        else: kw = dict(argv=['bash', '--noediting', '--rcfile', '{rcfile}', '-i'], rc=SHELL_RC, appendenv=appendenv)
        if self.cwd: kw['cwd'] = self.cwd
        if self.size: kw['cols'], kw['rows'] = self.size
        self.tc = JupyAsyncTerminalClient(self.url)
        await self.tc.start_terminal(name=f'ipyai-{uuid.uuid4().hex[:8]}', **kw)
        await self.tc.connect()
        self._pump = asyncio.create_task(self._pump_loop(), name='shell-pump')
        self._writer = asyncio.create_task(self._writer_loop(), name='shell-writer')
        return self

    async def _pump_loop(self):
        """Move every incoming ws frame into the relay queue. The None sentinel lands in `finally`,
        so a transport failure ends the stream the same way a clean eof does: `relay` returns 'eof',
        the app shows its shell-exited note (with `self.error`'s reason), and the next shell use
        respawns. Swallow-after-record: the failure is surfaced through that path, not re-raised."""
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
        "Propagate a new terminal size (same signature as the old local `Job.resize`)."
        self._out.put_nowait(('size', rows, cols))

    async def relay(self, write=None, mirror=None, in_fd=None):
        """Shuttle bytes between the app's terminal and the shell until the shell prints its prompt:
        returns ('prompt', exit_code, pwd) -- or 'eof' if the shell itself died. Output goes through
        `write` (None streams nothing to the screen) and tees into `mirror`; `in_fd` (the real tty's
        fd, raw mode) relays keystrokes in. The sentinel is stripped from the sinks (it is boundary
        metadata, not output); a partial escape tail is held back across frames so a sentinel split
        over two chunks cannot leak or be missed. A `gap` frame becomes a visible note in the sinks."""
        if self.dead: return 'eof'
        loop = asyncio.get_running_loop()
        buf = b''
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
                item = await self._in.get()
                if item is None or isinstance(item, dict) and item.get('type') == 'eof':
                    self.dead = True
                    _sink(buf)
                    return 'eof'
                if isinstance(item, dict):
                    if item.get('type') == 'gap': _sink(GAP_NOTE % item.get('bytes', 0))
                    continue
                buf += item
                m = SENTINEL.search(buf)
                if m:
                    _sink(buf[:m.start()])
                    try: code = int(m.group(1))
                    except ValueError: code = 0
                    return ('prompt', code, m.group(2).decode(errors='replace'))
                esc = buf.rfind(b'\x1b')
                keep = esc if esc != -1 and len(buf) - esc < 64 else len(buf)  # hold back a possible partial sentinel
                _sink(buf[:keep])
                buf = buf[keep:]
        finally:
            if in_fd is not None: loop.remove_reader(in_fd)

    async def close(self):
        "Cancel the tasks, delete the owned terminal (the gateway's terminate ladder ends its jobs), close up."
        for t in (self._pump, self._writer):
            if t is not None: t.cancel()
        if self.tc is not None:
            with suppress(Exception): await self.tc.shutdown_terminal()  # a dead pty still needs its registry entry deleted
            await self.tc.aclose()

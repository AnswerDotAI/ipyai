"The shell layer: three-mode routing, and the persistent shell running `!` submissions under borrows."
import asyncio, os
from teleprint.testing import EmuTty
from ipyai.assistant import route
from ipyai.cli import App

def test_route_modes():
    "Prefix overrides from any other mode; home interpretation otherwise; embedded `!` stays code."
    assert route('x = 1', 'code') == ('code', 'x = 1')
    assert route('.what is x?', 'code') == ('prompt', 'what is x?')
    assert route('!ls -la', 'code') == ('job', 'ls -la')
    assert route('  !ls', 'code') == ('job', 'ls')
    assert route('x = !ls', 'code') == ('code', 'x = !ls')            # embedded: IPython SList semantics
    assert route('!ls\n!pwd', 'code') == ('job', 'ls\n!pwd')          # leading !: one shell script
    assert route('!', 'code') == ('code', '!')                        # nothing to run
    assert route('%fg 2', 'code') == ('code', '%fg 2')                # %fg died with the jobs machinery
    assert route('%matplotlib inline', 'code') == ('code', '%matplotlib inline')
    assert route('hello', 'prompt') == ('prompt', 'hello')
    assert route(';x = 1', 'prompt') == ('code', 'x = 1')
    assert route('!ls', 'prompt') == ('job', 'ls')
    assert route('%time 1', 'prompt') == ('code', '%time 1')
    assert route('.still a prompt', 'prompt') == ('prompt', '.still a prompt')
    assert route('ls -la', 'shell') == ('job', 'ls -la')
    assert route('jobs', 'shell') == ('job', 'jobs')
    assert route('!!', 'shell') == ('job', '!!')                      # history expansion passes through at home
    assert route(';x = 1', 'shell') == ('code', 'x = 1')
    assert route('.explain', 'shell') == ('prompt', 'explain')
    assert route('%ipyai model', 'shell') == ('code', '%ipyai model') # magics reach the kernel from any mode

async def _until(pred, timeout=25):
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        if pred(): return
        await asyncio.sleep(0.05)
    raise TimeoutError('condition did not settle')

def _outs(app): return [b for b in app.comp.blocks.values() if b.tag == 'out']

def test_shell_end_to_end():
    "A `!` submission runs in the persistent shell under a borrow: raw bytes on glass, residue as a model-only block, prompt back."
    async def go():
        tty = EmuTty(60, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'!echo hello from job\r')
            await _until(lambda: _outs(app))
            assert 'hello from job' in tty.term.text()      # streamed raw during the borrow
            blk = _outs(app)[0]
            assert blk.committed and 'hello from job' in str(blk.body[0])
            assert app.shell is not None and app.fg_job is None
            assert '»»»' in tty.term.text()                 # tail repainted after reanchor
    asyncio.run(go())

def test_shell_state_persists_and_cwd_syncs():
    "cd sticks across submissions (one shell), and the kernel's cwd follows the shell's."
    import tempfile
    async def go(tmp):
        tty = EmuTty(70, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(f'!cd {tmp} && export TP_T=42\r'.encode())
            await _until(lambda: app.shell_pwd == tmp)
            app.comp.on_bytes(b'!echo $TP_T in $PWD\r')
            await _until(lambda: any(f'42 in {tmp}' in (b.source or '') for b in _outs(app)))
            app.comp.on_bytes(b'import os; print(os.getcwd())\r')     # the kernel followed the shell's cd
            await _until(lambda: tmp in tty.term.contents())
    with tempfile.TemporaryDirectory() as td: asyncio.run(go(os.path.realpath(td)))

def test_embedded_bang_stays_kernel():
    "`x = !ls` runs kernel-side: an SList lands in `x`, and no shell is ever spawned."
    async def go():
        tty = EmuTty(70, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'x = !echo kernelside\r')
            await _until(lambda: not app.k.busy and not app.buf.text)
            app.comp.on_bytes(b'x\r')
            await _until(lambda: 'kernelside' in tty.term.contents())
            assert app.shell is None
    asyncio.run(go())

def test_shell_exit_code_and_respawn():
    "A failing command reports its exit; `exit` kills the shell and the next submission gets a fresh one."
    async def go():
        tty = EmuTty(60, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'!false\r')
            await _until(lambda: 'exit 1' in tty.term.contents())
            first = app.shell.pid
            app.comp.on_bytes(b'!exit\r')
            await _until(lambda: app.shell is None)
            app.comp.on_bytes(b'!echo back up\r')
            await _until(lambda: any('back up' in (b.source or '') for b in _outs(app)))
            assert app.shell.pid != first
    asyncio.run(go())

def test_bg_job_and_quit_gate():
    "`&` backgrounds inside the shell; C-D warns once about live children, an immediate second C-D quits."
    async def go():
        tty = EmuTty(60, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'!sleep 30 &\r')
            await _until(lambda: app.fg_job is None and app.shell is not None)
            await _until(lambda: app._shell_children())
            app.comp.on_bytes(b'\x04')                       # C-D: gated
            await _until(lambda: any('C-D again' in (b.source or str(b.body[:1])) for b in app.comp.blocks.values()))
            assert not app.done.is_set()
            app.comp.on_bytes(b'\x04')                       # immediate second: quits, SIGHUPs the shell
            await _until(lambda: app.done.is_set())
    asyncio.run(go())

def test_stop_and_resume():
    "ctrl-Z stops the child (the shell's prompt is the boundary); `!fg` resumes it; ctrl-C ends it."
    async def go():
        tty = EmuTty(60, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'!sleep 30\r')
            await _until(lambda: app.fg_job is not None)
            os.write(app.shell.master_fd, b'\x1a')           # ^Z via the pty line discipline
            await _until(lambda: app.fg_job is None)         # the stop bounced us back to the prompt
            app.comp.on_bytes(b'!fg\r')
            await _until(lambda: app.fg_job is not None)
            os.write(app.shell.master_fd, b'\x03')           # ^C the resumed child
            await _until(lambda: app.fg_job is None)
    asyncio.run(go())

def test_f2_editor_roundtrip(monkeypatch, tmp_path):
    "F2 hands the composer to $EDITOR through the shell borrow, reloads on clean exit, records nothing."
    ed = tmp_path/'ed.sh'
    ed.write_text('#!/bin/sh\nprintf "x = 99" > "$1"\n')
    ed.chmod(0o755)
    monkeypatch.setenv('EDITOR', str(ed))
    async def go():
        tty = EmuTty(60, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.buf.insert('draft')
            await app.edit_buffer()
            assert app.buf.text == 'x = 99'                    # the editor's result landed in the composer
            assert app.buf.cursor == len('x = 99')
            assert not [b for b in app.comp.blocks.values() if b.tag in ('sh', 'out')]  # no transcript record
    asyncio.run(go())

def test_f2_abandon_on_nonzero_exit(monkeypatch, tmp_path):
    "A nonzero editor exit (vim's :cq) leaves the composer untouched."
    ed = tmp_path/'ed.sh'
    ed.write_text('#!/bin/sh\nprintf "junk" > "$1"; exit 1\n')
    ed.chmod(0o755)
    monkeypatch.setenv('EDITOR', str(ed))
    async def go():
        tty = EmuTty(60, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.buf.insert('keep me')
            await app.edit_buffer()
            assert app.buf.text == 'keep me'
    asyncio.run(go())

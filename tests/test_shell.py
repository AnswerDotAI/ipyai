import os
import pytest
from teleprint.testing import EmuTty
from ipyai.cli import App
from ipyai.shell import Framer
from .helpers import send, until

@pytest.mark.parametrize('shell', ['bash', 'zsh'])
async def test_persistent_shell_lifecycle(repl, monkeypatch, tmp_path, shell):
    app = repl
    monkeypatch.setenv('SHELL', shell)
    cwd = os.path.realpath(tmp_path/('d' * 100)/('e' * 100))
    os.makedirs(cwd)
    await send(app, f'!cd {cwd}\nexport TP_T=42\necho ready')
    sh, name = app.ctl.shell, app.ctl.shell.tc.name
    await send(app, '!echo "$TP_T in $PWD"')
    assert f'42 in {cwd}' in app.ctl.dlg.messages[-1].ai_output
    assert await app.k.kc.eval("__import__('os').getcwd()", call_=False) == cwd
    await send(app, 'x = !echo kernelside')
    assert app.ctl.shell is sh and await app.k.kc.retr('x') == ['kernelside']
    app.tty.term.resize(97, 41)
    app._resized()
    await send(app, '!stty size')
    assert '41 97' in app.ctl.dlg.messages[-1].ai_output
    await send(app, r"!printf '\033[?1049hSECRET DRAWING\033[?1049l'; echo visible after")
    assert 'visible after' in app.ctl.dlg.messages[-1].ai_output and 'SECRET' not in app.ctl.dlg.messages[-1].ai_output
    await send(app, '!seq 1 20000')
    assert app.ctl.dlg.messages[-1].ai_output.rstrip().endswith('20000')

    app.comp.on_bytes(b"!sh -c 'echo go; exec sleep 30'\r")
    await until(lambda: app.ctl.fg is not None and 'go' in app.ctl.fg[1].contents())
    sh.write(b'\x1a')
    await until(lambda: not app.busy)
    await send(app, '!jobs')
    assert 'sleep 30' in app.ctl.dlg.messages[-1].ai_output
    app.comp.on_bytes(b'!fg\r')
    await until(lambda: app.ctl.fg is not None and 'sleep' in app.ctl.fg[1].contents())
    sh.write(b'\x03')
    await until(lambda: not app.busy)
    await send(app, '!false')
    assert 'exit 1' in app.ctl.dlg.messages[-1].ai_output
    await send(app, '!exit')
    assert app.ctl.shell is None
    assert name not in [t['name'] for t in await sh.tc.list_terminals()]
    await send(app, '!sleep 30 &')
    sh = app.ctl.shell
    assert sh.tc.name != name
    app.comp.on_bytes(b'\x04')
    assert not app.done.is_set()
    app.comp.on_bytes(b'\x04')
    await until(app.done.is_set)
    await app.ctl.close()
    assert sh.tc.name not in [t['name'] for t in await sh.tc.list_terminals()]

def test_sentinel_framing():
    pwd = '/tmp/' + 'd' * 150
    stream = b'out\x1b[31mred\x1b]7770;2;' + pwd.encode() + b'\x07after'
    for size in (1, 3, 7, len(stream)):
        f, got = Framer(), []
        for i in range(0, len(stream), size): got += f.feed(stream[i:i + size])
        assert b''.join(x for x in got if isinstance(x, bytes)) == b'out\x1b[31mredafter'
        assert [x for x in got if isinstance(x, tuple)] == [(2, pwd)]
    f = Framer()
    assert f.feed(b'tail\x1b]77') == [b'tail'] and f.flush() == b'\x1b]77'

async def test_editor_handoff(monkeypatch, tmp_path):
    editor = tmp_path/'editor.sh'
    monkeypatch.setenv('EDITOR', str(editor))
    with EmuTty(60, 16) as tty:
        app = App(tty, history=None)
        app.paint()
        for status, expected in [(0, 'x = 99'), (1, 'keep me')]:
            editor.write_text(f'#!/bin/sh\nprintf "x = 99" > "$1"\nexit {status}\n')
            editor.chmod(0o755)
            app.buf.text = 'keep me'
            await app.edit_buffer()
            assert app.buf.text == expected and not app.ctl.dlg.messages and not app.comp.blocks

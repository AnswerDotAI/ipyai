import asyncio, os, time
from teleprint.testing import EmuTty
from aidialog.ipynb import read_ipynb
from ipyai.cli import App
from ipyai.kernel import KernelSession
from ipyai.history import History
from ipyai.session import list_sessions, resolve_session
from ipyai.controller import NEW_KERNEL
from .helpers import send, until

async def test_comm_settings_and_session_resume(conversation, gateway, tmp_path):
    app, _ = conversation
    await send(app, 'x = 42')
    await send(app, '%ipyai model sonnet')
    assert app.ctl.assistant.model == 'sonnet'
    await send(app, 'ack = %ipyai think h')
    assert app.ctl.assistant.think == 'h' and await app.k.kc.retr('ack') == 'think = h'
    await send(app, '%ipyai prompt')
    assert app.mode == 'prompt'
    await send(app, '%ipyai nonsense')
    assert 'unknown %ipyai command' in app.tty.term.contents()
    original = app.ctl.dlg.path_
    assert resolve_session(original.stem[:6]) == original and (tmp_path/'.ipyai'/'.gitignore').exists()
    saved = read_ipynb(original)
    assert saved.meta['ipyai']['model'] == 'sonnet' and saved.meta['ipyai']['kernel_id'] == app.k.kid
    await send(app, '%ipyai reset')
    assert len(app.ctl.dlg) == 0 and await app.k.kc.retr('x') == 42
    await send(app, ';x = 99')
    newer = app.ctl.dlg.path_
    os.utime(original, (time.time() - 60,) * 2)
    assert list_sessions()[0][0] == newer and History().items == ['x = 99', 'ack = %ipyai think h', 'x = 42']
    picker = asyncio.create_task(app.choose_session(''))
    await until(lambda: app.picker is not None)
    app.comp.on_bytes(b'2')
    assert await picker == original
    app.ctl.resume(original)
    assert await app.k.kc.retr('x') == 99  # resume never reruns saved cells
    async with KernelSession(url=gateway) as other:
        with EmuTty(60, 16) as tty:
            resumed = App(tty, kernel=other, history=None)
            resumed.ctl.resume(original)
            assert resumed.ctl.dlg.messages[-1].content == NEW_KERNEL
            resumed.ctl.resume(original)
            assert sum(m.content == NEW_KERNEL for m in resumed.ctl.dlg.messages) == 1

    await app.k.kc.reply('import ipyai.magic as _m; _m._TIMEOUT = 0.5', store_history=False)
    app.ctl.on_comm = None
    await send(app, '%ipyai model haiku')
    assert 'no ipyai host attached' in app.tty.term.contents() and app.ctl.assistant.model == 'sonnet'

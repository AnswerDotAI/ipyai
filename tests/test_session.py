import asyncio, os, time, pytest
from aidialog.ipynb import read_ipynb
from teleprint.testing import EmuTty
from ipyai.cli import App
from ipyai.kernel import KernelSession
from ipyai.history import History
from ipyai.session import list_sessions
from ipyai.controller import NEW_KERNEL
from ipyai.tools import KernelTools
import ipyai.config as config
from .helpers import send, until

async def test_session_resume_preserves_kernel(conversation, gateway, tmp_path):
    app, _ = conversation
    await app.k.setup()
    await send(app, 'x = 42')
    await send(app, '%ipyai model sonnet')
    await send(app, 'ack = %ipyai think h')
    assert app.ctl.assistant.think == 'h' and await app.k.kc.retr('ack') == 'think = h'
    await send(app, '%ipyai prompt')
    assert app.mode == 'prompt'
    original = app.ctl.dlg.path_
    meta = read_ipynb(original).meta['ipyai']
    assert (meta['model'], meta['think'], meta['kernel_id']) == ('sonnet', 'h', app.k.kid)
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
    attached = await KernelSession(url=gateway).start(kernel=app.k.kid[:8])
    try: assert not attached.owned and await attached.kc.retr('x') == 99
    finally: await attached.close()
    assert await app.k.mgr.is_alive(app.k.kid)
    config.STARTUP_PATH.parent.mkdir()
    config.STARTUP_PATH.write_text('startup_ran = 7\nstartup_file = __file__\n')
    other = await KernelSession(url=gateway).start(cwd=tmp_path, env=dict(IPYAI_TEST_MARK='yes'))
    try:
        await other.exec("def bash(**kw): return 'preseeded'")
        await other.setup()
        values = await other.kc.eval("(__import__('os').getcwd(), __import__('os').environ['IPYAI_TEST_MARK'], startup_ran, startup_file)", call_=False)
        assert values == (str(tmp_path), 'yes', 7, str(config.STARTUP_PATH))
        tools = KernelTools(other.kc)
        assert (await tools.call('bash')).text == 'preseeded'
        config.STARTUP_PATH.write_text('1/0\n')
        with pytest.raises(RuntimeError, match=config.STARTUP_PATH.name): await other.setup()
        with EmuTty(60, 16) as tty:
            resumed = App(tty, kernel=other, history=None)
            resumed.ctl.resume(original)
            assert resumed.ctl.dlg.messages[-1].content == NEW_KERNEL
            resumed.ctl.resume(original)
            assert sum(m.content == NEW_KERNEL for m in resumed.ctl.dlg.messages) == 1
    finally: await other.close()
    assert not await other.mgr.is_alive(other.kid)

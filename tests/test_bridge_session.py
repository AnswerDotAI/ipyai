"Integration: `KernelTools` against a live gateway ipymini, and session-file persistence."
import asyncio, os, pytest, time
from ipyai.kernel import KernelSession
from ipyai.tools import KernelTools
from ipyai.assistant import LAST_RESPONSE
from ipyai.session import new_session_path, save_session, list_sessions, resolve_session
from ipyai.history import History
from aidialog.dialog import Dialog
from aidialog.ipynb import read_ipynb
from aidialog.msg_parts import ToolResponse, InputImage
import ipyai.config as config
from teleprint.testing import EmuTty
from ipyai.cli import App
from ipyai.controller import NEW_KERNEL


async def test_py_tool_and_writeback(gateway):
    "The py tool runs as a plain cell with normal output, and sidecar xpush lands LAST_RESPONSE for the user."
    async with KernelSession(url=gateway) as k:
        await k.setup()
        tools = KernelTools(k.kc)
        assert 'py' in await tools.names()
        out = await tools.call('py', code='zz = 6*7\nprint("side effect")\nzz')
        assert '<stdout>\nside effect\n</stdout>' in out and '<execute_result>\n42\n</execute_result>' in out  # tagged, so the model can tell a print from a value
        assert await k.kc.retr('zz') == 42  # the tool ran in the USER'S namespace
        assert 'zz = 6*7\nprint("side effect")\nzz' not in await k.kc.retr('In')  # model cells stay out of the user's history
        err = await tools.call('py', code='1/0')
        assert 'ZeroDivisionError' in err and 'division by zero' in err
        img = await tools.call('py', code='from IPython.display import Image, display\nfrom aidialog.dialog import tiny_png\ndisplay(Image(data=tiny_png))')
        assert isinstance(img, ToolResponse) and any(isinstance(p, InputImage) for p in img.content)
        k.kc.xpush(**{LAST_RESPONSE: 'the reply text'})
        assert await k.kc.retr(LAST_RESPONSE) == 'the reply text'


async def test_startup_file(gateway):
    "An owned kernel runs `config.STARTUP_PATH` at setup with `__file__` bound, the way clikernel runs its startup.py."
    config.STARTUP_PATH.write_text('startup_ran = 7\nstartup_file = __file__\n')
    async with KernelSession(url=gateway) as k:
        await k.setup()
        assert await k.kc.retr('startup_ran') == 7
        assert await k.kc.retr('startup_file') == str(config.STARTUP_PATH)


async def test_startup_file_error_names_the_file(gateway):
    config.STARTUP_PATH.write_text('1/0\n')
    async with KernelSession(url=gateway) as k:
        with pytest.raises(RuntimeError, match='startup.py'): await k.setup()


async def test_start_cwd_env(gateway, tmp_path):
    "`start(cwd=, env=)` places an owned kernel: it starts in that directory with those entries over our environment."
    k = await KernelSession(url=gateway).start(cwd=tmp_path, env=dict(IPYAI_TEST_MARK='yes'))
    try:
        await k.exec('import os')
        assert await k.kc.retr('os.getcwd()') == str(tmp_path)
        assert await k.kc.retr('os.environ["IPYAI_TEST_MARK"]') == 'yes'
    finally: await k.close()


async def test_py_concurrent_calls(gateway):
    "Parallel `py` calls from one model turn each get their own result, and one call's error does not abort the ones queued behind it."
    async with KernelSession(url=gateway) as k:
        tools = KernelTools(k.kc)
        calls = (tools.call('py', code=c) for c in ['y = 6*7', '1/0', 'y', '1+1'])
        res = await asyncio.wait_for(asyncio.gather(*calls), 20)
        assert 'ZeroDivisionError' in res[1] and [res[0], *res[2:]] == ['', '42', '2']


def _saved(msgs, **meta):
    "A session file holding `msgs` (msg_type, content, output) triples, in the current directory."
    d = Dialog(name='t')
    for t, c, o in msgs: d.mk_message(c, msg_type=t, output=o)
    d.path_ = new_session_path()
    if meta: d.meta = {'ipyai': meta}
    save_session(d)
    return d.path_


def test_session_files(tmp_path):
    "Session round-trip: saving creates the self-ignored dir, listing and prefix resolution find it, meta rides along."
    p = _saved([('code', 'x = 1', ''), ('prompt', 'why?', 'Because.')], kernel_id='k1', model='m')
    assert (tmp_path/'.ipyai'/'.gitignore').exists()
    rows = list_sessions()
    assert len(rows) == 1 and rows[0][2] == 1 and rows[0][3] == 'why?'
    assert resolve_session(p.stem[:6]) == p
    with pytest.raises(FileNotFoundError): resolve_session('nope-nothing')
    d2 = read_ipynb(p)
    assert d2.meta['ipyai'] == dict(kernel_id='k1', model='m')
    assert [m.msg_type for m in d2.messages] == ['code', 'prompt']
    assert d2.messages[1].ai_res == 'Because.'


async def test_startup_picker():
    "Bare -r: no session starts fresh and one is chosen without asking; with several, the picker owns keys: a digit picks a row, Enter the newest, n starts fresh."
    tty = EmuTty(70, 14)
    app = App(tty, history=None)
    assert await app.choose_session('') is None
    older = _saved([('prompt', 'older one', '')])
    os.utime(older, (time.time() - 60,) * 2)
    assert await app.choose_session('') == older
    newer = _saved([('prompt', 'first prompt', '')])
    async def choose(keys):
        t = asyncio.ensure_future(app.choose_session(''))
        await asyncio.sleep(0)                    # the picker opens
        scr = tty.term.text()
        app.comp.on_bytes(keys)
        return scr, await t
    scr, got = await choose(b'2')                 # digit picks row 2
    assert 'resume a session in this directory' in scr and newer.stem[:8] in scr and 'older one' in scr
    assert got == older and app.picker is None
    assert (await choose(b'\r'))[1] == newer     # Enter: newest
    assert (await choose(b'n'))[1] is None       # n: fresh
    assert 'resume a session' not in tty.term.text()  # the transient evaporated


def test_resume_notes_a_new_kernel():
    "Resuming on a kernel other than the stamped one tells the AI the namespace is empty, once for each run of code."
    app = App(EmuTty(60, 12), history=None)
    p = _saved([('code', 'x = 1', '')], kernel_id='gone')
    app.ctl.resume(p)
    msgs = app.ctl.dlg.messages
    assert [m.msg_type for m in msgs] == ['code', 'note'] and msgs[-1].content == NEW_KERNEL
    assert 'namespace is empty' in app.tty.term.text()
    app.ctl.resume(p)                              # no code has run since the note: no second note
    assert [m.msg_type for m in app.ctl.dlg.messages] == ['code', 'note']
    app.ctl.resume(_saved([('code', 'y = 2', '')]))  # stamped with this same (absent) kernel: no note
    assert [m.msg_type for m in app.ctl.dlg.messages] == ['code']

def test_history_modes():
    "History mines session files, each composer mode scoped to its own message shape."
    _saved([('code', 'x = 1', ''), ('code', '!ls -la', ''), ('prompt', 'why?', 'B.'), ('note', 'a note', '')])
    assert History().items == ['x = 1']
    assert History(mode='shell').items == ['ls -la']
    assert History(mode='prompt').items == ['why?']
    assert History().suggest('x') == ' = 1'

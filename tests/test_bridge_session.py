"Integration: the KernelBridge against a live gateway ipymini, and session-file persistence."
import asyncio, pytest
from ipyai.kernel import KernelSession
from ipyai.bridge import setup_tools
from ipyai.assistant import LAST_RESPONSE
from ipyai.session import Session, list_sessions, resolve_session
from ipyai.history import History
from aidialog.dialog import Dialog
from aidialog.ipynb import read_ipynb
from aidialog.msg_parts import ToolResponse, InputImage
import ipyai.config as config
from ipyai.kernel_bridge import KernelBridge


async def test_bridge_and_writeback(gateway):
    "The py tool runs the model's code as a plain cell, so its outputs come back as the user would see them: text, tracebacks, and images; set_vars lands LAST_RESPONSE for the user."
    async with KernelSession(url=gateway) as k:
        bridge, tools = await setup_tools(k)
        names = await tools.names()
        assert 'py' in names
        out = await tools.call_text('py', {'code': 'zz = 6*7\nprint("side effect")\nzz'})
        assert '<stdout>\nside effect\n</stdout>' in out and '<execute_result>\n42\n</execute_result>' in out  # tagged, so the model can tell a print from a value
        assert await bridge.read_var('zz') == 42  # the tool ran in the USER'S namespace
        assert 'zz = 6*7\nprint("side effect")\nzz' not in await bridge.read_var('In')  # model cells stay out of the user's history (and that is how kernel-side rules tell them apart)
        err = await tools.call_text('py', {'code': '1/0'})
        assert 'ZeroDivisionError' in err and 'division by zero' in err
        img = await tools.call_text('py', {'code': 'from IPython.display import Image, display\nfrom aidialog.dialog import tiny_png\ndisplay(Image(data=tiny_png))'})
        assert isinstance(img, ToolResponse) and any(isinstance(p, InputImage) for p in img.content)
        bridge.set_vars(**{LAST_RESPONSE: 'the reply text'})
        assert await bridge.read_var(LAST_RESPONSE) == 'the reply text'
        # a normal cell still renders normally after bridge traffic (no iopub leakage)
        outs = []
        await k.run_cell('cellZ', 'print("clean")', outs.append)
        assert any('clean' in o.get('text', '') for o in outs if o['output_type'] == 'stream')


async def test_startup_file(gateway):
    "An owned kernel runs `config.STARTUP_PATH` at seeding with `__file__` bound, the way clikernel runs its startup.py."
    config.STARTUP_PATH.write_text('startup_ran = 7\nstartup_file = __file__\n')
    async with KernelSession(url=gateway) as k:
        bridge, _ = await setup_tools(k)
        assert await bridge.read_var('startup_ran') == 7
        assert await bridge.read_var('startup_file') == str(config.STARTUP_PATH)


async def test_start_cwd_env(gateway, tmp_path):
    "`start(cwd=, env=)` places an owned kernel: it starts in that directory with those entries over our environment."
    k = await KernelSession(url=gateway).start(cwd=tmp_path, env=dict(IPYAI_TEST_MARK='yes'))
    try:
        bridge = KernelBridge(k.kc)
        await bridge._exec('import os')
        assert await bridge.read_var('os.getcwd()') == str(tmp_path)
        assert await bridge.read_var('os.environ["IPYAI_TEST_MARK"]') == 'yes'
    finally: await k.close()


async def test_py_concurrent_calls(gateway):
    "Parallel `py` calls from one model turn each get their own result, and one call's error does not abort the ones queued behind it."
    async with KernelSession(url=gateway) as k:
        _, tools = await setup_tools(k)
        calls = (tools.call_text('py', dict(code=c)) for c in ['y = 6*7', '1/0', 'y', '1+1'])
        res = await asyncio.wait_for(asyncio.gather(*calls), 20)
        assert 'ZeroDivisionError' in res[1] and [res[0], *res[2:]] == ['', '42', '2']


def test_session_files(tmp_path, monkeypatch):
    "Session round-trip: save creates the self-ignored dir, listing and prefix resolution find it, meta rides along."
    monkeypatch.chdir(tmp_path)
    s = Session()
    d = Dialog(name='t')
    d.mk_message('x = 1', msg_type='code')
    d.mk_message('why?', msg_type='prompt', output='Because.')
    s.save(d, kernel_id='k1', model='m')
    assert (tmp_path/'.ipyai'/'.gitignore').exists()
    rows = list_sessions()
    assert len(rows) == 1 and rows[0][2] == 1 and rows[0][3] == 'why?'
    assert resolve_session(s.path.stem[:6]) == s.path
    with pytest.raises(FileNotFoundError): resolve_session('nope-nothing')
    d2 = read_ipynb(s.path)
    assert d2.meta['ipyai'] == dict(kernel_id='k1', model='m')
    assert [m.msg_type for m in d2.messages] == ['code', 'prompt']
    assert d2.messages[1].ai_res == 'Because.'


def test_history_modes(tmp_path, monkeypatch):
    "History mines session files, each composer mode scoped to its own message shape."
    monkeypatch.chdir(tmp_path)
    s = Session()
    d = Dialog(name='t')
    d.mk_message('x = 1', msg_type='code')
    d.mk_message('!ls -la', msg_type='code')
    d.mk_message('why?', msg_type='prompt', output='B.')
    d.mk_message('a note', msg_type='note')
    s.save(d)
    assert History().items == ['x = 1']
    assert History(mode='shell').items == ['ls -la']
    assert History(mode='prompt').items == ['why?']
    assert History().suggest('x') == ' = 1'

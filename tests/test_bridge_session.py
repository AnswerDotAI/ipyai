"Integration: the KernelBridge against a live gateway ipymini, and session-file persistence."
import pytest
from ipyai.kernel import KernelSession
from ipyai.bridge import setup_tools
from ipyai.assistant import LAST_RESPONSE
from ipyai.session import Session, list_sessions, resolve_session
from ipyai.history import History
from aidialog.dialog import Dialog
from aidialog.ipynb import read_ipynb


async def test_bridge_and_writeback(gateway):
    "Silent kernel-side exec: the py tool runs live code, set_vars lands LAST_RESPONSE for the user."
    async with KernelSession(url=gateway) as k:
        bridge, tools = await setup_tools(k.kc)
        names = await tools.names()
        assert 'py' in names
        out = await tools.call_text('py', {'code': 'zz = 6*7\nprint("side effect")\nzz'})
        assert 'side effect' in out and '42' in out
        assert await bridge.read_var('zz') == 42  # the tool ran in the USER'S namespace
        bridge.set_vars(**{LAST_RESPONSE: 'the reply text'})
        assert await bridge.read_var(LAST_RESPONSE) == 'the reply text'
        # a normal cell still renders normally after bridge traffic (no iopub leakage)
        outs = []
        await k.run('print("clean")', outs.append)
        assert any('clean' in o.get('text', '') for o in outs if o['output_type'] == 'stream')


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

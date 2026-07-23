"cp3b integration: the ConBridge against a live ipymini, and the persistence store on a private db."
import asyncio, json
from teleprint.testing import FakeTty
from ipyai.cli import App
from ipyai.kernel import KernelSession
from ipyai.bridge import setup_tools
from ipyai.store import Store, nbformat_outputs
from ipyai.assistant import Assistant, LAST_RESPONSE
from tests.test_assistant import mk_cfg, FakeChatFactory

def test_bridge_and_writeback():
    "Silent kernel-side exec: the py tool runs live code, set_vars lands LAST_RESPONSE for the user."
    async def go():
        async with KernelSession() as k:
            bridge, tools = await setup_tools(k.kc)
            names = await tools.names()
            assert 'py' in names
            out = await tools.call_text('py', {'code': 'zz = 6*7\nprint("side effect")\nzz'})
            assert 'side effect' in out and '42' in out
            assert await bridge.read_var('zz') == 42  # the tool ran in the USER'S namespace
            await bridge.set_vars(**{LAST_RESPONSE: 'the reply text'})
            assert await bridge.read_var(LAST_RESPONSE) == 'the reply text'
            path, session = await bridge.history_db_info()
            assert path.endswith('history.sqlite') and isinstance(session, int)
            # a normal cell still renders normally after bridge traffic (no iopub leakage)
            outs = []
            await k.run('print("clean")', lambda mt, c: outs.append((mt, c)))
            assert any('clean' in c.get('text', '') for mt, c in outs if mt == 'stream')
    asyncio.run(go())

def test_store_roundtrip(tmp_path):
    db = tmp_path/'hist.sqlite'
    st = Store(db, 7, cwd='/tmp/x', backend='codex')
    st.save_cell(1, 'x = 1', [('stream', {'name': 'stdout', 'text': 'hi\n'}),
                              ('execute_result', {'data': {'text/plain': '1'}, 'metadata': {}, 'execution_count': 1})])
    st.save_prompt('why?', 'why?', 'Because.', 1)
    st.save_cell(2, 'y = 2', [('error', {'ename': 'E', 'evalue': 'boom', 'traceback': ['tb']})])
    evs = st.load_session(7)
    assert [e['kind'] for e in evs] == ['cell', 'prompt', 'cell']
    assert evs[0]['outputs'][0] == dict(output_type='stream', name='stdout', text='hi\n')
    assert evs[1]['response'] == 'Because.'
    rows = st.sessions(cwd='/tmp/x')
    assert rows and rows[0][0] == 7 and rows[0][3] == 1
    assert st.sessions(cwd='/nowhere') == []
    st.close()

def test_resume_into_app(tmp_path):
    "Resume prints the stored session as blocks and rebuilds the Dialog to continue it."
    async def go():
        db = tmp_path/'hist.sqlite'
        st = Store(db, 3, cwd='.', backend='codex')
        st.save_cell(1, 'x = 41', [('execute_result', {'data': {'text/plain': '41'}, 'metadata': {}, 'execution_count': 1})])
        st.save_prompt('add one', 'add one', 'Use `x + 1`:\n\n```python\nx + 1\n```\n', 1)
        tty = FakeTty(60, 18)
        fake = FakeChatFactory([dict(text='later')])
        app = App(tty, history=None, assistant=Assistant(cfg=mk_cfg(), chat_factory=fake, system_prompt='sp'))
        app.assistant.store = st
        app.paint()
        app.load_session(3)
        scr = tty.term.text()
        assert '>>> x = 41' in scr and 'add one' in scr and 'x + 1' in scr
        a = app.assistant
        assert [m.msg_type for m in a.dlg.messages] == ['code', 'prompt']
        assert a.dlg.messages[-1].ai_output.startswith('Use `x + 1`')
        assert a.last_response.startswith('Use `x + 1`')
        app.comp.on_bytes(b'\x1bW')  # paste bindings work over the resumed reply
        assert app.buf.text == 'x + 1'
        # continuing the conversation carries the resumed history
        app.buf.clear()
        app.comp.on_bytes(b'.and now?\r')
        for _ in range(100):
            await asyncio.sleep(0.02)
            if fake.calls: break
        call = fake.calls[0]
        assert len(call['hist']) == 2 and call['hist'][1].startswith('Use `x + 1`')
        assert 'x = 41' in '\n'.join(p for p in call['hist'][0] if isinstance(p, str))
    asyncio.run(go())

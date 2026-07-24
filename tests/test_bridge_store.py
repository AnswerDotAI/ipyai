"cp3b integration: the ConBridge against a live ipymini, and the persistence store on a private db."
import asyncio, json
from teleprint.testing import EmuTty
from ipyai.cli import App
from ipyai.kernel import KernelSession
from ipyai.bridge import setup_tools
from ipyai.store import Store, nbformat_outputs
from ipyai.assistant import Assistant, LAST_RESPONSE
from tests.test_assistant import mk_cfg, StubChatFactory

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
        tty = EmuTty(60, 18)
        stub = StubChatFactory([dict(text='later')])
        app = App(tty, history=None, assistant=Assistant(cfg=mk_cfg(), chat_factory=stub, sp='sp'))
        app.assistant.store = st
        app.paint()
        app.load_session(3)
        scr = tty.term.text()
        assert '»»» x = 41' in scr and 'add one' in scr and 'x + 1' in scr
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
            if stub.calls: break
        call = stub.calls[0]
        assert len(call['hist']) == 2 and call['hist'][1].startswith('Use `x + 1`')
        assert 'x = 41' in '\n'.join(p for p in call['hist'][0] if isinstance(p, str))
    asyncio.run(go())

def test_reset_and_save_load(tmp_path):
    "%ipyai reset clears the Dialog and repoints the store; load silently re-runs code to rebuild kernel state."
    class StubKernel:
        busy, on_comm = False, None
        def __init__(self): self.ran = []
        async def run(self, code, on_output): self.ran.append(code)
    db = tmp_path/'hist.sqlite'
    st = Store(db, 3, cwd='.', backend='codex/gpt-5.4')
    st.save_cell(1, 'x = 41', [('execute_result', {'data': {'text/plain': '41'}, 'metadata': {}, 'execution_count': 1})])
    st.save_prompt('add one', 'add one', 'Use `x + 1`.', 1)
    tty = EmuTty(60, 18)
    app = App(tty, kernel=StubKernel(), history=None, assistant=Assistant(cfg=mk_cfg(), chat_factory=StubChatFactory(), sp='sp'))
    app.assistant.store = st
    app.paint()
    app.load_session(3)
    a = app.assistant
    p = tmp_path/'sess.ipynb'
    app._ipyai_cmd(['save', str(p)])   # the load round-trip below verifies the save
    app._ipyai_cmd(['reset', '9'])   # the kernel-side magic supplies the new session number
    assert len(a.dlg) == 0 and a.n_cells == 0 and st.session == 9   # reset cleared the dialog and switched session
    assert {r[0] for r in st.sessions()} == {3, 9}
    a.add_cell('%ipyai save foo', [])   # housekeeping commands never become dialog messages
    assert len(a.dlg) == 0 and a.n_cells == 0
    from aidialog.ipynb import read_ipynb, write_ipynb
    d = read_ipynb(p)
    d.mk_message('%ipyai reset', msg_type='code')   # a stale housekeeping cell in an old saved file
    write_ipynb(d, p)
    scr = tty.term.text()   # what load_session painted; load itself must add nothing
    app._ipyai_cmd(['load', str(p)])
    assert [m.msg_type for m in a.dlg.messages] == ['code', 'prompt']   # %ipyai cell filtered out on load
    assert a.dlg.messages[0].content == 'x = 41'
    assert a.last_response == 'Use `x + 1`.'
    assert tty.term.text() == scr   # loading paints nothing: the .ipynb is the inspection surface
    asyncio.run(app.run_loaded())
    assert app.k.ran == ['x = 41']   # code cells re-ran to rebuild kernel state; %ipyai cells skipped

def test_image_cap():
    import base64, io
    from PIL import Image
    from ipyai.store import nbformat_outputs
    def b64img(w, h, fmt):
        buf = io.BytesIO()
        Image.new('RGB', (w, h), 'blue').save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode()
    def dims(o, mime): return Image.open(io.BytesIO(base64.b64decode(o['data'][mime]))).size
    big, small, jpg = b64img(2000, 1500, 'PNG'), b64img(4, 2, 'PNG'), b64img(2400, 1000, 'JPEG')
    outs = nbformat_outputs([('display_data', dict(data={'image/png': big})),
                             ('execute_result', dict(data={'image/png': small}, execution_count=1)),
                             ('display_data', dict(data={'image/jpeg': jpg}))])
    w, h = dims(outs[0], 'image/png')
    assert w * h <= 2_000_000 and abs(w / h - 2000 / 1500) < 0.01   # downscaled to fit, aspect kept
    assert outs[1]['data']['image/png'] == small                    # under the cap: untouched
    w, h = dims(outs[2], 'image/jpeg')
    assert w * h <= 2_000_000                                       # jpeg capped too, staying jpeg

def test_hide_toggle(tmp_path):
    "h in the transcript view flips `skipped` on the exchange's message: blocks dim, AI context excludes it, the store remembers, resume replays it dim."
    async def go():
        db = tmp_path/'hist.sqlite'
        st = Store(db, 3, cwd='.', backend='codex')
        st.save_cell(1, 'secret = 41', [('execute_result', {'data': {'text/plain': '41'}, 'metadata': {}, 'execution_count': 1})])
        st.save_prompt('add one', 'add one', 'Use `x + 1`.', 1)
        tty = EmuTty(60, 18)
        stub = StubChatFactory([dict(text='later')])
        app = App(tty, history=None, assistant=Assistant(cfg=mk_cfg(), chat_factory=stub, sp='sp'))
        app.assistant.store = st
        app.paint()
        app.load_session(3)
        a = app.assistant
        cellmsg = a.dlg.messages[0]
        blks = [b for b in app.comp.blocks.values() if getattr(b, 'msg_id', None) == cellmsg.id]
        assert len(blks) == 2                             # input + result blocks linked to ONE message
        app.comp.on_bytes(b'\x14')                        # ctrl-T: transcript view
        assert app.tv.active
        for _ in range(9):
            if app.tv.cur == blks[0].id: break
            app.comp.on_bytes(b'\x1b[A')                  # cursor up to the exchange's input block
        app.comp.on_bytes(b'h')
        assert cellmsg.skipped and all(b.dim for b in blks)   # both halves hidden together
        app.comp.on_bytes(b'h')
        assert not cellmsg.skipped and not any(b.dim for b in blks)   # and the flip reverses
        app.comp.on_bytes(b'h')
        app.comp.on_bytes(b'\x14')                        # leave the view
        app.comp.on_bytes(b'.and now?\r')
        for _ in range(100):
            await asyncio.sleep(0.02)
            if stub.calls: break
        flat = str(stub.calls[0]['hist'])
        assert 'secret' not in flat and 'x + 1' in flat   # hidden exchange gone; the rest of history intact
        # the store learned, so a fresh resume replays the hide
        tty2 = EmuTty(60, 18)
        app2 = App(tty2, history=None, assistant=Assistant(cfg=mk_cfg(), chat_factory=StubChatFactory(), sp='sp'))
        app2.assistant.store = Store(db, 99)   # the resumed instance writes NEW work under its own session...
        app2.paint()
        app2.load_session(3)
        m2 = app2.assistant.dlg.messages[0]
        assert m2.skipped == 1
        assert all(b.dim for b in app2.comp.blocks.values() if getattr(b, 'msg_id', None) == m2.id)
        app2.comp.on_bytes(b'\x14')
        for _ in range(9):
            if getattr(app2.comp.blocks.get(app2.tv.cur), 'msg_id', None) == m2.id: break
            app2.comp.on_bytes(b'\x1b[A')
        app2.comp.on_bytes(b'h')                          # unhide FROM THE RESUMED instance...
        assert not m2.skipped
        assert Store(db, session=None).load_session(3)[0]['meta'] == {}   # ...must update session 3's row, not session 99's
    asyncio.run(go())

def test_store_meta_migration(tmp_path):
    "A pre-meta db gains the meta columns in place on open; existing rows read back with empty meta."
    import apsw
    db = tmp_path/'old.sqlite'
    con = apsw.Connection(str(db))
    con.execute('CREATE TABLE ipyai_prompts (id INTEGER PRIMARY KEY AUTOINCREMENT, session INTEGER NOT NULL, '
                'line INTEGER NOT NULL DEFAULT 0, prompt TEXT NOT NULL, full_prompt TEXT NOT NULL, response TEXT NOT NULL)')
    con.execute('CREATE TABLE ipyai_cells (session INTEGER NOT NULL, line INTEGER NOT NULL, source TEXT NOT NULL, '
                'outputs TEXT NOT NULL, PRIMARY KEY (session, line))')
    con.execute("INSERT INTO ipyai_cells VALUES (5, 1, 'x = 1', '[]')")
    con.close()
    st = Store(db, 5, cwd='.')
    evs = st.load_session(5)
    assert evs[0]['meta'] == {}                # old row, no meta
    st.set_cell_meta(1, {'skipped': 1})
    assert st.load_session(5)[0]['meta'] == {'skipped': 1}
    st.close()

def test_edit_in_transcript_view(tmp_path):
    "e edits the cursor exchange: cell source, prompt content, or the whole reply; Dialog + store + blocks all update; Esc cancels."
    async def go():
        db = tmp_path/'hist.sqlite'
        st = Store(db, 3, cwd='.', backend='codex')
        st.save_cell(1, 'x = 41', [('execute_result', {'data': {'text/plain': '41'}, 'metadata': {}, 'execution_count': 1})])
        st.save_prompt('add one', 'add one', 'Use `x + 1`.', 1)
        tty = EmuTty(64, 20)
        app = App(tty, history=None, assistant=Assistant(cfg=mk_cfg(), chat_factory=StubChatFactory(), sp='sp'))
        app.assistant.store = st
        app.paint()
        app.load_session(3)
        a = app.assistant
        cellmsg, pmsg = a.dlg.messages
        app.comp.on_bytes(b'\x14')                        # transcript view
        for _ in range(9):
            if getattr(app.comp.blocks.get(app.tv.cur), 'msg_id', None) == cellmsg.id and app.comp.blocks[app.tv.cur].tag == 'in': break
            app.comp.on_bytes(b'\x1b[A')
        app.comp.on_bytes(b'e')
        assert app.editing and app.buf.text == 'x = 41'   # composer pre-filled with the source
        app.comp.on_bytes(b'\x1b[D')                      # arrow keys edit within the composer
        app.buf.text, app.buf.cursor = 'x = 43', 6
        app.comp.on_bytes(b'\r')                          # Enter writes back, stays in the view
        assert app.editing is None and app.tv.active and app.buf.text == ''
        assert cellmsg.content == 'x = 43'
        assert st.load_session(3)[0]['source'] == 'x = 43'
        blk = next(b for b in app.comp.blocks.values() if getattr(b, 'msg_id', None) == cellmsg.id and b.tag == 'in')
        assert blk.source == 'x = 43'
        # reply edit: land on a reply-side block of the prompt exchange
        for _ in range(9):
            b = app.comp.blocks.get(app.tv.cur)
            if getattr(b, 'msg_id', None) == pmsg.id and b.tag != 'ask': break
            app.comp.on_bytes(b'\x1b[B')
        app.comp.on_bytes(b'e')
        assert app.buf.text == 'Use `x + 1`.'             # the WHOLE reply markdown
        app.buf.text = 'Use `x + 2`.'
        app.comp.on_bytes(b'\r')
        assert pmsg.ai_res == 'Use `x + 2`.'
        assert st.load_session(3)[1]['response'] == 'Use `x + 2`.'
        assert a.last_response == 'Use `x + 2`.'          # it was the last prompt
        # Esc cancels without touching anything
        app.comp.on_bytes(b'e')
        app.buf.text = 'garbage'
        app.comp.on_bytes(b'\x1b')
        app.comp.flush_input()                            # lone ESC resolves on the parser's timeout arm
        app.comp.flush_input()
        assert app.editing is None and pmsg.ai_res == 'Use `x + 2`.' and app.tv.active
    asyncio.run(go())

def test_shift_jump_exchanges(tmp_path):
    "Shift-up/down in the transcript view moves between exchange starts, skipping output blocks."
    db = tmp_path/'hist.sqlite'
    st = Store(db, 3, cwd='.', backend='codex')
    st.save_cell(1, 'a = 1', [('execute_result', {'data': {'text/plain': '1'}, 'metadata': {}, 'execution_count': 1})])
    st.save_cell(2, 'b = 2', [('execute_result', {'data': {'text/plain': '2'}, 'metadata': {}, 'execution_count': 2})])
    st.save_prompt('why?', 'why?', 'Because.', 2)
    tty = EmuTty(60, 18)
    app = App(tty, history=None, assistant=Assistant(cfg=mk_cfg(), chat_factory=StubChatFactory(), sp='sp'))
    app.assistant.store = st
    app.paint()
    app.load_session(3)
    app.comp.on_bytes(b'\x14')
    starts = [bid for bid, b in app.comp.blocks.items() if b.tag in ('in', 'ask', 'sh')]
    app.comp.on_bytes(b'\x1b[1;2A')                   # shift-up
    assert app.tv.cur == starts[-1]                   # from the tail: the last exchange start (the ask)
    app.comp.on_bytes(b'\x1b[1;2A')
    assert app.tv.cur == starts[-2]                   # previous exchange, outputs skipped
    app.comp.on_bytes(b'\x1b[1;2B')                   # shift-down
    assert app.tv.cur == starts[-1]
    assert not app.tv.follow                          # structure motion unpins follow
    st.close()

def test_retry_prompt(tmp_path):
    "Alt-up arms retry on the last exchange; a prompt submit REPLACES the old turn in dialog, store, and block model."
    async def go():
        db = tmp_path/'hist.sqlite'
        st = Store(db, 3, cwd='.', backend='codex')
        st.save_cell(1, 'x = 41', [('execute_result', {'data': {'text/plain': '41'}, 'metadata': {}, 'execution_count': 1})])
        st.save_prompt('add one', 'add one', 'Use `x + 1`.', 1)
        tty = EmuTty(64, 20)
        stub = StubChatFactory([dict(text='fresh answer')])
        app = App(tty, history=None, assistant=Assistant(cfg=mk_cfg(), chat_factory=stub, sp='sp'))
        app.assistant.store = st
        app.paint()
        app.load_session(3)
        a = app.assistant
        old_prompt = a.dlg.messages[-1]
        app.comp.on_bytes(b'\x1br')                        # alt-r
        assert app.retry and app.retry[0] is old_prompt and app.mode == 'prompt'
        assert app.buf.text == 'add one'                  # recalled, ready to edit
        assert '↻' in tty.term.text()                     # armed: the status cell says so
        app.buf.text = 'add two'
        app.comp.on_bytes(b'\r')
        for _ in range(100):
            await asyncio.sleep(0.02)
            if stub.calls: break
        assert [m.msg_type for m in a.dlg.messages] == ['code', 'prompt']
        assert a.dlg.messages[-1].content == 'add two'    # replaced, not appended
        assert 'add one' not in str(stub.calls[0]['hist'])    # the AI never sees the old turn
        evs = Store(db, session=None).load_session(3)
        assert [e['kind'] for e in evs] == ['cell', 'prompt']
        assert evs[1]['prompt'] == 'add two'              # the old prompt row was dropped; the fresh turn saved its own
        assert not any(getattr(b, 'msg_id', None) == old_prompt.id for b in app.comp.blocks.values())
        assert app.retry is None and '↻' not in tty.term.text()
    asyncio.run(go())

def test_retry_code_and_cancel(tmp_path):
    "Alt-up on a code exchange replaces the cell; a mismatched-kind submit cancels the retry and appends."
    class StubKernel:
        busy, on_comm = False, None
        def __init__(self): self.ran = []
        async def run(self, code, on_output): self.ran.append(code)
        async def check(self, code): return 'complete', ''
    async def go():
        db = tmp_path/'hist.sqlite'
        st = Store(db, 3, cwd='.', backend='codex')
        tty = EmuTty(64, 20)
        stub = StubChatFactory([dict(text='later')])
        app = App(tty, kernel=StubKernel(), history=None, assistant=Assistant(cfg=mk_cfg(), chat_factory=stub, sp='sp'))
        app.assistant.store = st
        app.paint()
        a = app.assistant
        app.comp.on_bytes(b'y = 1\r')
        for _ in range(100):
            await asyncio.sleep(0.02)
            if a.dlg.messages: break
        assert st.load_session(3)[0]['source'] == 'y = 1'
        app.comp.on_bytes(b'\x1br')                        # alt-r: last exchange is the cell
        assert app.retry and app.buf.text == 'y = 1' and app.mode == 'code'
        app.buf.text = 'y = 2'
        app.comp.on_bytes(b'\r')
        for _ in range(100):
            await asyncio.sleep(0.02)
            if len(app.k.ran) == 2: break
        assert [m.content for m in a.dlg.messages] == ['y = 2']   # replaced in the dialog
        assert [e['source'] for e in st.load_session(3)] == ['y = 2']  # and in the store
        # mismatch: recall the cell, submit a prompt instead -> cancel, append, cell survives
        app.comp.on_bytes(b'\x1br')
        assert app.retry is not None
        app.buf.text = '.and now?'
        app.comp.on_bytes(b'\r')
        for _ in range(100):
            await asyncio.sleep(0.02)
            if stub.calls: break
        assert app.retry is None
        assert [m.msg_type for m in a.dlg.messages] == ['code', 'prompt']
        assert a.dlg.messages[0].content == 'y = 2'        # unharmed
        # Esc disarms without touching the composer
        app.comp.on_bytes(b'\x1br')
        assert app.retry is not None
        app.comp.on_bytes(b'\x1b')
        app.comp.flush_input(); app.comp.flush_input()
        assert app.retry is None
    asyncio.run(go())

def test_retry_from_transcript_view(tmp_path):
    "E in the transcript view on a prompt input arms retry and returns to the live screen; on outputs it just notes."
    db = tmp_path/'hist.sqlite'
    st = Store(db, 3, cwd='.', backend='codex')
    st.save_cell(1, 'x = 41', [('execute_result', {'data': {'text/plain': '41'}, 'metadata': {}, 'execution_count': 1})])
    st.save_prompt('add one', 'add one', 'Use `x + 1`.', 1)
    tty = EmuTty(64, 20)
    app = App(tty, history=None, assistant=Assistant(cfg=mk_cfg(), chat_factory=StubChatFactory(), sp='sp'))
    app.assistant.store = st
    app.paint()
    app.load_session(3)
    app.comp.on_bytes(b'\x14')
    for _ in range(9):                                    # cursor to a NON-ask block first
        b = app.comp.blocks.get(app.tv.cur)
        if b is not None and b.tag == 'in': break
        app.comp.on_bytes(b'\x1b[A')
    app.comp.on_bytes(b'E')
    assert app.retry is None and app.tv.active            # not a prompt input: noted, nothing armed
    app.comp.on_bytes(b'\x1b[1;2B')                       # shift-down to the ask block (next exchange start)
    assert app.comp.blocks[app.tv.cur].tag == 'ask'
    app.comp.on_bytes(b'E')
    assert app.retry is not None and not app.tv.active    # armed and back on the live screen
    assert app.buf.text == 'add one' and app.mode == 'prompt'
    st.close()

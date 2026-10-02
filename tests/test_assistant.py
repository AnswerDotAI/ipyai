"Routing, the reply view, the assistant end-to-end over a StubChat, paste bindings, ghost text."
import asyncio
from fastcore.xml import to_xml
from aidialog.dialog import INTERRUPTED
from aidialog.msg_parts import Text, Thinking, ToolUse, ToolResult, Msg, hist2fmt
from teleprint.testing import EmuTty
from ipyai.cli import App
from ipyai.assistant import Assistant, route, code_blocks

def mk_cfg(**kw):
    return dict(model='m', suggest_model='cm', think='l',
                code_theme='ansi_dark', prompt_mode=False) | kw

def tool_use(name, args, id='t1'): return ToolUse(id=id, name=name, arguments=args)
def tool_result(name, args, text, id='t1'): return ToolResult(id=id, name=name, arguments=args, text=text)

class StubChat:
    """Stands in for one fastllm AsyncChat: `await chat(msg, stream=True, ...)` yields the scripted
    items. The factory instance records every construction and call for assertions."""
    def __init__(self, factory, **kw):
        self.factory, self.kw = factory, kw
        self.use = None

    async def __call__(self, msg=None, stream=False, **call_kw):
        self.factory.calls.append(dict(self.kw, msg=msg, **call_kw))
        events, gate = self.factory.events, self.factory.gate
        if self.kw.get('model') == 'cm': events = [Text(self.factory.suggestion)]
        async def gen():
            for e in events:
                await asyncio.sleep(0)
                yield e
            if gate is not None: await gate.wait()
        return gen()

    def full(self, **kw):
        """The turn's canonical form via the real `hist2fmt`, from history shaped as fastllm's is: each
        tool round's results in their own message, and a new assistant message after each round"""
        msgs, asst, tool = [], [], []
        for e in self.factory.events:
            if isinstance(e, ToolResult):
                tool.append(e)
                continue
            if tool: msgs, asst, tool = msgs + [Msg('assistant', asst), Msg('tool', tool)], [], []
            if isinstance(e, (Text, ToolUse)): asst.append(e)
        if tool: msgs, asst = msgs + [Msg('assistant', asst), Msg('tool', tool)], []
        return hist2fmt(msgs + ([Msg('assistant', asst)] if asst else []))

class StubChatFactory:
    "Supplied as Assistant's chat_factory; scripts the turn stream and the suggestion model's reply."
    def __init__(self, events=(), suggestion='', gate=None):
        self.events, self.suggestion, self.gate = list(events), suggestion, gate
        self.calls = []

    def __call__(self, **kw): return StubChat(self, **kw)

def mk_app(events=(), suggestion='', gate=None, prompt_mode=False, rows=16):
    tty = EmuTty(64, rows)
    stub = StubChatFactory(events, suggestion, gate)
    app = App(tty, history=None, assistant=Assistant(cfg=mk_cfg(), chat_factory=stub, sp='sp'))
    app.mode = 'prompt' if prompt_mode else 'code'
    return tty, app, stub

async def _idle(app, pred=lambda: True):
    for _ in range(100):
        await asyncio.sleep(0.02)
        if not app.busy and pred(): return
    raise TimeoutError('the app stayed busy')

def _parts_text(call):
    "All str content sent for the turn: prompt parts plus str history entries."
    msg = call['msg']
    parts = msg if isinstance(msg, list) else [msg]
    return '\n'.join(p for p in parts if isinstance(p, str))

def _parts(app, m): return [k.split(':', 1)[1] for k in app.view.keys[m.id]]

def test_route():
    assert route('x = 1', 'code') == ('code', 'x = 1')
    assert route('.what is x?', 'code') == ('prompt', 'what is x?')
    assert route('what is x?', 'prompt') == ('prompt', 'what is x?')
    assert route(';x = 1', 'prompt') == ('code', 'x = 1')
    assert route('  ;x = 1', 'prompt') == ('code', '  x = 1')
    assert route('!ls', 'prompt') == ('job', 'ls')
    assert route('%time 1', 'prompt') == ('code', '%time 1')
    assert route('.still a prompt', 'prompt') == ('prompt', '.still a prompt')

def test_streaming_reply_spans():
    "Spans close incrementally: each top-level md block becomes its own block; the growing last span is replaced in place, never duplicated."
    tty, app, stub = mk_app(rows=14)
    app.paint()
    m = app.ctl.dlg.mk_message('q', msg_type='prompt')
    txt = ''
    for chunk in ['# He', 'ading\n\npara ', 'text\n\n```python\nx ', '= 1\nprint(x)\n```\n\ntail']:
        txt += chunk
        m.output = txt
        app.view.mark(m, live=True)
    app.view.mark(m, live=False)
    scr = tty.term.contents()
    for s in ('# Heading', 'para text', 'x = 1', 'print(x)', 'tail'):
        assert scr.count(s) == 1, (s, scr)
    assert _parts(app, m) == ['ask', 'r0', 'r1', 'r2', 'r3']

def test_tall_fence_folds():
    "A fence taller than the threshold folds while still streaming, and stays folded once final."
    tty, app, stub = mk_app(rows=12)   # collapse_at: 6
    app.paint()
    m = app.ctl.dlg.mk_message('q', msg_type='prompt')
    m.output = '```python\n' + '\n'.join(f'line{i} = {i}' for i in range(12))
    app.view.mark(m, live=True)
    assert app.comp.blocks[f'{m.id}:r0'].collapsed  # folded mid-stream: a giant fence cannot flood the screen
    m.output += '\n```\n\ndone'
    app.view.mark(m, live=False)
    fence = app.comp.blocks[f'{m.id}:r0']
    assert fence.collapsed and fence.height == 12
    assert '… (+' in tty.term.text()

async def test_turn_with_tools():
    "A turn's reply shows text spans and a folded tool block; the thinking placeholder is gone once the turn completes."
    tty, app, stub = mk_app(events=[Thinking('hmm\nlines'), Text('Check the value.\n\n'), tool_use('py', {'code': 'x*2'}),
                                    tool_result('py', {'code': 'x*2'}, '42'), Text('The answer is 42.')])
    app.paint()
    app.comp.on_bytes(b'.go\r')
    await _idle(app, lambda: stub.calls)
    m = app.ctl.dlg.messages[-1]
    assert _parts(app, m) == ['ask', 'r0', 'tool:t1', 'r1']
    assert app.comp.blocks[f'{m.id}:tool:t1'].collapsed
    scr = tty.term.text()
    assert 'py(code=x*2)' in scr and '42' not in scr.split('answer')[0]  # result folded away
    assert '🧠' not in scr

async def test_prompt_flow_and_ctx():
    "The routed prompt flow: ask block, reply blocks, dialog records the turn, ctx rides via dlg2hist."
    tty, app, stub = mk_app(events=[Text('The answer.\n')])
    app.paint()
    app.ctl.dlg.mk_message('x = 41 + 1', msg_type='code', output=[dict(output_type='stream', name='stdout', text='')])
    app.comp.on_bytes(b'.what is x?\r')
    await _idle(app, lambda: stub.calls)
    assert [k.split(':')[1] for k in app.comp.blocks][:2] == ['ask', 'r0']
    scr = tty.term.text()
    assert 'what is x?' in scr and 'The answer.' in scr
    call = stub.calls[0]
    assert call['model'] == 'm' and call['think'] == 'l'
    sent = _parts_text(call)
    assert 'x = 41 + 1' in sent          # the cell rides in the prompt's user parts (dlg2hist)
    assert call['msg'][-1].endswith('>what is x?</prompt>')  # the aidialog envelope wraps every prompt
    assert [m.msg_type for m in app.ctl.dlg.messages] == ['code', 'prompt']
    assert app.ctl.last_reply() == 'The answer.'
    assert app.ctl.dlg.messages[-1].ai_output == 'The answer.'
    # second turn: history carries turn one; the cell does not repeat in the new prompt parts
    app.comp.on_bytes(b'.and again?\r')
    await _idle(app, lambda: len(stub.calls) == 2)
    call2 = stub.calls[1]
    assert len(call2['hist']) == 2 and 'The answer.' in call2['hist'][1]
    assert 'x = 41 + 1' in '\n'.join(p for p in call2['hist'][0] if isinstance(p, str))
    assert 'x = 41 + 1' not in _parts_text(call2)

async def test_reply_stored_in_fastllm_form():
    "The stored reply is fastllm's canonical form: a tool round-trip is a `{.tool}` block that `fmt2hist` parses back."
    tty, app, stub = mk_app(events=[Text('Look:\n'), tool_use('py', {'code': '1+1'}),
                                    tool_result('py', {'code': '1+1'}, '2'), Text('Done.')])
    app.paint()
    app.comp.on_bytes(b'.check\r')
    await _idle(app, lambda: stub.calls)
    resp = app.ctl.last_reply()
    assert '```json {.tool}' in resp and 'Done.' in resp
    from aidialog.msg_parts import fmt2hist
    msgs = fmt2hist(resp)
    assert any(isinstance(p, ToolResult) for m in msgs for p in m.content)

async def test_interrupt_freezes_turn():
    gate = asyncio.Event()
    tty, app, stub = mk_app(events=[Text('partial text so far')], gate=gate)
    app.paint()
    app.comp.on_bytes(b'.go\r')
    for _ in range(100):
        await asyncio.sleep(0.02)
        if 'partial text' in tty.term.text(): break
    assert app.busy
    app.comp.on_bytes(b'\x03')   # ctrl-C: cancels the turn, not the app
    await _idle(app)
    assert 'interrupted' in tty.term.text()
    assert app.ctl.last_reply().endswith(INTERRUPTED)
    assert app.ctl.dlg.messages[-1].msg_type == 'prompt'   # the interrupted turn still records
    gate.set()

async def test_turn_stopped_before_output_gives_the_prompt_back():
    "Ctrl-C before a turn has produced anything leaves no message, and its text returns to the composer."
    tty, app, stub = mk_app(gate=asyncio.Event())
    app.paint()
    app.comp.on_bytes(b'.never mind\r')
    for _ in range(100):
        await asyncio.sleep(0.02)
        if stub.calls: break
    assert app.busy
    app.comp.on_bytes(b'\x03')
    await _idle(app)
    assert app.ctl.dlg.messages == [] and not app.comp.blocks
    assert app.buf.text == 'never mind' and app.mode == 'prompt'

async def test_retry_hide_and_edit_keep_identity():
    """A retried turn replaces the old one in the Dialog and on screen, and its new blocks belong to the new
    message: hiding and editing it from the transcript view reach that message."""
    tty, app, stub = mk_app(events=[Text('ok')], rows=24)
    app.paint()
    for p in (b'.first\r', b'.second\r'):
        app.comp.on_bytes(p)
        await _idle(app, lambda: stub.calls)
    old = app.ctl.dlg.messages[-1]
    app.comp.on_bytes(b'\x1br')                       # alt-r: recall the last exchange for retry
    assert app.buf.text == 'second' and app.retry
    app.comp.on_bytes(b'\x15second again\r')          # ctrl-u, retype, Enter: replaces it
    await _idle(app, lambda: len(stub.calls) == 3)
    assert [m.content for m in app.ctl.dlg.messages] == ['first', 'second again']
    m = app.ctl.dlg.messages[-1]
    assert not any(k.startswith(old.id) for k in app.comp.blocks)
    assert _parts(app, m) == ['ask', 'r0']
    app.comp.on_bytes(b'\x14')                        # ctrl-T: the cursor starts on the newest block, m's reply
    app.comp.on_bytes(b'h')
    await asyncio.sleep(0)                            # the view syncs on the next pass of the event loop
    assert m.skipped and app.comp.blocks[f'{m.id}:ask'].dim and app.comp.blocks[f'{m.id}:r0'].dim
    app.comp.on_bytes(b'e')                           # edit the whole reply
    assert app.buf.text == 'ok'
    app.comp.on_bytes(b'\x15fixed\r')
    await asyncio.sleep(0)
    assert m.ai_res == 'fixed' and 'fixed' in app.comp.blocks[f'{m.id}:r0'].source

async def test_turn_failing_before_output_gives_the_prompt_back():
    "A turn that fails before producing anything leaves no message, and its text returns to the composer."
    tty, app, stub = mk_app()
    def boom(**kw): raise RuntimeError('no backend')
    app.ctl.assistant._chat_factory = boom
    app.paint()
    app.comp.on_bytes(b'.hi\r')
    await _idle(app)
    assert app.ctl.dlg.messages == []
    assert app.buf.text == 'hi' and app.mode == 'prompt'
    assert 'no backend' in tty.term.text()

async def test_paste_bindings():
    tty, app, stub = mk_app()
    app.paint()
    app.ctl.dlg.mk_message('q', msg_type='prompt', output='Try:\n\n```python\na = 1\n```\n\nthen\n\n```python\nb = 2\n```\n')
    assert code_blocks(app.ctl.last_reply()) == ['a = 1', 'b = 2']
    app.comp.on_bytes(b'\x1bW')          # alt-shift-w: all blocks
    assert app.buf.text == 'a = 1\nb = 2'
    app.comp.on_bytes(b'\x15')           # ctrl-u clears
    app.comp.on_bytes(b'\x1b@')          # alt-shift-2: second block
    assert app.buf.text == 'b = 2'
    app.comp.on_bytes(b'\x1b[1;4A')      # alt-shift-up: cycle replaces the buffer
    assert app.buf.text == 'a = 1'
    app.comp.on_bytes(b'\x1b[1;4A')
    assert app.buf.text == 'b = 2'

async def test_ai_ghost_text():
    tty, app, stub = mk_app(suggestion='(reverse=True)')
    app.paint()
    app.comp.on_bytes(b'xs.sort')
    app.comp.on_bytes(b'\x1b.')          # alt-.: explicit AI suggestion
    for _ in range(100):
        await asyncio.sleep(0.02)
        if app.ai_sugg: break
    assert app.ai_sugg[2] == '(reverse=True)'
    assert '(reverse=True)' in tty.term.text()
    assert stub.calls[-1]['model'] == 'cm'
    app.comp.on_bytes(b'(')              # document changed: the suggestion is stale and gone
    assert app.buf.suggestion == ''

async def test_prompt_mode_ui():
    tty, app, stub = mk_app(events=[Text('ok')], prompt_mode=True)
    app.paint()
    assert tty.term.text().splitlines()[-1].startswith('›››')  # the mode shows in the marker (trailing space trimmed by text())
    app.comp.on_bytes(b'hello there\r')  # plain Enter submits: English is never incomplete
    await _idle(app, lambda: stub.calls)
    assert stub.calls[0]['msg'][-1].endswith('>hello there</prompt>')
    app.comp.on_bytes(b'\x1bc')          # M-c: direct-select code mode (M-p is no longer a toggle)
    assert app.mode == 'code'
    assert tty.term.text().splitlines()[-1].startswith('»»»')

def test_ctx_usage_status():
    "The ctx meter: the final request's size over the model window, painted into the dim status line."
    from fastllm.chat import UsageStats
    from ipyai.cli import _fmt_tk
    assert (_fmt_tk(950), _fmt_tk(34_200), _fmt_tk(1_000_000)) == ('950', '34.2k', '1M')
    tty, app, stub = mk_app()
    a = app.ctl.assistant
    assert a.ctx_usage is None                 # unknown model ('m'): no meter
    a.model = 'codex/gpt-5.5'   # the vendor rides in the model string
    assert a.ctx_usage is None                 # known model but no turn yet: still no meter
    a.last_req_use = UsageStats(prompt_tokens=30000, completion_tokens=2000)
    assert a.ctx_usage == (32000, 256000)

class StubKC:
    "A kernel client answering `eval_exprs` with scripted values, recording each round trip."
    def __init__(self): self.asked = []
    async def eval_exprs(self, vs):
        self.asked.append(list(vs))
        def val(e):
            if e.startswith('get_ipython().getoutput'): return 'shell-out'
            return 42 if e == 'x' else '<error type="NameError" desc="nope">\nnope</error>'
        return {e: val(e) for e in vs}

async def test_shell_refs_go_through_kernel():
    "`$` vars and `!`cmd`` refs resolve in ONE eval_exprs round trip; `!` keyed by ref form; NameError -> warning."
    tty, app, stub = mk_app()
    a, kc = app.ctl.assistant, StubKC()
    app.ctl.dlg.mk_message('use $`x` and $`nosuch` and !`echo hi`', msg_type='prompt')
    vh, warn = await a.vars_turn(app.ctl.dlg.messages, kc)
    body = to_xml(vh[0][0]) if vh else ''
    assert 'x' in body and '42' in body
    assert '!`echo hi`' in body and 'shell-out' in body
    assert 'nosuch' not in body and 'nosuch' in (warn or '')          # undefined: warned, not rendered
    assert len(kc.asked) == 1                                         # one round trip for everything
    assert any('getoutput' in e for e in kc.asked[0])                 # ! ran via the kernel, not subprocess

async def test_hidden_refs_never_reach_the_kernel():
    "A hidden prompt's `$` and `!` refs are neither evaluated nor sent: the AI sees only what is not hidden."
    tty, app, stub = mk_app(events=[Text('ok')])
    kc = StubKC()
    hidden = app.ctl.dlg.mk_message('old $`secret` and !`rm -rf x`', msg_type='prompt', output='sure')
    hidden.skipped = 1
    app.ctl.dlg.mk_message('now $`x`', msg_type='prompt')
    await app.ctl.assistant.start_turn(app.ctl.dlg.messages, kc)
    assert kc.asked == [['x']]
    assert 'secret' not in str(stub.calls[-1])

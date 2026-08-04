"cp3/cp4: routing, streaming reply rendering, the assistant end-to-end over a StubChat, paste bindings, ghost text."
import asyncio
from fastcore.xml import to_xml
from aidialog.msg_parts import Part, PartType
from teleprint.testing import EmuTty
from ipyai.cli import App, _gutter
from ipyai.reply import TurnRenderer
from ipyai.assistant import Assistant, route, code_blocks

def mk_cfg(**kw):
    return dict(model='m', suggest_model='cm', think='l',
                code_theme='ansi_dark', prompt_mode=False) | kw

def tool_use(name, args, id='t1'): return Part(type=PartType.tool_use, data=dict(id=id, name=name, arguments=args, server=False))
def tool_result(name, args, text, id='t1'): return Part(type=PartType.tool_result, text=text, data=dict(id=id, name=name, arguments=args, server=False))

class StubChat:
    """Stands in for one fastllm AsyncChat: `await chat(msg, stream=True, ...)` yields the scripted
    items. The factory instance records every construction and call for assertions."""
    def __init__(self, factory, **kw):
        self.factory, self.kw = factory, kw
        self.use = None

    async def __call__(self, msg=None, stream=False, **call_kw):
        self.factory.calls.append(dict(self.kw, msg=msg, **call_kw))
        events, gate = self.factory.events, self.factory.gate
        if self.kw.get('model') == 'cm': events = [dict(text=self.factory.suggestion)]
        async def gen():
            for e in events:
                await asyncio.sleep(0)
                yield e
            if gate is not None: await gate.wait()
        return gen()

class StubChatFactory:
    "Supplied as Assistant's chat_factory; scripts the turn stream and the suggestion model's reply."
    def __init__(self, events=(), suggestion='', gate=None):
        self.events, self.suggestion, self.gate = list(events), suggestion, gate
        self.calls = []

    def __call__(self, **kw): return StubChat(self, **kw)

def mk_app(events=(), suggestion='', gate=None, prompt_mode=False):
    tty = EmuTty(64, 16)
    stub = StubChatFactory(events, suggestion, gate)
    app = App(tty, history=None, assistant=Assistant(cfg=mk_cfg(), chat_factory=stub, sp='sp'))
    app.mode = 'prompt' if prompt_mode else 'code'
    return tty, app, stub

def _parts_text(call):
    "All str content sent for the turn: prompt parts plus str history entries."
    msg = call['msg']
    parts = msg if isinstance(msg, list) else [msg]
    return '\n'.join(p for p in parts if isinstance(p, str))

def test_route():
    assert route('x = 1', 'code') == ('code', 'x = 1')
    assert route('.what is x?', 'code') == ('prompt', 'what is x?')
    assert route('what is x?', 'prompt') == ('prompt', 'what is x?')
    assert route(';x = 1', 'prompt') == ('code', 'x = 1')
    assert route('  ;x = 1', 'prompt') == ('code', '  x = 1')
    assert route('!ls', 'prompt') == ('job', 'ls')
    assert route('%time 1', 'prompt') == ('code', '%time 1')
    assert route('.still a prompt', 'prompt') == ('prompt', '.still a prompt')

def test_streaming_reply_blocks():
    "Spans close incrementally: each top-level md block becomes its own block; the partial is replaced, not duplicated."
    tty = EmuTty(60, 14)
    app = App(tty, history=None)
    app.paint()
    tr = TurnRenderer(app.comp, _gutter, collapse_at=5)
    for chunk in ['# He', 'ading\n\npara ', 'text\n\n```python\nx ', '= 1\nprint(x)\n```\n\ntail']:
        tr.event(dict(text=chunk))
    tr.done()
    scr = tty.term.contents()
    for s in ('# Heading', 'para text', 'x = 1', 'print(x)', 'tail'):
        assert scr.count(s) == 1, (s, scr)
    tags = [b.tag for b in app.comp.blocks.values()]
    assert tags.count('ai') == 4

def test_tall_fence_folds():
    "A fence taller than the threshold collapses at finalize; the partial folds while still streaming."
    tty = EmuTty(60, 12)
    app = App(tty, history=None)
    app.paint()
    tr = TurnRenderer(app.comp, _gutter, collapse_at=4)
    tr.event('```python\n' + '\n'.join(f'line{i} = {i}' for i in range(12)))
    partial = [b for b in app.comp.blocks.values()][-1]
    assert partial.collapsed  # folded mid-stream: a giant fence cannot flood the screen
    tr.event('\n```\n\ndone')
    tr.done()
    blocks = list(app.comp.blocks.values())
    assert blocks[0].collapsed and blocks[0].height == 12
    assert '… (+' in tty.term.text()

def test_turn_events_interleave():
    "fastllm items: thinking dicts stream dim then fold, tool Parts print collapsed, text resumes the md flow."
    tty = EmuTty(60, 16)
    app = App(tty, history=None)
    app.paint()
    tr = TurnRenderer(app.comp, _gutter, collapse_at=5)
    for e in [dict(thinking='hmm\nlines'), dict(thinking='\nof thought'), dict(text='Check the value.\n\n'),
              tool_use('py', {'code': 'x*2'}), tool_result('py', {'code': 'x*2'}, '42'),
              dict(text='The answer is 42.')]:
        tr.event(e)
    tr.done()
    kinds = [(b.tag, b.collapsed) for b in app.comp.blocks.values()]
    assert kinds == [('think', True), ('ai', False), ('tool', True), ('ai', False)]
    scr = tty.term.text()
    assert "py(code=x*2)" in scr and '42' not in scr.split('answer')[0]  # result folded away

async def test_prompt_flow_and_ctx():
    "The routed prompt flow: ask block, reply blocks, dialog records the turn, ctx rides via dlg2hist."
    tty, app, stub = mk_app(events=[dict(text='The answer.\n')])
    app.paint()
    app.assistant.add_cell('x = 41 + 1', [dict(output_type='stream', name='stdout', text='')])
    app.comp.on_bytes(b'.what is x?\r')
    for _ in range(100):
        await asyncio.sleep(0.02)
        if not app.busy and stub.calls: break
    assert [b.tag for b in app.comp.blocks.values()][:2] == ['ask', 'ai']
    scr = tty.term.text()
    assert 'what is x?' in scr and 'The answer.' in scr
    call = stub.calls[0]
    assert call['model'] == 'm' and call['think'] == 'l'
    sent = _parts_text(call)
    assert 'x = 41 + 1' in sent          # the cell rides in the prompt's user parts (dlg2hist)
    assert call['msg'][-1].endswith('>what is x?</prompt>')  # the aidialog envelope wraps every prompt
    a = app.assistant
    assert [m.msg_type for m in a.dlg.messages] == ['code', 'prompt']
    assert a.last_response == 'The answer.'
    assert a.dlg.messages[-1].ai_output == 'The answer.'
    # second turn: history carries turn one; the cell does not repeat in the new prompt parts
    app.comp.on_bytes(b'.and again?\r')
    for _ in range(100):
        await asyncio.sleep(0.02)
        if len(stub.calls) == 2: break
    call2 = stub.calls[1]
    assert len(call2['hist']) == 2 and 'The answer.' in call2['hist'][1]
    assert 'x = 41 + 1' in '\n'.join(p for p in call2['hist'][0] if isinstance(p, str))
    assert 'x = 41 + 1' not in _parts_text(call2)

async def test_reply_stored_in_fastllm_form():
    "The formatter tee stores the canonical form: a tool round-trip lands as a details block in last_response."
    tty, app, stub = mk_app(events=[dict(text='Look:\n'), tool_use('py', {'code': '1+1'}),
                                    tool_result('py', {'code': '1+1'}, '2'), dict(text='Done.')])
    app.paint()
    app.comp.on_bytes(b'.check\r')
    for _ in range(100):
        await asyncio.sleep(0.02)
        if not app.busy and stub.calls: break
    resp = app.assistant.last_response
    assert '```json {.tool}' in resp and 'Done.' in resp
    from aidialog.msg_parts import fmt2hist
    msgs = fmt2hist(resp)   # the stored form round-trips
    assert any(p.type == PartType.tool_result for m in msgs for p in m.content)

async def test_interrupt_freezes_turn():
    gate = asyncio.Event()
    tty, app, stub = mk_app(events=[dict(text='partial text so far')], gate=gate)
    app.paint()
    app.comp.on_bytes(b'.go\r')
    for _ in range(100):
        await asyncio.sleep(0.02)
        if 'partial text' in tty.term.text(): break
    assert app.busy
    app.on_sigint()   # ctrl-C: cancels the consumer, not the app
    for _ in range(100):
        await asyncio.sleep(0.02)
        if not app.busy: break
    assert 'interrupted' in tty.term.text()
    assert app.assistant.last_response.endswith('*[Response interrupted]*')
    assert app.assistant.dlg.messages[-1].msg_type == 'prompt'   # the interrupted turn still records
    gate.set()

async def test_failed_turn_removes_pending_prompt():
    "A backend error must not leave a pending prompt message tainting the next dlg2hist."
    tty, app, stub = mk_app()
    def boom(**kw): raise RuntimeError('no backend')
    app.assistant._chat_factory = boom
    app.paint()
    app.comp.on_bytes(b'.hi\r')
    for _ in range(100):
        await asyncio.sleep(0.02)
        if not app.busy: break
    assert [m.msg_type for m in app.assistant.dlg.messages] == []

async def test_paste_bindings():
    tty, app, stub = mk_app()
    app.paint()
    app.assistant.last_response = 'Try:\n\n```python\na = 1\n```\n\nthen\n\n```python\nb = 2\n```\n'
    assert code_blocks(app.assistant.last_response) == ['a = 1', 'b = 2']
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
    tty, app, stub = mk_app(events=[dict(text='ok')], prompt_mode=True)
    app.paint()
    assert tty.term.text().splitlines()[-1].startswith('›››')  # the mode shows in the marker (trailing space trimmed by text())
    app.comp.on_bytes(b'hello there\r')  # plain Enter submits: English is never incomplete
    for _ in range(100):
        await asyncio.sleep(0.02)
        if stub.calls: break
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
    a = app.assistant
    assert a.ctx_usage is None                 # unknown model ('m'): no meter
    a.model = 'codex/gpt-5.5'   # the vendor rides in the model string
    assert a.ctx_usage is None                 # known model but no turn yet: still no meter
    a.last_req_use = UsageStats(prompt_tokens=30000, completion_tokens=2000)
    assert a.ctx_usage == (32000, 256000)


async def test_shell_refs_go_through_kernel():
    "`$` vars and `!`cmd`` refs resolve in ONE eval_exprs round trip; `!` keyed by ref form; NameError -> warning."
    class StubClient:
        def __init__(self): self.asked = []
        async def eval_exprs(self, vs):
            self.asked.append(list(vs))
            def val(e):
                if e.startswith('get_ipython().getoutput'): return 'shell-out'
                return 42 if e == 'x' else '<error type="NameError" desc="nope">\nnope</error>'
            return {e: val(e) for e in vs}
    class StubBridge:
        def __init__(self): self.client = StubClient()
    tty, app, stub = mk_app()
    a = app.assistant
    a.bridge = StubBridge()
    a.dlg.mk_message('use $`x` and $`nosuch` and !`echo hi`', msg_type='prompt')
    vh, warn = await a._vars_turn()
    body = to_xml(vh[0][0]) if vh else ''
    assert 'x' in body and '42' in body
    assert '!`echo hi`' in body and 'shell-out' in body
    assert 'nosuch' not in body and 'nosuch' in (warn or '')          # undefined: warned, not rendered
    assert len(a.bridge.client.asked) == 1                            # one round trip for everything
    assert any('getoutput' in e for e in a.bridge.client.asked[0])    # ! ran via the kernel, not subprocess

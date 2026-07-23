"cp3: routing, streaming reply rendering, the assistant end-to-end over a FakeBackend, paste bindings, ghost text."
import asyncio
from rich.text import Text
from teleprint.testing import FakeTty
from ipyai.cli import App, _gutter
from ipyai.reply import TurnRenderer
from ipyai.assistant import Assistant, route, code_blocks
from ipyai.backend_common import PreparedTurn, CommonStreamFormatter, TextResponse

def mk_cfg(**kw):
    return dict(_backend_name='codex-api', model='m', completion_model='cm', think='l',
                code_theme='ansi_dark', prompt_mode=False) | kw

class FakeBackend:
    "Scripted stream events in place of a live LLM; records every prepare_turn call for assertions."
    formatter_cls = CommonStreamFormatter
    def __init__(self, events=(), completion='', gate=None):
        self.events, self.completion, self.gate = list(events), completion, gate
        self.calls = []

    async def prepare_turn(self, *, prompt, model, think='l', provider_session_id=None, seed=None, tool_mode='on', ephemeral=False):
        self.calls.append(dict(prompt=prompt, seed=seed, model=model, provider_session_id=provider_session_id))
        events, gate = self.events, self.gate
        async def gen():
            for e in events:
                await asyncio.sleep(0)
                yield e
            if gate is not None: await gate.wait()
        return PreparedTurn(stream=gen(), _state={'provider_session_id': 'fake-123'})

    async def complete(self, prompt, *, model): return TextResponse(self.completion)

def mk_app(events=(), completion='', gate=None, prompt_mode=False):
    tty = FakeTty(64, 16)
    fake = FakeBackend(events, completion, gate)
    app = App(tty, history=None, assistant=Assistant(cfg=mk_cfg(), backend=fake, system_prompt='sp'))
    app.prompt_mode = prompt_mode
    return tty, app, fake

def test_route():
    assert route('x = 1', False) == ('code', 'x = 1')
    assert route('.what is x?', False) == ('prompt', 'what is x?')
    assert route('what is x?', True) == ('prompt', 'what is x?')
    assert route(';x = 1', True) == ('code', 'x = 1')
    assert route('  ;x = 1', True) == ('code', '  x = 1')
    assert route('!ls', True) == ('code', '!ls')
    assert route('%time 1', True) == ('code', '%time 1')
    assert route('.still a prompt', True) == ('prompt', '.still a prompt')

def test_streaming_reply_blocks():
    "Spans close incrementally: each top-level md block becomes its own block; the partial is replaced, not duplicated."
    tty = FakeTty(60, 14)
    app = App(tty, history=None)
    app.paint()
    tr = TurnRenderer(app.comp, _gutter, collapse_at=5)
    for chunk in ['# He', 'ading\n\npara ', 'text\n\n```python\nx ', '= 1\nprint(x)\n```\n\ntail']:
        tr.event(chunk)
    tr.done()
    scr = tty.term.contents()
    for s in ('# Heading', 'para text', 'x = 1', 'print(x)', 'tail'):
        assert scr.count(s) == 1, (s, scr)
    tags = [b.tag for b in app.comp.blocks.values()]
    assert tags.count('ai') == 4

def test_tall_fence_folds():
    "A fence taller than the threshold collapses at finalize; the partial folds while still streaming."
    tty = FakeTty(60, 12)
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
    tty = FakeTty(60, 16)
    app = App(tty, history=None)
    app.paint()
    tr = TurnRenderer(app.comp, _gutter, collapse_at=5)
    for e in [{'kind': 'thinking_start'}, {'kind': 'thinking_delta', 'delta': 'hmm\nlines\nof thought'},
              {'kind': 'thinking_end'}, 'Check the value.\n\n',
              {'kind': 'tool_start', 'name': 'python', 'input': {'code': 'x*2'}},
              {'kind': 'tool_complete', 'name': 'python', 'input': {'code': 'x*2'}, 'content': '42'},
              'The answer is 42.']:
        tr.event(e)
    tr.done()
    kinds = [(b.tag, b.collapsed) for b in app.comp.blocks.values()]
    assert kinds == [('think', True), ('ai', False), ('tool', True), ('ai', False)]
    scr = tty.term.text()
    assert "python(code='x*2')" in scr and '42' not in scr.split('answer')[0]  # result folded away

def test_prompt_flow_and_context():
    "The routed prompt flow: ask block, reply blocks, turn recorded, context carries the cell record."
    async def go():
        tty, app, fake = mk_app(events=['The answer.\n'])
        app.paint()
        app.assistant.add_cell('x = 41 + 1', [('stream', {'name': 'stdout', 'text': ''})])
        app.comp.on_bytes(b'.what is x?\r')
        for _ in range(100):
            await asyncio.sleep(0.02)
            if not app.busy and fake.calls: break
        assert [b.tag for b in app.comp.blocks.values()][:2] == ['ask', 'ai']
        scr = tty.term.text()
        assert 'what is x?' in scr and 'The answer.' in scr
        full = fake.calls[0]['prompt']
        assert '<code>x = 41 + 1</code>' in full and '<user-request>what is x?</user-request>' in full
        a = app.assistant
        assert len(a.turns) == 1 and a.turns[0].response == 'The answer.\n'
        assert a.provider_session_id == 'fake-123'
        assert a.last_response == 'The answer.\n'
        # second turn: seed carries the first, context resets past it
        app.comp.on_bytes(b'.and again?\r')
        for _ in range(100):
            await asyncio.sleep(0.02)
            if len(fake.calls) == 2: break
        seed = fake.calls[1]['seed']
        assert len(seed.turns) == 1 and seed.turns[0].prompt == 'what is x?'
        assert '<code>' not in fake.calls[1]['prompt']  # the cell already rode in turn one
    asyncio.run(go())

def test_interrupt_freezes_turn():
    async def go():
        gate = asyncio.Event()
        tty, app, fake = mk_app(events=['partial text so far'], gate=gate)
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
        assert app.assistant.turns[0].response.endswith('<system>user interrupted</system>')
        gate.set()
    asyncio.run(go())

def test_paste_bindings():
    async def go():
        tty, app, fake = mk_app()
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
    asyncio.run(go())

def test_ai_ghost_text():
    async def go():
        tty, app, fake = mk_app(completion='(reverse=True)')
        app.paint()
        app.comp.on_bytes(b'xs.sort')
        app.comp.on_bytes(b'\x1b.')          # alt-.: explicit AI completion
        for _ in range(100):
            await asyncio.sleep(0.02)
            if app.ai_sugg: break
        assert app.ai_sugg[2] == '(reverse=True)'
        assert '(reverse=True)' in tty.term.text()
        app.comp.on_bytes(b'(')              # document changed: the suggestion is stale and gone
        assert app.buf.suggestion == ''
    asyncio.run(go())

def test_prompt_mode_ui():
    async def go():
        tty, app, fake = mk_app(events=['ok'], prompt_mode=True)
        app.paint()
        assert tty.term.text().splitlines()[-1].startswith('ai>')  # the mode shows in the marker (trailing space trimmed by text())
        app.comp.on_bytes(b'hello there\r')  # plain Enter submits: English is never incomplete
        for _ in range(100):
            await asyncio.sleep(0.02)
            if fake.calls: break
        assert '<user-request>hello there</user-request>' in fake.calls[0]['prompt']
        app.comp.on_bytes(b'\x1bp')          # alt-p back to code mode
        assert not app.prompt_mode
        assert tty.term.text().splitlines()[-1].startswith('>>>')
    asyncio.run(go())

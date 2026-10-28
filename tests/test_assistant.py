import asyncio
from aidialog.msg_parts import Text, Thinking, ToolUse, ToolResult
from aidialog.dialog import INTERRUPTED
from aidialog.ipynb import read_ipynb
from ipyai.history import History
from .helpers import keys, send, until

async def test_conversation_context_and_live_view(conversation):
    app, model = conversation
    app.hist = History()
    await send(app, 'x = 42')
    gate = asyncio.Event()
    fence = '```python\n' + '\n'.join(f'line{i} = {i}' for i in range(12)) + '\n```'
    reply = '# Heading\n\n' + fence + '\n\nThe answer is 42.\n\n```python\nx*2\n```'
    model.queue(reply, [Thinking('hmm'), Text('# Head'), Text('ing\n\n'), Text(fence), Text('\n\nThe answer is 42.')], gate)
    app.comp.on_bytes(b'.explain $`x` and $`nosuch`\r')
    await until(lambda: 'The answer is 42.' in app.tty.term.text())
    m = app.ctl.dlg.messages[-1]
    assert app.busy and any(b.collapsed and 'line11' in b.source for b in app.comp.blocks.values())
    gate.set()
    await until(lambda: not app.busy)
    assert app.tty.term.contents().count('# Heading') == 1 and '🧠' not in app.tty.term.text()
    assert m.ai_res == reply and any(b.collapsed and 'line11' in b.source for b in app.comp.blocks.values())
    request = str(model.requests[-1])
    assert '42' in request and 'undefined' in request and 'nosuch' in request

    model.queue('It is still 42.')
    await send(app, '.and again?')
    request = model.requests[-1]
    assert 'x = 42' in str(request['hist']) and 'The answer is 42.' in str(request['hist'])
    assert 'x = 42' not in str(request['msg'])
    app.comp.on_bytes(b'\x1bc\x1b[1;3A')  # code-mode history comes from the saved session, not prompts
    assert app.buf.text == 'x = 42'
    app.comp.on_bytes(b'\x15x = ')
    assert app.buf.suggestion == '42'
    app.comp.on_bytes(b'\x1b[C')
    assert app.buf.text == 'x = 42'
    model.queue('(reverse=True)')
    app.comp.on_bytes(b'\x15xs.sort\x1b.')
    await until(lambda: app.ai_sugg is not None)
    assert '(reverse=True)' in app.tty.term.text()
    app.comp.on_bytes(b'(')
    assert app.buf.suggestion == ''

async def test_interrupted_turn_retry_and_edit(conversation):
    app, model = conversation
    model.queue(events=[], gate=asyncio.Event())
    app.comp.on_bytes(b'.never mind\r')
    await until(lambda: model.requests)
    await keys(app, b'\x03')
    assert app.ctl.dlg.messages == [] and app.buf.text == 'never mind' and app.mode == 'prompt'
    model.queue(events=[RuntimeError('no backend')])
    await keys(app, b'\r')
    assert app.ctl.dlg.messages == [] and app.buf.text == 'never mind' and 'no backend' in app.tty.term.text()

    gate = asyncio.Event()
    model.queue(events=[Text('partial text\n\n'), ToolUse(id='t1', name='py', arguments={'code': 'x*2'}),
        ToolResult(id='t1', name='py', arguments={'code': 'x*2'}, text='84')], gate=gate)
    app.comp.on_bytes(b'\x15continue\r')
    await until(lambda: 'partial text' in app.tty.term.text() and '⏳' in app.tty.term.text())  # a running tool's row
    old = app.ctl.dlg.messages[-1]
    await keys(app, b'\x03')
    assert old.ai_res.endswith(INTERRUPTED) and any(b.collapsed and 'x*2' in b.source for b in app.comp.blocks.values())

    model.queue('Try:\n\n```python\na = 1\n```\n\n```python\nb = 2\n```')
    await keys(app, b'\x1br\x15continue again\r')
    m = app.ctl.dlg.messages[-1]
    assert len(app.ctl.dlg.messages) == 1 and m.content == 'continue again'
    assert not any('partial text' in b.source for b in app.comp.blocks.values())
    app.comp.on_bytes(b'\x1bW')
    assert app.buf.text == 'a = 1\nb = 2'
    app.comp.on_bytes(b'\x15\x14e\x15fixed\r')
    await asyncio.sleep(0)
    assert m.ai_res == 'fixed' and 'fixed' in app.tty.term.text()
    app.comp.on_bytes(b'h')
    await asyncio.sleep(0)
    hidden = [b for b in app.comp.blocks.values() if b.source in (m.content, m.ai_res)]
    assert m.skipped and hidden and all(b.dim for b in hidden)
    app.ctl.edit(m, content='old $`secret`')
    app.comp.on_bytes(b'\x14')
    model.queue('ok')
    await send(app, 'new question')
    assert 'secret' not in str(model.requests[-1])
    saved = read_ipynb(app.ctl.dlg.path_)
    assert saved.messages[0].skipped and saved.messages[0].ai_res == 'fixed'

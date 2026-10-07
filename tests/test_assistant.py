import asyncio
from aidialog.msg_parts import Text, Thinking, ToolUse, ToolResult
from aidialog.dialog import INTERRUPTED
from aidialog.ipynb import read_ipynb
from ipyai.history import History
from .helpers import send, until

async def test_conversation_context_and_live_view(conversation):
    app, model = conversation
    app.hist = History()
    await send(app, 'x = 42')
    gate = asyncio.Event()
    fence = '```python\n' + '\n'.join(f'line{i} = {i}' for i in range(12)) + '\n```'
    reply = '# Heading\n\n' + fence + '\n\nThe answer is 42.\n\n```python\nx*2\n```'
    model.queue(reply, [Thinking('hmm'), Text('# Head'), Text('ing\n\n'), Text(fence), Text('\n\nThe answer is 42.')], gate)
    app.comp.on_bytes(b'.explain $`x` and $`nosuch` and !`echo shell-out`\r')
    await until(lambda: 'The answer is 42.' in app.tty.term.text())
    m = app.ctl.dlg.messages[-1]
    assert app.busy and any(b.collapsed and 'line11' in b.source for b in app.comp.blocks.values())
    gate.set()
    await until(lambda: not app.busy)
    assert app.tty.term.contents().count('# Heading') == 1 and '🧠' not in app.tty.term.text()
    assert app.comp.blocks[f'{m.id}:r1'].collapsed and m.ai_res == reply
    request = str(model.requests[-1])
    assert '42' in request and 'shell-out' in request and "aren't defined" in request and 'nosuch' in request

    model.queue('It is still 42.')
    await send(app, '.and again?')
    request = model.requests[-1]
    assert 'x = 42' in str(request['hist']) and reply in request['hist']
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

async def test_interrupted_turn_retry_and_edit(conversation, tmp_path):
    app, model = conversation
    model.queue(events=[], gate=asyncio.Event())
    app.comp.on_bytes(b'.never mind\r')
    await until(lambda: model.requests)
    app.comp.on_bytes(b'\x03')
    await until(lambda: not app.busy)
    assert app.ctl.dlg.messages == [] and app.buf.text == 'never mind' and app.mode == 'prompt'
    model.queue(events=[RuntimeError('no backend')])
    app.comp.on_bytes(b'\r')
    await until(lambda: len(model.requests) == 2 and not app.busy)
    assert app.ctl.dlg.messages == [] and app.buf.text == 'never mind' and 'no backend' in app.tty.term.text()

    gate = asyncio.Event()
    model.queue(events=[Text('partial text\n\n'), ToolUse(id='t1', name='py', arguments={'code': 'x*2'}),
        ToolResult(id='t1', name='py', arguments={'code': 'x*2'}, text='84')], gate=gate)
    app.comp.on_bytes(b'\x15continue\r')
    await until(lambda: 'partial text' in app.tty.term.text() and any(':tool:t1' in k for k in app.comp.blocks))
    old = app.ctl.dlg.messages[-1]
    assert app.comp.blocks[f'{old.id}:tool:t1'].collapsed
    app.comp.on_bytes(b'\x03')
    await until(lambda: not app.busy)
    assert old.ai_res.endswith(INTERRUPTED)

    model.queue('Try:\n\n```python\na = 1\n```\n\n```python\nb = 2\n```')
    app.comp.on_bytes(b'\x1br\x15continue again\r')
    await until(lambda: len(model.requests) == 4 and not app.busy)
    m = app.ctl.dlg.messages[-1]
    assert len(app.ctl.dlg.messages) == 1 and m.content == 'continue again'
    assert not any(k.startswith(old.id) for k in app.comp.blocks)
    app.comp.on_bytes(b'\x1bW')
    assert app.buf.text == 'a = 1\nb = 2'
    app.comp.on_bytes(b'\x15\x14e\x15fixed\r')
    await asyncio.sleep(0)
    assert m.ai_res == 'fixed' and 'fixed' in app.comp.blocks[f'{m.id}:r0'].source
    app.comp.on_bytes(b'h')
    await asyncio.sleep(0)
    assert m.skipped and all(app.comp.blocks[k].dim for k in app.view.keys[m.id])
    app.ctl.edit(m, content='old $`secret` and !`touch forbidden`')
    app.comp.on_bytes(b'\x14')
    model.queue('ok')
    await send(app, 'new question')
    assert 'secret' not in str(model.requests[-1]) and not (tmp_path/'forbidden').exists()
    saved = read_ipynb(app.ctl.dlg.path_)
    assert saved.messages[0].skipped and saved.messages[0].ai_res == 'fixed'

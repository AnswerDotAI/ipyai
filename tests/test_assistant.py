import json, re, httpx, pytest
from threading import Event, Thread
from wsgiref.simple_server import make_server

STEPS = [f'step-{i:02}' for i in range(16)]

@pytest.fixture(scope='module')
def model_gate():
    arrived, release = Event(), Event()
    release.set()
    return arrived, release

@pytest.fixture(scope='module')
def model_url(model_gate):
    "Ask for a multiplier through a kernel tool, then return the result the kernel supplied."
    def respond(env, start):
        if env['PATH_INFO'] == '/models':
            start('200 OK', [('Content-Type', 'application/json')])
            return [b'{"defaults":{"standard":"local"},"data":[{"id":"local","max_input_tokens":16000}]}']
        if env['PATH_INFO'] == '/fim':
            start('200 OK', [('Content-Type', 'application/json')])
            return [b'{"completions":["-81"]}']
        last = json.loads(env['wsgi.input'].read(int(env['CONTENT_LENGTH'])))['input'][-1]
        arrived, release = model_gate
        if not release.is_set():
            arrived.set()
            release.wait(10)
        result = last.get('output')
        text = f'The answer is {result}.' if result is not None else '\n\n'.join(f'{step} verifying native scrollback across a resize' for step in STEPS)
        events = [dict(type='response.output_text.delta', delta=text)]
        if result is None:
            call = dict(type='function_call', call_id='multiply', name='exec', arguments=json.dumps(dict(code="6 * int(input('Multiplier? '))")))
            events.append(dict(type='response.output_item.done', item=call))
        events.append(dict(type='response.completed', response={'messages':[{'role':'assistant','content':text}],
            'usage':{'input_tokens':2400 if result is not None else 1600}}))
        start('200 OK', [('Content-Type', 'text/event-stream')])
        return [f'data: {json.dumps(e)}\n\n'.encode() for e in events]

    with make_server('127.0.0.1', 0, respond) as server:
        Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
        try: yield f'http://127.0.0.1:{server.server_port}'
        finally: server.shutdown()

def test_conversation_context_and_live_view(terminal, gateway, tmp_path):
    session, line, screen = terminal
    pane = session.pane
    screen('»»»')
    pane.send_keys('abs()', 'Left', 'M-.')
    screen(r'(?m)^»»» abs\(-81\)\s*\Z')
    pane.send_keys('Right', 'Enter')
    screen(r'(?m)\b81$')
    pane.send('abs(-')
    screen(r'(?m)^»»» abs\(-81\)\s*\Z')
    pane.send_keys('End', '*2', 'Enter')
    screen(r'(?m)\b162$')
    fence = '\n'.join(f'line{i} = {i}' for i in range(14))
    tool = json.dumps(dict(id='t1', name='exec', args={'code':'6 * 7'}, result='42'))
    reply = f'# Example\n\n```python\n{fence}\n```\n\n```json {{.tool}}\n{tool}\n```\n\n```python\na = 1\n```\n\n```python\nb = 2\n```'
    with httpx.Client(base_url=gateway) as http:
        kernel, = http.get('/api/kernels').json()
        url = f"/api/messages?kid={kernel['id']}"
        added = http.post(url, json=dict(op='add', msgs=[dict(msg_type='prompt', content='Show me an example.')])).raise_for_status()
        cid, = added.json()['added_ids']
        http.post(url, json=dict(op='update', msgs=[dict(id=cid, msg=dict(output=reply))])).raise_for_status()

    view = screen(r'b = 2')
    assert 'line0 = 0' in view and 'line13 = 13' not in view and 'exec(' in view
    pane.send_keys('M-W')
    view = screen(r'(?m)^··· b = 2$')
    assert '··· a = 1' in view.splitlines()
    pane.send_keys('C-c', 'M-@')
    screen(r'(?m)^»»» a = 1\s*\Z')
    pane.send_keys('M-S-Down')
    screen(r'(?m)^»»» b = 2\s*\Z')
    pane.send_keys('C-c', 'C-t', 'e', 'C-u')
    line('Corrected reply.')
    screen('Corrected reply')
    pane.send_keys('h', 'S-Up', 'e')
    screen(r'(?m)^››› Show me an example\.\s*\Z')
    pane.send_keys('C-u')
    line('Show this to the next reader.')
    screen(r'(?s)Show this to the next reader\..*Corrected reply')
    pane.send_keys('C-t', 'C-d')
    assert pane.wait(5000) == 0

    code, doubled, cell = json.loads((tmp_path/'keys.ipynb').read_text())['cells']
    assert [''.join(c['source']) for c in (code, doubled)] == ['abs(-81)', 'abs(-81)*2']
    assert cell['id'] == cid and ''.join(cell['source']).endswith('Show this to the next reader.')
    assert cell['metadata']['skipped'] and ''.join(cell['outputs'][0]['data']['text/markdown']) == 'Corrected reply.'

def test_typed_prompt_streams_through_tool_and_resize(terminal, tmp_path, model_gate):
    session, line, screen = terminal
    pane = session.pane
    screen('»»»')
    line('  .Multiply six by my number.')
    screen(r'(?m)^Multiplier\? *$')
    assert 'step-15' in pane.capture().text
    session.windows[0].resize(width=36, height=10)
    screen(r'(?m)step-15 verifying native *$')
    line('7')
    screen(r'(?s)The answer is 42\..*Alt-C code')
    session.windows[0].resize(width=80, height=14)
    screen(r'ctx .* \(15%\)')
    arrived, release = model_gate
    release.clear()
    try:
        pane.send_keys('M-p')
        line('.gitignore')
        assert arrived.wait(5)
        pane.send_keys('C-c')
        screen(r'(?s)\[Alt-P prompt\][^\n]*\n››› \.gitignore\s*\Z')
    finally: release.set()
    pane.send_keys('C-u', 'C-d')
    assert pane.wait(5000) == 0

    cell, cancelled = json.loads((tmp_path/'keys.ipynb').read_text())['cells']
    assert ''.join(cancelled['source']).endswith('.gitignore')
    assert ''.join(cell['source']).endswith('Multiply six by my number.')
    reply = ''.join(cell['outputs'][0]['data']['text/markdown'])
    assert 'json {.tool}' in reply
    history = pane.capture().text
    for part in [*STEPS, 'The answer is 42.']: assert history.count(part) == reply.count(part) == 1, (part, history, reply)
    prose = reply.partition('```')[0]
    unguttered = re.sub(r'(?m)^(?:‹.‹|···) ', '', history)
    assert ''.join(prose.split()) in ''.join(unguttered.split())

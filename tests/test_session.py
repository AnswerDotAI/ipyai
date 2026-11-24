import json, os, subprocess, httpx, pytest
from websockets.sync.client import connect

def test_rustygate_dialog_session(ipyai, gateway, tmp_path):
    "An idle owner stops its process, a borrower leaves it running, and neither destroys another profile's terminal."
    def submit(lines):
        return subprocess.run([ipyai, '--gateway', gateway, 'work.ipynb'], input=lines, text=True, capture_output=True, timeout=20)
    def cells(): return json.loads((tmp_path/'work.ipynb').read_text())['cells']

    with httpx.Client(base_url=gateway) as http:
        peer = {'x-api-token':'another-user:test-signature'}
        http.post('/api/terminals', headers=peer, json={'name':'external','path':'work.ipynb'}).raise_for_status()
        run = submit('x = 6\nx * 7\n')
        assert run.returncode == 0 and run.stdout.splitlines() == ['42'], run.stderr
        assign, result = cells()
        assert result['outputs'][0]['data']['text/plain']=='42'

        previous = http.get('/api/kernels').json()
        assert all(k['execution_state']=='dead' for k in previous)
        opened = http.post('/api/kernels', json={'path':'work.ipynb'}).raise_for_status()
        kid = opened.json()['id']
        assert opened.status_code==201 and all(k['id']!=kid for k in previous)
        assert http.get('/api/terminals/external', headers=peer).raise_for_status().json()['alive']
        run = submit(f":run {assign['id']}\nx += 1\n1/0\nx * 7\n")
        assert run.returncode != 0 and 'ZeroDivisionError' in run.stdout and run.stdout.splitlines()[-1] == '49'
        assert http.get(f'/api/kernels/{kid}').is_success
        run = submit("from pathlib import Path; _ = Path('work.ipynb').write_text('unfinished external edit'); x\n")
        assert run.returncode == 0 and run.stdout.splitlines() == ['7'], run.stderr
        conflict, = tmp_path.glob('work.conflict-*.ipynb')
        assert conflict.read_text() == 'unfinished external edit'
        assert conflict.name in run.stderr, run.stderr

    saved = cells()
    assert [c['id'] for c in saved[:2]] == [assign['id'], result['id']]
    assert len(saved) == 6  # Running the saved assignment didn't append another copy.


def test_reconnect_preserves_running_cell_and_peer(terminal, gateway, tmp_path):
    session, line, screen = terminal
    screen('»»»')
    gate = tmp_path/'release'
    os.mkfifo(gate)
    with httpx.Client(base_url=gateway) as http:
        kernel, = http.get('/api/kernels').raise_for_status().json()
        endpoint = '/api/kernels/' + kernel['id']
        channels = gateway.replace('http://', 'ws://')+endpoint+'/channels?session_id='
        with connect(channels+'external') as peer, gate.open('r+b', buffering=0) as signal:
            line("runs = globals().get('runs', 0) + 1; print('started', runs, flush=True); open('release', 'rb', buffering=0).read(1)")
            screen(r'(?m)\bstarted 1$')
            while (msg := json.loads(peer.recv(timeout=5)))['header']['msg_type'] != 'execute_input': pass
            # Attaching with the same session ID closes only ipyai's socket, not its kernel or the observer.
            with connect(channels+msg['parent_header']['session']): screen('Reconnected')
            signal.write(b'!!')  # A replay can finish too; only the counter distinguishes it.
            screen(r"(?ms)b'!'$.*?^ +\[Alt-C code\]")
            line("f'execution count: {runs}'")
            screen(r"(?m)execution count: \d+'$")
            session.pane.send_keys('C-d')
            assert session.pane.wait(5000) == 0
            assert http.get(endpoint).raise_for_status().json()['connections'] == 1
    first, counter = json.loads((tmp_path/'keys.ipynb').read_text())['cells']
    assert first['outputs'][-1]['data']['text/plain'] == "b'!'"
    assert counter['outputs'][0]['data']['text/plain'] == "'execution count: 1'"


@pytest.mark.parametrize('gateway_token', ['session-test-token'])
def test_authenticated_scratch_notebooks(ipyai, gateway, gateway_token, tmp_path):
    project = tmp_path/'project'
    project.mkdir()
    (project/'answer.txt').write_text('from the project')
    token = tmp_path/'gateway.token'
    token.write_text(gateway_token+'\n')
    assert httpx.get(gateway+'/api/kernels').status_code in (401, 403)

    def submit(source):
        run = subprocess.run([ipyai, '--gateway', gateway, '--token-file', token], cwd=project,
                             input=source+'\n', text=True, capture_output=True, timeout=20)
        assert run.returncode == 0, run.stderr
        return run.stdout.strip()

    assert submit("print(open('answer.txt').read())") == 'from the project'
    scratch = tmp_path/'.ipyai'
    first, = scratch.glob('*.ipynb')
    before = first.read_bytes()
    assert submit(r"get_ipython().display_pub.publish({'text/plain':'binary preview'}, buffers=[b'\xff\x00'])") == 'binary preview'
    second, = set(scratch.glob('*.ipynb')) - {first}
    result, = json.loads(second.read_text())['cells']
    assert result['outputs'][0]['data']['text/plain'] == 'binary preview'
    assert first.read_bytes() == before

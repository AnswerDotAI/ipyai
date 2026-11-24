import re, pytest
from fastmux.core import new_session
from pathlib import Path
from fastws.cargo import build
from jupyasyncclient import start_gateway

@pytest.fixture(scope="session")
def ipyai():
    artifacts = build(Path(__file__).resolve().parents[1], library=False, binaries=True)
    return next(a['executable'] for a in artifacts if a['target']['name'] == 'ipyai' and a['executable'])

@pytest.fixture
def model_url(): return 'http://127.0.0.1:1'

@pytest.fixture
def gateway_token(): return None

@pytest.fixture
def gateway(tmp_path, model_url, gateway_token):
    # Every test uses loopback and a fake key, never local provider credentials.
    args = ('env', f'HOME={tmp_path}', 'AAI_USER_KEY=ipyai:test', f'SOLVELP_URL={model_url}', f'IPYTHONDIR={tmp_path}/ipython',
        'rustygate', '--root', str(tmp_path), '--settings-file', str(tmp_path/'settings.json'))
    gateway = start_gateway(args, token=gateway_token)
    try: yield gateway.url
    finally: gateway.stop()

@pytest.fixture
def terminal(ipyai, gateway, tmp_path):
    session = new_session(cmd=[ipyai, '--gateway', gateway, 'keys.ipynb'], cwd=tmp_path, env={'EDITOR':'ed'}, width=70, height=14, remain=True)
    pane = session.pane
    def screen(pattern):
        view = pane.poll(wait_ms=5000, until=pattern, screen=True).text
        assert re.search(pattern, view), view
        return view
    def line(text):
        pane.send(text)
        pane.send_keys('Enter')
    try: yield session, line, screen
    finally: session.kill()

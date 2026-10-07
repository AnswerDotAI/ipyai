import asyncio, os, tempfile, shutil
import pytest, pytest_asyncio
from teleprint.testing import EmuTty
from ipyai.cli import App
from ipyai.kernel import KernelSession
from ipyai.assistant import Assistant
import ipyai.config as config
from .helpers import ScriptedModel

def pytest_configure(config):
    config.ipyai_ipython_dir = tempfile.mkdtemp(prefix='ipyai-test-ipy-')
    config.ipyai_old_ipython_dir = os.environ.get('IPYTHONDIR')
    os.environ['IPYTHONDIR'] = config.ipyai_ipython_dir

def pytest_unconfigure(config):
    shutil.rmtree(config.ipyai_ipython_dir)
    if config.ipyai_old_ipython_dir is None: os.environ.pop('IPYTHONDIR', None)
    else: os.environ['IPYTHONDIR'] = config.ipyai_old_ipython_dir

@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    cfg = tmp_path/'config'
    monkeypatch.setattr(config, 'CONFIG_DIR', cfg)
    for name, filename in [('CONFIG_PATH', 'config.json'), ('SYSP_PATH', 'sysp.txt'), ('STARTUP_PATH', 'startup.py')]:
        monkeypatch.setattr(config, name, cfg/filename)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('IPYAI_MODEL', raising=False)

@pytest.fixture(scope='session')
def gateway():
    from rustygate.tools import start_gateway
    g = start_gateway()
    try: yield g.url
    finally: g.stop()

@pytest_asyncio.fixture
async def repl(gateway):
    with EmuTty(60, 16, bg=(0xfa, 0xfa, 0xf4)) as tty:
        tty.write(b'$ ipyai\r\n')
        async with KernelSession(url=gateway) as k:
            await k.setup()
            app = App(tty, kernel=k, history=None)
            await app.comp.start()
            app.comp.on_resize = app._resized
            app.paint()
            try: yield app
            finally:
                if op := app.ctl.op:
                    app.ctl.cancel()
                    await asyncio.gather(op.task, return_exceptions=True)
                await app.ctl.close()
                await asyncio.sleep(0)
                app.comp.stop()

@pytest.fixture
def conversation(repl):
    model = ScriptedModel()
    cfg = dict(model='m', suggest_model='cm', think='l', code_theme='ansi_dark', prompt_mode=False)
    repl.ctl.assistant = Assistant(cfg=cfg, chat_factory=model, sp='sp')
    return repl, model

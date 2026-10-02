import os, tempfile

import pytest, pytest_asyncio

import ipyai.config as config


_IPYTHONDIR_SESSION = None


def pytest_configure(config):
    "Redirect IPYTHONDIR for the whole test session so no test run pollutes the user's real ~/.ipython."
    global _IPYTHONDIR_SESSION
    _IPYTHONDIR_SESSION = tempfile.mkdtemp(prefix="ipyai-test-ipy-")
    os.environ["IPYTHONDIR"] = _IPYTHONDIR_SESSION


def pytest_unconfigure(config):
    import shutil
    if _IPYTHONDIR_SESSION: shutil.rmtree(_IPYTHONDIR_SESSION, ignore_errors=True)


@pytest.fixture(autouse=True)
def temp_config_paths(tmp_path, monkeypatch):
    "Isolate tests from the user's setup: their config files and `IPYAI_MODEL` never apply, and each test runs in its own directory, so no session files are left behind."
    cfg = tmp_path/"config"
    cfg.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "CONFIG_DIR", cfg)
    monkeypatch.setattr(config, "CONFIG_PATH", cfg/"config.json")
    monkeypatch.setattr(config, "SYSP_PATH", cfg/"sysp.txt")
    monkeypatch.setattr(config, "STARTUP_PATH", cfg/"startup.py")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('IPYAI_MODEL', raising=False)
    yield


_USER_NAMES = "[k for k in globals() if not k.startswith('_')]"


@pytest.fixture(scope="session")
def gateway():
    "A rustygate subprocess for the whole test session (the jupyasyncclient test pattern)."
    from rustygate.tools import start_gateway
    g = start_gateway()   # free port per xdist worker
    yield g.url
    g.stop()


@pytest.fixture(scope="session", autouse=True)
def _gateway_env(gateway):
    "Point every bare KernelSession() at the test gateway: no test may ever touch a live gateway."
    os.environ['IPYAI_GATEWAY'] = gateway
    yield
    os.environ.pop('IPYAI_GATEWAY', None)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def session_kernel(gateway):
    "One kernel, set up as the app sets up an owned kernel, for the whole session, on pytest-asyncio's session loop."
    from ipyai.kernel import KernelSession
    ks = await KernelSession(url=gateway).start()
    await ks.setup()
    baseline = set(await ks.kc.eval(_USER_NAMES, call_=False) or [])
    yield dict(ks=ks, baseline=baseline)
    await ks.close()


@pytest_asyncio.fixture(loop_scope="session")
async def kernel_tools(session_kernel):
    "`KernelTools` over the session kernel; teardown clears any user_ns names the test added."
    from ipyai.tools import KernelTools
    ks = session_kernel["ks"]
    yield KernelTools(ks.kc)
    extras = [k for k in (await ks.kc.eval(_USER_NAMES, call_=False) or []) if k not in session_kernel["baseline"]]
    if extras: await ks.exec("\n".join(f"globals().pop({n!r}, None)" for n in extras))

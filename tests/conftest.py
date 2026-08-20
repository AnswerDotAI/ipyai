import os, tempfile

import pytest, pytest_asyncio

import ipyai.config as config
from ipyai.bridge import setup_tools


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
    "Isolate config/sysp so tests never read or write the user's real XDG ipyai config."
    cfg = tmp_path/"config"
    cfg.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "CONFIG_DIR", cfg)
    monkeypatch.setattr(config, "CONFIG_PATH", cfg/"config.json")
    monkeypatch.setattr(config, "SYSP_PATH", cfg/"sysp.txt")
    monkeypatch.setattr(config, "STARTUP_PATH", cfg/"startup.py")
    yield


async def _prepare_kernel_bridge(client):
    "Seed the test kernel exactly as the app does (`setup_tools`): ipykernel_helper, the `py` tool, the custom tool imports."
    bridge, _ = await setup_tools(client)
    return bridge


async def _snapshot_globals(bridge):
    return set(await bridge.read_var("[k for k in globals() if not k.startswith('_')]") or [])


async def _clear_extras(bridge, baseline):
    extras = await bridge.read_var(
        "[k for k in globals() if not k.startswith('_') and k not in %r]" % list(baseline)) or []
    if extras: await bridge._exec("\n".join(f"globals().pop({n!r}, None)" for n in extras))


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
    "One kernel + bridge for the whole session, on pytest-asyncio's session loop."
    from ipyai.kernel import KernelSession
    ks = await KernelSession(url=gateway).start()
    bridge = await _prepare_kernel_bridge(ks.kc)
    baseline = await _snapshot_globals(bridge)
    yield dict(ks=ks, client=ks.kc, bridge=bridge, baseline=baseline)
    await ks.close()


@pytest_asyncio.fixture(loop_scope="session")
async def kernel_bridge(session_kernel):
    "Session kernel bridge; teardown clears any user_ns names the test added."
    yield session_kernel["bridge"]
    await _clear_extras(session_kernel["bridge"], session_kernel["baseline"])

import asyncio, os, tempfile

import pytest

import ipyai.config as config
from ipyai.kernel_bridge import CUSTOM_TOOL_NAMES, KernelBridge


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
    yield


_KERNEL_BOOTSTRAP = ("from IPython import get_ipython\n"
    "_ip = get_ipython()\n"
    "try: _ip.extension_manager.load_extension('ipykernel_helper.core')\n"
    "except Exception: pass\n"
    "try: _ip.extension_manager.load_extension('safepyrun')\n"
    "except Exception: pass\n")


async def _prepare_kernel_bridge(client):
    bridge = KernelBridge(client)
    await bridge._exec(_KERNEL_BOOTSTRAP)
    present = set(await bridge.present_names(CUSTOM_TOOL_NAMES))
    await bridge.seed_tools(skip=present)
    await bridge.available_names(force=True)
    return bridge


async def _snapshot_globals(bridge):
    exprs = await bridge._exec("", expressions={"_r": "[k for k in globals() if not k.startswith('_')]"})
    return set(exprs.get("_r") or [])


async def _clear_extras(bridge, baseline):
    exprs = await bridge._exec("", expressions={
        "_r": "[k for k in globals() if not k.startswith('_') and k not in %r]" % list(baseline)})
    extras = exprs.get("_r") or []
    if extras: await bridge._exec("\n".join(f"globals().pop({n!r}, None)" for n in extras))


@pytest.fixture(scope="session")
def gateway():
    "An in-thread jupygate for the whole test session (the clikernel test pattern)."
    from jupygate.core import create_app, serve
    server = serve(create_app(), port=0, in_thread=True)   # port 0: a free port per xdist worker
    yield server.url
    server.should_exit = True


@pytest.fixture(scope="session", autouse=True)
def _gateway_env(gateway):
    "Point every bare KernelSession() at the test gateway: no test may ever touch a live jupygate."
    os.environ['IPYAI_GATEWAY'] = gateway
    yield
    os.environ.pop('IPYAI_GATEWAY', None)


@pytest.fixture(scope="session")
def session_kernel(gateway):
    from ipyai.kernel import KernelSession
    loop = asyncio.new_event_loop()

    async def _setup():
        ks = KernelSession(url=gateway)
        await ks.start()
        bridge = await _prepare_kernel_bridge(ks.kc)
        baseline = await _snapshot_globals(bridge)
        return ks, bridge, baseline

    ks,bridge,baseline = loop.run_until_complete(_setup())
    try: yield dict(ks=ks, client=ks.kc, bridge=bridge, baseline=baseline, loop=loop)
    finally:
        try: loop.run_until_complete(ks.close())
        except Exception: pass
        try: loop.close()
        except Exception: pass


@pytest.fixture
def kernel_bridge(session_kernel, request):
    "Session kernel bridge with per-test teardown that clears any user_ns names the test added."
    bridge = session_kernel["bridge"]
    baseline = session_kernel["baseline"]
    loop = session_kernel["loop"]

    def _finalize():
        try: loop.run_until_complete(_clear_extras(bridge, baseline))
        except Exception: pass

    request.addfinalizer(_finalize)
    return bridge


@pytest.fixture
def kernel_loop(session_kernel): return session_kernel["loop"]

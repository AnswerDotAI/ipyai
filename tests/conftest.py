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
    "try: _ip.extension_manager.load_extension('safepyrun')\n"
    "except Exception: pass\n"
    "try: _ip.extension_manager.load_extension('ipythonng')\n"
    "except Exception: pass\n"
    "_ip.history_manager.db_log_output = True\n")


async def _prepare_kernel_bridge(client):
    bridge = KernelBridge(client)
    await bridge._exec(_KERNEL_BOOTSTRAP)
    present = set(await bridge.present_names(CUSTOM_TOOL_NAMES))
    await bridge.seed_tools(skip=present)
    await bridge.available_names(force=True)
    return bridge


async def _snapshot_globals(bridge):
    exprs,_ = await bridge._exec("", expressions={"_r": "[k for k in globals() if not k.startswith('_')]"})
    return set(exprs.get("_r") or [])


async def _clear_extras(bridge, baseline):
    exprs,_ = await bridge._exec("", expressions={
        "_r": "[k for k in globals() if not k.startswith('_') and k not in %r]" % list(baseline)})
    extras = exprs.get("_r") or []
    if extras: await bridge._exec("\n".join(f"globals().pop({n!r}, None)" for n in extras))


@pytest.fixture(scope="session")
def session_kernel():
    from jupyter_client.manager import KernelManager
    from jupyter_client.asynchronous.client import AsyncKernelClient
    km = KernelManager()
    km.start_kernel(extra_arguments=["--HistoryManager.enabled=True"])
    loop = asyncio.new_event_loop()

    async def _setup():
        client = AsyncKernelClient()
        client.load_connection_file(km.connection_file)
        client.start_channels()
        await client.wait_for_ready(timeout=30)
        bridge = await _prepare_kernel_bridge(client)
        baseline = await _snapshot_globals(bridge)
        return client, bridge, baseline

    client,bridge,baseline = loop.run_until_complete(_setup())
    try: yield dict(manager=km, client=client, bridge=bridge, baseline=baseline, loop=loop)
    finally:
        try: loop.run_until_complete(client.stop_channels())
        except Exception:
            try: client.stop_channels()
            except Exception: pass
        try: km.shutdown_kernel(now=False)
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

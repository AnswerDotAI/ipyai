"Separate kernel process: spawn + bootstrap + authoritative tool injection + shutdown."
import asyncio

from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.manager import KernelManager

from ipyai.kernel_bridge import KernelBridge


_BOOTSTRAP = ("from IPython import get_ipython\n"
    "_ip = get_ipython()\n"
    "try: _ip.extension_manager.load_extension('safepyrun')\n"
    "except Exception: pass\n"
    "_ip.history_manager.db_log_output = True\n")


def test_spawn_bootstrap_overwrites_existing_tools_shutdown():
    km = KernelManager()
    km.start_kernel(extra_arguments=["--HistoryManager.enabled=True"])
    loop = asyncio.new_event_loop()

    try:
        async def _go():
            client = AsyncKernelClient()
            client.load_connection_file(km.connection_file)
            client.start_channels()
            await client.wait_for_ready(timeout=30)
            bridge = KernelBridge(client)

            await bridge._exec(_BOOTSTRAP)
            await bridge.inject_tools()
            names = set(await bridge.available_names(force=True))
            assert {"pyrun", "bash"} <= names

            await bridge._exec("def bash(**kw): return 'sentinel-preseeded'")
            res = await bridge.call_tool("bash", {})
            assert "sentinel-preseeded" in res, f"preseeded bash should be active before reinjection; got {res!r}"

            await bridge.inject_tools()
            res = await bridge.call_tool("bash", dict(cmd="printf 'real\\n'", as_dict=True))
            assert "real" in res and "sentinel-preseeded" not in res, f"inject_tools should overwrite preseeded bash; got {res!r}"

            try: await client.stop_channels()
            except Exception: client.stop_channels()

        loop.run_until_complete(_go())
    finally:
        km.shutdown_kernel(now=False)
        try: loop.close()
        except Exception: pass
        assert km.is_alive() is False, "kernel should be shut down"

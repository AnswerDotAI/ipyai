"Separate kernel process: spawn + bootstrap + skip-existing-on-inject + shutdown."
import asyncio

from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.manager import KernelManager

from ipyai.kernel_bridge import CUSTOM_TOOL_NAMES, KernelBridge


_BOOTSTRAP = ("from IPython import get_ipython\n"
    "_ip = get_ipython()\n"
    "try: _ip.extension_manager.load_extension('safepyrun')\n"
    "except Exception: pass\n"
    "_ip.history_manager.db_log_output = True\n")


def test_spawn_bootstrap_skip_inject_shutdown():
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
            present_after_bootstrap = set(await bridge.present_names(CUSTOM_TOOL_NAMES))
            assert "python" in present_after_bootstrap, "safepyrun extension should seed python"

            await bridge._exec("def bash(**kw): return 'sentinel-preseeded'")
            present_with_preseed = set(await bridge.present_names(CUSTOM_TOOL_NAMES))
            assert "bash" in present_with_preseed, "preseeded callable should count as present"

            await bridge.inject_tools(skip=present_with_preseed)

            res = await bridge.call_tool("bash", {})
            assert "sentinel-preseeded" in res, f"inject_tools with skip should have preserved preseeded bash; got {res!r}"

            await bridge._exec("globals().pop('bash', None)")
            await bridge.inject_tools(skip=set(await bridge.present_names(CUSTOM_TOOL_NAMES)))
            names = set(await bridge.available_names(force=True))
            assert "bash" in names, "after removing preseed and re-injecting, real bash should land"

            try: await client.stop_channels()
            except Exception: client.stop_channels()

        loop.run_until_complete(_go())
    finally:
        km.shutdown_kernel(now=False)
        try: loop.close()
        except Exception: pass
        assert km.is_alive() is False, "kernel should be shut down"

"Separate kernel process: attach via --existing-style connection-file; verify pre-attach code is history-readable, post-attach iopub is captureable, and detaching does not shut the kernel down."
import asyncio, sqlite3

from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.manager import KernelManager

from ipyai.kernel_bridge import CUSTOM_TOOL_NAMES, KernelBridge
from ipyai.shell import IPyAIHistory


_BOOTSTRAP_EXTS = ("from IPython import get_ipython\n"
    "_ip = get_ipython()\n"
    "try: _ip.extension_manager.load_extension('safepyrun')\n"
    "except Exception: pass\n"
    "_ip.history_manager.db_log_output = True\n")


def test_attach_existing_kernel_without_shutdown():
    km = KernelManager()
    km.start_kernel(extra_arguments=["--HistoryManager.enabled=True"])
    loop = asyncio.new_event_loop()

    try:
        async def _go():
            primary = AsyncKernelClient()
            primary.load_connection_file(km.connection_file)
            primary.start_channels()
            await primary.wait_for_ready(timeout=30)
            pb = KernelBridge(primary)
            await pb._exec(_BOOTSTRAP_EXTS)
            msg_id = primary.execute("hidden = 'walnut'\nq = 7\nprint('pre-attach')", silent=False, store_history=True)
            end = loop.time() + 10
            while loop.time() < end:
                try: msg = await asyncio.wait_for(primary.get_shell_msg(), timeout=0.5)
                except asyncio.TimeoutError: continue
                if msg["parent_header"].get("msg_id") == msg_id: break

            secondary = AsyncKernelClient()
            secondary.load_connection_file(km.connection_file)
            secondary.start_channels()
            await secondary.wait_for_ready(timeout=30)
            sb = KernelBridge(secondary)

            present = set(await sb.present_names(CUSTOM_TOOL_NAMES))
            assert "python" in present, "secondary client should see python already present (from primary's bootstrap)"

            val = await sb.read_var("hidden")
            assert val == "walnut"
            q = await sb.read_var("q")
            assert q == 7

            msg_id = secondary.history(hist_access_type="range", session=0, start=0, stop=100, output=False)
            reply = None
            end = loop.time() + 5
            while loop.time() < end:
                try: msg = await asyncio.wait_for(secondary.get_shell_msg(), timeout=0.5)
                except asyncio.TimeoutError: continue
                if msg["parent_header"].get("msg_id") == msg_id:
                    reply = msg
                    break
            assert reply is not None, "history_request should get a reply"
            hist_codes = [o[2] for o in (reply["content"].get("history") or [])]
            assert any("hidden" in c for c in hist_codes), f"pre-attach code should be visible in history: {hist_codes}"

            _, out = await sb._exec("print('post-attach')", capture_stream=True)
            assert "post-attach" in out

            try: await secondary.stop_channels()
            except Exception: secondary.stop_channels()
            assert km.is_alive(), "primary kernel must still be alive after secondary detach"

            try: await primary.stop_channels()
            except Exception: primary.stop_channels()

        loop.run_until_complete(_go())
    finally:
        km.shutdown_kernel(now=False)
        try: loop.close()
        except Exception: pass


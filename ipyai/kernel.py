"Kernel lifecycle over a rustygate gateway: jupyasyncclient + ipymini, iopub rendered as it arrives."
import os
from fastcore.utils import first
from jupyasyncclient.multimanager import JupyAsyncMultiKernelManager
from jupyasyncclient import JupyAsyncKernelClient

DEFAULT_URL = 'http://127.0.0.1:8787'   # rustygate's default port; IPYAI_GATEWAY overrides


class KernelSession:
    """One gateway kernel and its ws client; incremental iopub for UIs (messages surface as they
    arrive, not drained at completion). rustygate is a hard runtime prerequisite, like any Jupyter
    server: an unreachable gateway fails loudly at `start` with the command to run."""
    def __init__(self, url=None):
        self.url = url or os.environ.get('IPYAI_GATEWAY', DEFAULT_URL)
        self.mgr, self.kc, self.kid, self.owned, self.busy = None, None, None, False, False
        self.on_comm = None   # host-side comm handler: (msg_type, content) for comm traffic seen on iopub
        self.on_stdin = None  # async (prompt, password) -> str, answering kernel input_requests

    async def start(self, kernel='', cwd=None, env=None):
        """Create an owned kernel (closed on exit), or attach to `kernel` by id prefix (taken as
        found: not seeded, never stopped by us). Owned kernels start in `cwd` (ours if None) with
        `env` laid over our environment; ownership is `connect`'s: it stamps `kc.owned`, honored at close."""
        self.mgr = JupyAsyncMultiKernelManager(self.url)
        try: ks = await self.mgr.list_kernels()   # reachability and auth fail here, loudly
        except Exception as e: raise ConnectionError(f'no rustygate gateway at {self.url} (start one with `rustygate`): {e}') from e
        if kernel:
            kid = first(k['id'] for k in ks if k['id'].startswith(kernel))
            if not kid: raise ValueError(f'no kernel matching {kernel!r} on {self.url}: {[k["id"][:8] for k in ks]}')
            self.kc = await JupyAsyncKernelClient.connect(self.url, kernel=kid)
        else: self.kc = await JupyAsyncKernelClient.connect(self.url, cwd=str(cwd or os.getcwd()), env=dict(os.environ, **(env or {})))
        self.kid, self.owned = self.kc.kernel_id, self.kc.owned
        if self.owned:   # seed the REPL services (sig_help etc.); attached kernels are taken as found
            try: await self.kc.reply("get_ipython().extension_manager.load_extension('ipykernel_helper.core')",
                                     silent=True, store_history=False, timeout=10)
            except Exception: pass
        return self

    async def run(self, code, on_output):
        """Execute `code`, calling `on_output(out)` per nbformat-shaped output as it arrives: a thin
        adapter over `kc.run`, which owns parent-id filtering, reply+idle completion, stdin via
        `on_stdin` (absent means `allow_stdin=False`), comm passthrough via `on_comm`, and
        dead-kernel detection (`DeadKernelError` on silence from a dead kernel)."""
        self.busy = True
        try:
            async for o in self.kc.run(code, on_stdin=self.on_stdin, on_comm=self.on_comm): on_output(o)
        finally: self.busy = False

    async def interrupt(self): await self.mgr.interrupt_kernel(self.kid)

    async def close(self):
        "Close the client (`kc.__aexit__` is the ownership contract: an owned kernel shuts down, an attached one survives)."
        if self.kc is not None: await self.kc.__aexit__()

    async def __aenter__(self): return await self.start()
    async def __aexit__(self, *exc): await self.close()

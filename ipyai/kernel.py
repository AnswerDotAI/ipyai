"""Kernel lifecycle over a rustygate gateway: one owned or attached kernel, its setup, and the one inbound message hook."""
import os
from contextlib import suppress
from jupyasyncclient.multimanager import JupyAsyncMultiKernelManager
from jupyasyncclient import JupyAsyncKernelClient
from . import config
from .tools import KernelTools, SEED_IMPORTS

DEFAULT_URL = 'http://127.0.0.1:8787'   # rustygate's default port; IPYAI_GATEWAY overrides
kernel_env = dict(PYTHONSAFEPATH='1')

def _startup_src(path):
    "The startup file's source wrapped so `__file__` is bound to its path during the run, and absent after (clikernel's shape)"
    return f"""__file__ = {str(path)!r}
try: exec(compile({path.read_text()!r}, __file__, 'exec'))
finally: del __file__"""

class KernelSession:
    """One gateway kernel and its ws client. rustygate is a hard runtime prerequisite, like any Jupyter
    server: an unreachable gateway fails loudly at `start` with the command to run. Every inbound kernel
    message goes once to `on_jmsg`, whatever request it belongs to."""
    def __init__(self, url=None):
        self.url = url or os.environ.get('IPYAI_GATEWAY', DEFAULT_URL)
        self.mgr, self.kc, self.kid, self.owned = None, None, None, False
        self.on_jmsg = None

    async def start(self, kernel='', cwd=None, env=None):
        """Create an owned kernel (closed on exit, with ipykernel_helper's REPL services loaded), or attach to
        `kernel` by id prefix (taken as found: nothing loaded, never stopped by us). An unknown or ambiguous
        prefix raises `ValueError`. Owned kernels start in `cwd` (ours if None) with `env` laid over our environment."""
        self.mgr = JupyAsyncMultiKernelManager(self.url)
        try: ks = await self.mgr.list_kernels()   # reachability and auth fail here, loudly
        except Exception as e: raise ConnectionError(f'no rustygate gateway at {self.url} (start one with `rustygate`): {e}') from e
        if kernel:
            ids = [k['id'] for k in ks if k['id'].startswith(kernel)]
            if len(ids) != 1:
                shown = [i[:8] for i in (ids or [k['id'] for k in ks])]
                raise ValueError(f"{'ambiguous' if ids else 'no'} kernel matching {kernel!r} on {self.url}: {shown}")
            self.kc = await JupyAsyncKernelClient.connect(self.url, kernel=ids[0])
        else: self.kc = await JupyAsyncKernelClient.connect(self.url, cwd=str(cwd or os.getcwd()), env=dict(os.environ, **kernel_env, **(env or {})))
        self.kid, self.owned = self.kc.kernel_id, self.kc.owned
        self.kc.on_jmsg = self._on_jmsg
        if self.owned:
            with suppress(RuntimeError): await self.exec("get_ipython().extension_manager.load_extension('ipykernel_helper.core')", timeout=10)
        return self

    def _on_jmsg(self, jmsg):
        if self.on_jmsg is not None: self.on_jmsg(jmsg)

    async def exec(self, code, timeout=20):
        "Run `code` silently, outside the user's history; a kernel error raises `RuntimeError`."
        cts = (await self.kc.reply(code, silent=True, store_history=False, timeout=timeout))['content']
        if cts.get('status') != 'ok': raise RuntimeError(cts.get('evalue') or cts.get('ename') or 'kernel execute failed')

    async def setup(self):
        """Set up an owned kernel for the AI and the app: the `%ipyai` magic, the user's `config.STARTUP_PATH`
        (an error there names the file and raises), then imports for the custom tools the kernel does not
        already define."""
        with suppress(RuntimeError): await self.exec("get_ipython().extension_manager.load_extension('ipyai.magic')")
        if config.STARTUP_PATH.exists():
            try: await self.exec(_startup_src(config.STARTUP_PATH), timeout=60)
            except RuntimeError as e: raise RuntimeError(f'{config.STARTUP_PATH}: {e}') from None
        present = await KernelTools(self.kc).names()
        for name, stmt in SEED_IMPORTS.items():
            if name in present: continue
            with suppress(RuntimeError): await self.exec(stmt)

    async def interrupt(self): await self.mgr.interrupt_kernel(self.kid)

    async def close(self):
        "Close the client (`kc.__aexit__` is the ownership contract: an owned kernel shuts down, an attached one survives)."
        if self.kc is not None: await self.kc.__aexit__()

    async def __aenter__(self): return await self.start()
    async def __aexit__(self, *exc): await self.close()

"Kernel lifecycle over a rustygate gateway: jupyasyncclient + ipymini, iopub rendered as it arrives."
import logging, os
from fastcore.utils import first, rtoken_hex
from fastcore.nbio import msg2out
from jupyasyncclient.multimanager import JupyAsyncMultiKernelManager
from jupyasyncclient import JupyAsyncKernelClient
from jupywire.route import OUTPUT_MSGS, COMM_MSGS

DEFAULT_URL = 'http://127.0.0.1:8787'   # rustygate's default port; IPYAI_GATEWAY overrides
log = logging.getLogger(__name__)


class KernelSession:
    """One gateway kernel and its ws client; incremental iopub for UIs (messages surface as they
    arrive, not drained at completion). rustygate is a hard runtime prerequisite, like any Jupyter
    server: an unreachable gateway fails loudly at `start` with the command to run."""
    def __init__(self, url=None):
        self.url = url or os.environ.get('IPYAI_GATEWAY', DEFAULT_URL)
        self.mgr, self.kc, self.kid, self.owned, self.busy = None, None, None, False, False
        self.on_comm = None   # host-side comm handler: (msg_type, content) for comm traffic seen on iopub
        self.on_stdin = None  # async (prompt, password) -> str, answering kernel input_requests
        self.on_cell_msg = None  # (cell_id, jmsg) observer for merged-stream traffic tagged `{cell_id}.{token}`


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
        self.kc.on_jmsg = self._on_jmsg
        return self

    def _on_jmsg(self, jmsg):
        """A message left unmatched by `route` whose parent msg_id is `{cell_id}.{token}` is offered to the
        `on_cell_msg` observer (post-idle traffic, e.g. a background thread's print); the rest is bridge
        plumbing, dropped. stdin never lands here: each run owns its requests via `run`'s `on_stdin` hook."""
        pid = jmsg.get('parent_header', {}).get('msg_id') or ''
        if '.' not in pid: return
        cid = pid.split('.', 1)[0]
        if self.on_cell_msg is not None:
            try: self.on_cell_msg(cid, jmsg)
            except Exception: log.exception('on_cell_msg failed')

    def _answer_stdin(self, m):
        "run()'s `on_stdin` hook: adapt the wire `input_request` to the app's `(prompt, password)` handler."
        c = m['content']
        return self.on_stdin(c.get('prompt', ''), c.get('password', False))

    async def run_cell(self, cid, code, allow_stdin=None, **kw):
        """Execute `code` tagged as cell `cid` (msg_id `{cid}.{token}`), yielding each nbformat output
        as it arrives. Comm traffic goes to `on_comm`, and every message to the `on_cell_msg` observer.
        Several cells can be in flight at once; each collects only its own traffic."""
        if allow_stdin is None: allow_stdin = self.on_stdin is not None
        stdin = self._answer_stdin if allow_stdin and self.on_stdin is not None else None
        self.busy = True
        try:
            async for m in self.kc.run(code, msg_id=f'{cid}.{rtoken_hex(4)}', on_stdin=stdin, **kw):
                if self.on_cell_msg is not None: self.on_cell_msg(cid, m)
                typ = m['msg_type']
                if typ in OUTPUT_MSGS: yield msg2out(m)
                elif typ in COMM_MSGS and self.on_comm is not None: self.on_comm(typ, m['content'])
        finally: self.busy = False

    async def interrupt(self): await self.mgr.interrupt_kernel(self.kid)

    async def close(self):
        "Close the client (`kc.__aexit__` is the ownership contract: an owned kernel shuts down, an attached one survives)."
        if self.kc is not None: await self.kc.__aexit__()

    async def __aenter__(self): return await self.start()
    async def __aexit__(self, *exc): await self.close()

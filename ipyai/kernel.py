"Kernel lifecycle over a jupygate gateway: jupyasyncclient + ipymini, iopub rendered as it arrives."
import asyncio, os
from fastcore.utils import first
from jupyasyncclient.multimanager import JupyAsyncMultiKernelManager

DEFAULT_URL = 'http://127.0.0.1:8787'   # jupygate's default port; IPYAI_GATEWAY overrides
OUTPUT_MSGS = ('stream', 'display_data', 'execute_result', 'error')
COMM_MSGS = ('comm_open', 'comm_msg', 'comm_close')

class DeadKernelError(RuntimeError): pass

class KernelSession:
    """One gateway kernel and its ws client; incremental iopub for UIs (messages surface as they
    arrive, not drained at completion). jupygate is a hard runtime prerequisite, like any Jupyter
    server: an unreachable gateway fails loudly at `start` with the command to run."""
    def __init__(self, url=None):
        self.url = url or os.environ.get('IPYAI_GATEWAY', DEFAULT_URL)
        self.mgr, self.kc, self.kid, self.owned, self.busy = None, None, None, False, False
        self.on_comm = None   # host-side comm handler: (msg_type, content) for COMM_MSGS seen on iopub
        self.on_stdin = None  # async (prompt, password) -> str, answering kernel input_requests

    async def start(self, kernel=''):
        """Create an owned kernel (closed on exit), or attach to `kernel` by id prefix (taken as
        found: not seeded, never stopped by us). Owned kernels start in our cwd and environment."""
        self.mgr = JupyAsyncMultiKernelManager(self.url)
        try: ks = await self.mgr.list_kernels()   # reachability and auth fail here, loudly
        except Exception as e: raise ConnectionError(f'no jupygate gateway at {self.url} (start one with `jupygate`): {e}') from e
        if kernel:
            self.kid = first(k['id'] for k in ks if k['id'].startswith(kernel))
            if not self.kid: raise ValueError(f'no kernel matching {kernel!r} on {self.url}: {[k["id"][:8] for k in ks]}')
            self.owned = False
        else:
            self.kid = await self.mgr.start_kernel(cwd=os.getcwd(), env=dict(os.environ))
            self.owned = True
        self.kc = self.mgr.client(self.kid)
        self.kc.start_channels()
        await self.kc.wait_for_ready()
        if self.owned:   # seed the REPL services (sig_help etc.); attached kernels are taken as found
            try: await self.kc.reply("get_ipython().extension_manager.load_extension('ipykernel_helper.core')",
                                     silent=True, store_history=False, timeout=10)
            except Exception: pass
        return self

    async def run(self, code, on_output):
        """Execute `code`, calling `on_output(msg_type, content)` per iopub output message as it
        arrives. Completion means both the execute_reply and the idle status OF THIS EXECUTION:
        iopub is a shared broadcast, so status and outputs are filtered by parent msg id -- a stray
        idle left by an out-of-band bridge exec must not end the loop early. Comm traffic is
        deliberately unfiltered. stdin is answered through `on_stdin` ('' without a handler), so
        kernel-side `input()` works. A dead gateway kernel surfaces as DeadKernelError: the ws
        client reconnects on its own, so a 1s silence check polls liveness like conkernel did."""
        self.busy = True
        try:
            mid = self.kc.execute(code, allow_stdin=self.on_stdin is not None)
            chans = dict(iopub=self.kc.get_iopub_msg, shell=self.kc.get_shell_msg, stdin=self.kc.get_stdin_msg)
            pend = {ch: asyncio.ensure_future(f(timeout=None)) for ch, f in chans.items()}
            done = idle = False
            try:
                while not (done and idle):
                    ready, _ = await asyncio.wait(pend.values(), return_when=asyncio.FIRST_COMPLETED, timeout=1)
                    if not ready:
                        if not await self.mgr.is_alive(self.kid): raise DeadKernelError('kernel died while executing')
                        continue
                    for t in ready:
                        ch = first(k for k, v in pend.items() if v is t)
                        msg = t.result()
                        pend[ch] = asyncio.ensure_future(chans[ch](timeout=None))
                        mt, c = msg['msg_type'], msg['content']
                        mine = msg.get('parent_header', {}).get('msg_id') == mid
                        if ch == 'shell':
                            if mt == 'execute_reply' and mine: done = True
                        elif ch == 'stdin':
                            if mt == 'input_request' and mine:
                                v = await self.on_stdin(c.get('prompt', ''), c.get('password', False)) if self.on_stdin else ''
                                self.kc.input(v)
                        elif mt == 'status' and c.get('execution_state') == 'idle' and mine: idle = True
                        elif mt in OUTPUT_MSGS and mine: on_output(mt, c)
                        elif mt in COMM_MSGS and self.on_comm is not None: self.on_comm(mt, c)
            finally:
                for t in pend.values(): t.cancel()
        finally: self.busy = False

    async def complete(self, code, pos):
        "Completion matches and the replace-start offset, via `complete_request`"
        r = await self.kc.complete(code, pos, reply=True, timeout=5)
        c = r['content']
        return c.get('matches', []), c.get('cursor_start', pos)

    async def check(self, code):
        "('complete'|'incomplete'|'invalid'|'unknown', indent) via `is_complete_request`"
        r = await self.kc.is_complete(code, reply=True, timeout=5)
        c = r['content']
        return c.get('status', 'unknown'), c.get('indent', '')

    async def inspect(self, code, pos, detail=0):
        "Inspection text ('' when nothing found) via `inspect_request` -- signatures and docs, ANSI-styled"
        r = await self.kc.inspect(code, pos, detail_level=detail, reply=True, timeout=5)
        c = r['content']
        return c.get('data', {}).get('text/plain', '') if c.get('found') else ''

    async def interrupt(self): await self.mgr.interrupt_kernel(self.kid)

    async def close(self):
        "Close the client; shut the kernel down only if we created it (attached kernels are left running)."
        if self.kc is not None: await self.kc.aclose()
        if self.owned and self.mgr is not None:
            try: await self.mgr.shutdown_kernel(self.kid)
            except Exception: pass   # gateway gone: nothing to shut down
        if self.mgr is not None: await self.mgr.aclose()

    async def __aenter__(self): return await self.start()
    async def __aexit__(self, *exc): await self.close()

"Kernel lifecycle and incremental execution: conkernelclient + ipymini, iopub rendered as it arrives."
import asyncio, sys, uuid
from queue import Empty
from conkernelclient import ConKernelManager, DeadKernelError
from jupyter_client.kernelspec import KernelSpec

DEFAULT_KERNEL = 'ipymini'
OUTPUT_MSGS = ('stream', 'display_data', 'execute_result', 'error')
COMM_MSGS = ('comm_open', 'comm_msg', 'comm_close')

class ModuleKernelManager(ConKernelManager):
    "Launches `python -m <module> -f <connection_file>`, so no kernelspec install is needed"
    def __init__(self, module=DEFAULT_KERNEL, **kw):
        super().__init__(**kw)
        self._kernel_spec = KernelSpec(language='python', display_name=module,
            argv=[sys.executable, '-Xfrozen_modules=off', '-m', module, '-f', '{connection_file}'])

class KernelSession:
    "One kernel and client; incremental iopub for UIs (messages surface as they arrive, not drained at completion)."
    def __init__(self, module=DEFAULT_KERNEL):
        self.module, self.km, self.kc, self.busy = module, None, None, False
        self.on_comm = None  # host-side comm handler: (msg_type, content) for COMM_MSGS seen on iopub

    async def start(self):
        self.km = ModuleKernelManager(module=self.module)
        await self.km.start_kernel()
        self.kc = await self.km.client().start_channels()
        return self

    async def run(self, code, on_output):
        """Execute `code`, calling `on_output(msg_type, content)` per iopub output message as it arrives.
        Completion means both the shell reply and the idle status OF THIS EXECUTION: iopub is a shared
        broadcast, so status and outputs are filtered by parent msg id -- a stray idle left by an
        out-of-band `eval_expr` (e.g. the shell layer's cwd sync) must not end the loop early, or this
        cell's outputs orphan until the next run paints them one cell late. Comm traffic is deliberately
        unfiltered. A ZMQ peer dying is silent -- no EOF, the reply just never comes -- so liveness is
        polled while waiting (conkernel's lesson)."""
        self.busy = True
        try:
            mid = uuid.uuid4().hex
            t = asyncio.ensure_future(self.kc.execute(code, reply=True, timeout=None, msg_id=mid))
            idle = False
            while not (t.done() and idle):
                try: msg = await self.kc.get_iopub_msg(timeout=1)
                except Empty:
                    if not await self.km.is_alive():
                        t.cancel()
                        raise DeadKernelError('kernel died while executing')
                    continue
                mt, c = msg['msg_type'], msg['content']
                mine = msg.get('parent_header', {}).get('msg_id') == mid
                if mt == 'status' and c.get('execution_state') == 'idle' and mine: idle = True
                elif mt in OUTPUT_MSGS and mine: on_output(mt, c)
                elif mt in COMM_MSGS and self.on_comm is not None: self.on_comm(mt, c)
        finally: self.busy = False

    async def complete(self, code, pos):
        "Completion matches and the replace-start offset, via `complete_request`"
        r = await self.kc.shell_request('complete_request', code=code, cursor_pos=pos)
        c = r['content']
        return c.get('matches', []), c.get('cursor_start', pos)

    async def check(self, code):
        "('complete'|'incomplete'|'invalid'|'unknown', indent) via `is_complete_request`"
        r = await self.kc.shell_request('is_complete_request', code=code)
        c = r['content']
        return c.get('status', 'unknown'), c.get('indent', '')

    async def inspect(self, code, pos, detail=0):
        "Inspection text ('' when nothing found) via `inspect_request` -- signatures and docs, ANSI-styled"
        r = await self.kc.shell_request('inspect_request', code=code, cursor_pos=pos, detail_level=detail)
        c = r['content']
        return c.get('data', {}).get('text/plain', '') if c.get('found') else ''

    async def interrupt(self): await self.km.interrupt_kernel()

    async def close(self):
        if self.kc is not None: self.kc.stop_channels()
        if self.km is not None and await self.km.is_alive(): await self.km.shutdown_kernel()

    async def __aenter__(self): return await self.start()
    async def __aexit__(self, *exc): await self.close()

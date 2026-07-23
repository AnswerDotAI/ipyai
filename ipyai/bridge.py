"KernelBridge over conkernelclient: silent kernel-side execution for tools, variable reads, and namespace writes."
import asyncio, uuid
from queue import Empty
from .kernel_bridge import KernelBridge, _expr_value, _EXEC_TIMEOUT
from .tooling import ToolRegistry

PYTHON_TOOL_SRC = '''
def py(code:str):  # `py`, the solveit name (codex reserves the function name `python` model-side)
    "Execute `code` in the user's live IPython session; returns captured output, the result repr, and any error."
    from IPython.utils.capture import capture_output
    ip = get_ipython()
    with capture_output() as cap: res = ip.run_cell(code, store_history=False)
    parts = [cap.stdout, cap.stderr]
    parts += [o.data.get('text/plain', '') for o in cap.outputs]  # the displayed result lands here under capture
    if res.result is not None and not cap.outputs: parts.append(repr(res.result))
    err = res.error_in_exec or res.error_before_exec
    if err is not None: parts.append(f'{type(err).__name__}: {err}')
    return '\\n'.join(p for p in parts if p) or 'OK'
'''

class ConBridge(KernelBridge):
    "KernelBridge whose `_exec` speaks ConKernelClient: reply via its pending-queue reader, iopub drained by msg_id."
    async def _exec(self, code, *, expressions=None, capture_stream=False, timeout=_EXEC_TIMEOUT):
        async with self._exec_lock:
            msg_id = uuid.uuid4().hex
            rep = self.client.execute(code, reply=True, timeout=timeout, msg_id=msg_id,
                                      silent=True, store_history=False, user_expressions=expressions or {})
            stream = [] if capture_stream else None
            iop = asyncio.create_task(self._drain_iopub(msg_id, stream, timeout))
            try: reply = await rep
            finally:
                if not iop.done(): iop.cancel()
            content = reply['content']
            if content.get('status') != 'ok':
                raise RuntimeError(content.get('evalue') or content.get('ename') or 'kernel execute failed')
            exprs = {k: _expr_value(v) for k, v in (content.get('user_expressions') or {}).items()}
            return exprs, ''.join(stream) if stream is not None else ''

    async def _drain_iopub(self, msg_id, stream_buf, timeout):
        "Consume this request's iopub until idle so its messages never leak into the next cell's renderer."
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            try: msg = await asyncio.wait_for(self.client.get_iopub_msg(), timeout=1.0)
            except (asyncio.TimeoutError, Empty): continue
            if msg['parent_header'].get('msg_id') != msg_id: continue
            if msg['msg_type'] == 'stream':
                if stream_buf is not None: stream_buf.append(msg['content'].get('text', ''))
            elif msg['msg_type'] == 'status' and msg['content'].get('execution_state') == 'idle': return

    async def set_vars(self, **vals):
        "Assign values into the kernel's user namespace, silently."
        await self._exec('\n'.join(f'{k} = {v!r}' for k, v in vals.items()))

async def setup_tools(client):
    "Wire a ConBridge to a live kernel client: define the `py` tool, import the others, return (bridge, registry)."
    bridge = ConBridge(client)
    try: await bridge._exec(PYTHON_TOOL_SRC)
    except Exception: pass
    await bridge.inject_tools(skip=('py', 'python'))
    return bridge, ToolRegistry(bridge)

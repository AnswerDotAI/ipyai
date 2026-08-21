"Wire the kernel bridge to a live gateway client: seed the tool layer, return (bridge, registry)."
from .kernel_bridge import KernelBridge
from .tooling import ToolRegistry

PYTHON_TOOL_SRC = '''
async def py(code:str):  # `py`, the solveit name (codex reserves the function name `python` model-side)
    "Execute `code` in the user's live IPython session; returns captured output, the result repr, and any error."
    from IPython.utils.capture import capture_output
    ip = get_ipython()
    tc = ip.transform_cell(code)
    with capture_output() as cap: res = await ip.run_cell_async(code, store_history=False, transformed_cell=tc)
    parts = [cap.stdout, cap.stderr]
    parts += [o.data.get('text/plain', '') for o in cap.outputs]  # the displayed result lands here under capture
    if res.result is not None and not cap.outputs: parts.append(repr(res.result))
    err = res.error_in_exec or res.error_before_exec
    if err is not None: parts.append(f'{type(err).__name__}: {err}')
    return '\\n'.join(p for p in parts if p) or 'OK'
'''

SEED_SRC = "get_ipython().extension_manager.load_extension('ipykernel_helper.core')"

async def setup_tools(client):
    "Seed a live kernel (ipykernel_helper services + the `py` tool + the custom tool imports) and return (bridge, registry)."
    bridge = KernelBridge(client)
    for src in (SEED_SRC, PYTHON_TOOL_SRC):
        try: await bridge._exec(src)
        except Exception: pass
    await bridge.seed_tools(skip=('py', 'python'))
    return bridge, ToolRegistry(bridge)

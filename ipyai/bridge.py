"Wire the kernel bridge to a live gateway client: seed the tool layer, return (bridge, registry)."
from .kernel_bridge import KernelBridge
from .tooling import ToolRegistry
from . import config

PYTHON_TOOL_SRC = '''
async def py(code:str):  # `py`, the solveit name (codex reserves the function name `python` model-side)
    "Execute `code` as a cell in the user's live IPython session; its printed text, results, errors, and images come back as the user would see them."
    ip = get_ipython()
    await ip.run_cell_async(code, store_history=False, transformed_cell=ip.transform_cell(code))
'''

SEED_SRC = "get_ipython().extension_manager.load_extension('ipykernel_helper.core')"

def _startup_src(path):
    "The startup file's source wrapped so `__file__` is bound to its path during the run, and absent after (clikernel's shape)"
    return f'''__file__ = {str(path)!r}
try: exec(compile({path.read_text()!r}, __file__, 'exec'))
finally: del __file__'''

async def setup_tools(client):
    """Seed a live kernel and return (bridge, registry): ipykernel_helper services, the `py` tool, the user's
    `config.STARTUP_PATH` (if present; a failure there names the file and stops the launch), then the custom tool imports."""
    bridge = KernelBridge(client)
    for src in (SEED_SRC, PYTHON_TOOL_SRC):
        try: await bridge._exec(src)
        except Exception: pass
    if config.STARTUP_PATH.exists():
        try: await bridge._exec(_startup_src(config.STARTUP_PATH), timeout=60)   # startup files import the tooling stack
        except RuntimeError as e: raise RuntimeError(f'{config.STARTUP_PATH}: {e}') from None
    await bridge.seed_tools(skip=('py',))
    return bridge, ToolRegistry(bridge)

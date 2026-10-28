"""The model's tools: `py`, which ipyai runs as a cell, and the custom tools the kernel defines. ipyai reaches those
through the kernel's `schemas` and `call` operations, specified in kernmini's DEV.md."""
from fastcore.funccall import get_schema
from fastcore.utils import first, rtoken_hex
from aihist import tool_result

__all__ = ['KernelTools', 'CUSTOM_TOOL_NAMES', 'SEED_IMPORTS', 'PY_SCHEMA']

CUSTOM_TOOL_NAMES = ("bash", "start_bgterm", "write_stdin", "close_bgterm", "lnhashview_file", "exhash_file", "list_pyskills")
SEED_IMPORTS = dict(bash="from safecmd import bash", start_bgterm="from ptymini.bg import start_bgterm",
    write_stdin="from ptymini.bg import write_stdin", close_bgterm="from ptymini.bg import close_bgterm",
    lnhashview_file="from exhash import lnhashview_file", exhash_file="from exhash import exhash_file",
    list_pyskills="from pyskills import list_pyskills")

def py(
    code:str, # Python source to run
):
    "Execute `code` as a cell in the user's live IPython session; its printed text, results, errors, and images come back as the user would see them."

# `py` is the solveit name: codex reserves `python` model-side
PY_SCHEMA = dict(type='function', function=get_schema(py, pname='parameters'))

class KernelTools:
    """The tools a turn offers the model, and their dispatch into the kernel `kc`. `aim_info` gates images
    in tool results. `inflight` counts tool requests still running, so cancelling a turn knows to interrupt."""
    def __init__(self, kc, aim_info=None): self.kc, self.aim_info, self.inflight = kc, aim_info, 0

    async def request(self, code='', op=None):
        "Outputs of running `code`, or the kernel operation `op`, outside the user's history. An error doesn't abort requests queued behind it."
        return await self.kc.exec_outs(code, metadata=op, msg_id=f'tool{rtoken_hex(4)}', allow_stdin=False, store_history=False, stop_on_error=False)

    async def op_json(self, op):
        "The `application/json` result of kernel operation `op`, or `{}`"
        outs = await self.request(op=op)
        return first(o['data']['application/json'] for o in outs if o['output_type'] == 'execute_result') or {}

    async def schemas(self):
        "OpenAI-style schemas for `py`, plus each custom tool the kernel defines, from one `schemas` operation"
        found = await self.op_json(dict(schemas=list(CUSTOM_TOOL_NAMES)))
        return [PY_SCHEMA] + [dict(type='function', function=s) for s in found.values() if isinstance(s, dict)]

    def ns(self, schemas):
        "fastllm's tool namespace for `schemas`: one async caller per tool"
        def caller(name):
            async def _f(**kwargs): return await self.call(name, **kwargs)
            _f.__name__ = name
            return _f
        return {(n := s['function']['name']): caller(n) for s in schemas}

    async def call(self, name, **kwargs):
        """Run tool `name` and return its outputs as the model sees them, through aihist's `tool_result`, as Solveit's turns
        give them: canonical results with images gated by `aim_info` and independent Markdown/full flags.
        `py` runs as a cell, other tools through the `call` operation."""
        self.inflight += 1
        try: outs = await (self.request(kwargs['code']) if name == 'py' else self.request(op=dict(call=dict(name=name, args=kwargs))))
        finally: self.inflight -= 1
        return tool_result(outs, self.aim_info, code=name == 'py')

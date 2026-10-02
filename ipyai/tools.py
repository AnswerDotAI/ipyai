"""The model's tools: `py`, which ipyai runs as a kernel request, and the custom tools the kernel defines,
called through `EvalOps.eval` so each result comes back in its own request."""
import json
from fastcore.funccall import get_schema
from fastcore.nbio import msg2out, render_text
from fastcore.utils import rtoken_hex
from jupywire.route import OUTPUT_MSGS
from aidialog.dialog import Message
from aidialog.hist import output_parts, merge_media
from aidialog.msg_parts import FullResponse, ToolResponse

__all__ = ['KernelTools', 'CUSTOM_TOOL_NAMES', 'SEED_IMPORTS', 'PY_SCHEMA']

CUSTOM_TOOL_NAMES = ("bash", "start_bgterm", "write_stdin", "close_bgterm", "lnhashview_file", "exhash_file", "list_pyskills")
SEED_IMPORTS = dict(bash="from safecmd import bash", start_bgterm="from ptymini.bg import start_bgterm",
    write_stdin="from ptymini.bg import write_stdin", close_bgterm="from ptymini.bg import close_bgterm",
    lnhashview_file="from exhash import lnhashview_file", exhash_file="from exhash import exhash_file",
    list_pyskills="from pyskills import list_pyskills")
TB_MAXLEN = 180   # most characters shown per traceback line in `py` results, clikernel's budget
TOOL_TIMEOUT = 600

def py(
    code:str, # Python source to run
):
    "Execute `code` as a cell in the user's live IPython session; its printed text, results, errors, and images come back as the user would see them."

# `py` is the solveit name: codex reserves `python` model-side
PY_SCHEMA = dict(type='function', function=get_schema(py, pname='parameters'))

class KernelTools:
    """The tools a turn offers the model, and their dispatch into the kernel `kc`. `aim_info` gates images
    in `py` results. `inflight` counts tool requests still running, so cancelling a turn knows to interrupt."""
    def __init__(self, kc, aim_info=None): self.kc, self.aim_info, self.inflight = kc, aim_info, 0

    async def names(self):
        "`py`, plus each custom tool defined and callable in the kernel's namespace"
        r = await self.kc.eval("[n for n in %r if callable(globals().get(n))]" % list(CUSTOM_TOOL_NAMES), call_=False)
        return ['py'] + (list(r) if isinstance(r, list) else [])

    async def schemas(self, names):
        "OpenAI-style schemas for `names`: `py`'s is static, the rest come from the kernel"
        custom = [n for n in names if n != 'py']
        s = (await self.kc.get_schemas(fs=custom)) if custom else {}
        return [PY_SCHEMA] + [v for v in s.values() if not isinstance(v, str)]

    def ns(self, names):
        "fastllm's tool namespace for `names`: one async caller per tool"
        def caller(name):
            async def _f(**kwargs): return await self.call(name, **kwargs)
            _f.__name__ = name
            return _f
        return {n: caller(n) for n in names}

    async def call(self, name, **kwargs):
        "Run tool `name`, returning its result as text, `FullResponse` or `ToolResponse`"
        self.inflight += 1
        try:
            if name == 'py': return await self.run_py(kwargs['code'])
            res = await self.kc.eval(f'(lambda a: call_tool(globals()[{name!r}], a))', kwargs, timeout_=TOOL_TIMEOUT)
            if type(res).__name__ == 'FullResponse': return FullResponse(res)
            return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False, default=str)
        finally: self.inflight -= 1

    async def run_py(self, code):
        """Run `code` as a cell outside the user's history, rendered as clikernel does: tagged text, capped
        tracebacks, images gated by `aim_info`. An error does not abort calls queued behind it."""
        run = self.kc.run(code, msg_id=f'py{rtoken_hex(4)}', allow_stdin=False, store_history=False, stop_on_error=False)
        outs = [msg2out(m) async for m in run if m['msg_type'] in OUTPUT_MSGS]
        res = merge_media(render_text(outs, tb_maxlen=TB_MAXLEN), output_parts(Message(code, output=outs), self.aim_info))
        return res if isinstance(res, str) else ToolResponse(res)

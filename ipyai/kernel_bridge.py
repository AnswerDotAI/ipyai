"Kernel-facing bridge: tool discovery, tool calls, and variable reads over a `JupyAsyncKernelClient` (whose eval family does the wire work)."
import json
from aidialog.dialog import Message
from aidialog.hist import output_parts, merge_media
from aidialog.msg_parts import FullResponse, ToolResponse
from fastcore.nbio import render_text


TB_MAXLEN = 180   # most characters shown per traceback line in `py` results, clikernel's budget

# `py` is the ipyaing kernel tool (the solveit name; codex reserves `python` model-side)
CUSTOM_TOOL_NAMES = ("py", "bash", "start_bgterm", "write_stdin", "close_bgterm", "lnhashview_file", "exhash_file", "list_pyskills")
_SEED_IMPORTS = dict(bash="from safecmd import bash", start_bgterm="from ptymini.bg import start_bgterm",
    write_stdin="from ptymini.bg import write_stdin", close_bgterm="from ptymini.bg import close_bgterm",
    lnhashview_file="from exhash import lnhashview_file", exhash_file="from exhash import exhash_file",
    list_pyskills="from pyskills import list_pyskills")
_EXEC_TIMEOUT = 20
_TOOL_TIMEOUT = 600


class KernelBridge:
    "Gives ToolRegistry a namespace-shaped interface over the kernel: silent executes for reads and tool calls (nothing in history), except `py`, which runs as a plain cell."
    def __init__(self, client):
        self.client = client
        self._schemas = None
        self._names = None
        self.aim_info = None   # model capability dict gating images in `py` results; the Assistant sets it each turn

    async def _exec(self, code, *, timeout=_EXEC_TIMEOUT):
        cts = (await self.client.reply(code, silent=True, store_history=False, timeout=timeout))["content"]
        if cts.get("status") != "ok":
            raise RuntimeError(cts.get("evalue") or cts.get("ename") or "kernel execute failed")

    async def present_names(self, names):
        "Return subset of `names` already defined and callable in the kernel's user_ns."
        r = await self.client.eval("[n for n in %r if n in globals() and callable(globals()[n])]" % list(names), _call=False)
        return list(r) if isinstance(r, list) else []

    async def seed_tools(self, skip=()):
        "Import the custom tool names (other than py, which is defined by the host)."
        skip = set(skip)
        for stmt in (_SEED_IMPORTS[n] for n in CUSTOM_TOOL_NAMES if n in _SEED_IMPORTS and n not in skip):
            try: await self._exec(stmt)
            except Exception: pass
        return await self.available_names(force=True)

    async def available_names(self, force=False):
        if self._names is not None and not force: return self._names
        self._names = await self.present_names(CUSTOM_TOOL_NAMES)
        self._schemas = None
        return self._names

    async def schemas(self):
        if self._schemas is not None: return self._schemas
        names = await self.available_names()
        s = (await self.client.get_schemas(fs=names)) if names else {}
        self._schemas = [v for v in s.values() if not isinstance(v, str)]
        return self._schemas

    async def call_tool(self, name, args=None):
        names = await self.available_names()
        if name not in names: raise NameError(f"{name!r} is not defined in the kernel namespace")
        if name == 'py': return await self.run_py(args['code'])
        await self._exec(f"_ipyai_r = await call_tool(globals()[{name!r}], {(args or {})!r})", timeout=_TOOL_TIMEOUT)
        full_e = "any(c.__name__=='FullResponse' for c in type(_ipyai_r).__mro__)"
        exprs = await self.client.eval_exprs(vs=['_ipyai_r', full_e])
        res = exprs.get('_ipyai_r')
        text = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False, default=str)
        return FullResponse(text) if exprs.get(full_e) is True else text

    async def run_py(self, code):
        """The `py` tool: run the model's `code` as a plain cell, the same path as a user cell (so the kernel sees
        exactly the model's code, and nothing is captured kernel-side), and render the nbformat outputs for the
        model: `render_text` (tagged when a cell has several outputs, tracebacks capped), the same text clikernel
        shows, plus images as media parts gated by `aim_info`. fastllm runs a turn's tool calls in parallel, so the
        kernel queues them; `stop_on_error=False` keeps one call's error from aborting the rest. `store_history=False`
        keeps the model's cells out of the user's `In` history, and is what kernel-side rules use to tell the
        model's cells from the user's."""
        outs = await self.client.run(code, store_history=False, stop_on_error=False)
        res = merge_media(render_text(outs, tb_maxlen=TB_MAXLEN), output_parts(Message(code, output=outs), self.aim_info))
        return res if isinstance(res, str) else ToolResponse(res)

    async def read_var(self, name):
        "Value of live expression `name` (`foo` or `foo.bar(...)`); an `<error .../>` string when it raises."
        return (await self.client.eval_exprs(vs=[name])).get(name)

    async def read_vars(self, names): return await self.client.eval_exprs(vs=list(names))

    def set_vars(self, **vals):
        "Assign values into the kernel's user namespace, silently."
        self.client.xpush(**vals)

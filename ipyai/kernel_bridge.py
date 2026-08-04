"Kernel-facing bridge: tool discovery, tool calls, and variable reads over a `JupyAsyncKernelClient` (whose eval family does the wire work)."
import ast, json


# `py` is the ipyaing kernel tool (the solveit name; codex reserves `python` model-side); `python` remains
# discoverable for legacy kernels where the safepyrun extension seeds it. Only one is ever defined at a time.
CUSTOM_TOOL_NAMES = ("py", "python", "bash", "start_bgterm", "write_stdin", "close_bgterm", "lnhashview_file", "exhash_file", "list_pyskills")
_SEED_IMPORTS = dict(bash="from safecmd import bash", start_bgterm="from bgterm import start_bgterm",
    write_stdin="from bgterm import write_stdin", close_bgterm="from bgterm import close_bgterm",
    lnhashview_file="from exhash import lnhashview_file", exhash_file="from exhash import exhash_file",
    list_pyskills="from pyskills import list_pyskills")
_EXEC_TIMEOUT = 20
_TOOL_TIMEOUT = 600


def _literal(text):
    try: return ast.literal_eval(text)
    except Exception: return text


def _expr_value(expr):
    if expr.get("status") != "ok": raise RuntimeError(expr.get("evalue", "kernel expression error"))
    data = expr.get("data") or {}
    if "application/json" in data: return data["application/json"]
    return _literal(data.get("text/plain", ""))


class KernelBridge:
    "Gives ToolRegistry a namespace-shaped interface over the kernel; silent executes, nothing in history."
    def __init__(self, client):
        self.client = client
        self._schemas = None
        self._names = None

    async def _exec(self, code, *, expressions=None, timeout=_EXEC_TIMEOUT):
        cts = (await self.client.reply(code, user_expressions=expressions or {}, silent=True,
                                       store_history=False, timeout=timeout))["content"]
        if cts.get("status") != "ok":
            raise RuntimeError(cts.get("evalue") or cts.get("ename") or "kernel execute failed")
        return {k: _expr_value(v) for k, v in (cts.get("user_expressions") or {}).items()}

    async def present_names(self, names):
        "Return subset of `names` already defined and callable in the kernel's user_ns."
        r = await self.client.eval("[n for n in %r if n in globals() and callable(globals()[n])]" % list(names), _call=False)
        return list(r) if isinstance(r, list) else []

    async def seed_tools(self, skip=()):
        "Import the custom tool names (other than py/python, which are defined by the host)."
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
        code = f"_ipyai_r = await call_tool(globals()[{name!r}], {(args or {})!r})"
        exprs = await self._exec(code, expressions={"_r": "_ipyai_r",
            "_full": "any(c.__name__=='FullResponse' for c in type(_ipyai_r).__mro__)"}, timeout=_TOOL_TIMEOUT)
        res = exprs.get("_r")
        text = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False, default=str)
        if exprs.get("_full"):
            from aidialog.msg_parts import FullResponse
            return FullResponse(text)
        return text

    async def read_var(self, name):
        "Return the value of a live expression (`name` may be `foo` or `foo.bar(...)`), raising if it raises."
        exprs = await self._exec("", expressions={"_r": name})
        return exprs.get("_r")

    async def read_vars(self, names):
        exprs = await self._exec("", expressions={f"_v{i}": name for i, name in enumerate(names)})
        return {name: exprs.get(f"_v{i}") for i, name in enumerate(names)}

    async def set_vars(self, **vals):
        "Assign values into the kernel's user namespace, silently."
        await self.client.xpush(**vals)

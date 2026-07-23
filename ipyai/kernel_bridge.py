"Kernel-facing bridge: abstracts tool discovery, tool calls, and variable reads over a jupyter_client AsyncKernelClient."
import ast, asyncio, json
from queue import Empty


# `py` is the ipyaing kernel tool (the solveit name; codex reserves `python` model-side); `python` remains
# discoverable for legacy kernels where the safepyrun extension seeds it. Only one is ever defined at a time.
CUSTOM_TOOL_NAMES = ("py", "python", "bash", "start_bgterm", "write_stdin", "close_bgterm", "lnhashview_file", "exhash_file", "list_pyskills")
_INJECT_IMPORTS = dict(bash="from safecmd import bash", start_bgterm="from bgterm import start_bgterm",
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


async def _get_shell_reply(client, msg_id, timeout=_EXEC_TIMEOUT):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while True:
        remaining = end - loop.time()
        if remaining <= 0: raise TimeoutError("kernel shell reply timeout")
        try: msg = await asyncio.wait_for(client.get_shell_msg(), timeout=min(remaining, 1.0))
        except (asyncio.TimeoutError, Empty): continue
        if msg["parent_header"].get("msg_id") == msg_id: return msg


async def _drain_iopub_until_idle(client, msg_id, stream_buf, timeout=_EXEC_TIMEOUT):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while True:
        remaining = end - loop.time()
        if remaining <= 0: raise TimeoutError("kernel iopub idle timeout")
        try: msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=min(remaining, 1.0))
        except (asyncio.TimeoutError, Empty): continue
        if msg["parent_header"].get("msg_id") != msg_id: continue
        if msg["msg_type"] == "stream":
            if stream_buf is not None: stream_buf.append(msg["content"].get("text", ""))
        elif msg["msg_type"] == "status" and msg["content"].get("execution_state") == "idle": return


class KernelBridge:
    "Runs code in a remote ipykernel via jupyter_client; gives ToolRegistry a namespace-shaped interface."
    def __init__(self, client):
        self.client = client
        self._schemas = None
        self._names = None
        self._exec_lock = asyncio.Lock()  # shell/iopub channels are shared across all execute requests

    async def _exec(self, code, *, expressions=None, capture_stream=False, timeout=_EXEC_TIMEOUT):
        async with self._exec_lock:
            msg_id = self.client.execute(code, silent=True, store_history=False, user_expressions=expressions or {})
            stream = [] if capture_stream else None
            iop = asyncio.create_task(_drain_iopub_until_idle(self.client, msg_id, stream, timeout=timeout))
            reply = await _get_shell_reply(self.client, msg_id, timeout=timeout)
            try: await iop
            except Exception: iop.cancel()
            content = reply["content"]
            if content.get("status") != "ok":
                raise RuntimeError(content.get("evalue") or content.get("ename") or "kernel execute failed")
            exprs = {k: _expr_value(v) for k,v in (content.get("user_expressions") or {}).items()}
            return exprs, "".join(stream) if stream is not None else ""

    async def present_names(self, names):
        "Return subset of `names` already defined and callable in the kernel's user_ns."
        probe = "[n for n in %r if n in globals() and callable(globals()[n])]" % list(names)
        exprs,_ = await self._exec("", expressions={"_r": probe})
        return list(exprs.get("_r") or [])

    async def inject_tools(self, skip=()):
        "Import the custom tool names (other than python, which must come from an extension)."
        skip = set(skip)
        stmts = [_INJECT_IMPORTS[n] for n in CUSTOM_TOOL_NAMES if n in _INJECT_IMPORTS and n not in skip]
        for stmt in stmts:
            try: await self._exec(stmt)
            except Exception: pass
        return await self.available_names(force=True)

    async def available_names(self, force=False):
        if self._names is not None and not force: return self._names
        exprs,_ = await self._exec("", expressions={"_r": "[n for n in %r if n in globals() and callable(globals()[n])]" % list(CUSTOM_TOOL_NAMES)})
        self._names = list(exprs.get("_r") or [])
        self._schemas = None
        return self._names

    async def schemas(self):
        if self._schemas is not None: return self._schemas
        names = await self.available_names()
        if not names:
            self._schemas = []
            return self._schemas
        code = "from toolslm.funccall import get_schema_nm as _ipyai_gs"
        probe = ("[dict(type='function', function=_ipyai_gs(n, globals(), pname='parameters')) "
            "for n in %r if n in globals()]" % names)
        exprs,_ = await self._exec(code, expressions={"_r": probe})
        self._schemas = list(exprs.get("_r") or [])
        return self._schemas

    async def call_tool(self, name, args=None):
        names = await self.available_names()
        if name not in names: raise NameError(f"{name!r} is not defined in the kernel namespace")
        code = f"""_ipyai_args = {(args or {})!r}
_ipyai_fn = globals()[{name!r}]
_ipyai_r = _ipyai_fn(**_ipyai_args)
if hasattr(_ipyai_r, '__await__'): _ipyai_r = await _ipyai_r
"""
        exprs,_ = await self._exec(code, expressions={"_r": "_ipyai_r",
            "_full": "any(c.__name__=='FullResponse' for c in type(_ipyai_r).__mro__)"}, timeout=_TOOL_TIMEOUT)
        res = exprs.get("_r")
        text = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False, default=str)
        if exprs.get("_full"):
            from lisette.core import FullResponse
            return FullResponse(text)
        return text

    async def read_var(self, name):
        "Return repr of a live value by expression (`name` may be `foo` or `foo.bar(...)`)."
        exprs,_ = await self._exec("", expressions={"_r": name})
        return exprs.get("_r")

    async def read_vars(self, names):
        exprs,_ = await self._exec("", expressions={f"_v{i}": name for i,name in enumerate(names)})
        return {name: exprs.get(f"_v{i}") for i,name in enumerate(names)}

    async def history_db_info(self):
        "Return (hist_file_path, session_number) from the kernel's HistoryManager."
        exprs,_ = await self._exec("", expressions={
            "_path": "str(get_ipython().history_manager.hist_file)",
            "_sess": "get_ipython().history_manager.session_number"})
        return exprs.get("_path"), exprs.get("_sess")

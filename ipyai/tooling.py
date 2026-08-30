"Shared tool registry + local namespace bridge. Kernel-backed bridge lives in kernel_bridge.py."
import inspect, json

from aidialog.msg_parts import ToolResponse
from .kernel_bridge import CUSTOM_TOOL_NAMES


def _result_text(res): return res if isinstance(res, (str, ToolResponse)) else json.dumps(res, ensure_ascii=False, default=str)


class LocalBridge:
    "Dict-backed bridge used for tests and in-process callers."
    def __init__(self, ns): self.ns = ns

    async def available_names(self, force=False): return [o for o in CUSTOM_TOOL_NAMES if callable(self.ns.get(o))]

    async def schemas(self):
        from fastcore.funccall import get_schema_nm   # test-time only: the kernel bridge gets schemas kernel-side
        out = []
        for name in await self.available_names():
            try: out.append(dict(type="function", function=get_schema_nm(name, self.ns, pname="parameters")))
            except Exception: continue
        return out

    async def call_tool(self, name, args=None):
        fn = self.ns.get(name)
        if not callable(fn): raise NameError(f"{name!r} is not defined in the active namespace")
        res = fn(**(args or {}))
        if inspect.isawaitable(res): res = await res
        return _result_text(res)

    async def read_var(self, expr):
        import ast
        try: tree = ast.parse(expr, mode="eval")
        except SyntaxError: return self.ns.get(expr)
        if isinstance(tree.body, ast.Name): return self.ns.get(expr)
        try: return eval(expr, self.ns)
        except Exception: return None

class ToolRegistry:
    "Uniform tool surface for AI backends; delegates to a bridge (LocalBridge or KernelBridge)."
    def __init__(self, bridge):
        self.bridge = bridge
        self._names_cache = None
        self._schemas_cache = None

    @classmethod
    def from_ns(cls, ns): return cls(LocalBridge(ns))

    @property
    def ns(self): return getattr(self.bridge, "ns", None)

    async def names(self, force=False):
        if self._names_cache is None or force: self._names_cache = await self.bridge.available_names(force=force)
        return self._names_cache

    async def openai_schemas(self):
        if self._schemas_cache is None: self._schemas_cache = await self.bridge.schemas()
        return self._schemas_cache

    async def call_text(self, name, args=None):
        res = await self.bridge.call_tool(name, args or {})
        return _result_text(res)

    def invalidate(self): self._names_cache = self._schemas_cache = None

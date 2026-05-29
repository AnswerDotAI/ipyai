import json, warnings

from litellm.types.utils import ModelResponse
from fastllm.chat import (AsyncChat as FastLLMAsyncChat, AsyncStreamFormatter as FastLLMAsyncStreamFormatter,
    contents, mk_tr_details)

from .backend_common import BaseBackend, ConversationSeed, compact_tool, seed_to_flat_history
from .lisette_compat import full_response_sentinel_text, is_full_response, strip_full_response_sentinel


class _BridgeNS(dict):
    "Dict-shaped proxy so chat's ns-based tool-call path routes through the ToolRegistry bridge. Pass-through — any `FullResponse` a tool returns survives, and plain-str results go through chat's default truncation."
    def __init__(self, registry):
        super().__init__()
        self._reg = registry

    def __contains__(self, name): return False

    def get(self, name, default=None):
        async def _caller(**kwargs): return await self._reg.call_text(name, kwargs)
        _caller.__name__ = name
        return _caller

    def __getitem__(self, name): return self.get(name)


# litellm's responses-endpoint handler falls back to `model_construct` on chat/responses
# usage-shape mismatches, which later emits pydantic serializer warnings when the partially
# typed object is dumped. Upstream bug, harmless here.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)


class AsyncStreamFormatter(FastLLMAsyncStreamFormatter):
    "Streams via fastllm's formatter so `outp` keeps full fastllm `<details>` tool blocks (round-trippable through `fmt2hist`), but emits a compact `🔧 ...` line per tool result for the live terminal display."
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_text = ""
        self.display_text = ""

    def _emit(self, text):
        if text: self.display_text += text
        return text or ""

    def _tool_result_for_outp(self, o):
        content = o.get("content")
        if is_full_response(content): return {**o, "content": full_response_sentinel_text(content)}
        return o

    def format_item(self, o):
        if isinstance(o, ModelResponse):
            if tcs := getattr(contents(o), "tool_calls", None):
                self.tcs = {tc.id: tc for tc in tcs}
                return ""
        if isinstance(o, dict) and "tool_call_id" in o:
            tc = self.tcs.pop(o["tool_call_id"], None)
            if tc is not None:
                self.outp += mk_tr_details(self._tool_result_for_outp(o), tc, mx=self.mx)
                self.final_text = self.outp
                args = json.loads(tc.function.arguments or "{}")
                return self._emit(compact_tool(tc.function.name, args, strip_full_response_sentinel(o.get("content") or "")))
        res = super().format_item(o)
        self.final_text = self.outp
        return self._emit(res)


class _FastLLMBackend(BaseBackend):
    formatter_cls = AsyncStreamFormatter

    def _make_chat(self, **kw): raise NotImplementedError

    async def prepare_turn(self, *, prompt, model, think="l", provider_session_id=None, seed=None, tool_mode="on", ephemeral=False):
        seed = seed or ConversationSeed()
        tools = (await self.tools.openai_schemas()) or None if tool_mode != "off" else None
        ns = _BridgeNS(self.tools) if tool_mode != "off" else {}
        chat = self._make_chat(model=model, sp=self.ctx.system_prompt, hist=seed_to_flat_history(seed) or None, ns=ns, tools=tools)
        stream = await chat(prompt, stream=True, think=think, max_steps=21)
        return self.prepared_turn(stream, provider_session_id=provider_session_id)


class ClaudeAPIBackend(_FastLLMBackend):
    def _make_chat(self, **kw): return FastLLMAsyncChat(**kw, cache=True)


class CodexAPIBackend(_FastLLMBackend):
    "Codex API backend via fastllm's codex vendor, using `~/.codex/auth.json`. Accepts short names like `gpt-5.4` or prefixed names like `codex/gpt-5.4`."
    def _make_chat(self, model, **kw):
        if "/" in model: _,model = model.split("/", 1)
        return FastLLMAsyncChat(model=model, vendor_name="codex", **kw)

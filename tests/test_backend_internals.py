"Fast backend-internal unit tests: api_client formatter overrides, Claude CLI backend internals, Codex backend internals, MCP socket server."
import asyncio, json

from litellm.types.utils import Choices, Message, ModelResponse
from safepyrun import RunPython

import ipyai.claude_client as claude, ipyai.codex_client as codex
from ipyai.api_client import AsyncStreamFormatter, CodexAPIBackend, _BridgeNS
from ipyai.backend_common import COMPLETION_THINK, compact_tool, tool_call
from ipyai.mcp_server import ToolSocketServer
from ipyai.tooling import ToolRegistry
from fastllm.chat import FullResponse, fmt2hist, tool_dtls_tag


# ---- api_client (fastllm formatter overrides) ----

def _resp_with_tc(call_id="call_1", name="pyrun", arguments='{"code":"2+2"}'):
    msg = Message(role="assistant", content=None,
        tool_calls=[dict(id=call_id, type="function", function=dict(name=name, arguments=arguments))])
    return ModelResponse(choices=[Choices(message=msg, index=0, finish_reason="tool_calls")])


async def _run_api(items):
    fmt = AsyncStreamFormatter()
    out = []

    async def _agen():
        for x in items: yield x

    async for chunk in fmt.format_stream(_agen()): out.append(chunk)
    return fmt, out


def test_tool_call_truncates_long_param_values():
    "Long arg values must be truncated for display so the 🔧 line stays readable; short values render unchanged."
    assert tool_call("pyrun", dict(code="1+1")) == 'pyrun(code="1+1")'
    long = "x" * 5000
    rendered = tool_call("pyrun", dict(code=long))
    assert long not in rendered, f"long arg must not appear verbatim: {rendered!r}"
    assert "…" in rendered or "..." in rendered, f"truncation marker expected: {rendered!r}"
    assert len(rendered) < 100, f"rendered tool_call must be short, got {len(rendered)} chars"


def test_compact_tool_truncates_args_in_display_summary():
    "compact_tool() feeds the 🔧 line; its args section must be truncated."
    long = "y" * 3000
    summary = compact_tool("pyrun", dict(code=long), "ok")
    assert long not in summary, f"full arg should not leak into compact_tool: {summary!r}"
    assert "🔧 pyrun(" in summary


def test_api_display_truncates_long_tool_args_but_outp_keeps_full():
    "In api_client display path the compact 🔧 line must show truncated args, while outp (LLM-facing) keeps the full args via mk_tr_details."
    long = "z" * 3000
    arguments = json.dumps({"code": long})
    resp = _resp_with_tc(call_id="call_3", name="pyrun", arguments=arguments)
    tool_msg = {"tool_call_id": "call_3", "content": "ok"}
    fmt,out = asyncio.run(_run_api([resp, tool_msg]))
    joined = "".join(out)

    assert long not in joined, "display must not contain the full arg"
    assert long not in fmt.display_text
    assert long in fmt.outp, "outp must keep full arg for LLM replay"


def test_codex_api_backend_uses_chatgpt_provider_via_async_chat(shell):
    "CodexChat has been replaced by AsyncChat + codex model aliases (chatgpt/* via LiteLLM). Unprefixed names get `chatgpt/` prefixed so existing configs like 'gpt-5.4' keep working; already-resolved `chatgpt/...` passes through untouched."
    from fastllm.chat import AsyncChat as FastLLMAsyncChat, codex55
    backend = CodexAPIBackend(shell=shell)
    chat = backend._make_chat(model="gpt-5.4", sp="", hist=None, ns={}, tools=None)
    assert isinstance(chat, FastLLMAsyncChat), f"expected FastLLMAsyncChat, got {type(chat).__name__}"
    assert chat.model == "gpt-5.4", f"short model must normalize back to the base model name: {chat.model!r}"

    chat2 = backend._make_chat(model=codex55, sp="", hist=None, ns={}, tools=None)
    assert chat2.model == codex55, f"already-resolved alias must pass through: {chat2.model!r}"


def test_api_tool_start_marker_suppressed_in_display_and_outp():
    "ModelResponse with tool_calls captures tcs but emits no `⏳` chunk and adds nothing to outp."
    fmt,out = asyncio.run(_run_api([_resp_with_tc()]))
    assert out == [""]
    assert "⏳" not in fmt.outp
    assert "⏳" not in fmt.display_text
    assert "call_1" in fmt.tcs


def test_api_full_tool_result_preserved_in_outp_compact_in_display():
    "Tool result must live in full inside outp/final_text (for replay) while terminal display sees only the compact one-liner."
    long = "x" * 5000
    resp = _resp_with_tc(call_id="call_1", name="pyrun", arguments='{"code":"big"}')
    tool_msg = {"tool_call_id": "call_1", "content": FullResponse(long)}
    fmt,out = asyncio.run(_run_api([resp, tool_msg]))
    joined = "".join(out)

    assert long in fmt.outp, "full tool result must live in outp for replay"
    assert fmt.final_text == fmt.outp
    assert tool_dtls_tag in fmt.outp, "outp must use fastllm's <details> block format so fmt2hist can round-trip"

    assert '🔧 pyrun(code="big")' in joined, f"display must show compact one-liner: {joined!r}"
    assert long not in joined, "display must NOT include the full tool result"
    assert long not in fmt.display_text


def test_api_outp_round_trips_via_fmt2hist_with_full_tool_content():
    "Replaying outp through fastllm's fmt2hist must yield a real tool message whose content is the FULL original payload."
    long = "y" * 3000
    resp = _resp_with_tc(call_id="call_2", name="bash", arguments='{"cmd":"ls"}')
    tool_msg = {"tool_call_id": "call_2", "content": FullResponse(long)}
    fmt,_ = asyncio.run(_run_api([resp, tool_msg]))

    msgs = fmt2hist(fmt.outp)
    tool_results = [m for m in msgs if getattr(m, "role", None) == "tool"]
    assert tool_results, f"fmt2hist should yield a tool result message: {msgs}"
    assert tool_results[0].content[0].text == long, "replayed tool content must be full, not truncated"


def test_bridge_ns_does_not_wrap_plain_str_results():
    "BridgeNS must not force-wrap tool results; truncation should use fastllm's per-tool opt-in."
    ns = {"pyrun": lambda code: code}
    reg = ToolRegistry.from_ns(ns)
    bns = _BridgeNS(reg)
    caller = bns.get("pyrun")
    res = asyncio.run(caller(code="z" * 5000))
    assert type(res) is str, f"plain-str tool return must stay plain str, got {type(res).__name__}"


def test_bridge_ns_preserves_full_response_from_tool():
    "A tool that opts into no-truncation by returning FullResponse must have that type preserved end-to-end."
    ns = {"notebook_xml": lambda: FullResponse("<ipython-notebook>...</ipython-notebook>")}
    reg = ToolRegistry.from_ns(ns)
    bns = _BridgeNS(reg)
    caller = bns.get("notebook_xml")
    res = asyncio.run(caller())
    assert isinstance(res, FullResponse), f"FullResponse tool return must survive the bridge, got {type(res).__name__}"


# ---- claude_client (jsonl cleanup predicates + formatter) ----

async def _aiter(*items):
    for o in items: yield o


def _jsonl(tmp_path, name, *objs):
    p = tmp_path/name
    p.write_text("".join(json.dumps(o) + "\n" for o in objs))
    return p


def test_session_belongs_to_us_matches_any_session_id(tmp_path):
    p = _jsonl(tmp_path, "a.jsonl", dict(sessionId="abc", type="user"), dict(sessionId="abc", type="assistant"))
    assert claude._session_belongs_to_us(p, {"abc"}) is True
    assert claude._session_belongs_to_us(p, {"xyz"}) is False


def test_session_belongs_to_us_empty_or_bad_file(tmp_path):
    empty = tmp_path/"empty.jsonl"
    empty.write_text("")
    bad = tmp_path/"bad.jsonl"
    bad.write_text("not json\n")
    assert claude._session_belongs_to_us(empty, {"abc"}) is False
    assert claude._session_belongs_to_us(bad, {"abc"}) is False


def test_ai_title_stub_identifies_title_only_file(tmp_path):
    stub = _jsonl(tmp_path, "stub.jsonl", dict(type="ai-title", aiTitle="x", sessionId="z"))
    mixed = _jsonl(tmp_path, "mixed.jsonl", dict(type="ai-title", sessionId="z"), dict(type="user", sessionId="z"))
    assert claude._is_ai_title_stub(stub) is True
    assert claude._is_ai_title_stub(mixed) is False


def test_claude_tool_name_strips_mcp_prefix():
    assert claude._tool_name("mcp__ipy__pyrun") == "pyrun"
    assert claude._tool_call("mcp__ipy__pyrun", dict(code="1+1")) == 'pyrun(code="1+1")'


def test_claude_compact_tool_leaves_blank_line_after_summary():
    assert claude._compact_tool("mcp__ipy__pyrun", dict(code="1+1"), "2") == '\n\n🔧 pyrun(code="1+1") => 2\n\n'


async def test_claude_async_stream_formatter_shows_live_tool_and_stores_compact_summary():
    fmt = claude.AsyncStreamFormatter()
    fmt.is_tty = True
    seen = []
    done = dict(kind="tool_complete", name="mcp__ipy__pyrun", input=dict(code="1+1"), content="2")
    stream = _aiter(dict(kind="tool_start", name="mcp__ipy__pyrun", input=dict(code="1+1")), done, "2")

    async for _ in fmt.format_stream(stream): seen.append(fmt.display_text)

    assert seen[0] == '⌛ `pyrun(code="1+1")`'
    assert '🔧 pyrun(code="1+1") => 2\n\n2' in fmt.final_text
    assert seen[-1].endswith("\n\n2")


# ---- codex_client (formatter + complete path + turn consumer) ----

class FakeCodexClient:
    def __init__(self):
        self.started = []
        self.turns = []

    async def start_thread(self, **kwargs):
        self.started.append(kwargs)
        return "thread_1"

    async def turn_stream(self, *args, **kwargs):
        self.turns.append((args, kwargs))
        yield "done"


def test_codex_tool_name_strips_mcp_prefix():
    assert codex._tool_name("mcp__ipy__pyrun") == "pyrun"
    assert codex._tool_call("mcp__ipy__pyrun", dict(code="1+1")) == 'pyrun(code="1+1")'


def test_codex_compact_tool_leaves_blank_line_after_summary():
    assert codex._compact_tool("mcp__ipy__pyrun", dict(code="1+1"), "2") == '\n\n🔧 pyrun(code="1+1") => 2\n\n'


async def test_codex_async_stream_formatter_shows_live_tool_and_stores_compact_summary():
    fmt = codex.AsyncStreamFormatter()
    fmt.is_tty = True
    seen = []
    done = dict(kind="tool_complete", name="mcp__ipy__pyrun", input=dict(code="1+1"), content="2")
    stream = _aiter(dict(kind="tool_start", name="mcp__ipy__pyrun", input=dict(code="1+1")), done, "2")

    async for _ in fmt.format_stream(stream): seen.append(fmt.display_text)

    assert seen[0] == '⌛ `pyrun(code="1+1")`'
    assert '🔧 pyrun(code="1+1") => 2\n\n2' in fmt.final_text
    assert seen[-1].endswith("\n\n2")


async def test_codex_complete_uses_toolless_ephemeral_turn(shell, monkeypatch):
    fake = FakeCodexClient()
    monkeypatch.setattr(codex, "get_codex_client", lambda: fake)
    backend = codex.CodexBackend(shell=shell, system_prompt="system")

    res = await backend.complete("hi", model="gpt-5.4-mini")

    assert str(res) == "done"
    assert fake.started == [dict(model="gpt-5.4-mini", sp="system", dynamic_tools=None, ephemeral=True, cwd=backend.ctx.cwd)]
    assert fake.turns == [(("thread_1", "hi"), dict(tools=None, think=COMPLETION_THINK, cwd=backend.ctx.cwd))]


async def test_codex_archives_created_thread_ids():
    client = codex._CodexAppServer()
    seen = []

    async def _ensure_initialized(): pass
    async def _request(method, params):
        seen.append((method, params))
        return {}

    client.ensure_initialized = _ensure_initialized
    client.request = _request
    client.created_thread_ids.add("thread_1")

    await client.archive_thread("thread_1")

    assert seen == [("thread/archive", dict(threadId="thread_1"))]
    assert client.created_thread_ids == set()


async def test_codex_consume_turn_emits_tool_start_and_complete_events():
    client = codex._CodexAppServer()
    client.events = asyncio.Queue()
    thread_id,turn_id = "thread_1","turn_1"
    started = dict(method="item/started", params=dict(threadId=thread_id, turnId=turn_id,
        item=dict(type="dynamicToolCall", id="tool_1", tool="mcp__ipy__pyrun", arguments=dict(code="1+1"))))
    done = dict(method="item/completed", params=dict(threadId=thread_id, turnId=turn_id,
        item=dict(type="dynamicToolCall", id="tool_1", tool="mcp__ipy__pyrun", arguments=dict(code="1+1"), contentItems=[dict(type="inputText", text="2")])))
    msgs = [started, done, dict(method="turn/completed", params=dict(threadId=thread_id, turnId=turn_id, turn=dict(id=turn_id)))]
    for msg in msgs: await client.events.put(msg)

    chunks = [o async for o in client._consume_turn(thread_id, turn_id, {})]

    assert chunks == [dict(kind="tool_start", name="mcp__ipy__pyrun", input=dict(code="1+1")),
        dict(kind="tool_complete", name="mcp__ipy__pyrun", input=dict(code="1+1"), content="2")]


# ---- mcp_server (unix-socket RPC) ----

async def _rpc(sock_path, method, params=None):
    reader, writer = await asyncio.open_unix_connection(sock_path)
    try:
        writer.write((json.dumps(dict(id=1, method=method, params=params or {})) + "\n").encode())
        await writer.drain()
        return json.loads(await reader.readline())
    finally:
        writer.close()
        try: await writer.wait_closed()
        except Exception: pass


async def _with_server(ns, fn):
    srv = await ToolSocketServer(ToolRegistry.from_ns(ns)).start()
    try: return await fn(srv.sock_path)
    finally: await srv.stop()


def test_socket_list_tools():
    ns = {"pyrun": RunPython(g={})}
    resp = asyncio.run(_with_server(ns, lambda p: _rpc(p, "list_tools")))
    names = [o["name"] for o in resp["result"]]
    assert "pyrun" in names


def test_socket_call_tool_pyrun():
    ns = {}
    ns["pyrun"] = RunPython(g=ns)
    ns["hidden"] = "walnut"
    resp = asyncio.run(_with_server(ns, lambda p: _rpc(p, "call_tool", dict(name="pyrun", args=dict(code="hidden")))))
    assert resp["result"]["isError"] is False
    assert "walnut" in resp["result"]["content"][0]["text"]


def test_socket_unknown_method():
    resp = asyncio.run(_with_server({}, lambda p: _rpc(p, "nonsense")))
    assert "error" in resp

import asyncio, shutil

import pytest

from safepyrun import RunPython

from ipyai.mcp_server import ToolSocketServer
from ipyai.tooling import ToolRegistry


async def _bridge_roundtrip(ns):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    srv = await ToolSocketServer(ToolRegistry.from_ns(ns)).start()
    try:
        params = StdioServerParameters(command=shutil.which("ipyai-mcp-bridge"), args=[], env=dict(IPYAI_MCP_SOCK=srv.sock_path))
        async with stdio_client(params) as (rx,tx):
            async with ClientSession(rx, tx) as session:
                await session.initialize()
                listed = await session.list_tools()
                names = [t.name for t in listed.tools]
                called = await session.call_tool("python", dict(code="hidden"))
                text = called.content[0].text if called.content else ""
                return names, called.isError, text
    finally: await srv.stop()


def test_bridge_lists_and_callspython():
    if shutil.which("ipyai-mcp-bridge") is None: pytest.skip("ipyai-mcp-bridge not installed")
    ns = {}
    ns["python"] = RunPython(g=ns)
    ns["hidden"] = "walnut"
    names,is_error,text = asyncio.run(_bridge_roundtrip(ns))
    assert "python" in names
    assert is_error is False
    assert "walnut" in text

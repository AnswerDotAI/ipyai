"Uses the session kernel fixture. Verifies tool-bridge dispatch, variable-ref reads, and iopub output buffer shape."
import asyncio


def test_bridge_runspython_and_reads_vars(kernel_bridge, kernel_loop):
    loop = kernel_loop

    async def _go():
        await kernel_bridge._exec("x = 41\ny = x + 1")

        val = await kernel_bridge.read_var("y")
        assert val == 42

        vals = await kernel_bridge.read_vars(["x", "y"])
        assert vals == {"x": 41, "y": 42}

        names = await kernel_bridge.available_names(force=True)
        assert "python" in names, f"python missing from {names}"

        result = await kernel_bridge.call_tool("python", dict(code="2 + 3"))
        assert "5" in result

        bash_res = await kernel_bridge.call_tool("bash", dict(cmd="printf 'x\\n'", as_dict=True))
        assert "x" in bash_res, f"bool tool arg should be marshalled to Python True: {bash_res!r}"

        schemas = await kernel_bridge.schemas()
        python_schema = next(s for s in schemas if s["function"]["name"] == "python")
        assert "parameters" in python_schema["function"]

    loop.run_until_complete(_go())


def test_call_tool_uses_longer_timeout_than_probe_exec(kernel_bridge, kernel_loop, monkeypatch):
    "Tool calls can legitimately run longer than the probe/exec default — `call_tool` must use a tool-specific timeout so a slow tool does not trip `_EXEC_TIMEOUT`."
    import ipyai.kernel_bridge as kb
    monkeypatch.setattr(kb, "_EXEC_TIMEOUT", 0.3)
    monkeypatch.setattr(kb, "CUSTOM_TOOL_NAMES", tuple(list(kb.CUSTOM_TOOL_NAMES) + ["slow_tool"]))

    async def _go():
        await kernel_bridge._exec("import time\ndef slow_tool(): time.sleep(1.2); return 'done'\n", timeout=5)
        await kernel_bridge.available_names(force=True)
        res = await kernel_bridge.call_tool("slow_tool", {})
        assert res == "done", f"slow tool should complete; got {res!r}"

    kernel_loop.run_until_complete(_go())


def test_bridge_preserves_full_response_from_kernel_tool(kernel_bridge, kernel_loop, monkeypatch):
    "A kernel-side tool that opts out of truncation with `FullResponse` must have its type preserved across the bridge, so downstream truncation skips it."
    import ipyai.kernel_bridge as kb
    from aidialog.msg_parts import FullResponse
    monkeypatch.setattr(kb, "CUSTOM_TOOL_NAMES", tuple(list(kb.CUSTOM_TOOL_NAMES) + ["notebook_xml"]))

    async def _go():
        payload = "<ipynb>" + ("x" * 5000) + "</ipynb>"
        await kernel_bridge._exec(
            "from aidialog.msg_parts import FullResponse\n"
            f"def notebook_xml(): return FullResponse({payload!r})\n")
        names = await kernel_bridge.available_names(force=True)
        assert "notebook_xml" in names, f"monkeypatch should expose notebook_xml: {names}"

        res = await kernel_bridge.call_tool("notebook_xml", {})

        assert isinstance(res, FullResponse), f"FullResponse type must survive the kernel bridge, got {type(res).__name__}"
        assert str(res) == payload

    kernel_loop.run_until_complete(_go())


def test_iopub_buffer_captures_stream_and_display(session_kernel):
    "Teeing iopub via install_iopub_tee populates the shell's output_buffer."
    from collections import defaultdict
    loop = session_kernel["loop"]
    client = session_kernel["client"]

    captured = defaultdict(str)

    def _append(ec, text):
        if text is None: return
        captured[ec] += text

    def _capture(msg):
        typ = msg.get("msg_type")
        content = msg.get("content") or {}
        parent = msg.get("parent_header") or {}
        ec = parent.get("execution_count") or content.get("execution_count")
        if typ == "stream": _append(ec, content.get("text"))
        elif typ == "execute_result":
            data = content.get("data") or {}
            if "text/plain" in data: _append(ec, data["text/plain"])

    async def _go():
        msg_id = client.execute("print('hello ipyai'); 5+5", silent=False, store_history=False)
        start = loop.time()
        while loop.time() - start < 10:
            try: msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=0.5)
            except asyncio.TimeoutError: continue
            if msg["parent_header"].get("msg_id") != msg_id: continue
            _capture(msg)
            if msg["msg_type"] == "status" and msg["content"].get("execution_state") == "idle": break

    loop.run_until_complete(_go())

    joined = "".join(captured.values())
    assert "hello ipyai" in joined
    assert "10" in joined

def test_run_ignores_foreign_iopub():
    "iopub is a shared broadcast: a stray idle/output from an out-of-band execute (the cwd sync) must not end run() early or paint."
    from ipyai.kernel import KernelSession
    class StubKC:
        def __init__(self):
            self.q = asyncio.Queue()
            self.shell = asyncio.Queue()
        def execute(self, code, allow_stdin=False):
            mid = 'mine'
            for m in [dict(msg_type='status', parent_header=dict(msg_id='foreign'), content=dict(execution_state='idle')),
                      dict(msg_type='execute_result', parent_header=dict(msg_id='foreign'), content=dict(data={'text/plain': 'FOREIGN'})),
                      dict(msg_type='execute_result', parent_header=dict(msg_id=mid), content=dict(data={'text/plain': 'MINE'})),
                      dict(msg_type='status', parent_header=dict(msg_id=mid), content=dict(execution_state='idle'))]:
                self.q.put_nowait(m)
            self.shell.put_nowait(dict(msg_type='execute_reply', parent_header=dict(msg_id=mid), content=dict(status='ok')))
            return mid
        async def get_iopub_msg(self, timeout=None): return await self.q.get()
        async def get_shell_msg(self, timeout=None): return await self.shell.get()
        async def get_stdin_msg(self, timeout=None): return await asyncio.Future()
    class StubMgr:
        async def is_alive(self, kid): return True
    ks = KernelSession()
    ks.kc, ks.mgr = StubKC(), StubMgr()
    outs = []
    asyncio.run(ks.run('x', lambda mt, c: outs.append(c['data']['text/plain'])))
    assert outs == ['MINE']   # the foreign output never painted, and the foreign idle did not end the loop

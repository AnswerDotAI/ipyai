"Uses the session kernel fixture. Verifies tool dispatch through `KernelTools`."
import asyncio, pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")  # the session_kernel fixture's objects live on the session loop


async def test_names_schemas_and_calls(kernel_tools):
    names = await kernel_tools.names()
    assert names[0] == 'py' and 'bash' in names, names
    assert '5' in await kernel_tools.call('py', code='2 + 3')
    res = await kernel_tools.call('bash', cmd="printf 'x\\n'", as_dict=True)
    assert 'x' in res, f"bool tool arg should be marshalled to Python True: {res!r}"
    schemas = await kernel_tools.schemas(names)
    py_schema = next(s for s in schemas if s['function']['name'] == 'py')
    assert 'code' in py_schema['function']['parameters']['properties']
    assert any(s['function']['name'] == 'bash' for s in schemas)


async def test_concurrent_tool_results_stay_apart(session_kernel, kernel_tools):
    "Tool calls running at the same time each get their own result: no shared kernel variable carries them."
    await session_kernel['ks'].exec("import time\ndef slow_echo(x): time.sleep(0.2); return x")
    res = await asyncio.wait_for(asyncio.gather(*(kernel_tools.call('slow_echo', x=f'v{i}') for i in range(4))), 20)
    assert res == ['v0', 'v1', 'v2', 'v3']


async def test_full_response_survives(session_kernel, kernel_tools):
    "A kernel-side tool that opts out of truncation with `FullResponse` keeps that type, so downstream truncation skips it."
    from aidialog.msg_parts import FullResponse
    payload = "<ipynb>" + ("x" * 5000) + "</ipynb>"
    await session_kernel['ks'].exec(f"from aidialog.msg_parts import FullResponse\ndef notebook_xml(): return FullResponse({payload!r})")
    res = await kernel_tools.call('notebook_xml')
    assert isinstance(res, FullResponse), f"FullResponse type must survive the call, got {type(res).__name__}"
    assert str(res) == payload

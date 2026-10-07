import asyncio
import pytest
from ipyai.kernel import KernelSession
from ipyai.tools import KernelTools
from aidialog.msg_parts import FullResponse, ToolResponse, InputImage
import ipyai.config as config
from .helpers import until

async def test_tools_preserve_namespace_outputs_and_results(repl):
    k, tools = repl.k, KernelTools(repl.k.kc)
    out = await tools.call('py', code='zz = 6*7\nprint("side effect")\nzz')
    assert '<stdout>\nside effect\n</stdout>' in out and '<execute_result>\n42\n</execute_result>' in out
    assert await k.kc.retr('zz') == 42 and 'zz = 6*7\nprint("side effect")\nzz' not in await k.kc.retr('In')
    assert repl.ctl.dlg.messages == []  # helper output must not become a user cell
    calls = [tools.call('py', code=c) for c in ['y = 6*7', '1/0', 'y', '1+1']]
    results = await asyncio.wait_for(asyncio.gather(*calls), 20)
    assert 'ZeroDivisionError' in results[1] and [results[0], *results[2:]] == ['', '42', '2']
    img = await tools.call('py', code='from IPython.display import Image, display\nfrom aidialog.dialog import tiny_png\ndisplay(Image(data=tiny_png))')
    assert isinstance(img, ToolResponse) and any(isinstance(p, InputImage) for p in img.content)
    assert 'bash' in await tools.names()
    assert any(s['function']['name'] == 'bash' for s in await tools.schemas(await tools.names()))
    assert 'value' in await tools.call('bash', cmd="printf 'value\\n'", as_dict=True)
    await k.exec('import time\ndef slow_echo(x): time.sleep(0.02); return x')
    assert await asyncio.gather(*(tools.call('slow_echo', x=f'v{i}') for i in range(4))) == ['v0', 'v1', 'v2', 'v3']
    await k.exec('from aidialog.msg_parts import FullResponse\ndef notebook_xml(): return FullResponse("x"*5000)')
    result = await tools.call('notebook_xml')
    assert isinstance(result, FullResponse) and result == 'x' * 5000

async def test_owned_kernel_setup_and_attachment(gateway, tmp_path):
    k = await KernelSession(url=gateway).start(cwd=tmp_path, env=dict(IPYAI_TEST_MARK='yes'))
    try:
        config.STARTUP_PATH.parent.mkdir()
        config.STARTUP_PATH.write_text('startup_ran = 7\nstartup_file = __file__\n')
        await k.exec("def bash(**kw): return 'preseeded'")
        await k.setup()
        values = await k.kc.eval("(__import__('os').getcwd(), __import__('os').environ['IPYAI_TEST_MARK'], startup_ran, startup_file)", call_=False)
        assert values == (str(tmp_path), 'yes', 7, str(config.STARTUP_PATH))
        assert await KernelTools(k.kc).call('bash') == 'preseeded'
        got = []
        k.on_jmsg = lambda m: got.append((m.get('parent_header', {}).get('msg_id'), m['msg_type']))
        k.kc.execute("print('tagged'); 6*7", msg_id='unawaited')
        await until(lambda: ('unawaited', 'execute_result') in got)
        await k.kc.reply("'mine'", msg_id='awaited')
        assert {mt for pid, mt in got if pid == 'unawaited'} >= {'stream', 'execute_input', 'execute_result'}
        assert got.count(('awaited', 'execute_result')) == 1
        config.STARTUP_PATH.write_text('1/0\n')
        with pytest.raises(RuntimeError, match='startup.py'): await k.setup()
        att = await KernelSession(url=gateway).start(kernel=k.kid[:8])
        try: assert not att.owned and await att.kc.retr('startup_ran') == 7
        finally: await att.close()
        assert await k.mgr.is_alive(k.kid)
        with pytest.raises(ValueError, match='no kernel matching'): await KernelSession(url=gateway).start(kernel='zzzz-none')
    finally: await k.close()
    assert not await k.mgr.is_alive(k.kid)

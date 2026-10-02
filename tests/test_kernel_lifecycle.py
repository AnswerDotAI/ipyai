"Gateway kernel lifecycle: spawn an owned kernel, set it up, route every message through `on_jmsg`, shut down on close."
import asyncio
from jupyasyncclient.multimanager import JupyAsyncMultiKernelManager
from ipyai.kernel import KernelSession
from ipyai.tools import KernelTools


async def _until(pred, timeout=10):
    for _ in range(int(timeout / 0.05)):
        if pred(): return
        await asyncio.sleep(0.05)
    raise TimeoutError('condition did not settle')


async def test_spawn_setup_route_shutdown(gateway):
    ks = await KernelSession(url=gateway).start()
    assert ks.owned
    await ks.exec("def bash(**kw): return 'sentinel-preseeded'")
    await ks.setup()   # a tool the kernel already defines is left alone
    assert 'sentinel-preseeded' in await KernelTools(ks.kc).call('bash')

    got = []
    ks.on_jmsg = lambda m: got.append((m.get('parent_header', {}).get('msg_id'), m['msg_type']))
    ks.kc.execute("print('tagged'); 6*7", msg_id='cellA.abc123')   # nobody awaits this request
    await _until(lambda: ('cellA.abc123', 'execute_result') in got)
    assert {'stream', 'execute_input', 'execute_result'} <= {mt for pid, mt in got if pid == 'cellA.abc123'}
    await ks.kc.reply("'mine'", msg_id='cellB.abc123')                # a request a run() owns reaches on_jmsg too, once
    assert [mt for pid, mt in got if pid == 'cellB.abc123'].count('execute_result') == 1

    async def answer(jmsg): return 'blue'
    await ks.kc.reply("print('got', input('fav? '))", msg_id='cellC.abc123', on_stdin=answer)
    assert ('cellC.abc123', 'stream') in got

    kid = ks.kid
    await ks.close()
    m = JupyAsyncMultiKernelManager(gateway)
    assert not await m.is_alive(kid), "an owned kernel is shut down on close"

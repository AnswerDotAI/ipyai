"Gateway kernel lifecycle: spawn an owned kernel, seed with skip semantics, pump cell-tagged traffic, shut down on close."
import asyncio
from jupyasyncclient.multimanager import JupyAsyncMultiKernelManager
from ipyai.kernel import KernelSession
from ipyai.kernel_bridge import CUSTOM_TOOL_NAMES, KernelBridge


async def test_spawn_seed_pump_shutdown(gateway):
    ks = await KernelSession(url=gateway).start()
    assert ks.owned
    bridge = KernelBridge(ks.kc)
    await bridge._exec("def bash(**kw): return 'sentinel-preseeded'")
    present = set(await bridge.present_names(CUSTOM_TOOL_NAMES))
    assert 'bash' in present, "preseeded callable should count as present"
    await bridge.seed_tools(skip=present)
    res = await bridge.call_tool('bash', {})
    assert 'sentinel-preseeded' in res, f"seed_tools with skip should have preserved preseeded bash; got {res!r}"
    await bridge._exec("globals().pop('bash', None)")
    await bridge.seed_tools(skip=set(await bridge.present_names(CUSTOM_TOOL_NAMES)))
    names = set(await bridge.available_names(force=True))
    assert 'bash' in names, "after removing preseed and re-seeding, real bash should land"

    # the session pump: `{cell_id}.{token}` traffic reaches on_cell_msg; the bridge plumbing above never does
    got = []
    ks.on_cell_msg = lambda cid, m: got.append((cid, m['msg_type']))
    ks.kc.execute("print('tagged'); 6*7", msg_id='cellA.abc123')
    for _ in range(100):
        if any(mt == 'execute_result' for _, mt in got): break
        await asyncio.sleep(0.05)
    types = [mt for cid, mt in got if cid == 'cellA']
    assert {'stream', 'execute_input', 'execute_result'} <= set(types), f'cell traffic not routed: {got}'
    assert all(cid == 'cellA' for cid, mt in got), f'untagged traffic leaked: {got}'
    outs = []
    ret = await ks.run_cell('cellB', "'mine'", outs.append)   # a live tagged cell: streamed and returned
    assert 'mine' in str(outs) and outs == ret
    assert {cid for cid, mt in got} == {'cellA', 'cellB'}, f'unexpected cell ids: {got}'

    kid = ks.kid
    await ks.close()
    m = JupyAsyncMultiKernelManager(gateway)
    assert not await m.is_alive(kid), "an owned kernel is shut down on close"

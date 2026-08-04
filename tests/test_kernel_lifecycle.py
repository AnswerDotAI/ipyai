"Gateway kernel lifecycle: spawn an owned kernel, seed with skip semantics, shut down on close."
from jupyasyncclient.multimanager import JupyAsyncMultiKernelManager
from ipyai.kernel import KernelSession
from ipyai.kernel_bridge import CUSTOM_TOOL_NAMES, KernelBridge


async def test_spawn_bootstrap_skip_seed_shutdown(gateway):
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
    kid = ks.kid
    await ks.close()
    m = JupyAsyncMultiKernelManager(gateway)
    assert not await m.is_alive(kid), "an owned kernel is shut down on close"
    await m.aclose()

"Attach to an existing gateway kernel by id prefix: taken as found, and our close never shuts it down."
from jupyasyncclient.multimanager import JupyAsyncMultiKernelManager
from ipyai.kernel import KernelSession
from ipyai.kernel_bridge import KernelBridge


async def test_attach_existing_kernel_without_shutdown(gateway):
    owner = await KernelSession(url=gateway).start()
    ob = KernelBridge(owner.kc)
    await ob._exec("hidden = 'walnut'\nq = 7")
    att = await KernelSession(url=gateway).start(kernel=owner.kid[:8])
    assert not att.owned and att.kid == owner.kid
    sb = KernelBridge(att.kc)
    assert await sb.read_var('hidden') == 'walnut'   # live state is the point of attaching
    assert await sb.read_var('q') == 7
    outs = [o async for o in att.run_cell('cellY', "print('post-attach')")]
    assert any('post-attach' in o.get('text', '') for o in outs if o['output_type'] == 'stream')
    await att.close()   # attached: the kernel must survive our close
    assert await owner.mgr.is_alive(owner.kid), "kernel must still be alive after attached-client close"
    kid = owner.kid
    await owner.close()  # owned: now it goes
    m = JupyAsyncMultiKernelManager(gateway)
    assert not await m.is_alive(kid)

"Attach to an existing gateway kernel by id prefix: taken as found, and our close never shuts it down."
import asyncio
from jupyasyncclient.multimanager import JupyAsyncMultiKernelManager
from ipyai.kernel import KernelSession
from ipyai.kernel_bridge import KernelBridge


def test_attach_existing_kernel_without_shutdown(gateway):
    async def go():
        owner = await KernelSession(url=gateway).start()
        ob = KernelBridge(owner.kc)
        await ob._exec("hidden = 'walnut'\nq = 7")
        att = await KernelSession(url=gateway).start(kernel=owner.kid[:8])
        assert not att.owned and att.kid == owner.kid
        sb = KernelBridge(att.kc)
        assert await sb.read_var('hidden') == 'walnut'   # live state is the point of attaching
        assert await sb.read_var('q') == 7
        outs = []
        await att.run("print('post-attach')", lambda mt, c: outs.append((mt, c)))
        assert any('post-attach' in c.get('text', '') for mt, c in outs if mt == 'stream')
        await att.close()   # attached: the kernel must survive our close
        assert await owner.mgr.is_alive(owner.kid), "kernel must still be alive after attached-client close"
        kid = owner.kid
        await owner.close()  # owned: now it goes
        m = JupyAsyncMultiKernelManager(gateway)
        assert not await m.is_alive(kid)
        await m.aclose()
    asyncio.run(go())

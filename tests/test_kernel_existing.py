"Attach to an existing gateway kernel by id prefix: taken as found, and our close never shuts it down."
import pytest
from jupyasyncclient.multimanager import JupyAsyncMultiKernelManager
from ipyai.kernel import KernelSession


async def test_attach_existing_kernel_without_shutdown(gateway):
    owner = await KernelSession(url=gateway).start()
    await owner.exec("hidden = 'walnut'\nq = 7")
    att = await KernelSession(url=gateway).start(kernel=owner.kid[:8])
    assert not att.owned and att.kid == owner.kid
    assert await att.kc.eval('hidden', call_=False) == 'walnut'   # live state is the point of attaching
    assert await att.kc.eval('q', call_=False) == 7
    await att.close()   # attached: the kernel must survive our close
    assert await owner.mgr.is_alive(owner.kid), "kernel must still be alive after attached-client close"
    with pytest.raises(ValueError, match='no kernel matching'): await KernelSession(url=gateway).start(kernel='zzzz-none')
    kid = owner.kid
    await owner.close()  # owned: now it goes
    m = JupyAsyncMultiKernelManager(gateway)
    assert not await m.is_alive(kid)

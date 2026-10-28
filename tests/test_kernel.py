import asyncio
from aidialog.msg_parts import ToolResult, InputImage
from ipyai.tools import KernelTools
from .helpers import until

async def test_tools_preserve_namespace_outputs_and_results(repl):
    k, tools = repl.k, KernelTools(repl.k.kc)
    await k.setup()
    code = 'zz = 6*7\nprint("side effect")\nzz'
    out = (await tools.call('py', code=code)).text
    assert '<stdout>\nside effect\n</stdout>' in out and '<execute_result>\n42\n</execute_result>' in out
    assert await k.kc.retr('zz') == 42 and code not in await k.kc.retr('In')
    assert repl.ctl.dlg.messages == []  # helper output must not become a user cell
    calls = [tools.call('py', code=c) for c in ['y = 6*7', '1/0', 'y', '1+1']]
    results = [r.text for r in await asyncio.wait_for(asyncio.gather(*calls), 20)]
    assert 'ZeroDivisionError' in results[1] and [results[0], *results[2:]] == ['', '42', '2']
    img = await tools.call('py', code='from IPython.display import Image, display\nfrom aidialog.dialog import tiny_png\ndisplay(Image(data=tiny_png))')
    assert isinstance(img, ToolResult) and any(isinstance(p, InputImage) for p in img.text)
    assert any(s['function']['name'] == 'bash' for s in await tools.schemas())
    assert 'value' in (await tools.call('bash', cmd="printf 'value\\n'", as_dict=True)).text
    await k.exec(r'''from IPython.display import Markdown
class NotebookReport(Markdown): full = True
def notebook_xml(): return NotebookReport('x'*6000)
def echo(x): return x''')
    assert [r.text for r in await asyncio.gather(*(tools.call('echo', x=f'v{i}') for i in range(4)))] == ['v0', 'v1', 'v2', 'v3']
    result = await tools.call('notebook_xml')
    assert result.md and result.full and result.text == 'x' * 6000
    received = []
    k.on_jmsg = lambda m: received.append((m.get('parent_header', {}).get('msg_id'), m['msg_type']))
    k.kc.execute("print('tagged'); 6*7", msg_id='unawaited')
    await until(lambda: ('unawaited', 'execute_result') in received)
    await k.kc.reply("'mine'", msg_id='awaited')
    assert {mt for pid, mt in received if pid == 'unawaited'} >= {'stream', 'execute_input', 'execute_result'}
    assert received.count(('awaited', 'execute_result')) == 1

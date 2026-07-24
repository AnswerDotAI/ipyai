"The %ipyai magic end-to-end: kernel-side async magic -> comm -> host app dispatch -> ack as cell output."
import asyncio
from teleprint.testing import EmuTty
from ipyai.cli import App
from ipyai.assistant import Assistant

def mk_cfg(**kw):
    return dict(model='m', suggest_model='cm', think='l',
                code_theme='ansi_dark', prompt_mode=False) | kw

async def _settle(app, pred, timeout=25):
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        await asyncio.sleep(0.05)
        if not app.k.busy and pred(): return
    raise TimeoutError('kernel run did not settle')

def _mk_app():
    tty = EmuTty(70, 24)
    return tty, App(tty, history=None, assistant=Assistant(cfg=mk_cfg()))

def test_ipyai_magic_e2e():
    "Settings, setter/getter, prompt toggle, unknown command -- all through a real ipymini kernel."
    async def go():
        tty, app = _mk_app()
        async with app.k:
            app.paint()
            await app.k.run('%load_ext ipyai.magic', lambda *a: None)  # what attach_assistant execs
            app.comp.on_bytes(b'%ipyai\r')
            await _settle(app, lambda: 'suggest_model = cm' in tty.term.contents())
            assert 'think = l' in tty.term.contents()
            app.comp.on_bytes(b'%ipyai model sonnet\r')
            await _settle(app, lambda: app.assistant.model == 'sonnet')  # the setter applied through the round-trip
            app.comp.on_bytes(b'%ipyai think\r')  # getter form: no value shows the current one
            await _settle(app, lambda: 'think = l' in tty.term.contents())
            assert app.assistant.think == 'l'
            app.comp.on_bytes(b'%ipyai prompt\r')
            await _settle(app, lambda: app.mode == 'prompt')
            app.comp.on_bytes(b'%ipyai prompt\r')  # % lines still reach the kernel in prompt mode
            await _settle(app, lambda: app.mode == 'code')
            app.assistant.dlg.mk_message('x = 1', msg_type='code')
            app.comp.on_bytes(b'%ipyai reset\r')  # kernel side runs history_manager.new_session() for real
            await _settle(app, lambda: 'x = 1' not in str(app.assistant.dlg.messages))  # reset swapped in a fresh dialog
            assert len(app.assistant.dlg) == 0  # and the reset command itself was not recorded
            app.comp.on_bytes(b'%ipyai nonsense\r')
            await _settle(app, lambda: 'unknown %ipyai command' in tty.term.contents())
            assert app.assistant.model == 'sonnet'  # the failed command changed nothing
    asyncio.run(go())

def test_ipyai_magic_assignment():
    "x = %ipyai ... captures the ack (async line magic assignment form through the kernel)."
    async def go():
        tty, app = _mk_app()
        async with app.k:
            app.paint()
            await app.k.run('%load_ext ipyai.magic', lambda *a: None)
            app.comp.on_bytes(b'x = %ipyai think s\r')
            await _settle(app, lambda: not app.k.busy and app.assistant.think == 's')
            assert await app.k.kc.eval_expr('str(x)') == 'think = s'
    asyncio.run(go())

def test_ipyai_magic_no_host():
    "With no host listening the magic fails loudly instead of hanging or silently no-opping."
    async def go():
        tty, app = _mk_app()
        async with app.k:
            app.paint()
            await app.k.run('%load_ext ipyai.magic', lambda *a: None)
            await app.k.run('import ipyai.magic as _m; _m._TIMEOUT = 0.5', lambda *a: None)
            app.k.on_comm = None  # nobody home
            app.comp.on_bytes(b'%ipyai model haiku\r')
            await _settle(app, lambda: 'no ipyai host attached' in tty.term.contents())
            assert app.assistant.model == 'm'
    asyncio.run(go())

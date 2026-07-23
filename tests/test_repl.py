"Checkpoint 0 end-to-end: the App on a FakeTty with a real ipymini kernel."
import asyncio
from teleprint.testing import FakeTty
from ipyai.cli import App

async def _settle(app, pred, timeout=25):
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        await asyncio.sleep(0.05)
        if not app.k.busy and pred(): return
    raise TimeoutError('kernel run did not settle')

def test_checkpoint0_repl():
    async def go():
        tty = FakeTty(60, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'6*7\r')
            await _settle(app, lambda: '42' in tty.term.text())
            scr = tty.term.text().splitlines()
            assert '>>> 6*7' in scr and any(l.endswith('42') for l in scr)
            app.comp.on_bytes(b'print("hi kernel")\r')
            await _settle(app, lambda: 'hi kernel' in tty.term.text())
            app.comp.on_bytes(b'1/0\r')
            await _settle(app, lambda: 'ZeroDivisionError' in tty.term.contents())
            app.comp.on_bytes(b'import o\t')
            await _settle(app, lambda: app.menu is not None)
            assert 'os' in app.menu.matches
            assert tty.term.cursor[1] == app.comp._park - app.comp._coff
    asyncio.run(go())

def test_checkpoint1_multiline():
    "Smart Enter via is_complete_request; alt-enter always inserts a newline."
    async def go():
        tty = FakeTty(60, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'def f():\r')  # incomplete: Enter continues with auto-indent
            await _settle(app, lambda: '\n' in app.buf.text)
            assert app.buf.text == 'def f():\n    '
            assert '...' in tty.term.text().splitlines()  # continuation line is painted
            app.comp.on_bytes('return 6*7\r'.encode())
            await _settle(app, lambda: app.buf.text.count('\n') == 2)  # still incomplete: block open
            app.comp.on_bytes(b'\r')  # blank line closes the block: submits
            await _settle(app, lambda: not app.buf.text and not app.k.busy)
            app.comp.on_bytes(b'f()\r')
            await _settle(app, lambda: '42' in tty.term.text())
            # alt-enter: unconditional newline even though '1+1' is complete
            app.comp.on_bytes(b'1+1\x1b\r')
            await asyncio.sleep(0.2)
            assert app.buf.text == '1+1\n'
    asyncio.run(go())

def test_completion_menu_and_inspect():
    "Tab auto-selects the first match; Tab/shift+Tab cycle; Enter accepts; shift+Tab bare inspects."
    async def go():
        tty = FakeTty(70, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'import o\t')
            await _settle(app, lambda: app.menu is not None)
            m = app.menu.matches
            assert 'os' in m
            assert app.menu.i == 0 and app.buf.text == 'import ' + m[0]  # first match auto-selected
            app.comp.on_bytes(b'\t')
            assert app.menu.i == 1 and app.buf.text == 'import ' + m[1]
            app.comp.on_bytes(b'\x1b[Z')      # shift+tab with menu open: cycle back
            assert app.menu.i == 0
            app.comp.on_bytes(b'\r')          # Enter accepts: menu gone, nothing submitted
            assert app.menu is None and app.buf.text == 'import ' + m[0]
            assert not app.k.busy
            app.comp.on_bytes(b'\x15print')   # ctrl+u clear, then a name to inspect
            app.comp.on_bytes(b'\x1b[Z')      # shift+tab with no menu: signature tooltip
            await _settle(app, lambda: app.tip is not None)
            assert 'print' in tty.term.text().rsplit('>>> ', 1)[-1]  # signature painted below the prompt
            app.comp.on_bytes(b'(')           # typing dismisses the tooltip
            assert app.tip is None and app.buf.text == 'print('
            app.comp.on_bytes(b'"a", \x1b[Z')  # shift+tab inside the call: Signature panel
            await _settle(app, lambda: app.tip is not None)
            from teleprint.widgets import Signature
            assert isinstance(app.tip, Signature) and app.tip.name == 'print'
            assert app.tip.active == 0  # *args soaks the positional overflow
    asyncio.run(go())

def test_matplotlib_end_to_end():
    "A real matplotlib figure arrives as one placeholder image block (deduped across display_data/execute_result)."
    async def go():
        tty = FakeTty(60, 16)
        app = App(tty, history=None)
        assert app.detect_kitty()
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'%matplotlib inline\r')
            await _settle(app, lambda: not app.k.busy)
            code = 'import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([1,2,3])\nfig'
            app.comp.print_block(code, tag='in')  # bypass Enter/check for a multiline cell
            await app.run_cell(code)
            scr = tty.term.contents()
            assert '\U0010eeee' in scr
            assert len([b for b in app.comp.blocks.values() if b.tag == 'image']) == 1  # shown once, not twice
    asyncio.run(go())

def test_interleaved_stream_and_display():
    "print / display / print: three blocks, in order, no only-last-block-can-grow crash."
    async def go():
        tty = FakeTty(60, 16)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            code = "from IPython.display import display\nprint('aaa')\ndisplay('mid')\nprint('bbb')"
            app.comp.print_block(code, tag='in')
            await app.run_cell(code)
            scr = tty.term.contents().splitlines()
            find = lambda s: next(i for i, l in enumerate(scr) if l.endswith(s))
            ia, im, ib = find('aaa'), find("'mid'"), find('bbb')
            assert ia < im < ib
            outs = [b for b in app.comp.blocks.values() if b.tag in ('out', 'result')]
            assert len(outs) == 3  # two stream blocks split around the display
    asyncio.run(go())

def test_transcript_mode_submit():
    "ctrl-T browses; typing lands in the shared composer; Enter submits and returns to the live screen."
    async def go():
        tty = FakeTty(50, 12)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'print("early bird")\r')
            await _settle(app, lambda: 'early bird' in tty.term.text())
            app.comp.on_bytes(b'\x14')             # ctrl-T: enter the transcript view
            assert app.tv.active
            app.comp.on_bytes(b'1+1')              # typing goes to the composer while browsing
            assert app.buf.text == '1+1'
            assert '1+1' in tty.term.text().splitlines()[-1]
            app.comp.on_bytes(b'\x1b[A')           # up: block cursor moves, not history
            assert app.buf.text == '1+1'
            app.comp.on_bytes(b'\r')               # Enter with content: submit and leave
            assert not app.tv.active
            await _settle(app, lambda: any(l.endswith('2') for l in tty.term.text().splitlines()))
    asyncio.run(go())

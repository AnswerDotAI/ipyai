"Milestone 0 end-to-end: the App on a EmuTty with a real ipymini kernel."
import asyncio, base64
from teleprint.testing import EmuTty
from ipyai.cli import App

async def _settle(app, pred, timeout=25):
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        await asyncio.sleep(0.05)
        if not app.k.busy and pred(): return
    raise TimeoutError('kernel run did not settle')

async def test_milestone0_repl():
    tty = EmuTty(60, 14)
    app = App(tty, history=None)
    async with app.k:
        app.paint()
        app.comp.on_bytes(b'6*7\r')
        await _settle(app, lambda: '42' in tty.term.text())
        scr = tty.term.text().splitlines()
        assert '»»» 6*7' in scr and any(l.endswith('42') for l in scr)
        app.comp.on_bytes(b'print("hi kernel")\r')
        await _settle(app, lambda: 'hi kernel' in tty.term.text())
        app.comp.on_bytes(b'1/0\r')
        await _settle(app, lambda: 'ZeroDivisionError' in tty.term.contents())
        app.comp.on_bytes(b'import o\t')
        await _settle(app, lambda: app.menu is not None)
        assert 'os' in app.menu.matches
        assert tty.term.cursor == (app.comp._cursor[1], app.comp._cursor[0])  # parked where the frame said

async def test_milestone1_multiline():
    "Smart Enter via is_complete_request; alt-enter always inserts a newline."
    tty = EmuTty(60, 14)
    app = App(tty, history=None)
    async with app.k:
        app.paint()
        app.comp.on_bytes(b'def f():\r')  # incomplete: Enter continues with auto-indent
        await _settle(app, lambda: '\n' in app.buf.text)
        assert app.buf.text == 'def f():\n    '
        assert '···' in tty.term.text().splitlines()  # continuation line is painted
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

async def test_burst_input_not_flattened():
    """Interior newlines in one key burst (tmux send-keys, unbracketed paste) serialize through
    is_complete instead of concatenating: input queues while the enter round-trip is in flight."""
    tty = EmuTty(70, 14)
    app = App(tty, history=None)
    async with app.k:
        app.paint()
        app.comp.on_bytes(b'def area(w, h):\n    "rect area"\n    return w * h\n\n')
        await _settle(app, lambda: not app.buf.text)
        assert 'SyntaxError' not in tty.term.contents()
        app.comp.on_bytes(b'area(6, 7)\r')
        await _settle(app, lambda: '42' in tty.term.text())

async def test_completion_menu_and_inspect():
    "Tab auto-selects the first match; Tab/shift+Tab cycle; Enter accepts; shift+Tab bare inspects."
    tty = EmuTty(70, 14)
    app = App(tty, history=None)
    async with app.k:
        app.paint()
        app.comp.on_bytes(b'import o\t')
        await _settle(app, lambda: app.menu is not None)
        m = app.menu.matches
        assert 'os' in m
        assert app.menu.i == 0 and app.buf.text == 'import ' + m[0]  # first match auto-selected
        scr = tty.term.text().splitlines()
        assert 'matches)' in scr[-3] and scr[-1].startswith('»»»')  # the menu row sits directly above the status row
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
        scr = tty.term.text().splitlines()
        assert any('print' in l for l in scr[:-2])  # signature rows sit above the status line, never in the tail
        app.comp.on_bytes(b'(')           # typing dismisses the tooltip
        assert app.tip is None and app.buf.text == 'print('
        app.comp.on_bytes(b'"a", \x1b[Z')  # shift+tab inside the call: Signature panel
        await _settle(app, lambda: app.tip is not None)
        from teleprint.widgets import Signature
        assert isinstance(app.tip, Signature) and app.tip.name == 'print'
        assert app.tip.active == 0  # *args soaks the positional overflow

async def test_matplotlib_end_to_end():
    "A real matplotlib figure arrives as one placeholder image block (deduped across display_data/execute_result)."
    tty = EmuTty(60, 16)
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

async def test_interleaved_stream_and_display():
    "print / display / print: three blocks, in order, no only-last-block-can-grow crash."
    tty = EmuTty(60, 16)
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

async def test_transcript_mode_submit():
    "ctrl-T browses; typing lands in the shared composer; Enter submits and returns to the live screen."
    tty = EmuTty(50, 12)
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
        app.comp.on_bytes(b'\x14')             # back in: search and copy ride the input blocks' sources
        writes, w = [], tty.write
        tty.write = lambda d: (writes.append(d), w(d))
        app.comp.on_bytes(b'/early\r')
        assert app.tv.cur == next(b.id for b in app.comp.blocks.values() if b.source == 'print("early bird")')
        app.comp.on_bytes(b'y')
        b64 = base64.b64encode(b'print("early bird")').decode()
        assert any(isinstance(d, str) and d.startswith('\x1b]52;c;' + b64) for d in writes)
        tty.write = w
        app.comp.on_bytes(b'\x1b')
        app.comp.flush_input()                 # first loop timeout arms the pending ESC...
        app.comp.flush_input()                 # ...and the second resolves it: leave
        assert not app.tv.active

def test_empty_composer_hint():
    "An empty composer hints the two ways out of the current mode -- and only those two."
    tty = EmuTty(60, 10)
    app = App(tty, history=None)
    app.paint()
    assert 'M-p prompt · M-s shell' in tty.term.text()   # code mode: the escapes, not the home mode
    app.comp.on_bytes(b'x')
    assert 'M-p prompt' not in tty.term.text()           # any text replaces the hint
    app.comp.on_bytes(b'\x7f')
    assert 'M-p prompt · M-s shell' in tty.term.text()   # empty again: hint returns
    app._set_mode('shell')
    app.paint()
    assert 'M-c code · M-p prompt' in tty.term.text()

def test_throbber_cell():
    "The status line's first cell: a braille spinner while busy, a plain space when idle; no busy text segment."
    tty = EmuTty(120, 10)   # wide enough that the hint text (C-C included) survives the one-row truncation
    app = App(tty, history=None)
    app.paint()
    status = next(l for l in tty.term.text().splitlines() if '[code]' in l)
    assert status.startswith('  [code]')                  # idle: one quiet cell
    app.k.busy = True
    app.paint()
    status = next(l for l in tty.term.text().splitlines() if '[code]' in l)
    assert status[0] in App._SPIN and status[1] == ' '    # busy: the spinner cell
    assert 'responding' not in status and 'running' not in status
    assert 'C-C interrupts' in status                     # discoverability moved into the hints


async def test_nonzero_origin_start():
    "A real launch paints below pre-existing shell output: start()'s CPR anchors the region at the cursor row, not row 0."
    tty = EmuTty(60, 14)
    tty.write(b'$ ls\r\nREADME.md\r\n')      # prior shell history: the emulated cursor is now on row 2
    app = App(tty, history=None)
    try:
        await app.comp.start()
        assert app.comp._top == 2            # anchored by the CPR reply
        app.paint()
        scr = tty.term.text().splitlines()
        assert 'README.md' in scr[1]         # prior content above the region is untouched
        assert '[code]' in tty.term.text()   # and the app's chrome painted below it
    finally: app.comp.stop()                 # start() registered real signal handlers; leave no dispositions behind

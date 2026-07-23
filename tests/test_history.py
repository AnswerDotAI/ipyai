"History provider and its App wiring, against a temp IPython-schema db (no kernel needed)."
import apsw
from teleprint.testing import FakeTty
from ipyai.history import History
from ipyai.cli import App

def mk_hist(tmp_path):
    p = tmp_path/'hist.sqlite'
    con = apsw.Connection(str(p))
    con.execute('CREATE TABLE history (session integer, line integer, source text, source_raw text)')
    rows = [(1, 1, 'x = 1', 'x = 1'), (1, 2, 'print(x)', 'print(x)'), (2, 1, 'x = 1', 'x = 1'), (2, 2, 'import os', 'import os')]
    for r in rows: con.execute('INSERT INTO history VALUES (?,?,?,?)', r)
    con.close()
    return History(path=p)

def test_provider(tmp_path):
    h = mk_hist(tmp_path)
    assert h.items == ['import os', 'x = 1', 'print(x)']  # newest first, deduped
    assert h.suggest('import ') == 'os'
    assert h.suggest('zzz') == ''
    assert h.prev('draft') == 'import os'
    assert h.prev('') == 'x = 1'
    assert h.next() == 'import os'
    assert h.next() == 'draft'  # stash restored at the live edge
    assert h.next() is None

def test_app_wiring(tmp_path):
    tty = FakeTty(50, 10)
    app = App(tty, history=mk_hist(tmp_path))
    app.paint()
    app.comp.on_bytes(b'\x1b[1;3A')  # alt+up: newest history item
    assert app.buf.text == 'import os'
    app.comp.on_bytes(b'\x1b[A')     # plain up on a single line: also history
    assert app.buf.text == 'x = 1'
    app.comp.on_bytes(b'\x1b[1;3B\x1b[1;3B')  # alt+down twice: back to the (empty) stash
    assert app.buf.text == ''
    app.comp.on_bytes(b'import ')    # ghost text appears from history
    assert app.buf.suggestion == 'os'
    assert 'import os' in tty.term.text()  # ghost painted after the cursor
    app.comp.on_bytes(b'\x1b[C')     # right at end accepts
    assert app.buf.text == 'import os' and app.buf.suggestion == ''

PNG_1PX = ('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==')

def test_show_image(tmp_path):
    import base64
    tty = FakeTty(50, 10)
    app = App(tty, history=mk_hist(tmp_path))
    app.paint()
    assert app.detect_kitty()  # the ghostty emulator answers the graphics probe
    app.show_image(base64.b64decode(PNG_1PX))
    assert '\U0010eeee' in tty.term.text()  # the kitty placeholder char landed as a block

def test_show_image_fallback(tmp_path):
    import base64
    tty = FakeTty(50, 10)
    app = App(tty, history=None)
    app.paint()
    app.kitty = False
    app.show_image(base64.b64decode(PNG_1PX))
    scr = tty.term.text()
    assert 'lacks kitty' in scr and '\U0010eeee' not in scr

def test_theme_detection():
    light = App(FakeTty(50, 10, bg=(0xfa, 0xfa, 0xf4)), history=None)
    assert light.detect_theme() == 'ansi_light'
    dark = App(FakeTty(50, 10, bg=(0x1e, 0x1e, 0x2e)), history=None)
    assert dark.detect_theme() == 'ansi_dark'
    silent = App(FakeTty(50, 10), history=None)  # no bg configured: no OSC 11 reply
    assert silent.detect_theme() == 'ansi_dark'  # silence stays dark
    assert light.detect_kitty()  # probes still compose after the refactor

def test_input_line_not_bold():
    "The prompt marker is bold; the highlighted code after it must not inherit that as a base style."
    tty = FakeTty(60, 10)
    app = App(tty, history=None)
    app.buf.insert('for i in range(5):')
    app.paint()
    line = next(segs for bid, segs in app.comp._lines if bid is None and any('>>>' in s.text for s in segs))
    for s in line:
        if '>>>' in s.text: assert s.style and s.style.bold
        elif s.text.strip():  assert not (s.style and s.style.bold), f'bold leaked into {s.text!r}'

def test_refresh_and_local(tmp_path):
    h = mk_hist(tmp_path)
    con = apsw.Connection(str(tmp_path/'hist.sqlite'))
    con.execute("INSERT INTO history VALUES (3, 1, 'new_line', 'new_line')")
    con.close()
    assert h.prev('draft') == 'new_line'  # entering nav refreshes: this-session lines appear
    h.reset_nav()
    h.add_local('typed_now')
    assert h.items[0] == 'typed_now'
    assert h.suggest('typed_') == 'now'

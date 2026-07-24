"History provider and its App wiring, against a temp IPython-schema db (no kernel needed)."
import apsw
from teleprint.testing import EmuTty
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
    tty = EmuTty(50, 10)
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
    tty = EmuTty(50, 10)
    app = App(tty, history=mk_hist(tmp_path))
    app.paint()
    assert app.detect_kitty()  # the ghostty emulator answers the graphics probe
    app.show_image(base64.b64decode(PNG_1PX))
    assert '\U0010eeee' in tty.term.text()  # the kitty placeholder char landed as a block

def test_show_image_fallback(tmp_path):
    import base64
    tty = EmuTty(50, 10)
    app = App(tty, history=None)
    app.paint()
    app.kitty = False
    app.show_image(base64.b64decode(PNG_1PX))
    scr = tty.term.text()
    assert '\U0010eeee' not in scr   # no kitty placeholder emitted in the fallback path

def _jpeg(w, h):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', (w, h), 'red').save(buf, format='JPEG')
    return buf.getvalue()

def test_show_image_jpeg():
    tty = EmuTty(50, 10)
    app = App(tty, history=None)
    app.paint()
    assert app.detect_kitty()
    app.show_image(_jpeg(3, 2))            # converts to PNG for the kitty transmit
    assert '\U0010eeee' in tty.term.text()

def test_on_out_jpeg_fallback():
    import base64
    tty = EmuTty(50, 10)
    app = App(tty, history=None)
    app.paint()
    app.kitty = False
    app.cell_imgs = set()
    app.on_out('display_data', dict(data={'image/jpeg': base64.b64encode(_jpeg(3, 2)).decode()}))
    assert '3x2px' in tty.term.text()      # image/jpeg routed and measured like png

def test_theme_detection():
    light = App(EmuTty(50, 10, bg=(0xfa, 0xfa, 0xf4)), history=None)
    assert light.detect_theme() == 'ansi_light'
    dark = App(EmuTty(50, 10, bg=(0x1e, 0x1e, 0x2e)), history=None)
    assert dark.detect_theme() == 'ansi_dark'
    silent = App(EmuTty(50, 10), history=None)  # no bg configured: no OSC 11 reply
    assert silent.detect_theme() == 'ansi_dark'  # silence stays dark
    assert light.detect_kitty()  # probes still compose after the refactor

def test_input_line_not_bold():
    "The prompt marker is bold; the highlighted code after it must not inherit that as a base style."
    tty = EmuTty(60, 10)
    app = App(tty, history=None)
    app.buf.insert('for i in range(5):')
    app.paint()
    line = next(segs for bid, segs in app.comp._tail if any('»»»' in s.text for s in segs))
    for s in line:
        if '»»»' in s.text: assert s.style and s.style.bold
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

def test_status_truncates():
    "A status line wider than the screen becomes one ellipsis-truncated row, never a wrap."
    tty = EmuTty(40, 10)
    app = App(tty, history=None)
    app.paint()
    lines = tty.term.text().splitlines()
    i = next(j for j, l in enumerate(lines) if l.startswith('»»»'))  # the prompt row now carries the mode hint
    assert lines[i - 1].endswith('…')  # one truncated row directly above the prompt
    assert i == 1                      # and nothing wrapped above it

def test_cwd_scoping(tmp_path):
    "With cwd set, only sessions annotated to that directory count; unannotated sessions vanish (junk-suggestion fix)."
    p = tmp_path/'hist.sqlite'
    con = apsw.Connection(str(p))
    con.execute('CREATE TABLE history (session integer, line integer, source text, source_raw text)')
    con.execute('CREATE TABLE ipyai_sessions (session integer, cwd text)')
    for r in [(1, 1, 'here_a', 'here_a'), (2, 1, 'elsewhere_b', 'elsewhere_b'), (3, 1, 'junk_test_line', 'junk_test_line')]:
        con.execute('INSERT INTO history VALUES (?,?,?,?)', r)
    con.execute("INSERT INTO ipyai_sessions VALUES (1, '/proj/here')")
    con.execute("INSERT INTO ipyai_sessions VALUES (2, '/proj/elsewhere')")   # session 3: unannotated (bare IPython)
    con.close()
    h = History(path=p, cwd='/proj/here')
    assert h.items == ['here_a']
    assert History(path=p, cwd='/proj/nowhere').items == []
    assert History(path=p).items == ['junk_test_line', 'elsewhere_b', 'here_a']  # unscoped keeps the old view

def test_cwd_scoping_without_table(tmp_path):
    "A db no ipyai ever touched: scoped history starts empty rather than erroring."
    p = tmp_path/'virgin.sqlite'
    con = apsw.Connection(str(p))
    con.execute('CREATE TABLE history (session integer, line integer, source text, source_raw text)')
    con.execute("INSERT INTO history VALUES (1, 1, 'x', 'x')")
    con.close()
    h = History(path=p, cwd='/anywhere')
    assert h.items == []
    h.add_local('fresh line')            # the live session still fills history immediately
    assert h.suggest('fresh') == ' line'

def test_startup_picker(tmp_path):
    "The picker owns keys while open: digit picks a row, Enter the newest, n fresh; rows are click targets."
    tty = EmuTty(70, 14)
    app = App(tty, history=None)
    picked = []
    app.load_session = picked.append
    app.picker = [(41, '/p', 'claude-opus-4-6', 3, 'first prompt'), (40, '/p', 'gpt-5.6-luna', 1, 'older one')]
    app.paint()
    scr = tty.term.text()
    assert 'resume a session in this directory' in scr and 'claude-opus-4-6' in scr and 'older one' in scr
    app.comp.on_bytes(b'2')                       # digit picks row 2
    assert picked == [40] and app.picker is None
    app.picker = [(41, '/p', 'm', 3, 'p'), (40, '/p', 'm', 1, 'q')]
    app.paint()
    app.comp.on_bytes(b'\r')                      # Enter: newest
    assert picked == [40, 41]
    app.picker = [(41, '/p', 'm', 3, 'p')]
    app.paint()
    app.comp.on_bytes(b'n')                       # fresh: nothing loaded
    assert picked == [40, 41] and app.picker is None
    assert 'resume a session' not in tty.term.text()  # the transient evaporated

def test_mode_scoped_history(tmp_path):
    "Each mode navigates its own past: code from the kernel's table, prompt from ipyai_prompts, shell from `!` cells."
    p = tmp_path/'hist.sqlite'
    con = apsw.Connection(str(p))
    con.execute('CREATE TABLE history (session integer, line integer, source text, source_raw text)')
    con.execute('CREATE TABLE ipyai_sessions (session integer, cwd text)')
    con.execute('CREATE TABLE ipyai_prompts (id integer primary key, session integer, line integer, prompt text, full_prompt text, response text)')
    con.execute('CREATE TABLE ipyai_cells (session integer, line integer, source text, outputs text)')
    con.execute("INSERT INTO ipyai_sessions VALUES (1, '/proj')")
    con.execute("INSERT INTO history VALUES (1, 1, 'x = 1', 'x = 1')")
    con.execute("INSERT INTO ipyai_prompts VALUES (NULL, 1, 2, 'what is x?', 'full', 'resp')")
    con.execute("INSERT INTO ipyai_cells VALUES (1, 3, '!ls -la', '[]')")
    con.execute("INSERT INTO ipyai_cells VALUES (1, 4, '!# background output', '[]')")  # pseudo-cell: never history
    con.execute("INSERT INTO ipyai_cells VALUES (1, 5, 'x = 1', '[]')")                 # code cell: not shell history
    con.close()
    h = History(path=p, cwd='/proj')
    assert h.items == ['x = 1']
    h.mode = 'prompt'; h.refresh()
    assert h.items == ['what is x?']
    h.mode = 'shell'; h.refresh()
    assert h.items == ['ls -la']

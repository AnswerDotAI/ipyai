"History provider (session-file mining) and its App wiring, plus App UI paths that need no kernel."
import os, time
from teleprint.testing import EmuTty
from aidialog.dialog import Dialog
from ipyai.history import History
from ipyai.session import new_session_path, save_session
from ipyai.cli import App

def _save(root, sources, age=0):
    "One session file of code messages, its mtime pushed `age` seconds into the past."
    d = Dialog(name='t')
    for src in sources: d.mk_message(src, msg_type='code')
    d.path_ = new_session_path(root)
    save_session(d)
    t = time.time() - age
    os.utime(d.path_, (t, t))

def mk_hist(root):
    _save(root, ['x = 1', 'print(x)'], age=60)
    _save(root, ['x = 1', 'import os'], age=30)
    return History(root=root)

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

def test_scoping_and_refresh(tmp_path):
    "Directory scoping is the filesystem: another root sees nothing; nav entry refreshes from disk."
    h = mk_hist(tmp_path)
    other = tmp_path/'elsewhere'
    other.mkdir()
    assert History(root=other).items == []
    _save(tmp_path, ['new_line'])
    assert h.prev('draft') == 'new_line'  # entering nav refreshes: new session lines appear
    h.reset_nav()
    h.add_local('typed_now')
    assert h.items[0] == 'typed_now'
    assert h.suggest('typed_') == 'now'

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

async def test_theme_detection():
    light = App(EmuTty(50, 10, bg=(0xfa, 0xfa, 0xf4)), history=None)
    assert await light.detect_theme() == 'ansi_light'
    dark = App(EmuTty(50, 10, bg=(0x1e, 0x1e, 0x2e)), history=None)
    assert await dark.detect_theme() == 'ansi_dark'
    silent = App(EmuTty(50, 10), history=None)  # no bg configured: no OSC 11 reply
    assert await silent.detect_theme() == 'ansi_dark'  # silence stays dark
    assert await light.detect_kitty()  # probes still compose

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

def test_status_truncates():
    "A status line wider than the screen becomes one ellipsis-truncated row, never a wrap."
    tty = EmuTty(40, 10)
    app = App(tty, history=None)
    app.paint()
    lines = tty.term.text().splitlines()
    i = next(j for j, l in enumerate(lines) if l.startswith('»»»'))  # the prompt row now carries the mode hint
    assert lines[i - 1].endswith('…')  # one truncated row directly above the prompt
    assert i == 1                      # and nothing wrapped above it


from .helpers import keys, send, until

async def test_notebook_input_and_outputs(repl):
    app, tty = repl, repl.tty
    assert '$ ipyai' in tty.term.text()
    assert await app.detect_theme() == 'ansi_light'
    assert await app.detect_kitty()

    await keys(app, b'def area(w, h):\n    "rect area"\n    return w*h\n\n')
    await send(app, 'area(6, 7)')
    assert any(l.endswith('42') for l in tty.term.contents().splitlines())
    await send(app, '1/0')
    assert 'ZeroDivisionError' in tty.term.contents()

    app.comp.on_bytes(b'import o\t')
    await until(lambda: app.menu is not None)
    matches = app.menu.matches
    assert 'os' in matches
    app.comp.on_bytes(b'\t\x1b[Z\r')
    assert app.menu is None and app.buf.text == 'import ' + matches[0] and not app.busy
    app.comp.on_bytes(b'\x15print("a", \x1b[Z')
    await until(lambda: app.tip is not None)
    assert app.tip.name == 'print'
    app.comp.on_bytes(b')')
    assert app.tip is None
    app.comp.on_bytes(b'\x15')

    await send(app, "from IPython.display import display\nprint('aaa')\ndisplay('mid')\nprint('bbb')")
    screen = tty.term.contents()
    assert screen.index('aaa\n') < screen.index("'mid'\n") < screen.index('bbb\n')
    await send(app, 'import threading\nrelease = threading.Event()\ndef later():\n    release.wait()\n    print("late output")\nthreading.Thread(target=later).start()')
    late = app.ctl.dlg.messages[-1]
    await send(app, 'release.set()')
    await until(lambda: 'late output' in late.ai_output)
    assert late.ai_output.count('late output') == 1 and 'late output' not in app.ctl.dlg.messages[-1].ai_output
    await send(app, '%matplotlib inline')
    await send(app, 'import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([1,2,3])\nfig')
    assert '\U0010eeee' in tty.term.contents()
    m = app.ctl.dlg.messages[-1]
    assert len([o for o in m.output if 'image/png' in o.get('data', {})]) == 2
    assert sum(it.kind == 'image' for it in app.view.items(m)) == 1

    await keys(app, b'\x147*8\r')
    assert not app.tv.active and any(l.endswith('56') for l in tty.term.contents().splitlines())
    app.comp.on_bytes(b'x' * 70)
    tty.term.resize(30, 8)
    app._resized()
    assert tty.term.cursor[0] == (len(app.buf.text) + 4) % tty.size[0] and tty.term.cursor[1] < tty.size[1]
    app.buf.clear()
    app.comp.on_bytes(b'line\x1b\r' * 10)
    assert app.buf.text.count('\n') == 10 and tty.term.cursor[1] < 8
    assert 'line' in tty.term.text().splitlines()[-2]

async def test_interrupt_and_private_stdin(repl):
    app = repl
    app.comp.on_bytes(b'import time; print("sleeping", flush=True); time.sleep(30)\r')
    await until(lambda: app.busy and 'sleeping\n' in app.tty.term.contents())
    await keys(app, b'\x03')
    assert 'KeyboardInterrupt' in app.tty.term.contents()
    for code, answer in [("x = input('fav? ')", 'blue'), ("import getpass; pw = getpass.getpass('pw: ')", 'hunter2')]:
        app.comp.on_bytes(code.encode() + b'\r')
        await until(lambda: app.inp is not None)
        app.comp.on_bytes(answer.encode())
        if answer == 'hunter2': assert answer not in app.tty.term.text() and '•••••••' in app.tty.term.text()
        await keys(app, b'\r')
    assert await app.k.kc.retr('x') == 'blue'
    assert 'hunter2' not in str([m.output for m in app.ctl.dlg.messages])

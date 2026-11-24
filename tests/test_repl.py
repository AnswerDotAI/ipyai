import json, os, re
from pyghostty import Terminal

def emulated_screen(pane):
    term = Terminal(pane.width, pane.height)
    term.feed(pane.ansi.replace('\n', '\r\n'))
    return term

def test_interrupt_and_private_stdin(terminal, tmp_path):
    session, line, screen = terminal
    pane = session.pane
    def check_mode(active):
        term = emulated_screen(pane)
        rows = term.text().splitlines()
        bar = next(row for row in rows if active in row)
        for y, row in enumerate(rows):
            start = row.find(active)
            for x in range(pane.width):
                assert bool(term.style(x, y)['underline']) == (start >= 0 and start <= x < start + len(active)), row
        term.close()
        return [bar.index(label) for label in ('Alt-C code', 'Alt-P prompt', 'Alt-S shell')]
    screen('»»»')
    cols = check_mode('[Alt-C code]')
    pane.click('Alt-P prompt')
    screen('›››')
    assert check_mode('[Alt-P prompt]') == cols
    pane.click('Alt-S shell')
    screen(r'\$\$\$')
    assert check_mode('[Alt-S shell]') == cols
    line(';6 * 7')
    screen(r'(?m)\b42$')
    pane.send_keys('M-c')
    screen('»»»')

    gate = tmp_path/'ask'
    os.mkfifo(gate)
    with gate.open('r+b', buffering=0) as signal:
        line("import getpass; open('ask','rb').read(1); answer = getpass.getpass('Password? ')")
        spinning = r'(?m)^([\u2801-\u28ff]) +\[Alt-C code\]'
        first = re.search(spinning, screen(spinning))[1]
        screen(rf'(?m)^(?!{first})[\u2801-\u28ff] +\[Alt-C code\]')
        assert check_mode('[Alt-C code]') == cols
        pane.send_keys('C-t')
        screen('e edit')
        signal.write(b'!')
        screen(r'(?m)^Password\?')
    pane.send('never-display-me')
    assert 'never-display-me' not in screen('•{16}')
    pane.send_keys('C-u', 'C-y', 'Enter')
    screen('»»»')
    pane.send_keys('C-y')  # The password field's cut buffer isn't the notebook composer's.
    line('len(answer)')
    screen(r'(?m)\b16$')

    pane.send_keys('M-Up')
    screen(r'»»» len\(answer\)')
    pane.send_keys('C-u', 'C-t', '7', '*', '8')
    assert 'len(answer)' in screen(r'(?s)Esc to browse.*»»» 7\*8')
    pane.send_keys('Escape', 'i', 'Enter')
    screen(r'(?m)\b56$')
    pane.send_keys('M-r')
    screen(r'»»» 7\*8')
    pane.send_keys('C-u')
    line('7*9')
    screen(r'(?m)\b63$')
    pane.send_keys('C-t', 'h', 'C-t')

    line('print(*range(1000,1020), sep=chr(10))')
    folded = r'1000.*\+[0-9]+ lines'
    screen(folded)
    first_row = re.compile(r'^«\d«(?= 1000)', re.M)
    pane.click(first_row)
    screen(r'(?m)\b1019$')
    pane.wheel(1, wait_ms=1000, settle_ms=100, screen=True)
    assert pane.fmt('#{pane_mode}') == 'copy-mode'
    pane.send_keys('q')
    pane.send_keys('C-t')
    assert not first_row.search(screen(r'(?ms)\b1019$.*e edit'))
    pane.wheel(4)
    screen(r'(?m)^«\d« 1000$')
    pane.send_keys('/', '101[89]', 'Enter')
    screen(r'(?m)\b1019$')
    pane.wheel(4)
    screen(r'(?m)^«\d« 1000$')
    pane.click(first_row)
    assert '1019' not in screen(folded)
    pane.send_keys('C-t')
    line('import time; print("starting"); time.sleep(30)')
    screen(r'(?m)\bstarting$')
    pane.send_keys('C-c')
    screen('KeyboardInterrupt')
    pane.send_keys('C-d')
    assert pane.wait(5000) == 0
    saved = (tmp_path/'keys.ipynb').read_text()
    assert 'never-display-me' not in saved
    cells = json.loads(saved)['cells']
    retries = [c for c in cells if ''.join(c['source']) in ('7*8', '7*9')]
    assert len(retries) == 1 and ''.join(retries[0]['source']) == '7*9' and retries[0]['metadata']['skipped']

def test_notebook_input_and_outputs(terminal, tmp_path):
    session, line, screen = terminal
    pane = session.pane
    screen('»»»')
    pane.send_keys('def area(w, 高):', 'Enter', '    return w*高')
    screen(r'(?m)^··· +return w\*高$')
    capture = pane.screen()
    term = emulated_screen(pane)
    row = term.text().splitlines()[capture.cy]
    assert capture.cx == len(row) + 1  # 高 occupies two terminal cells
    assert term.style(0, capture.cy)['bold']
    assert term.style(row.index('return'), capture.cy)['fg'] != term.style(row.index('w*高'), capture.cy)['fg']
    term.close()
    pane.send_keys('Enter', 'Enter')
    screen(r'(?ms)return w\*高.*^»»» *$')
    line('area(8, 9)')
    screen(r'(?m)\b72$')

    pane.send('import o')
    pane.send_keys('Tab')
    choice = re.search(r'»»» import (o\w+)', screen(r'»»» import o\w+')).group(1)
    pane.send_keys('Tab')
    screen(rf'»»» import (?!{choice}\b)o\w+')
    pane.send_keys('BTab', 'Enter')
    screen(f'»»» import {choice}')
    pane.send_keys('C-u')
    pane.send('print(')
    pane.send_keys('BTab')
    assert '72' in screen(r'\bflush\b')  # The documentation excerpt leaves recent output visible.
    term = emulated_screen(pane)
    y, signature = next((i,s) for i,s in enumerate(term.text().splitlines()) if s.startswith('print('))
    assert term.style(signature.index('*args'), y)['bold'] and not term.style(signature.index('sep'), y)['bold']
    term.close()
    pane.send_keys('C-u')

    worker = "Thread(target=lambda: (go.wait(), clear_output(wait=True), print('late'+' value')))"
    line(f"from IPython.display import clear_output; from threading import Event, Thread; go = Event(); {worker}.start(); print('armed')")
    screen(r'(?m)\barmed$')
    line('go.set()')
    screen(r'(?m)\blate value$')
    pane.send_keys('C-d')
    assert pane.wait(5000) == 0
    *_, parent, release = json.loads((tmp_path/'keys.ipynb').read_text())['cells']
    def streams(cell): return ''.join(''.join(o.get('text', '')) for o in cell['outputs'])
    assert streams(parent).strip() == 'late value' and 'late value' not in streams(release)

def test_notes_and_recalled_drafts(terminal, tmp_path):
    session, line, screen = terminal
    pane = session.pane
    screen('»»»')
    line(':not-a-command')
    screen('unknown command')
    pane.send_keys('C-u')
    line('value = 42')
    assert 'unknown command' in screen(r'(?ms)value = 42.*^»»» *$')
    line(':note')
    pane.send_keys('Escape', '[200~', 'Why this value?', 'C-m', 'C-j', '!first draft', 'Escape', '[201~')
    screen(r'(?m)^··· !first draft$')
    pane.send_keys('F2')
    screen(r'(?m)^\d+$')  # ed reports the size of the draft it opened
    pane.send_keys('$s/first draft/a note, not shell/', 'Enter', 'w', 'Enter', 'q', 'Enter')
    screen(r'(?m)^··· !a note, not shell$')
    pane.send_keys('Enter', 'M-c')
    screen(r'(?s)!a note, not shell.*Alt-C code.*\n»»»\s*$')
    line(r'"two\n" "lines"')
    screen(r'(?s)two.*lines.*Alt-C code.*\n»»»\s*$')
    pane.send_keys('M-r')
    screen(r'(?m)^»»» value = 42\s*\Z')
    pane.send_keys('Escape', wait_ms=1000, until=r'(?m)Alt-S shell\s*\n»»» value = 42\s*\Z')
    pane.send_keys('C-a', 'M-f', 'C-k')
    pane.send(' + 1')
    pane.send_keys('C-e', 'Enter')
    screen(r'(?m)\b43$')
    pane.send('stash')
    pane.send_keys('M-Up')
    screen(r'»»» value \+ 1')
    pane.send_keys('C-w')
    pane.send('2')
    pane.send_keys('M-Down', 'Enter')
    screen(r'(?m)\b44$')
    pane.send_keys('C-d')
    assert pane.wait(5000) == 0
    a, note, literal, b, c = json.loads((tmp_path/'keys.ipynb').read_text())['cells']
    assert note['cell_type'] == 'markdown' and ''.join(note['source']) == 'Why this value?\n!a note, not shell'
    assert literal['cell_type'] == 'markdown' and ''.join(literal['source']) == 'two\nlines'
    assert [''.join(x['source']) for x in (a,b,c)] == ['value = 42', 'value + 1', 'value + 2']

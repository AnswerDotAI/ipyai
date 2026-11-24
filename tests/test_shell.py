import httpx, os, select

def test_persistent_shell_lifecycle(terminal, gateway, tmp_path):
    session, line, screen = terminal
    pane = session.pane
    (tmp_path/'initial'/'room').mkdir(parents=True)
    screen('»»»')
    terminals = f'{gateway}/api/terminals'
    httpx.post(terminals, json={'name':'other-client', 'path':'keys.ipynb'}).raise_for_status()
    line("import os; os.chdir('initial'); print('kernel starts in', os.path.basename(os.getcwd()))")
    screen(r'(?m)\bkernel starts in initial$')
    ready, release = tmp_path/'shell-ready', tmp_path/'shell-release'
    for path in (ready, release): os.mkfifo(path)
    startup = f"printf x > '{ready}'; read -r ipyai_startup_release < '{release}'"
    for rc in ('.bashrc', '.zshrc'): (tmp_path/rc).write_text(startup)
    with ready.open('r+b', buffering=0) as started, release.open('r+b', buffering=0) as resume:
        line('!echo "shell starts in ${PWD##*/}"')
        assert select.select([started], [], [], 5)[0] and started.read(1) == b'x'
        view = pane.poll(wait_ms=1000, settle_ms=100, screen=True).text
        assert 'kernel starts in initial' in view and 'opening shell' in view
        pane.send_keys('C-]')
        screen(r'Alt-S shell\s*\n»»»\s*\Z')
        line(':shell')
        resume.write(b'\n')
        screen(r'(?m)\bshell starts in initial$')
    line("!cd room; export IPYAI_CHECK=42; printf '%s%s\\n' ex ported")
    screen(r'(?m)\bexported$')
    line('!echo "$IPYAI_CHECK in ${PWD##*/}"')
    screen(r'(?m)\b42 in room$')
    pane.send_keys('M-s')
    line('hits=$((hits + 1)); echo history:$hits')
    screen(r'(?m)\bhistory:1$')
    line('!!')
    screen(r'(?m)\bhistory:2$')
    line('echo broken )')
    screen(r'(?i)syntax error|parse error')
    pane.send_keys('for i in 3 4; do', 'M-Enter', '  echo multiline:$i', 'M-Enter', 'done', 'Enter')
    screen(r'(?m)\bmultiline:4$')
    pane.send_keys('M-c')
    line("import os; print('kernel in', os.path.basename(os.getcwd()))")
    screen(r'(?m)\bkernel in room$')

    line(r"""!sh -c 'printf "%s%s\n" before stop; kill -STOP $$; printf "%s%s\n" after resume'""")
    screen(r'(?i)\b(stopped|suspended)\b')
    line(r"!jobs; printf '%s%s\n' table end")
    screen(r'(?m)\btableend$')
    line('!fg')
    screen(r'(?m)\bafterresume$')
    line(':shell')
    screen(r'(?m)^\$\$\$ *\Z')
    pane.send(r'printf "direct-%s\n" "$IPYAI_')
    pane.send_keys('C-]')
    screen(r'(?m)^»»» *\Z')
    line(':shell')
    screen(r'\$IPYAI_ *\Z')
    pane.send_keys('CHECK"', 'Enter')
    screen(r'(?m)^direct-42$')
    pane.send_keys('C-]')
    screen(r'(?m)^»»» *\Z')
    session.windows[0].resize(width=47, height=13)
    line('!stty size')
    screen(r'(?m)\b13 47$')
    kernel, = httpx.get(f'{gateway}/api/kernels').json()
    api = f"{gateway}/api/kernels/{kernel['id']}/cells"
    def cells(): return httpx.get(api).raise_for_status().json()['cells']
    cid = cells()[-1]['id']
    line(f':skip {cid}')
    line(':note Discard on retry')
    screen(r'Discard on retry')
    line('123 + 123')
    screen(r'(?m)\b246$')
    pane.send_keys('C-t', '/', 'stty size', 'Enter', 'E')
    screen(r'(?m)^\$\$\$ stty size\s*\Z')
    pane.send_keys('C-u')
    line("""sh -c 'printf "Numb%s: " er; read -r v; echo got:"$v"'""")
    screen(r'(?m)^Number: *$')
    assert cells()[-1]['id'] == cid
    pane.send_keys('C-]')
    screen(r'(?m)^\$\$\$ *\Z')
    line(':note Keep during retry')
    screen('Keep during retry')
    line(':shell')
    screen(r'(?m)^Number: *$')
    line('42')
    screen(r'(?s)got:42.*Alt-S shell.*\n\$\$\$\s*\Z')
    revised, kept = cells()[-2:]
    assert revised['id'] == cid and revised['metadata']['skipped'] and 'got:42' in ''.join(revised['outputs'][0]['text'])
    assert 'Keep during retry' in ''.join(kept['source'])
    pane.send_keys('M-c')
    line(r"!printf '\033[?1049hprivate%s\033[?1049l' drawing; seq 1600 1640")
    screen(r'1600.*\+[0-9]+ lines')
    screen('»»»')
    pane.send_keys('C-d')
    assert pane.wait(5000) == 0
    saved = httpx.get(f'{gateway}/api/contents/keys.ipynb', params={'fields':'content'}).raise_for_status().json()['content']
    assert '1601' in saved and '1639' in saved and 'privatedrawing' not in saved
    assert [term['name'] for term in httpx.get(terminals).json()] == ['other-client']
    kernel, = httpx.get(f'{gateway}/api/kernels').raise_for_status().json()
    assert kernel['execution_state'] == 'idle'

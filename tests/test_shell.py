"The shell layer over the gateway: a persistent bash/zsh as a rustygate terminal, sentinel boundaries, emulator-cleaned residue."
import asyncio, pyghostty
from ipyai.shell import GateShell


async def _boot(url, sh='bash', size=(80, 24), cwd=None):
    s = await GateShell(url, size=size, cwd=cwd, sh=sh).start()
    with pyghostty.Terminal(*size) as boot: assert (await asyncio.wait_for(s.relay(mirror=boot), 15))[0] == 'prompt'
    return s


async def _run(s, cmd, size=(80, 24)):
    "One command through the shell; returns (result, cleaned residue)."
    with pyghostty.Terminal(*size) as m:
        s.write(cmd.encode() + b'\n')
        res = await asyncio.wait_for(s.relay(mirror=m), 15)
        return res, m.contents().strip()


async def _quit(s):
    s.write(b'exit\n')
    assert await asyncio.wait_for(s.relay(), 15) == 'eof'
    await s.close()


async def _shell_roundtrip(url, sh):
    "Boundary sentinel with exit code + pwd; state persists; the written command never echoes."
    s = await _boot(url, sh, cwd='/tmp')
    res, _ = await _run(s, 'cd /Users && TP_X=7')
    assert res == ('prompt', 0, '/Users')
    res, out = await _run(s, 'echo "$TP_X in $PWD"')
    assert res[0] == 'prompt' and out == '7 in /Users'   # output only: no command echo
    res, _ = await _run(s, 'false')
    assert res[1] == 1
    await _quit(s)


async def test_shell_bash(gateway): await _shell_roundtrip(gateway, 'bash')
async def test_shell_zsh(gateway): await _shell_roundtrip(gateway, 'zsh')


async def test_shell_own_job_control(gateway):
    "fg/bg/jobs/ctrl-Z are the shell's builtins: stop a child, see it in `jobs`, resume and end it."
    s = await _boot(gateway)
    s.write(b'sleep 30\n')
    await asyncio.sleep(0.4)
    s.write(b'\x1a')                                     # ^Z: the shell stops it and prompts (= boundary)
    assert (await asyncio.wait_for(s.relay(), 15))[0] == 'prompt'
    res, out = await _run(s, 'jobs')
    assert 'sleep 30' in out
    s.write(b'fg\n')
    await asyncio.sleep(0.4)
    s.write(b'\x03')                                     # ^C ends the resumed child
    res = await asyncio.wait_for(s.relay(), 15)
    assert res[0] == 'prompt' and res[1] != 0            # 130: died by SIGINT
    await _quit(s)


async def test_altscreen_erases_itself_from_residue(gateway):
    "Jeremy's vim rule needs no detection: the emulator's alt-screen semantics drop it from contents()."
    s = await _boot(gateway)
    res, out = await _run(s, r"printf '\033[?1049hSECRET DRAWING\033[?1049l'; echo visible after")
    assert res[0] == 'prompt'
    assert 'visible after' in out and 'SECRET' not in out
    await _quit(s)


async def test_chatty_output_never_stalls(gateway):
    "The old bg-stall regression, gateway edition: a huge burst relays through a headless mirror without deadlock."
    s = await _boot(gateway)
    with pyghostty.Terminal(80, 24, scrollback=100_000) as m:
        s.write(b'seq 1 20000\n')
        res = await asyncio.wait_for(s.relay(mirror=m), 30)
        assert res[0] == 'prompt' and res[1] == 0
        assert m.contents().splitlines()[-1] == '20000'
    await _quit(s)


async def test_resize_reaches_shell_children(gateway):
    "resize sends set_size; a command inside the shell sees the new size."
    s = await _boot(gateway, size=(50, 12))
    s.resize(97, 41)
    res, out = await _run(s, 'stty size')
    assert res[0] == 'prompt' and '41 97' in out          # stty reports rows cols
    await _quit(s)


async def test_owned_terminal_deleted_on_close(gateway):
    "Fresh-per-session ownership: close() removes the terminal from the gateway registry."
    s = await _boot(gateway)
    names = [t['name'] for t in await s.tc.list_terminals()]
    assert s.tc.name in names
    await s.close()
    names = [t['name'] for t in await s.tc.list_terminals()]
    assert s.tc.name not in names

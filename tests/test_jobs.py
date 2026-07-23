"cp5 jobs layer: bare-`!` lines as pty jobs on the App, embedded `!` staying kernel-side."
import asyncio, os
from teleprint.testing import FakeTty
from ipyai.assistant import route
from ipyai.cli import App

def test_route_job_kinds():
    "Single-line bare-`!` (and `%fg`) go to the jobs layer; everything embedded or multiline stays code."
    assert route('!ls -la', False) == ('job', 'ls -la')
    assert route('  !ls', False) == ('job', 'ls')
    assert route('%fg 2', False) == ('job', '%fg 2')
    assert route('x = !ls', False) == ('code', 'x = !ls')          # embedded: IPython SList semantics
    assert route('!ls\n!pwd', False) == ('code', '!ls\n!pwd')      # multiline: kernel-side
    assert route('!', False) == ('code', '!')                      # nothing to run
    assert route('%fgrep foo', False) == ('code', '%fgrep foo')    # %fg is a word, not a prefix
    assert route('%matplotlib inline', False) == ('code', '%matplotlib inline')
    assert route('!ls', True) == ('job', 'ls')                     # prompt mode routes `!` lines the same
    assert route('hello', True) == ('prompt', 'hello')

async def _until(pred, timeout=25):
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        if pred(): return
        await asyncio.sleep(0.05)
    raise TimeoutError('condition did not settle')

def _outs(app): return [b for b in app.comp.blocks.values() if b.tag == 'out']

def test_fg_job_end_to_end():
    "A bare-`!` line runs on the pty: raw bytes on glass, mirror residue as a model-only block, prompt back."
    async def go():
        tty = FakeTty(60, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'!echo hello from job\r')
            await _until(lambda: _outs(app))
            assert 'hello from job' in tty.term.text()      # streamed raw during the borrow
            blk = _outs(app)[0]
            assert blk.committed and 'hello from job' in str(blk.body[0])
            assert not app.jobs and app.fg_job is None
            assert '>>> ' in tty.term.text()                # tail repainted after reanchor
    asyncio.run(go())

def test_embedded_bang_stays_kernel():
    "`x = !ls` runs kernel-side: an SList lands in `x`, and no job is ever spawned."
    async def go():
        tty = FakeTty(70, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'x = !echo kernelside\r')
            await _until(lambda: not app.k.busy and not app.buf.text)
            app.comp.on_bytes(b'x\r')
            await _until(lambda: 'kernelside' in tty.term.contents())
            assert app._njob == 0
    asyncio.run(go())

def test_bg_job_reaps_and_prints():
    "`cmd &` drains headless, then the reap prints the residue as a normal block."
    async def go():
        tty = FakeTty(60, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'!echo bg done &\r')
            await _until(lambda: not app.jobs and app._njob == 1)
            scr = tty.term.contents()
            assert '[1] done: echo bg done' in scr
            assert 'bg done' in scr                          # bg wrote nothing on glass until the reap printed it
    asyncio.run(go())

def test_stop_then_fg_resume():
    "^Z stops the job into the jobs table; %fg resumes the borrow; a signal exit reports itself."
    async def go():
        tty = FakeTty(60, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(b'!sleep 30\r')
            await _until(lambda: app.fg_job is not None)
            job = app.fg_job[0]
            os.write(job.master_fd, b'\x1a')                 # ^Z via the pty line discipline (FakeTty has no fd)
            await _until(lambda: app.jobs)
            assert job.state == 'stopped' and 'stopped: sleep 30' in tty.term.contents()
            app.comp.on_bytes(b'%fg\r')
            await _until(lambda: app.fg_job is not None)
            import signal as _signal
            os.killpg(job.pgid, _signal.SIGTERM)
            await _until(lambda: app.fg_job is None and not app.jobs)
            assert 'signal 15' in tty.term.contents()
    asyncio.run(go())

def test_job_cwd_follows_kernel():
    "The jobs layer queries the kernel's cwd per spawn, so `%cd`-style moves are seen by bare `!`."
    async def go(tmp):
        tty = FakeTty(80, 14)
        app = App(tty, history=None)
        async with app.k:
            app.paint()
            app.comp.on_bytes(f'import os; os.chdir({tmp!r})\r'.encode())
            await _until(lambda: not app.k.busy and not app.buf.text)
            app.comp.on_bytes(b'!pwd\r')
            await _until(lambda: _outs(app))
            assert tmp in str(_outs(app)[0].body[0])
    import tempfile
    with tempfile.TemporaryDirectory() as td: asyncio.run(go(os.path.realpath(td)))

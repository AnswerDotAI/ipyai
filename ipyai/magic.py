"""The %ipyai line magic: control the ipyai host app from inside the kernel, over a jupyter comm.

The magic is a normal kernel-side line magic (registered by the app at attach time), so
tab-completion of the name and `%ipyai?` help come from the kernel's own machinery. Every
command is request/reply: the magic sends on the 'ipyai' comm and awaits the host's reply with a
timeout, so "no ipyai host attached" fails loudly and results return as ordinary cell output."""
import asyncio
from fastcore.basics import PrettyString
from IPython.core.error import UsageError

_TIMEOUT = 5  # seconds to wait for the host's ack
_state = dict(comm=None, n=0, pending={})


def _on_msg(msg):
    d = msg.get('content', {}).get('data', {})
    fut = _state['pending'].pop(d.get('req'), None)
    if fut is not None and not fut.done(): fut.set_result(d)


def _comm():
    if _state['comm'] is None:
        from comm import create_comm
        c = create_comm(target_name='ipyai')
        c.on_msg(_on_msg)
        _state['comm'] = c
    return _state['comm']


async def ipyai(line=''):
    """Control the attached ipyai app. Settings are session-only (config.json is not written).

    %ipyai                        current settings and commands
    %ipyai model NAME             set the turn model
    %ipyai suggest_model NAME     set the inline-suggestion model
    %ipyai think LEVEL            set think effort
    %ipyai code_theme NAME        set the code highlight theme, future blocks only ('auto' redetects)
    %ipyai prompt                 toggle prompt mode (same as alt-p)
    %ipyai sessions               list past ipyai sessions for this directory
    %ipyai reset                  start a fresh conversation (and a new resumable session file)
    %ipyai save PATH              export the session dialog as a .ipynb
    %ipyai load PATH              import a dialog .ipynb into the session

    Setters double as getters: `%ipyai model` (no value) shows the current value."""
    cmd = line.split()
    _state['n'] += 1
    req = _state['n']
    fut = asyncio.get_running_loop().create_future()
    _state['pending'][req] = fut
    _comm().send(dict(cmd=cmd, req=req))
    try: d = await asyncio.wait_for(fut, _TIMEOUT)
    except asyncio.TimeoutError:
        _state['pending'].pop(req, None)
        c, _state['comm'] = _state['comm'], None  # the host may never have seen this comm's open (e.g. a model cell created it): re-open fresh next time
        try: c.close()
        except Exception: pass
        raise RuntimeError(f'no ipyai host attached (no reply after {_TIMEOUT}s)') from None
    if d.get('error'): raise UsageError(d['error'])
    return PrettyString(d.get('text', ''))


def load_ipython_extension(ip): ip.register_magic_function(ipyai, 'line', 'ipyai')

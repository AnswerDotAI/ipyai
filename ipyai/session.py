"Session persistence: each session is one Dialog .ipynb under `./.ipyai/sessions/`."
import os, uuid
from pathlib import Path
from aidialog.ipynb import read_ipynb, write_ipynb

def sessions_dir(root='.'): return Path(root)/'.ipyai'/'sessions'

class Session:
    "One session file: the Dialog in memory is the model, this file its only projection (saved whole on every event)."
    def __init__(self, root='.', path=None):
        d = sessions_dir(root)
        d.mkdir(parents=True, exist_ok=True)
        gi = d.parent/'.gitignore'
        if not gi.exists(): gi.write_text('*\n')   # self-excluding, pytest-cache style
        self.dir = d
        self.path = Path(path) if path else d/f'{uuid.uuid4().hex}.ipynb'

    def save(self, dlg, **meta):
        "Write `dlg` whole (`write_ipynb` is atomic); `meta` (kernel_id, model, think) rides in the notebook metadata."
        if meta: dlg.meta = {**dlg.meta, 'ipyai': {**dlg.meta.get('ipyai', {}), **meta}}
        write_ipynb(dlg, self.path)

def list_sessions(root='.'):
    "Session files newest first: (path, mtime, n_prompts, first prompt)."
    d = sessions_dir(root)
    if not d.exists(): return []
    out = []
    for p in sorted(d.glob('*.ipynb'), key=os.path.getmtime, reverse=True):
        dlg = read_ipynb(p)
        if dlg is None: continue
        prompts = [m.content for m in dlg.messages if m.msg_type == 'prompt']
        out.append((p, os.path.getmtime(p), len(prompts), prompts[0] if prompts else ''))
    return out

def resolve_session(prefix, root='.'):
    "The unique session file whose name starts with `prefix` (`.ipynb` optional)."
    d = sessions_dir(root)
    ms = sorted(d.glob(f'{Path(prefix).stem}*.ipynb'))
    if len(ms) != 1: raise FileNotFoundError(f"{'ambiguous' if ms else 'no'} session matching {prefix!r} in {d}")
    return ms[0]

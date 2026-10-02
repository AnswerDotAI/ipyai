"Session files: each session is one Dialog .ipynb under `./.ipyai/sessions/`, saved to its `path_`."
import os, uuid
from pathlib import Path
from aidialog.ipynb import read_ipynb

def sessions_dir(root='.'): return Path(root)/'.ipyai'/'sessions'

def new_session_path(root='.'):
    "A fresh session file path under `root`. Nothing is created until `save_session` writes it."
    return sessions_dir(root)/f'{uuid.uuid4().hex}.ipynb'

def save_session(dlg):
    "Write `dlg` whole and atomically to its `path_`, first creating its directory and a self-ignoring `.gitignore` beside it (pytest-cache style)."
    d = Path(dlg.path_).parent
    d.mkdir(parents=True, exist_ok=True)
    gi = d.parent/'.gitignore'
    if not gi.exists(): gi.write_text('*\n')
    dlg.save()

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

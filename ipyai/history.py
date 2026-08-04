"History provider: recent inputs mined from this directory's session files, mode-scoped by message type."
import os
from aidialog.ipynb import read_ipynb
from .session import sessions_dir

class History:
    """Newest-first unique recent inputs: arrow navigation with a stash, and prefix suggestions for
    ghost text. `mode` picks the source, so each composer mode navigates its own past: 'code' reads
    code messages, 'prompt' the prompts, 'shell' the recorded `!` cells (leading `!` stripped, `!#`
    pseudo-cells excluded). Directory scoping is the filesystem: only this root's `.ipyai/sessions/`
    files are read, so unrelated dirs never ghost-suggest here."""
    def __init__(self, root='.', recent=2000, mode='code', nfiles=20):
        self.dir, self.recent, self.mode, self.nfiles = sessions_dir(root), recent, mode, nfiles
        self.pos, self.stash, self.items = None, '', []
        self.refresh()

    def _mine(self, m):
        "The history item a message contributes to the current mode, or None."
        if self.mode == 'prompt': return m.content if m.msg_type == 'prompt' else None
        if m.msg_type != 'code': return None
        sh = m.content.startswith('!')
        if self.mode == 'shell': return m.content[1:] if sh and not m.content.startswith('!#') else None
        return None if sh else m.content

    def refresh(self):
        "Reload newest-first unique items from the newest `nfiles` session files."
        files = sorted(self.dir.glob('*.ipynb'), key=os.path.getmtime, reverse=True) if self.dir.exists() else []
        seen, fresh = set(), []
        for p in files[:self.nfiles]:
            dlg = read_ipynb(p)
            if dlg is None: continue
            for m in reversed(dlg.messages):
                if (s := self._mine(m)) and s not in seen:
                    seen.add(s)
                    fresh.append(s)
                if len(fresh) >= self.recent: break
        self.items = fresh

    def add_local(self, source):
        "Prepend a just-submitted line immediately: cheaper than a re-read, and correct before any save lands."
        source = source.rstrip('\n')
        if not source: return
        if source in self.items: self.items.remove(source)
        self.items.insert(0, source)

    def suggest(self, prefix):
        "Ghost tail (first line only) of the newest item extending `prefix`, or ''."
        if not prefix: return ''
        return next((s[len(prefix):].split('\n')[0] for s in self.items
                     if s.startswith(prefix) and s != prefix), '')

    def prev(self, current):
        "Older item (stashing the live edit and refreshing from disk on the first step), or None."
        if self.pos is None:
            self.refresh()
            if not self.items: return None
            self.stash, self.pos = current, 0
        elif self.pos < len(self.items) - 1: self.pos += 1
        if not self.items: return None
        return self.items[self.pos]

    def next(self):
        "Newer item; past the newest, restore and return the stash. None at the live edge."
        if self.pos is None: return None
        self.pos -= 1
        if self.pos < 0:
            self.pos = None
            return self.stash
        return self.items[self.pos]

    def reset_nav(self):
        self.pos = None

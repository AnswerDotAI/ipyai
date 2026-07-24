"History provider: IPython's real history.sqlite, read read-only via apsw (faithful SQLite semantics: the kernel's writer thread owns the file)."
import apsw
from pathlib import Path

def hist_path():
    return Path.home()/'.ipython'/'profile_default'/'history.sqlite'

class History:
    "Newest-first unique recent inputs: arrow navigation with a stash, and prefix suggestions for ghost text."
    def __init__(self, path=None, recent=2000, cwd=None, mode='code'):
        self.con = apsw.Connection(str(path or hist_path()), flags=apsw.SQLITE_OPEN_READONLY)
        self.con.setbusytimeout(250)
        self.recent, self.cwd, self.mode = recent, cwd, mode
        self.pos, self.stash, self.items = None, '', []
        self.refresh()

    def refresh(self):
        """Reload the newest-first unique items. `mode` picks the source, so each composer mode
        navigates its own past: 'code' reads the kernel's history table, 'prompt' the ipyai_prompts
        store, 'shell' the recorded `!` cells (leading `!` stripped, `!#` pseudo-cells excluded) --
        the three are naturally disjoint, since prompts and `!` lines never execute in the kernel.
        With `cwd`, only sessions annotated to this directory count (the `ipyai_sessions` JOIN) --
        unannotated sessions (bare IPython, test runs) fall outside, which also kills junk
        ghost-suggestions from unrelated dirs; before any ipyai session ever ran here, history
        simply starts empty (`add_local` fills it live)."""
        scope = 'session IN (SELECT session FROM ipyai_sessions WHERE cwd = ?)'
        try:
            if self.mode == 'prompt': q, args = f'SELECT prompt FROM ipyai_prompts WHERE {scope}', (self.cwd,)
            elif self.mode == 'shell':
                q = f"SELECT substr(source, 2) FROM ipyai_cells WHERE source LIKE '!%' AND source NOT LIKE '!#%' AND {scope}"
                args = (self.cwd,)
            elif self.cwd is not None: q, args = f'SELECT source_raw FROM history WHERE {scope}', (self.cwd,)
            else: q, args = 'SELECT source_raw FROM history', ()
            rows = self.con.execute(f'{q} ORDER BY session DESC, line DESC LIMIT ?', (*args, self.recent)).fetchall()
        except apsw.SQLError:  # no ipyai tables yet: no ipyai ever ran against this db
            rows = []
        seen, fresh = set(), []
        for (s,) in rows:
            if s and s not in seen:
                seen.add(s)
                fresh.append(s)
        self.items = fresh

    def add_local(self, source):
        "Prepend a just-submitted line immediately: the kernel's own write can lag its flush thread."
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
        "Older item (stashing the live edit and refreshing from the db on the first step), or None."
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

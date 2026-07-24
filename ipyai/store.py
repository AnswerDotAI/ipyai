"""Transcript persistence: `ipyai_*` tables inside the kernel's own history.sqlite (DEV design).

The kernel's HistorySavingThread is a concurrent writer, so: apsw (real SQLite semantics),
busy timeout, short single-statement transactions, and NEVER touch the db's journal mode.
Cell outputs persist as verbatim nbformat outputs JSON arrays -- the iopub dicts we already
consume, so export-to-ipynb is nearly a SELECT."""
import json
import apsw

_SCHEMA = ["""CREATE TABLE IF NOT EXISTS ipyai_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session INTEGER NOT NULL,
    line INTEGER NOT NULL DEFAULT 0,
    prompt TEXT NOT NULL,
    full_prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    meta TEXT)""",
"""CREATE TABLE IF NOT EXISTS ipyai_cells (
    session INTEGER NOT NULL,
    line INTEGER NOT NULL,
    source TEXT NOT NULL,
    outputs TEXT NOT NULL,
    meta TEXT,
    PRIMARY KEY (session, line))""",
"""CREATE TABLE IF NOT EXISTS ipyai_sessions (
    session INTEGER PRIMARY KEY,
    cwd TEXT,
    backend TEXT)  -- holds the flat model string; column name kept for existing dbs"""]

MAX_IMAGE_PX = 2_000_000  # persistence cap: display renders full size, the stored record downscales to fit

def _cap_images(data):
    "A data dict with any image/png or image/jpeg entry over MAX_IMAGE_PX total pixels downscaled to fit."
    for mime in ('image/png', 'image/jpeg'):
        if mime not in data: continue
        import base64, io
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(data[mime])))
        if img.width * img.height <= MAX_IMAGE_PX: continue
        scale = (MAX_IMAGE_PX / (img.width * img.height)) ** 0.5
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        buf = io.BytesIO()
        img.save(buf, format='JPEG' if mime == 'image/jpeg' else 'PNG')
        data = {**data, mime: base64.b64encode(buf.getvalue()).decode()}
    return data


def nbformat_outputs(outputs):
    "Raw iopub (msg_type, content) records as an nbformat outputs array."
    out = []
    for mt, c in outputs:
        if mt == 'stream': out.append(dict(output_type='stream', name=c.get('name', 'stdout'), text=c.get('text', '')))
        elif mt == 'display_data': out.append(dict(output_type='display_data', data=_cap_images(c.get('data', {})), metadata=c.get('metadata', {})))
        elif mt == 'execute_result':
            out.append(dict(output_type='execute_result', data=_cap_images(c.get('data', {})), metadata=c.get('metadata', {}),
                            execution_count=c.get('execution_count')))
        elif mt == 'error':
            out.append(dict(output_type='error', ename=c.get('ename', ''), evalue=c.get('evalue', ''), traceback=c.get('traceback', [])))
    return out

class Store:
    "One session's writer/reader on the shared history.sqlite; every write is one short implicit transaction."
    def __init__(self, path, session=None, cwd=None, backend=None):
        self.con = apsw.Connection(str(path))
        self.con.setbusytimeout(2000)
        self.session = session
        for stmt in _SCHEMA: self.con.execute(stmt)
        for t in ('ipyai_cells', 'ipyai_prompts'):  # pre-meta dbs gain the column in place
            if 'meta' not in {r[1] for r in self.con.execute(f'PRAGMA table_info({t})')}:
                self.con.execute(f'ALTER TABLE {t} ADD COLUMN meta TEXT')
        if session is not None: self.set_session(session, cwd=cwd, backend=backend)

    def set_session(self, session, cwd=None, backend=None):
        "Point subsequent writes at `session`, inserting its row (%ipyai reset uses this mid-run)."
        self.session = session
        self.con.execute('INSERT OR REPLACE INTO ipyai_sessions (session, cwd, backend) VALUES (?, ?, ?)',
                         (session, cwd, backend))

    def save_cell(self, line, source, outputs):
        self.con.execute('INSERT OR REPLACE INTO ipyai_cells (session, line, source, outputs) VALUES (?, ?, ?, ?)',
                         (self.session, line, source, json.dumps(nbformat_outputs(outputs), ensure_ascii=False)))

    def save_prompt(self, prompt, full_prompt, response, line):
        "Insert one prompt turn, returning its rowid (the handle for later `set_prompt_meta`)."
        self.con.execute('INSERT INTO ipyai_prompts (session, line, prompt, full_prompt, response) VALUES (?, ?, ?, ?, ?)',
                         (self.session, line, prompt, full_prompt, response))
        return self.con.last_insert_rowid()

    def set_cell_meta(self, line, meta, session=None):
        "Overwrite one cell's message metadata (solveit-shaped: skipped/pinned etc), e.g. after a hide toggle. `session` targets a resumed session's rows; default this one's."
        self.con.execute('UPDATE ipyai_cells SET meta=? WHERE session=? AND line=?',
                         (json.dumps(meta), self.session if session is None else session, line))

    def set_prompt_meta(self, rowid, meta):
        self.con.execute('UPDATE ipyai_prompts SET meta=? WHERE id=?', (json.dumps(meta), rowid))

    def update_cell(self, line, source, session=None):
        "Rewrite one cell's source after an edit (outputs and meta stay); same session rule as set_cell_meta."
        self.con.execute('UPDATE ipyai_cells SET source=? WHERE session=? AND line=?',
                         (source, self.session if session is None else session, line))

    def update_prompt(self, rowid, prompt=None, response=None):
        "Rewrite a prompt turn's text sides after an edit; None leaves a side unchanged."
        if prompt is not None: self.con.execute('UPDATE ipyai_prompts SET prompt=? WHERE id=?', (prompt, rowid))
        if response is not None: self.con.execute('UPDATE ipyai_prompts SET response=? WHERE id=?', (response, rowid))

    def truncate(self, line=None, prompt_id=None, sessions=None):
        """Conversation rewind: drop the target and everything after it, across `sessions` (a resumed
        dialog spans two). A cell target passes its `line` (dropped inclusively); a prompt target passes
        its `prompt_id` (its own line is looked up here). Returns the target's line, for the caller's
        cell-counter rollback."""
        ss = list(sessions or [self.session])
        ph = ','.join('?' * len(ss))
        if prompt_id is not None:
            line = next(iter(self.con.execute('SELECT line FROM ipyai_prompts WHERE id=?', (prompt_id,))))[0]
            self.con.execute(f'DELETE FROM ipyai_cells WHERE session IN ({ph}) AND line > ?', (*ss, line))
            self.con.execute(f'DELETE FROM ipyai_prompts WHERE session IN ({ph}) AND (line > ? OR id >= ?)', (*ss, line, prompt_id))
        else:
            self.con.execute(f'DELETE FROM ipyai_cells WHERE session IN ({ph}) AND line >= ?', (*ss, line))
            self.con.execute(f'DELETE FROM ipyai_prompts WHERE session IN ({ph}) AND line >= ?', (*ss, line))
        return line

    def load_session(self, session):
        "A past session's events, oldest first: dicts of kind 'cell' (source, outputs) or 'prompt' (prompt, response), each with its message `meta`."
        cells = [dict(kind='cell', line=l, source=s, outputs=json.loads(o), meta=json.loads(m or '{}')) for l, s, o, m in
                 self.con.execute('SELECT line, source, outputs, meta FROM ipyai_cells WHERE session=? ORDER BY line', (session,))]
        prompts = [dict(kind='prompt', line=l, prompt=p, full_prompt=f, response=r, id=i, meta=json.loads(m or '{}')) for l, p, f, r, i, m in
                   self.con.execute('SELECT line, prompt, full_prompt, response, id, meta FROM ipyai_prompts WHERE session=? ORDER BY id', (session,))]
        return sorted(cells + prompts, key=lambda o: (o['line'], 0 if o['kind'] == 'cell' else 1))

    def sessions(self, cwd=None):
        "Past ipyai sessions (newest first): (session, cwd, backend, n_prompts, last_prompt)."
        q = '''SELECT s.session, s.cwd, s.backend, COUNT(p.id), MAX(p.prompt)
               FROM ipyai_sessions s LEFT JOIN ipyai_prompts p ON p.session = s.session
               {} GROUP BY s.session ORDER BY s.session DESC'''
        if cwd is not None: return list(self.con.execute(q.format('WHERE s.cwd = ?'), (cwd,)))
        return list(self.con.execute(q.format('')))

    def close(self): self.con.close()

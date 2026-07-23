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
    response TEXT NOT NULL)""",
"""CREATE TABLE IF NOT EXISTS ipyai_cells (
    session INTEGER NOT NULL,
    line INTEGER NOT NULL,
    source TEXT NOT NULL,
    outputs TEXT NOT NULL,
    PRIMARY KEY (session, line))""",
"""CREATE TABLE IF NOT EXISTS ipyai_sessions (
    session INTEGER PRIMARY KEY,
    cwd TEXT,
    backend TEXT)"""]

def nbformat_outputs(outputs):
    "Raw iopub (msg_type, content) records as an nbformat outputs array."
    out = []
    for mt, c in outputs:
        if mt == 'stream': out.append(dict(output_type='stream', name=c.get('name', 'stdout'), text=c.get('text', '')))
        elif mt == 'display_data': out.append(dict(output_type='display_data', data=c.get('data', {}), metadata=c.get('metadata', {})))
        elif mt == 'execute_result':
            out.append(dict(output_type='execute_result', data=c.get('data', {}), metadata=c.get('metadata', {}),
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
        if session is not None:
            self.con.execute('INSERT OR REPLACE INTO ipyai_sessions (session, cwd, backend) VALUES (?, ?, ?)',
                             (session, cwd, backend))

    def save_cell(self, line, source, outputs):
        self.con.execute('INSERT OR REPLACE INTO ipyai_cells (session, line, source, outputs) VALUES (?, ?, ?, ?)',
                         (self.session, line, source, json.dumps(nbformat_outputs(outputs), ensure_ascii=False)))

    def save_prompt(self, prompt, full_prompt, response, line):
        self.con.execute('INSERT INTO ipyai_prompts (session, line, prompt, full_prompt, response) VALUES (?, ?, ?, ?, ?)',
                         (self.session, line, prompt, full_prompt, response))

    def load_session(self, session):
        "A past session's events, oldest first: dicts of kind 'cell' (source, outputs) or 'prompt' (prompt, response)."
        cells = [dict(kind='cell', line=l, source=s, outputs=json.loads(o)) for l, s, o in
                 self.con.execute('SELECT line, source, outputs FROM ipyai_cells WHERE session=? ORDER BY line', (session,))]
        prompts = [dict(kind='prompt', line=l, prompt=p, full_prompt=f, response=r) for l, p, f, r in
                   self.con.execute('SELECT line, prompt, full_prompt, response FROM ipyai_prompts WHERE session=? ORDER BY id', (session,))]
        return sorted(cells + prompts, key=lambda o: (o['line'], 0 if o['kind'] == 'cell' else 1))

    def sessions(self, cwd=None):
        "Past ipyai sessions (newest first): (session, cwd, backend, n_prompts, last_prompt)."
        q = '''SELECT s.session, s.cwd, s.backend, COUNT(p.id), MAX(p.prompt)
               FROM ipyai_sessions s LEFT JOIN ipyai_prompts p ON p.session = s.session
               {} GROUP BY s.session ORDER BY s.session DESC'''
        if cwd is not None: return list(self.con.execute(q.format('WHERE s.cwd = ?'), (cwd,)))
        return list(self.con.execute(q.format('')))

    def close(self): self.con.close()

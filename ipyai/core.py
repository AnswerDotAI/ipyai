import asyncio, ast, atexit, json, os, re, signal, sqlite3, sys, uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastcore.xdg import xdg_config_home
from fastcore.xtras import frontmatter
from IPython import get_ipython
from IPython.core.inputtransformer2 import leading_empty_lines
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from rich.console import Console
from rich.file_proxy import FileProxy
from rich.live import Live
from rich.markdown import Markdown, TableDataElement

from .backend_common import CommonStreamFormatter, ConversationSeed, PromptTurn, StartupEvent, print_unexpected_error, thinking_to_blockquote
from .backends import BACKENDS, DEFAULT_BACKEND, backend_spec, normalize_backend_name
from .tooling import LocalBridge, ToolRegistry


FileProxy.isatty = lambda self: self.rich_proxied_file.isatty()


def _tde_on_text(self, context, text):
    if isinstance(text, str): self.content.append(text, context.current_style)
    else: self.content.append_text(text)


TableDataElement.on_text = _tde_on_text

DEFAULT_THINK = "m"
DEFAULT_CODE_THEME = "monokai"
DEFAULT_LOG_EXACT = False
DEFAULT_PROMPT_MODE = False
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant running inside terminal IPython through ipyai.

The user may give you:
- a `<context>` block containing recent executed Python code, outputs, and notes
- a `<user-request>` block containing the actual request
- `<variable>` blocks containing live interpreter values
- `<shell>` blocks containing shell command output

Treat `<note>` blocks as user-authored context, not executable code.

Use tools when they materially improve correctness:
- use live Python tooling such as `python` when interpreter state matters
- use available shell/file tools for repository work
- use web tools when fresh web context matters

Respond concisely and practically. Markdown is rendered in a terminal with Rich."""
_COMPLETION_SP = "You are a code completion engine for IPython. Return only the completion text to insert at the cursor."

MAGIC_NAME = "ipyai"
LAST_PROMPT = "_ai_last_prompt"
LAST_RESPONSE = "_ai_last_response"
CONTROLLER_NS = "_ipyai"
CONTROLLER_ATTR = "_ipyai_controller"
RESET_LINE_NS = "_ipyai_reset_line"
PROMPTS_TABLE = "claude_prompts"
PROMPTS_COLS = ["id", "session", "prompt", "full_prompt", "response", "history_line"]
SESSIONS_TABLE = "sessions"
HISTORY_TABLE = "history"
_PROMPTS_SQL = f"""CREATE TABLE IF NOT EXISTS {PROMPTS_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    full_prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    history_line INTEGER NOT NULL DEFAULT 0)"""
CONFIG_DIR = xdg_config_home()/"ipyai"
CONFIG_PATH = CONFIG_DIR/"config.json"
SYSP_PATH = CONFIG_DIR/"sysp.txt"
LOG_PATH = CONFIG_DIR/"exact-log.jsonl"

__all__ = """IPyAIController LAST_PROMPT LAST_RESPONSE create_controller resume_session
prompt_from_lines transform_dots transform_prompt_mode
astream_to_stdout CONFIG_PATH SYSP_PATH LOG_PATH""".split()

_prompt_template = "{context}<user-request>{prompt}</user-request>"
_var_re = re.compile(r"\$`(\w+(?:\([^`]*\))?)`")
_shell_re = re.compile(r"(?<![\w`])!`([^`]+)`")
_status_attrs = "backend model completion_model think code_theme log_exact prompt_mode".split()
SOLVEIT_REPLY_SEP = "\n\n##### 🤖Reply🤖<!-- SOLVEIT_SEPARATOR_7f3a9b2c -->\n\n"
SOLVEIT_MODE_KEY = "solveit_dialog_mode"
SOLVEIT_VER_KEY = "solveit_ver"
_CWD_KEY = "cwd"
_BACKEND_KEY = "backend"
_PROVIDER_SESSION_KEY = "provider_session_id"


def _extract_code_blocks(text):
    from mistletoe import Document
    from mistletoe.block_token import CodeFence
    return [child.children[0].content.strip() for child in Document(text).children
        if isinstance(child, CodeFence) and child.language in ("python", "py") and child.children and child.children[0].content.strip()]


def is_dot_prompt(lines): return bool(lines) and lines[0].startswith(".")


def prompt_from_lines(lines):
    if not is_dot_prompt(lines): return None
    first,*rest = lines
    return "".join([first[1:], *rest]).replace("\\\n", "\n")


def transform_dots(lines, magic=MAGIC_NAME):
    prompt = prompt_from_lines(lines)
    if prompt is None: return lines
    return [f"get_ipython().run_cell_magic({magic!r}, '', {prompt!r})\n"]


def transform_prompt_mode(lines, magic=MAGIC_NAME):
    if not lines: return lines
    first = lines[0]
    stripped = first.lstrip()
    if not stripped or stripped == "\n": return lines
    if stripped.startswith(("!", "%")): return lines
    if stripped.startswith(";"): return [first.replace(";", "", 1)] + lines[1:]
    text = "".join(lines).replace("\\\n", "\n")
    return [f"get_ipython().run_cell_magic({magic!r}, '', {text!r})\n"]


def _tag(name, content="", **attrs):
    ats = "".join(f' {k}="{v}"' for k,v in attrs.items())
    return f"<{name}{ats}>{content}</{name}>"


def _is_ipyai_input(source):
    src = (source or "").lstrip()
    return src.startswith(".") or src.startswith("%ipyai") or src.startswith("%%ipyai")


def _is_note(source):
    try: tree = ast.parse(source)
    except SyntaxError: return False
    return (len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str))


def _note_str(source): return ast.parse(source).body[0].value.value


def _var_names(text): return set(_var_re.findall(text or ""))


def _exposed_vars(text):
    fm,_ = frontmatter(text)
    names = _var_names(text)
    if fm and fm.get("exposed-vars"): names |= set(str(fm["exposed-vars"]).split())
    return names


def _var_refs(prompt, hist, notes=None):
    names = _var_names(prompt)
    for o in hist: names |= _var_names(o["prompt"])
    for note in (notes or []): names |= _exposed_vars(note)
    return names


_MISSING = object()


def _is_missing_var_error(exc):
    msg = str(exc)
    return isinstance(exc, (NameError, KeyError)) or "not defined" in msg or "is not defined" in msg


async def _eval_vars(names, bridge):
    if not names: return {}
    out = {}
    for name in names:
        try: val = await bridge.read_var(name)
        except Exception as e:
            if not _is_missing_var_error(e): print_unexpected_error(f"Variable reference failed for {name!r}", e)
            val = _MISSING
        out[name] = _MISSING if val is None and name.isidentifier() and not await _probe_defined(bridge, name) else val
    return out


async def _probe_defined(bridge, name):
    if not name.isidentifier(): return True
    try:
        exprs,_ = await bridge._exec("", expressions={"_r": f"{name!r} in globals()"})
        return bool(exprs.get("_r"))
    except Exception as e:
        print_unexpected_error(f"Variable existence probe failed for {name!r}", e)
        return False


def _format_var_xml(values):
    parts = []
    for name in sorted(values):
        val = values[name]
        if val is _MISSING or val is None: continue
        parts.append(f'<variable name="{name}" type="{type(val).__name__}">{val}</variable>')
    return "".join(parts)


def _shell_names(text): return set(_shell_re.findall(text or ""))


def _shell_cmds(text):
    fm,_ = frontmatter(text)
    names = _shell_names(text)
    if fm and fm.get("shell-cmds"):
        sc = str(fm["shell-cmds"])
        names |= set(sc.split("\n")) if "\n" in sc else {sc}
    return names


def _shell_refs(prompt, hist, notes=None):
    names = _shell_names(prompt)
    for o in hist: names |= _shell_names(o["prompt"])
    for note in (notes or []): names |= _shell_cmds(note)
    return names


def _run_shell_refs(cmds):
    if not cmds: return ""
    import subprocess
    parts = []
    for cmd in sorted(cmds):
        try: out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.rstrip()
        except Exception as e: out = f"Error: {e}"
        parts.append(f'<shell cmd="{cmd}">{out}</shell>')
    return "".join(parts)


def _event_sort_key(o): return o.get("line", 0), 0 if o.get("kind") == "code" else 1


def _display_text(text): return thinking_to_blockquote(text)


def _try_invalidate(app, label="Prompt redraw failed"):
    try: app.invalidate()
    except Exception as e: print_unexpected_error(label, e)


def _markdown_renderable(text, code_theme, markdown_cls=Markdown):
    return markdown_cls(text, code_theme=code_theme, inline_code_theme=code_theme, inline_code_lexer="python")


async def _astream_to_live_markdown(chunks, out, code_theme, formatter=None, partial=None, console_cls=Console, markdown_cls=Markdown, live_cls=Live):
    console = console_cls(file=out, force_terminal=True)
    text = ""
    live = live_cm = None
    try:
        async for chunk in chunks:
            if chunk:
                text += chunk
                if partial is not None: partial.append(chunk)
            display_text = getattr(formatter, "display_text", None) if formatter is not None else None
            current = text if display_text is None else display_text
            if not current: continue
            renderable = _markdown_renderable(_display_text(current), code_theme, markdown_cls)
            if live is None:
                live_cm = live_cls(renderable, console=console, auto_refresh=False, transient=True, redirect_stdout=False, redirect_stderr=False,
                    vertical_overflow="ellipsis")
                live = live_cm.__enter__()
            else: live.update(renderable, refresh=True)
    finally:
        if live_cm is not None: live_cm.__exit__(None, None, None)
    final_text = getattr(formatter, "final_text", text)
    final_display = getattr(formatter, "display_text", "") or final_text
    if final_display:
        console.print(_markdown_renderable(_display_text(final_display), code_theme, markdown_cls))
        out.flush()
    return final_text


async def astream_to_stdout(stream, formatter_cls: Callable[..., CommonStreamFormatter]=CommonStreamFormatter, out=None,
    code_theme=DEFAULT_CODE_THEME, partial=None, console_cls=Console, markdown_cls=Markdown, live_cls=Live):
    out = sys.stdout if out is None else out
    fmt = formatter_cls()
    is_tty = getattr(out, "isatty", lambda: False)()
    if hasattr(fmt, "is_tty"): fmt.is_tty = is_tty
    chunks = fmt.format_stream(stream)
    if is_tty: return await _astream_to_live_markdown(chunks, out, code_theme, formatter=fmt, partial=partial, console_cls=console_cls,
        markdown_cls=markdown_cls, live_cls=live_cls)
    res = []
    async for chunk in chunks:
        if not chunk: continue
        out.write(chunk)
        out.flush()
        res.append(chunk)
        if partial is not None: partial.append(chunk)
    written = "".join(res)
    if written and not written.endswith("\n"):
        out.write("\n")
        out.flush()
    return getattr(fmt, "final_text", written)


@contextmanager
def _suppress_output_history(shell):
    "Prevent IPython's display_pub from recording the AI stream into output history."
    pub = getattr(shell, "display_pub", None)
    if pub is None or not hasattr(pub, "_is_publishing"):
        yield
        return
    old = pub._is_publishing
    pub._is_publishing = True
    try: yield
    finally: pub._is_publishing = old


def _validate_level(name, value, default):
    value = (value or default).strip().lower()
    if value not in {"l", "m", "h"}: raise ValueError(f"{name} must be one of h/m/l, got {value!r}")
    return value


def _validate_bool(name, value, default):
    if value is None: return default
    if isinstance(value, bool): return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}: return True
    if value in {"0", "false", "no", "off"}: return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _default_models():
    return {name: dict(model=spec.default_model, completion_model=spec.default_completion_model, think=DEFAULT_THINK)
        for name,spec in BACKENDS.items()}


def _default_config():
    return dict(backend=DEFAULT_BACKEND, models=_default_models(),
        code_theme=DEFAULT_CODE_THEME, log_exact=DEFAULT_LOG_EXACT, prompt_mode=DEFAULT_PROMPT_MODE)


def _ensure_config_dir(path=None):
    path = Path(path or CONFIG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent


def load_config(path=None, backend_name=None):
    path = Path(path or CONFIG_PATH)
    _ensure_config_dir(path)
    cfg = _default_config()
    if path.exists():
        data = json.loads(path.read_text())
        if not isinstance(data, dict): raise ValueError(f"Invalid config format in {path}")
        if "models" in data:
            for bk,vals in data["models"].items():
                if bk in cfg["models"]: cfg["models"][bk].update(vals)
        for k in ("backend", "code_theme", "log_exact", "prompt_mode"):
            if k in data: cfg[k] = data[k]
    else: path.write_text(json.dumps(cfg, indent=2) + "\n")
    backend_name = normalize_backend_name(backend_name or cfg["backend"])
    spec = backend_spec(backend_name)
    mcfg = cfg["models"].get(backend_name, {})
    cfg["model"] = str(mcfg.get("model", "") or os.environ.get("IPYAI_MODEL", "") or spec.default_model).strip()
    cfg["completion_model"] = str(mcfg.get("completion_model", "") or spec.default_completion_model).strip()
    cfg["think"] = _validate_level("think", mcfg.get("think"), DEFAULT_THINK)
    cfg["code_theme"] = str(cfg["code_theme"]).strip() or DEFAULT_CODE_THEME
    cfg["log_exact"] = _validate_bool("log_exact", cfg["log_exact"], DEFAULT_LOG_EXACT)
    cfg["prompt_mode"] = _validate_bool("prompt_mode", cfg["prompt_mode"], DEFAULT_PROMPT_MODE)
    cfg["_backend_name"] = backend_name
    return cfg


def load_sysp(path=None):
    path = Path(path or SYSP_PATH)
    _ensure_config_dir(path)
    if not path.exists(): path.write_text(DEFAULT_SYSTEM_PROMPT)
    return path.read_text()


def _cell_id(): return uuid.uuid4().hex[:8]


def _split_solveit_prompt(source):
    content,*reply = (source or "").split(SOLVEIT_REPLY_SEP, 1)
    return content, (reply[0] if reply else "")


def _event_to_cell(o):
    if o.get("kind") == "code":
        source = o.get("source", "")
        if _is_note(source):
            return dict(id=_cell_id(), cell_type="markdown", source=_note_str(source),
                metadata=dict(ipyai=dict(kind="code", line=o.get("line", 0), source=source)))
        return dict(id=_cell_id(), cell_type="code", source=source, metadata=dict(ipyai=dict(kind="code", line=o.get("line", 0))),
            outputs=[], execution_count=None)
    if o.get("kind") == "prompt":
        meta = dict(kind="prompt", line=o.get("line", 0), history_line=o.get("history_line", 0), prompt=o.get("prompt", ""),
            full_prompt=o.get("full_prompt", ""))
        source = o.get("prompt", "") + SOLVEIT_REPLY_SEP + o.get("response", "")
        return dict(id=_cell_id(), cell_type="markdown", source=source, metadata=dict(ipyai=meta, solveit_ai=True))


def _cell_to_event(cell):
    if cell.get("metadata", {}).get("solveit_ai"):
        meta = cell.get("metadata", {}).get("ipyai", {})
        prompt,response = _split_solveit_prompt(cell.get("source", ""))
        return dict(kind="prompt", line=meta.get("line", 0), history_line=meta.get("history_line", 0), prompt=meta.get("prompt", prompt),
            full_prompt=meta.get("full_prompt", ""), response=response)
    meta = cell.get("metadata", {}).get("ipyai", {})
    kind = meta.get("kind")
    if kind == "code":
        source = meta.get("source") or cell.get("source", "")
        return dict(kind="code", line=meta.get("line", 0), source=source)
    if kind == "prompt":
        return dict(kind="prompt", line=meta.get("line", 0), history_line=meta.get("history_line", 0), prompt=meta.get("prompt", ""),
            full_prompt=meta.get("full_prompt", ""), response=cell.get("source", ""))


def _load_notebook(path):
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Notebook not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict): raise ValueError(f"Invalid notebook format in {path}")
    return [e for c in data.get("cells", []) if (e := _cell_to_event(c)) is not None]


def _git_repo_root(path):
    p = Path(path).resolve()
    for d in [p, *p.parents]:
        if (d/".git").exists(): return str(d)
    return None


def _discover_plugin_dirs(cwd):
    dirs = []
    for d in [Path(cwd).resolve(), *Path(cwd).resolve().parents]:
        cand = d/".claude"/"plugins"
        if cand.exists() and cand.is_dir(): dirs.append(str(cand))
    return list(dict.fromkeys(dirs))


def _session_meta(remark):
    if not remark: return {}
    try: meta = json.loads(remark)
    except Exception: return {}
    return meta if isinstance(meta, dict) else {}


def _session_remark(remark=None, **updates):
    meta = _session_meta(remark)
    for k,v in updates.items():
        if v is None: meta.pop(k, None)
        else: meta[k] = v
    return json.dumps(meta, sort_keys=True) if meta else None


def _update_session_remark(db, session_id, **updates):
    if db is None: return
    row = db.execute(f"SELECT remark FROM {SESSIONS_TABLE} WHERE session=?", (session_id,)).fetchone()
    if not row: return
    db.execute(f"UPDATE {SESSIONS_TABLE} SET remark=? WHERE session=?", (_session_remark(row[0], **updates), session_id))


def _open_db(path):
    "Open a client-side sqlite connection to the kernel's shared history.sqlite and ensure claude_prompts exists. The kernel's HistoryManager owns creation of `sessions` and `history`."
    Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    with db:
        db.execute(_PROMPTS_SQL)
        db.execute(f"CREATE INDEX IF NOT EXISTS idx_{PROMPTS_TABLE}_session_id ON {PROMPTS_TABLE} (session, id)")
    return db


_LIST_SQL = f"""SELECT s.session, s.start, s.end, s.num_cmds,
    CASE WHEN json_valid(s.remark) THEN json_extract(s.remark, '$.{_CWD_KEY}') END,
    (SELECT prompt FROM {PROMPTS_TABLE} WHERE session=s.session ORDER BY id DESC LIMIT 1)
    FROM {SESSIONS_TABLE} s
    WHERE CASE WHEN json_valid(s.remark) THEN json_extract(s.remark, '$.{_CWD_KEY}') END{{w}}
      AND CASE WHEN json_valid(s.remark) THEN json_extract(s.remark, '$.{_BACKEND_KEY}') END=?
    ORDER BY s.session DESC LIMIT 20"""


def _list_sessions(db, cwd, backend_name):
    rows = db.execute(_LIST_SQL.format(w="=?"), (cwd, backend_name)).fetchall()
    if not rows:
        repo = _git_repo_root(cwd)
        if repo and repo != cwd: rows = db.execute(_LIST_SQL.format(w="=?"), (repo, backend_name)).fetchall()
    return rows


def _fmt_session(sid, start, ncmds, last_prompt, max_prompt=60):
    p = (last_prompt or "").replace("\n", " ")[:max_prompt]
    if last_prompt and len(last_prompt) > max_prompt: p += "..."
    return f"{sid:>6}  {str(start or '')[:19]:20}  {ncmds or 0:>5}  {p}"


def _pick_session(rows):
    from prompt_toolkit.shortcuts import radiolist_dialog
    values = [(sid, _fmt_session(sid, start, ncmds, lp)) for sid,start,end,ncmds,cwd,lp in rows]
    return radiolist_dialog(title="Resume session", text="Select a session to resume:", values=values, default=values[0][0]).run()


def resume_session(db, session_id, backend_name=None):
    "Validate that ipyai session `session_id` exists (and matches the requested backend); reopen it in the DB."
    row = db.execute(f"SELECT session, remark FROM {SESSIONS_TABLE} WHERE session=?", (session_id,)).fetchone()
    if not row: raise ValueError(f"Session {session_id} not found")
    meta = _session_meta(row[1])
    if backend_name and meta.get(_BACKEND_KEY) and meta.get(_BACKEND_KEY) != backend_name:
        raise ValueError(f"Session {session_id} uses backend {meta.get(_BACKEND_KEY)!r}, not {backend_name!r}")
    with db: db.execute(f"UPDATE {SESSIONS_TABLE} SET end=NULL WHERE session=?", (session_id,))


@magics_class
class AIMagics(Magics):
    def __init__(self, shell, ctrl):
        super().__init__(shell)
        self.ctrl = ctrl

    @line_magic(MAGIC_NAME)
    def ipyai_line(self, line=""): return self.ctrl.handle_line(line)

    @cell_magic(MAGIC_NAME)
    async def ipyai_cell(self, line="", cell=None): await self.ctrl.run_prompt(cell)


class IPyAIController:
    def __init__(self, shell, model=None, completion_model=None, think=None, code_theme=None, log_exact=None, system_prompt=None,
        prompt_mode=None, backend_name=None, backend_factory=None, bridge=None, session_number=None, db=None):
        self.shell,self.loaded = shell,False
        cfg = load_config(CONFIG_PATH, backend_name=backend_name)
        self.backend_name = cfg["_backend_name"]
        self.backend_spec = backend_spec(self.backend_name)
        self.backend_factory = backend_factory or self.backend_spec.factory
        self.prompt_mode = cfg["prompt_mode"] ^ bool(prompt_mode)
        self.model = model or cfg["model"] or self.backend_spec.default_model
        self.completion_model = completion_model or cfg["completion_model"] or self.backend_spec.default_completion_model
        self.think = _validate_level("think", think if think is not None else cfg["think"], DEFAULT_THINK)
        self.code_theme = str(code_theme or cfg["code_theme"]).strip() or DEFAULT_CODE_THEME
        self.log_exact = _validate_bool("log_exact", log_exact if log_exact is not None else cfg["log_exact"], DEFAULT_LOG_EXACT)
        self.system_prompt = system_prompt if system_prompt is not None else load_sysp(SYSP_PATH)
        self.plugin_dirs = _discover_plugin_dirs(os.getcwd())
        self.bridge = bridge if bridge is not None else LocalBridge(getattr(shell, "user_ns", {}))
        if db is None: raise ValueError("IPyAIController requires a sqlite db connection (shared with the kernel's HistoryManager)")
        self.db = db if isinstance(db, sqlite3.Connection) else _open_db(db)
        if session_number is None: raise ValueError("IPyAIController requires a session_number from the kernel's HistoryManager")
        self._session_number = session_number

    @property
    def backend(self): return self.backend_name

    @property
    def history_manager(self): return getattr(self.shell, "history_manager", None)

    @property
    def session_number(self): return self._session_number

    @property
    def reset_line(self): return self.shell.user_ns.get(RESET_LINE_NS, 0)

    def make_backend(self, system_prompt=None):
        return self.backend_factory(shell=self.shell, cwd=os.getcwd(), system_prompt=self.system_prompt if system_prompt is None else system_prompt,
            plugin_dirs=self.plugin_dirs, tools=ToolRegistry(self.bridge))

    @contextmanager
    def _db(self):
        with self.db: yield self.db

    def _ensure_session_row(self):
        with self._db() as db: _update_session_remark(db, self.session_number, cwd=os.getcwd(), backend=self.backend_name)

    def prompt_records(self, session=None):
        session = self.session_number if session is None else session
        cur = self.db.execute(f"SELECT id, prompt, full_prompt, response, history_line FROM {PROMPTS_TABLE} WHERE session=? ORDER BY id", (session,))
        return cur.fetchall()

    def prompt_rows(self, session=None): return [(p, r) for _,p,_,r,_ in self.prompt_records(session=session)]

    def last_prompt_line(self, session=None):
        rows = self.prompt_records(session=session)
        return rows[-1][4] if rows else self.reset_line

    def current_prompt_line(self): return max(getattr(self.shell, "execution_count", 1) - 1, 0)

    def current_input_line(self): return max(getattr(self.shell, "execution_count", 1), 1)

    def code_history(self, start, stop):
        hm = self.history_manager
        if hm is None or stop <= start: return []
        buf = getattr(self.shell, "output_buffer", None) or {}
        out = []
        for sess,line,pair in hm.get_range(session=0, start=start, stop=stop, raw=True, output=True):
            if isinstance(pair, (tuple, list)) and len(pair) == 2: src,outp = pair[0], pair[1]
            else: src,outp = pair,None
            if outp is None and line in buf: outp = buf[line]
            out.append((sess, line, (src, outp)))
        return out

    def full_history(self): return self.code_history(1, self.current_input_line()+1)

    def code_context(self, start, stop):
        parts = []
        for _,line,pair in self.code_history(start, stop):
            source,output = pair
            if not source or _is_ipyai_input(source): continue
            if _is_note(source): parts.append(_tag("note", _note_str(source)))
            else:
                parts.append(_tag("code", source))
                if output is not None: parts.append(_tag("output", output))
        return _tag("context", "".join(parts)) + "\n" if parts else ""

    def format_prompt(self, prompt, start, stop):
        ctx = self.code_context(start, stop)
        return _prompt_template.format(context=ctx, prompt=prompt.strip())

    def note_strings(self, start, stop):
        return [_note_str(src) for _,_,pair in self.code_history(start, stop) if (src := pair[0]) and _is_note(src)]

    def save_prompt(self, prompt, full_prompt, response, history_line):
        with self._db() as db:
            db.execute(f"INSERT INTO {PROMPTS_TABLE} (session, prompt, full_prompt, response, history_line) VALUES (?, ?, ?, ?, ?)",
                (self.session_number, prompt, full_prompt, response, history_line))

    def startup_events(self):
        events = []
        for _,line,pair in self.full_history():
            source,_ = pair
            if not source or _is_ipyai_input(source): continue
            events.append(dict(kind="code", line=line, source=source))
        for pid,prompt,full_prompt,response,history_line in self.prompt_records():
            events.append(dict(kind="prompt", id=pid, line=history_line+1, history_line=history_line, prompt=prompt, full_prompt=full_prompt, response=response))
        return sorted(events, key=_event_sort_key)

    def conversation_seed(self, prompt_records=None, startup_events=None):
        prompt_records = self.prompt_records() if prompt_records is None else prompt_records
        startup_events = self.startup_events() if startup_events is None else startup_events
        turns = tuple(PromptTurn(prompt=prompt, full_prompt=full_prompt, response=response, history_line=history_line)
            for _,prompt,full_prompt,response,history_line in prompt_records)
        events = tuple(StartupEvent(**{k:v for k,v in o.items() if k != "id"}) for o in startup_events)
        return ConversationSeed(turns=turns, startup_events=events)

    def save_notebook(self, path):
        path = Path(path)
        if path.suffix != ".ipynb": path = path.with_suffix(".ipynb")
        events = [{k:v for k,v in o.items() if k != "id"} for o in self.startup_events()]
        nb = dict(cells=[_event_to_cell(e) for e in events], metadata=dict(ipyai_version=1, solveit_ver=2, solveit_dialog_mode="standard"),
            nbformat=4, nbformat_minor=5)
        path.write_text(json.dumps(nb, indent=2) + "\n")
        return path, sum(o["kind"] == "code" for o in events), sum(o["kind"] == "prompt" for o in events)

    def _advance_execution_count(self):
        if hasattr(self.shell, "execution_count"): self.shell.execution_count += 1

    def load_notebook(self, path):
        path = Path(path)
        if path.suffix != ".ipynb": path = path.with_suffix(".ipynb")
        events = _load_notebook(path)
        ncode = nprompt = 0
        for o in sorted(events, key=_event_sort_key):
            if o.get("kind") == "code":
                source = o.get("source", "")
                if not source: continue
                res = self.shell.run_cell(source, store_history=True)
                ncode += 1
                if getattr(res, "success", True) is False: break
            elif o.get("kind") == "prompt":
                history_line = int(o.get("history_line", max(o.get("line", 1)-1, 0)))
                self.save_prompt(o.get("prompt", ""), o.get("full_prompt", ""), o.get("response", ""), history_line)
                self._advance_execution_count()
                nprompt += 1
        return path, ncode, nprompt

    def log_exact_exchange(self, prompt, response):
        if not self.log_exact: return
        rec = dict(ts=datetime.now(timezone.utc).isoformat(), session=self.session_number, prompt=prompt, response=response)
        with LOG_PATH.open("a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def reset_session_history(self):
        with self._db() as db:
            cur = db.execute(f"DELETE FROM {PROMPTS_TABLE} WHERE session=?", (self.session_number,))
            _update_session_remark(db, self.session_number, cwd=os.getcwd(), backend=self.backend_name, provider_session_id=None)
        self.shell.user_ns.pop(LAST_PROMPT, None)
        self.shell.user_ns.pop(LAST_RESPONSE, None)
        self.shell.user_ns[RESET_LINE_NS] = self.current_prompt_line()
        return cur.rowcount or 0

    def get_provider_session_id(self):
        row = self.db.execute(f"SELECT remark FROM {SESSIONS_TABLE} WHERE session=?", (self.session_number,)).fetchone()
        if not row: return None
        meta = _session_meta(row[0])
        return meta.get(_PROVIDER_SESSION_KEY) if meta.get(_BACKEND_KEY) == self.backend_name else None

    def set_provider_session(self, session_id):
        if not session_id: return
        with self._db() as db:
            _update_session_remark(db, self.session_number, cwd=os.getcwd(), backend=self.backend_name, provider_session_id=session_id)

    def _pt_cli(self): return getattr(self.shell, "pt_cli", None) or getattr(self.shell, "pt_app", None)

    def _install_lexer(self):
        "Swap the prompt_toolkit lexer to plain for AI-prompt lines; code lexer otherwise."
        pt = self._pt_cli()
        if pt is None: return
        orig = getattr(pt, "lexer", None)
        if orig is None: return
        from prompt_toolkit.lexers import Lexer, SimpleLexer
        plain = SimpleLexer()
        ctrl = self

        class _Dispatch(Lexer):
            def lex_document(self, document):
                text = document.text.lstrip()
                if ctrl.prompt_mode and not text.startswith((";", "!", "%")): return plain.lex_document(document)
                if text.startswith(".") or text.startswith("%%ipyai"): return plain.lex_document(document)
                return orig.lex_document(document)

        pt.lexer = _Dispatch()

    def _refresh_prompt(self):
        "Force pt_cli to redraw its prompt (prompt-mode toggle)."
        pt = self._pt_cli()
        if pt is None: return
        app = getattr(pt, "app", None)
        if app is not None: _try_invalidate(app)

    def _register_keybindings(self):
        pt = self._pt_cli()
        if pt is None: return
        auto_suggest = getattr(pt, "auto_suggest", None)
        if auto_suggest is not None:
            auto_suggest._ai_full_text = None
            _orig_get = auto_suggest.get_suggestion

            def _patched_get(buffer, document):
                from prompt_toolkit.auto_suggest import Suggestion
                text,full_text = document.text,auto_suggest._ai_full_text
                if full_text and full_text.startswith(text) and len(full_text) > len(text): return Suggestion(full_text[len(text):])
                auto_suggest._ai_full_text = None
                return _orig_get(buffer, document)

            auto_suggest.get_suggestion = _patched_get

        ns = self.shell.user_ns

        def _get_blocks(): return _extract_code_blocks(ns.get(LAST_RESPONSE, ""))

        @pt.key_bindings.add("escape", "W")
        def _paste_all(event):
            blocks = _get_blocks()
            if blocks: event.current_buffer.insert_text("\n".join(blocks))

        for i,ch in enumerate("!@#$%^&*(", 1):
            @pt.key_bindings.add("escape", ch)
            def _paste_nth(event, n=i):
                blocks = _get_blocks()
                if len(blocks) >= n: event.current_buffer.insert_text(blocks[n-1])

        cycle = dict(idx=-1, resp="")

        def _cycle(event, delta):
            resp = ns.get(LAST_RESPONSE, "")
            blocks = _get_blocks()
            if not blocks: return
            if resp != cycle["resp"]: cycle.update(idx=-1, resp=resp)
            cycle["idx"] = (cycle["idx"] + delta) % len(blocks)
            from prompt_toolkit.document import Document
            event.current_buffer.document = Document(blocks[cycle["idx"]])

        @pt.key_bindings.add("escape", "s-up")
        def _cycle_down(event): _cycle(event, 1)

        @pt.key_bindings.add("escape", "s-down")
        def _cycle_up(event): _cycle(event, -1)

        @pt.key_bindings.add("escape", "up")
        def _hist_back(event): event.current_buffer.history_backward()

        @pt.key_bindings.add("escape", "down")
        def _hist_fwd(event): event.current_buffer.history_forward()

        @pt.key_bindings.add("escape", ".")
        def _ai_suggest(event):
            buf,doc,app = event.current_buffer,event.current_buffer.document,event.app
            if not doc.text.strip(): return

            async def _do_complete():
                try:
                    text = await self._ai_complete(doc)
                    if text and buf.document == doc:
                        from prompt_toolkit.auto_suggest import Suggestion
                        if auto_suggest is not None: auto_suggest._ai_full_text = doc.text + text
                        buf.suggestion = Suggestion(text)
                        app.invalidate()
                except asyncio.CancelledError: raise
                except Exception as e:
                    print_unexpected_error("AI completion failed", e)
                    _try_invalidate(app)

            app.create_background_task(_do_complete())

        @pt.key_bindings.add("escape", "p")
        def _toggle_prompt(event):
            self._toggle_prompt_mode()
            event.app.invalidate()

    async def _ai_complete(self, document):
        prefix,suffix = document.text_before_cursor,document.text_after_cursor
        ctx = self.code_context(self.last_prompt_line()+1, self.current_prompt_line())
        parts = [ctx] if ctx else []
        parts.append(f"<current-input>\n<prefix>{prefix}</prefix>")
        if suffix.strip(): parts.append(f"<suffix>{suffix}</suffix>")
        parts.append("</current-input>")
        parts.append("Return only the completion text to insert immediately after the prefix.")
        res = await self.make_backend(system_prompt=_COMPLETION_SP).complete("\n".join(parts), model=self.completion_model)
        return (res.content or "").strip()

    def load(self):
        if self.loaded: return self
        self._ensure_session_row()
        cts = self.shell.input_transformer_manager.cleanup_transforms
        if self.prompt_mode:
            if transform_prompt_mode not in cts: cts.insert(0, transform_prompt_mode)
        elif transform_dots not in cts:
            idx = 1 if cts and cts[0] is leading_empty_lines else 0
            cts.insert(idx, transform_dots)
        self.shell.register_magics(AIMagics(self.shell, self))
        self.shell.user_ns[CONTROLLER_NS] = self
        self.shell.user_ns.setdefault(RESET_LINE_NS, 0)
        setattr(self.shell, CONTROLLER_ATTR, self)
        self._register_keybindings()
        self._install_lexer()
        self.loaded = True
        return self

    def unload(self):
        if not self.loaded: return self
        cts = self.shell.input_transformer_manager.cleanup_transforms
        if transform_dots in cts: cts.remove(transform_dots)
        if transform_prompt_mode in cts: cts.remove(transform_prompt_mode)
        if self.shell.user_ns.get(CONTROLLER_NS) is self: self.shell.user_ns.pop(CONTROLLER_NS, None)
        if getattr(self.shell, CONTROLLER_ATTR, None) is self: delattr(self.shell, CONTROLLER_ATTR)
        self.loaded = False
        return self

    def _show(self, attr): print(f"self.{attr}={getattr(self, attr)!r}")

    def _set(self, attr, value):
        setattr(self, attr, value)
        self._show(attr)

    def _toggle_prompt_mode(self):
        self.prompt_mode = not self.prompt_mode
        cts = self.shell.input_transformer_manager.cleanup_transforms
        if self.prompt_mode:
            if transform_prompt_mode not in cts: cts.insert(0, transform_prompt_mode)
            if transform_dots in cts: cts.remove(transform_dots)
        else:
            if transform_prompt_mode in cts: cts.remove(transform_prompt_mode)
            if transform_dots not in cts:
                idx = 1 if cts and cts[0] is leading_empty_lines else 0
                cts.insert(idx, transform_dots)
        self._invalidate_history_cache()
        self._refresh_prompt()

    def _invalidate_history_cache(self):
        "Clear the History cache AND reset the Buffer's _working_lines (preserving current text) so the next up-arrow reloads from the new mode's query."
        pt = self._pt_cli()
        buf = getattr(pt, "default_buffer", None) if pt else None
        hist = getattr(buf, "history", None)
        if hist is not None: hist._loaded = False
        if buf is not None: buf.reset(document=buf.document)

    def _show_help(self):
        cmds = [("(no args)", "Show current settings"), ("help", "Show this help"), ("model <name>", "Set model"),
            ("completion_model <name>", "Set completion model"), ("think <l|m|h>", "Set thinking level"), ("code_theme <name>", "Set code theme"),
            ("log_exact <bool>", "Set raw prompt/response logging"), ("prompt", "Toggle prompt mode"), ("save <file>", "Save session to .ipynb"),
            ("load <file>", "Load session from .ipynb"), ("reset", "Clear AI prompts from current session"), ("sessions", "List previous sessions")]
        print("Usage: %ipyai <command>\n")
        for cmd,desc in cmds: print(f"  {cmd:22s} {desc}")

    def handle_line(self, line):
        line = line.strip()
        if not line:
            for o in _status_attrs: self._show(o)
            print(f"{CONFIG_PATH=}")
            print(f"{SYSP_PATH=}")
            return print(f"{LOG_PATH=}")
        if line in _status_attrs: return self._show(line)
        if line == "prompt": return self._toggle_prompt_mode()
        if line == "reset":
            n = self.reset_session_history()
            return print(f"Deleted {n} AI prompts from session {self.session_number}.")
        if line == "sessions":
            rows = _list_sessions(self.db, os.getcwd(), self.backend_name)
            if not rows: return print("No sessions found for this directory.")
            print(f"{'ID':>6}  {'Start':20}  {'Cmds':>5}  {'Last prompt'}")
            for sid,start,end,ncmds,remark,lp in rows: print(_fmt_session(sid, start, ncmds, lp))
            return
        cmd,_,arg = line.partition(" ")
        clean = arg.strip()
        if cmd == "save":
            if not clean: return print("Usage: %ipyai save <filename>")
            path,ncode,nprompt = self.save_notebook(clean)
            return print(f"Saved {ncode} code cells and {nprompt} prompts to {path}.")
        if cmd == "load":
            if not clean: return print("Usage: %ipyai load <filename>")
            try:
                path,ncode,nprompt = self.load_notebook(clean)
                return print(f"Loaded {ncode} code cells and {nprompt} prompts from {path}.")
            except FileNotFoundError as e: return print(str(e))
        if cmd == "help": return self._show_help()
        if clean:
            vals = dict(model=lambda: clean, completion_model=lambda: clean or self.backend_spec.default_completion_model,
                code_theme=lambda: clean or DEFAULT_CODE_THEME,
                think=lambda: _validate_level("think", clean, self.think), log_exact=lambda: _validate_bool("log_exact", clean, self.log_exact))
            if cmd in vals: return self._set(cmd, vals[cmd]())
        print(f"Unknown command: {line!r}. Run %ipyai help for available commands.")

    async def run_prompt(self, prompt):
        prompt = (prompt or "").rstrip("\n")
        if not prompt.strip(): return None
        self._ensure_session_row()
        history_line = self.current_prompt_line()
        prompt_records = self.prompt_records()
        records = [dict(prompt=p, history_line=hl) for _,p,_,_,hl in prompt_records]
        notes,prev_line = [],self.reset_line
        for o in records:
            notes += self.note_strings(prev_line+1, o["history_line"])
            prev_line = o["history_line"]
        notes += self.note_strings(self.last_prompt_line()+1, history_line)
        var_names = _var_refs(prompt, records, notes=notes)
        values = await _eval_vars(var_names, self.bridge)
        missing_vars = sorted(n for n,v in values.items() if v is _MISSING)
        var_xml = _format_var_xml(values)
        shell_cmds = _shell_refs(prompt, records, notes=notes)
        shell_xml = _run_shell_refs(shell_cmds)
        warnings = _tag("warnings", f"The following symbols were referenced but aren't defined in the interpreter: {', '.join(missing_vars)}") + "\n" if missing_vars else ""
        full_prompt = warnings + var_xml + shell_xml + self.format_prompt(prompt, self.last_prompt_line()+1, history_line+1)
        self.shell.user_ns[LAST_PROMPT] = prompt
        backend = self.make_backend()
        partial = []
        stream = turn = None
        loop,task = asyncio.get_running_loop(),asyncio.current_task()
        signal_set = False
        try:
            turn = await backend.prepare_turn(prompt=full_prompt, model=self.model, think=self.think,
                provider_session_id=self.get_provider_session_id(), seed=self.conversation_seed(prompt_records=prompt_records))
            stream = turn.stream
            try:
                loop.add_signal_handler(signal.SIGINT, task.cancel)
                signal_set = True
            except (NotImplementedError, RuntimeError): pass
            try:
                with _suppress_output_history(self.shell):
                    text = await astream_to_stdout(stream, formatter_cls=backend.formatter_cls, code_theme=self.code_theme, partial=partial)
            except asyncio.CancelledError:
                text = "".join(partial) + "\n<system>user interrupted</system>"
                print("\nstopped")
            finally:
                if signal_set:
                    try: loop.remove_signal_handler(signal.SIGINT)
                    except Exception as e: print_unexpected_error("Failed to restore SIGINT handler", e)
                if stream is not None: await stream.aclose()
        except Exception as e:
            print_unexpected_error("AI prompt failed", e)
            raise
        self.shell.user_ns[LAST_RESPONSE] = text
        ng = getattr(self.shell, "_ipythonng_extension", None)
        if ng is not None: ng._pty_output = thinking_to_blockquote(text)
        if session_id := await turn.wait_provider_session_id(): self.set_provider_session(session_id)
        self.log_exact_exchange(full_prompt, text)
        self.save_prompt(prompt, full_prompt, text, history_line)
        self._advance_execution_count()
        return None


def _resume_command(session_id, backend_name, existing=None):
    cfg = load_config(CONFIG_PATH)
    default = cfg["_backend_name"]
    backend_part = f" -b {backend_name}" if backend_name != default else ""
    tail = f"--existing={existing}" if existing else f"-r {session_id}"
    return f"ipyai{backend_part} {tail}"


def create_controller(shell=None, resume=None, file=None, prompt_mode=False, backend=None, bridge=None,
    db=None, session_number=None, existing=None, **kwargs):
    "Build/load the IPyAIController. `db` is the client-side sqlite connection to the kernel's shared history.sqlite; `session_number` is the kernel's HistoryManager.session_number."
    shell = shell or get_ipython()
    if shell is None: raise RuntimeError("No active IPython shell found")
    cfg = load_config(CONFIG_PATH, backend_name=backend)
    backend_name = cfg["_backend_name"]
    db = db if isinstance(db, sqlite3.Connection) or db is None else _open_db(db)
    if resume is not None:
        if db is None: raise ValueError("resume requires a db connection")
        if resume == -1:
            rows = _list_sessions(db, os.getcwd(), backend_name)
            if rows and (chosen := _pick_session(rows)):
                resume_session(db, chosen, backend_name=backend_name)
                session_number = chosen
            else: print("No sessions found for this directory.")
        else:
            resume_session(db, resume, backend_name=backend_name)
            session_number = resume
    ctrl = getattr(shell, CONTROLLER_ATTR, None)
    if ctrl is not None and ctrl.backend_name != backend_name: ctrl.unload()
    if ctrl is None or ctrl.backend_name != backend_name:
        ctrl = IPyAIController(shell=shell, prompt_mode=prompt_mode, backend_name=backend_name, bridge=bridge,
            session_number=session_number, db=db, **kwargs)
    if not ctrl.loaded: ctrl.load()
    if file is not None:
        try:
            path,ncode,nprompt = ctrl.load_notebook(file)
            print(f"Loaded {ncode} code cells and {nprompt} prompts from {path}.")
        except FileNotFoundError as e: print(str(e))
    ctrl._ensure_session_row()
    if not getattr(shell, "_ipyai_atexit", False):
        sid = ctrl.session_number
        atexit.register(lambda: print(f"\nTo resume: {_resume_command(sid, ctrl.backend_name, existing=existing)}"))
        shell._ipyai_atexit = True
    return ctrl

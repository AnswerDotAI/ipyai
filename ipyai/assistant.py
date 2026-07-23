"""The assistant: mode routing, context assembly over the app's cell records, backend turns, kernel writeback.

The old IPyAIController minus prompt_toolkit and minus in-kernel transformers: routing happens at
submit time in the app, context is assembled from the cell records the app keeps beside the block
model (not from an iopub tee), and the reply reaches the kernel namespace by one silent execute."""
import ast, asyncio, re, os, subprocess
from .backends import backend_spec
from .backend_common import ConversationSeed, PromptTurn
from .config import load_config, load_sysp, COMPLETION_SP

LAST_PROMPT = '_ai_last_prompt'
LAST_RESPONSE = '_ai_last_response'
_var_re = re.compile(r"\$`(\w+(?:\([^`]*\))?)`")
_shell_re = re.compile(r"(?<![\w`])!`([^`]+)`")
_MISSING = object()

def route(text, prompt_mode):
    """The `.`/`;`/`!`/`%` dispatch, at submit time: ('prompt'|'code', payload).
    Code mode: a leading `.` sends the rest as a prompt. Prompt mode: everything is a prompt,
    except `;` (strip it, run as code) and `!`/`%` lines (shell/magics pass through untouched)."""
    if prompt_mode:
        s = text.lstrip()
        if s.startswith(';'): return 'code', text.replace(';', '', 1)
        if s.startswith(('!', '%')): return 'code', text
        return 'prompt', text
    if text.startswith('.'): return 'prompt', text[1:]
    return 'code', text

def _tag(name, content=''): return f'<{name}>{content}</{name}>'

def _is_note(source):
    "A cell that is one bare string literal is a note, not code (the old ipyai convention)."
    try: tree = ast.parse(source)
    except SyntaxError: return False
    return (len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str))

def _note_str(source): return ast.parse(source).body[0].value.value

def _text(t): return ''.join(t) if isinstance(t, list) else (t or '')

def output_text(outputs, mx=4000):
    "Flatten a cell's raw iopub (msg_type, content) records to context text, truncated in the middle."
    parts = []
    for mt, c in outputs:
        if mt == 'stream': parts.append(_text(c.get('text')))
        elif mt in ('execute_result', 'display_data'):
            data = c.get('data', {})
            if 'text/plain' in data: parts.append(_text(data['text/plain']) + '\n')
            elif 'image/png' in data: parts.append('[image]\n')
        elif mt == 'error': parts.append(f"{c.get('ename')}: {c.get('evalue')}\n")
    out = ''.join(parts).rstrip('\n')
    return out if len(out) <= mx else out[:mx//2] + '\n...[truncated]...\n' + out[-mx//2:]

def code_blocks(md):
    "Python fenced blocks of `md`, in order, via mdhtml structure (never regex)."
    import mdhtml
    return [b['text'].rstrip('\n') for b in mdhtml.blocks(md or '')
            if b['type'] == 'code_block' and b.get('lang') in ('python', 'py') and b.get('text', '').strip()]

async def _eval_vars(names, bridge):
    if not names or bridge is None: return {}
    out = {}
    for name in names:
        try: val = await bridge.read_var(name)
        except Exception: val = _MISSING
        out[name] = val
    return out

def _format_var_xml(values):
    return ''.join(f'<variable name="{n}" type="{type(v).__name__}">{v}</variable>'
                   for n, v in sorted(values.items()) if v is not _MISSING and v is not None)

def _run_shell_refs(cmds):
    if not cmds: return ''
    parts = []
    for cmd in sorted(cmds):
        try: out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.rstrip()
        except Exception as e: out = f'Error: {e}'
        parts.append(f'<shell cmd="{cmd}">{out}</shell>')
    return ''.join(parts)

class Assistant:
    """Owns the AI side of a session: config, backend, conversation turns, and the cell records
    (source + raw iopub outputs) the app appends as cells complete. `backend` injects a fake for tests."""
    def __init__(self, cfg=None, tools=None, bridge=None, cwd=None, backend=None, system_prompt=None):
        self.cfg = cfg or load_config()
        self.spec = backend_spec(self.cfg['_backend_name'])
        self.model, self.completion_model, self.think = self.cfg['model'], self.cfg['completion_model'], self.cfg['think']
        self.tools, self.bridge = tools, bridge
        self.cwd = cwd or os.getcwd()
        self.system_prompt = load_sysp() if system_prompt is None else system_prompt
        self._backend = backend
        self.turns = []          # PromptTurn records, oldest first
        self.provider_session_id = None
        self.cells = []          # dicts: source, outputs=[(msg_type, content), ...], line
        self._ctx_cells = 0      # cells already carried by an earlier turn's full_prompt
        self.last_response = ''
        self.store = None        # optional Store (persistence); set by the app after kernel start
        self._consumer = None    # the in-flight turn's stream-consumer task, cancel_turn's target

    def make_backend(self, system_prompt=None):
        if self._backend is not None: return self._backend
        sp = self.system_prompt if system_prompt is None else system_prompt
        return self.spec.factory(shell=None, cwd=self.cwd, system_prompt=sp, tools=self.tools)

    def add_cell(self, source, outputs):
        self.cells.append(dict(source=source, outputs=outputs, line=len(self.cells) + 1))
        if self.store:
            try: self.store.save_cell(len(self.cells), source, outputs)
            except Exception: pass

    def context(self):
        "Recent cells (since the last prompt) as the `<context>` block; earlier ones ride in the seed's turns."
        parts = []
        for c in self.cells[self._ctx_cells:]:
            src = c['source']
            if _is_note(src): parts.append(_tag('note', _note_str(src)))
            else:
                parts.append(_tag('code', src))
                out = output_text(c['outputs'])
                if out: parts.append(_tag('output', out))
        return _tag('context', ''.join(parts)) + '\n' if parts else ''

    def seed(self):
        return ConversationSeed(turns=tuple(self.turns))

    async def full_prompt(self, prompt):
        "Context + `$`var``/`!`cmd`` expansions + the user request, the old template."
        names = set(_var_re.findall(prompt))
        for t in self.turns: names |= set(_var_re.findall(t.prompt))
        values = await _eval_vars(names, self.bridge)
        missing = sorted(n for n, v in values.items() if v is _MISSING)
        warn = _tag('warnings', f"The following symbols were referenced but aren't defined in the interpreter: {', '.join(missing)}") + '\n' if missing else ''
        cmds = set(_shell_re.findall(prompt))
        for t in self.turns: cmds |= set(_shell_re.findall(t.prompt))
        return warn + _format_var_xml(values) + _run_shell_refs(cmds) + self.context() + _tag('user-request', prompt.strip())

    def cancel_turn(self):
        "Stop the in-flight AI turn (ctrl-C): the consumer task is cancelled, not the caller."
        if self._consumer is not None and not self._consumer.done():
            self._consumer.cancel()
            return True
        return False

    async def run_prompt(self, prompt, renderer):
        """One AI turn: build the full prompt, stream the backend through `renderer` (a TurnRenderer),
        record the turn, and land the reply kernel-side. `cancel_turn` freezes the partial mid-stream;
        the consumer runs as its own task so cancelling it never poisons this coroutine's cleanup awaits."""
        prompt = (prompt or '').rstrip('\n')
        if not prompt.strip(): return None
        full = await self.full_prompt(prompt)
        backend = self.make_backend()
        fmt = backend.formatter_cls()
        turn = await backend.prepare_turn(prompt=full, model=self.model, think=self.think,
            provider_session_id=self.provider_session_id, seed=self.seed())
        stream = turn.stream

        async def _consume():
            async for e in stream:
                fmt._format_event(e)
                renderer.event(e)
        self._consumer = asyncio.ensure_future(_consume())
        try:
            await self._consumer
            renderer.done()
            text = fmt.final_text
        except asyncio.CancelledError:
            if not self._consumer.cancelled(): raise  # the caller itself was cancelled: propagate
            renderer.stopped()
            text = fmt.final_text + '\n<system>user interrupted</system>'
        finally:
            self._consumer = None
            if aclose := getattr(stream, 'aclose', None): await aclose()
        if sid := await turn.wait_provider_session_id(): self.provider_session_id = sid
        self.turns.append(PromptTurn(prompt=prompt, full_prompt=full, response=text, history_line=len(self.cells)))
        self._ctx_cells = len(self.cells)
        self.last_response = text
        if self.bridge:
            try: await self.bridge.set_vars(**{LAST_PROMPT: prompt, LAST_RESPONSE: text})
            except Exception: pass
        if self.store:
            try: self.store.save_prompt(prompt, full, text, len(self.cells), self.provider_session_id)
            except Exception: pass
        return text

    async def ai_complete(self, prefix, suffix=''):
        "One-shot inline completion (Alt-.): recent context + the split input, via the small model."
        parts = [self.context(), f'<current-input>\n<prefix>{prefix}</prefix>']
        if suffix.strip(): parts.append(f'<suffix>{suffix}</suffix>')
        parts += ['</current-input>', 'Return only the completion text to insert immediately after the prefix.']
        res = await self.make_backend(system_prompt=COMPLETION_SP).complete('\n'.join(p for p in parts if p), model=self.completion_model)
        return (getattr(res, 'content', res) or '').strip()

"""The assistant: mode routing, an aidialog `Dialog` as the session model, fastllm turns, kernel writeback.

The session IS a Dialog: executed cells append as code/note messages (with verbatim nbformat
outputs), AI turns append as prompt messages whose output holds the reply in fastllm's canonical
formatted form. Context assembly is `dlg2hist` -- no hand-rolled XML -- and every backend is one
fastllm `AsyncChat` built from the backend table's routing kwargs."""
import asyncio, ast, os, re, subprocess
from aidialog.dialog import Dialog, prompt_output
from aidialog.hist import dlg2hist   # also activates the Message.to_parts/ai_output patches
from .backends import backend_spec
from .config import load_config, load_sysp, COMPLETION_SP

LAST_PROMPT = '_ai_last_prompt'
LAST_RESPONSE = '_ai_last_response'
_var_re = re.compile(r"\$`(\w+(?:\([^`]*\))?)`")
_shell_re = re.compile(r"(?<![\w`])!`([^`]+)`")
_MISSING = object()

def route(text, prompt_mode):
    """The `.`/`;`/`!`/`%` dispatch, at submit time: ('prompt'|'code'|'job', payload).
    Code mode: a leading `.` sends the rest as a prompt. Prompt mode: everything is a prompt,
    except `;` (strip it, run as code) and `!`/`%` lines (shell/magics, routed like code mode).
    A single-line bare-`!` command (and `%fg`) is a 'job': the app-side pty layer runs it on the
    real terminal. Multiline input or embedded `!` (e.g. `x = !ls`) stays kernel-side code, so
    IPython's own transformer keeps exact SList semantics."""
    if prompt_mode:
        s = text.lstrip()
        if s.startswith(';'): return 'code', text.replace(';', '', 1)
        if not s.startswith(('!', '%')): return 'prompt', text
    elif text.startswith('.'): return 'prompt', text[1:]
    s = text.strip()
    if s.startswith('!') and '\n' not in s and s[1:].strip(): return 'job', s[1:].strip()
    if s.split()[:1] == ['%fg'] and '\n' not in s: return 'job', s
    return 'code', text

def _is_note(source):
    "A cell that is one bare string literal is a note, not code (the old ipyai convention)."
    try: tree = ast.parse(source)
    except SyntaxError: return False
    return (len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str))

def _note_str(source): return ast.parse(source).body[0].value.value

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

class _BridgeNS(dict):
    """Dict-shaped proxy routing fastllm's ns-based tool dispatch through the ToolRegistry bridge:
    `ns[name]` yields an async caller, so kernel-side tools look like local callables."""
    def __init__(self, registry):
        super().__init__()
        self._reg = registry

    def get(self, name, default=None):
        async def _caller(**kwargs): return await self._reg.call_text(name, kwargs)
        _caller.__name__ = name
        return _caller

    def __getitem__(self, name): return self.get(name)

class Assistant:
    """Owns the AI side of a session: config, the session `Dialog`, and the fastllm chat per turn.
    `chat_factory` injects a fake chat for tests."""
    def __init__(self, cfg=None, tools=None, bridge=None, cwd=None, chat_factory=None, system_prompt=None):
        self.cfg = cfg or load_config()
        self.spec = backend_spec(self.cfg['_backend_name'])
        self.model, self.completion_model, self.think = self.cfg['model'], self.cfg['completion_model'], self.cfg['think']
        self.tools, self.bridge = tools, bridge
        self.cwd = cwd or os.getcwd()
        self.system_prompt = load_sysp() if system_prompt is None else system_prompt
        self._chat_factory = chat_factory
        self.dlg = Dialog(name=os.path.basename(self.cwd))   # the session model
        self.n_cells = 0         # cells recorded, the store's line counter
        self.last_response = ''
        self.last_use = None     # fastllm UsageStats from the latest turn (status-bar material)
        self.last_req_use = None  # UsageStats of the final request alone: its size is the current context (`last_use` sums a whole turn)
        self.store = None        # optional Store (persistence); set by the app after kernel start
        self._consumer = None    # the in-flight turn's stream-consumer task, cancel_turn's target

    @property
    def aim_info(self):
        "Model capability dict for dlg2hist media handling; {} when the model is unknown to fastllm."
        try:
            from fastllm.types import get_model_info
            return dict(get_model_info(self.model, self.spec.chat_kw.get('vendor_name')) or {})
        except Exception: return {}

    @property
    def ctx_usage(self):
        "(tokens the context now holds, model max input) from the latest request; None before the first turn or for unknown models."
        u, mx = self.last_req_use, self.aim_info.get('max_input_tokens')
        if not (u and mx): return None
        return u.prompt_tokens + u.completion_tokens, mx

    def _make_chat(self, model, sp, tools=None, ns=None, hist=None):
        if self._chat_factory is not None:
            return self._chat_factory(model=model, sp=sp, tools=tools, ns=ns, hist=hist)
        from fastllm.chat import AsyncChat
        kw = dict(self.spec.chat_kw)
        if kw.get('api_name') == 'claude_code':
            import fastllm_claude_code.core  # noqa: F401 -- registers the claude_code api (the bare package import does not)
        return AsyncChat(model=model, sp=sp, tools=tools or None, hist=hist or None,
                         ns=ns if ns is not None else {}, **kw)

    def add_cell(self, source, outputs):
        "Record one executed cell: raw iopub (msg_type, content) records become an nbformat code/note message."
        from .store import nbformat_outputs
        outs = nbformat_outputs(outputs)
        if _is_note(source): self.dlg.mk_message(_note_str(source), msg_type='note')
        else: self.dlg.mk_message(source, msg_type='code', output=outs)
        self.n_cells += 1
        if self.store:
            try: self.store.save_cell(self.n_cells, source, outputs)
            except Exception: pass

    async def _ref_parts(self):
        "`$`var``/`!`cmd`` expansions gathered across every prompt in the dialog, evaluated now."
        names, cmds = set(), set()
        for m in self.dlg.messages:
            if m.msg_type == 'prompt':
                names |= set(_var_re.findall(m.content))
                cmds |= set(_shell_re.findall(m.content))
        values = await _eval_vars(names, self.bridge)
        missing = sorted(n for n, v in values.items() if v is _MISSING)
        parts = []
        if missing: parts.append(f"<warnings>The following symbols were referenced but aren't defined in the interpreter: {', '.join(missing)}</warnings>")
        if xml := _format_var_xml(values): parts.append(xml)
        if xml := _run_shell_refs(cmds): parts.append(xml)
        return parts

    def cancel_turn(self):
        "Stop the in-flight AI turn (ctrl-C): the consumer task is cancelled, not the caller."
        if self._consumer is not None and not self._consumer.done():
            self._consumer.cancel()
            return True
        return False

    async def run_prompt(self, prompt, renderer):
        """One AI turn: append the prompt message, build history via `dlg2hist`, stream the chat through
        `renderer` (a TurnRenderer) while an `AsyncStreamFormatter` tees the canonical stored form, then
        land the reply in the message output and the kernel namespace. `cancel_turn` freezes the partial
        mid-stream; the consumer runs as its own task so cancelling it never poisons cleanup awaits."""
        from fastllm.chat import AsyncStreamFormatter
        prompt = (prompt or '').strip()
        if not prompt: return None
        pmsg = self.dlg.mk_message(prompt, msg_type='prompt')
        try:
            *hist, parts, _ = dlg2hist(self.dlg, self.aim_info)
            parts = await self._ref_parts() + parts
            tools = ((await self.tools.openai_schemas()) or None) if self.tools else None
            ns = _BridgeNS(self.tools) if tools else {}
            chat = self._make_chat(self.model, self.system_prompt, tools=tools, ns=ns, hist=hist)
            stream = await chat(parts, stream=True, think=self.think or None, max_steps=21)
        except BaseException:
            self.dlg.remove_msgs([pmsg])   # the turn never started: a retry must not double the prompt
            raise
        fmt = AsyncStreamFormatter()
        async def _consume():
            async for e in stream:
                fmt.format_item(e)
                renderer.event(e)
        self._consumer = asyncio.ensure_future(_consume())
        interrupted = False
        try:
            await self._consumer
            renderer.done()
        except asyncio.CancelledError:
            if not self._consumer.cancelled(): raise  # the caller itself was cancelled: propagate
            renderer.stopped()
            interrupted = True
        except BaseException:
            self.dlg.remove_msgs([pmsg])
            raise
        finally:
            self._consumer = None
            if aclose := getattr(stream, 'aclose', None): await aclose()
        text = fmt.outp.strip()
        if interrupted: text += '\n\n*[Response interrupted]*'  # ai_fmt strips this marker on replay
        pmsg.output = text    # the setter wraps prompt output and clears the ai_output cache
        self.last_response = text
        self.last_use = getattr(chat, 'use', None)
        self.last_req_use = getattr(chat, 'last_req_use', None)
        if self.bridge:
            try: await self.bridge.set_vars(**{LAST_PROMPT: prompt, LAST_RESPONSE: text})
            except Exception: pass
        if self.store:
            try: self.store.save_prompt(prompt, '\n\n'.join(p for p in parts if isinstance(p, str)), text, self.n_cells)
            except Exception: pass
        return text

    def _recent_xml(self):
        "Messages since the last completed prompt, in history XML: the context for inline completion."
        msgs = self.dlg.messages
        i = max((j + 1 for j, m in enumerate(msgs) if m.msg_type == 'prompt'), default=0)
        return '\n'.join(x for m in msgs[i:] if (x := m.hist_xml()))

    async def ai_complete(self, prefix, suffix=''):
        "One-shot inline completion (Alt-.): recent context + the split input, via the small model."
        parts = [self._recent_xml(), f'<current-input>\n<prefix>{prefix}</prefix>']
        if suffix.strip(): parts.append(f'<suffix>{suffix}</suffix>')
        parts += ['</current-input>', 'Return only the completion text to insert immediately after the prefix.']
        chat = self._make_chat(self.completion_model, COMPLETION_SP)
        rs = await chat('\n'.join(p for p in parts if p), stream=True)
        try: return ''.join([(o.get('text') or '') async for o in rs if isinstance(o, dict)]).strip()
        finally:
            if aclose := getattr(rs, 'aclose', None): await aclose()

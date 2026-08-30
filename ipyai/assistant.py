"""The assistant: mode routing, an aidialog `Dialog` as the session model, fastllm turns, kernel writeback.

The session IS a Dialog: executed cells append as code/note messages (with verbatim nbformat
outputs), AI turns append as prompt messages whose output holds the reply in fastllm's canonical
formatted form. Ctx assembly is `dlg2hist` -- no hand-rolled XML -- and every model is one
fastllm `AsyncChat` built from a flat vendor-prefixed model string (e.g. 'codex/gpt-5.4')."""
import asyncio, ast, os
from aidialog.dialog import Dialog, INTERRUPTED
from aidialog.msg_parts import Text
from aidialog.hist import dlg2hist, get_exprs, is_nameerr, vars_hist, warning_tag
from fastcore.xml import to_xml
from .config import load_config, load_sysp, SUGGEST_SP, render_sp

LAST_PROMPT = '_ai_last_prompt'
LAST_RESPONSE = '_ai_last_response'

def route(text, mode='code'):
    """The mode dispatch, at submit time: ('prompt'|'code'|'job', payload). Three modes --
    prompt (the AI), code (the kernel), shell (the persistent shell) -- with per-submission
    prefix overrides valid from any *other* mode: `.` sends a prompt, `;` runs code, a leading
    `!` runs shell (multiline fine: one shell script). Overrides only apply when they change
    mode, so a prompt legitimately starting with `.` (or shell history's `!!`) passes through
    at home. Embedded `!` (`x = !ls`) is ordinary code, keeping IPython's exact SList capture
    semantics kernel-side; `%` lines go to the kernel from every mode."""
    s = text.lstrip()
    if mode != 'prompt' and s.startswith('.'): return 'prompt', s[1:]
    if mode != 'code' and s.startswith(';'): return 'code', text.replace(';', '', 1)
    if mode != 'shell' and s.startswith('!') and s[1:].strip(): return 'job', s[1:]
    if s.startswith('%'): return 'code', text  # magics reach the kernel from every mode
    if mode == 'prompt': return 'prompt', text
    if mode == 'shell': return 'job', text
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
    `chat_factory` supplies a stub chat for tests."""
    def __init__(self, cfg=None, tools=None, bridge=None, cwd=None, chat_factory=None, sp=None):
        self.cfg = cfg or load_config()
        self.model, self.suggest_model, self.think = self.cfg['model'], self.cfg['suggest_model'], self.cfg['think']
        self.tools, self.bridge = tools, bridge
        self.cwd = cwd or os.getcwd()
        self.sp = load_sysp() if sp is None else sp
        self._chat_factory = chat_factory
        self.dlg = Dialog(name=os.path.basename(self.cwd))   # the session model
        self.n_cells = 0         # cells recorded, the store's line counter
        self.last_response = ''
        self.last_use = None     # fastllm UsageStats from the latest turn (status-bar material)
        self.last_req_use = None  # UsageStats of the final request alone: its size is the current ctx (`last_use` sums a whole turn)
        self.session = None      # optional Session (persistence); set by the app after kernel start
        self._consumer = None    # the in-flight turn's stream-consumer task, cancel_turn's target

    def save(self):
        "Persist the dialog to its session file (whole-file write, atomic); a no-op without a session."
        if self.session: self.session.save(self.dlg, model=self.model, think=self.think)

    def reset(self):
        "Start a fresh conversation: new empty Dialog, counters cleared. Kernel state is untouched."
        self.dlg = Dialog(name=os.path.basename(self.cwd))
        self.n_cells = 0
        self.last_response = ''
        self.last_use = self.last_req_use = None

    @property
    def aim_info(self):
        "Model capability dict for dlg2hist media handling; {} when the model is unknown to fastllm."
        try:
            from fastllm.types import get_model_info
            from fastllm.acomplete import split_vendor
            v, m = split_vendor(self.model)
            return dict(get_model_info(m, v) or {})
        except Exception: return {}

    @property
    def ctx_usage(self):
        "(tk the ctx now holds, model max input) from the latest request; None before the first turn or for unknown models."
        u, mx = self.last_req_use, self.aim_info.get('max_input_tokens')
        if not (u and mx): return None
        return u.prompt_tokens + u.completion_tokens, mx

    def _make_chat(self, model, sp, tools=None, ns=None, hist=None):
        if self._chat_factory is not None: return self._chat_factory(model=model, sp=sp, tools=tools, ns=ns, hist=hist)
        from fastllm.chat import AsyncChat
        from fastllm.acomplete import split_vendor
        v, _ = split_vendor(model)
        return AsyncChat(model=model, sp=sp, tools=tools or None, hist=hist or None,
            ns=ns if ns is not None else {}, cache=(v == 'anthropic'))

    def add_cell(self, source):
        "Record a cell about to run as a code/note message, returned so its id can tag the execute (None when nothing records)."
        if source.lstrip().startswith('%ipyai'): return  # housekeeping commands are not part of the conversation
        if _is_note(source): m = self.dlg.mk_message(_note_str(source), msg_type='note')
        else: m = self.dlg.mk_message(source, msg_type='code', output=[])
        self.n_cells += 1
        return m

    def finish_cell(self, m, outputs):
        "Complete `add_cell`'s message once the run ends: outputs land on code messages, and the dialog saves."
        if m is None: return
        if m.msg_type == 'code': m.output = list(outputs or [])
        self.save()

    async def _vars_turn(self):
        """The synthetic variables turn plus any missing-var warning: `$` and `!` refs both resolved
        in ONE kernel round trip (solveit's `prepare_context` shape; `!` via the kernel's own
        `getoutput`, keyed by full ref form), merged for `vars_hist`."""
        names = get_exprs(self.dlg.messages)
        cmds = get_exprs(self.dlg.messages, sigil='!')
        cmd_exprs = {c: f'get_ipython().getoutput({c!r}).n' for c in cmds}
        ns = {}
        if self.bridge is not None and (names or cmd_exprs):
            ns = await self.bridge.client.eval_exprs(vs=names + list(cmd_exprs.values())) or {}
        missing = sorted(v for v in names if is_nameerr(ns.get(v)))
        ns = {v: ns[v] for v in names if v in ns and v not in missing} | {f'!`{c}`': ns[e] for c, e in cmd_exprs.items() if e in ns}
        warn = warning_tag(f"The following symbols were referenced but aren't defined in the interpreter: {', '.join(missing)}." if missing else '')
        return vars_hist(self.aim_info, ns), (to_xml(warn, do_escape=False) if warn else None)

    def cancel_turn(self):
        "Stop the in-flight AI turn (ctrl-C): the consumer task is cancelled, not the caller."
        if self._consumer is not None and not self._consumer.done():
            self._consumer.cancel()
            return True
        return False

    async def run_prompt(self, prompt, renderer):
        """One AI turn: append the prompt message, build history via `dlg2hist`, stream the chat through
        `renderer` (a TurnRenderer), then land the reply in the message output and the kernel namespace.
        The stored form is `chat.full()`; an interrupt freezes the accumulated partial instead.
        The consumer runs as its own task so cancelling it leaves cleanup awaits running."""
        from fastllm.chat import StreamAccum
        from fastllm.acomplete import split_vendor
        prompt = (prompt or '').strip()
        if not prompt: return None
        pmsg = self.dlg.mk_message(prompt, msg_type='prompt')
        try:
            *hist, parts, _ = dlg2hist(self.dlg, self.aim_info)
            vh, warn = await self._vars_turn()
            hist = vh + hist
            if warn: parts.insert(0, warn)
            tools = ((await self.tools.openai_schemas()) or None) if self.tools else None
            ns = _BridgeNS(self.tools) if tools else {}
            if self.bridge: self.bridge.aim_info = self.aim_info
            chat = self._make_chat(self.model, render_sp(self.sp, split_vendor(self.model)[1]), tools=tools, ns=ns, hist=hist)
            stream = await chat(parts, stream=True, think=self.think or None, max_steps=21)
        except BaseException:
            self.dlg.remove_msgs([pmsg])   # the turn never started: a retry must not double the prompt
            raise
        acc = StreamAccum(chat)
        async def _consume():
            async for e in stream:
                acc(e)
                renderer.event(e)
        self._consumer = asyncio.create_task(_consume(), name='stream-consumer')  # bare deliberately: its exception is consumed at the await below
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
        text = (acc.txt if interrupted else chat.full()).strip()
        if interrupted: text += '\n\n' + INTERRUPTED  # ai_fmt strips this marker on replay
        pmsg.output = text    # the setter wraps prompt output and clears the ai_output cache
        self.last_response = text
        self.last_use = getattr(chat, 'use', None)
        self.last_req_use = getattr(chat, 'last_req_use', None)
        if self.bridge:
            try: self.bridge.client.xpush(**{LAST_PROMPT: prompt, LAST_RESPONSE: text})
            except Exception: pass
        self.save()
        return text

    def _recent_xml(self):
        "Messages since the last completed prompt, in history XML: the ctx for inline suggestion."
        msgs = self.dlg.messages
        i = max((j + 1 for j, m in enumerate(msgs) if m.msg_type == 'prompt'), default=0)
        return '\n'.join(x for m in msgs[i:] if (x := m.hist_xml()))

    async def suggest(self, prefix, suffix=''):
        "One-shot inline suggestion (Alt-.): recent ctx + the split input, via the small model."
        parts = [self._recent_xml(), f'<current-input>\n<prefix>{prefix}</prefix>']
        if suffix.strip(): parts.append(f'<suffix>{suffix}</suffix>')
        parts += ['</current-input>', 'Return only the suggestion text to insert immediately after the prefix.']
        chat = self._make_chat(self.suggest_model, SUGGEST_SP)
        rs = await chat('\n'.join(p for p in parts if p), stream=True)
        try: return ''.join([o.text or '' async for o in rs if isinstance(o, Text)]).strip()
        finally:
            if aclose := getattr(rs, 'aclose', None): await aclose()

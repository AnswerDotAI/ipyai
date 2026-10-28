"""The assistant: mode routing, and the AI side of a turn.

A turn's model input comes from the session Dialog through aihist's `build`, in the form Solveit's turns use. Every
model is one fastllm `AsyncChat` built from a flat vendor-prefixed model string (e.g. 'codex/gpt-5.4'). The
controller owns the Dialog and runs turns; this module builds them."""
import ast
from aidialog.msg_parts import Text
from aihist import build, get_exprs, hist_xml
from fastllm.responses import response_input
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

def is_note(source):
    "A cell that is one bare string literal is a note, not code (the old ipyai convention)."
    try: tree = ast.parse(source)
    except SyntaxError: return False
    return (len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str))

def note_str(source): return ast.parse(source).body[0].value.value

def code_blocks(md):
    "Python fenced blocks of `md`, in order, via mdhtml structure (never regex)."
    import mdhtml
    return [b['text'].rstrip('\n') for b in mdhtml.blocks(md or '')
        if b['type'] == 'code_block' and b.get('lang') in ('python', 'py') and b.get('text', '').strip()]

def ai_msgs(msgs):
    "The messages the AI sees: every one not hidden with `skipped`."
    return [m for m in msgs if not m.skipped]

class Assistant:
    """Model settings and the chats built from them. `chat_factory` supplies a stub chat for tests.
    `last_req_use` is the latest turn's final request usage, for the ctx meter."""
    def __init__(self, cfg=None, chat_factory=None, sp=None):
        self.cfg = cfg or load_config()
        self.model, self.suggest_model, self.think = self.cfg['model'], self.cfg['suggest_model'], self.cfg['think']
        self.sp = load_sysp() if sp is None else sp
        self._chat_factory = chat_factory
        self.last_req_use = None

    @property
    def aim_info(self):
        "Model capability dict for aihist's media handling; {} when the model is unknown to fastllm."
        try:
            from fastllm.types import get_model_info
            from fastllm.acomplete import split_vendor
            v, m = split_vendor(self.model)
            return dict(get_model_info(m, v) or {})
        except Exception: return {}

    @property
    def ctx_usage(self):
        "(tk the ctx held, model max input) from the latest request; None before the first turn or for unknown models."
        u, mx = self.last_req_use, self.aim_info.get('max_input_tokens')
        if not (u and mx): return None
        return u.prompt_tokens + u.completion_tokens, mx

    def make_chat(self, model, sp, tools=None, ns=None, hist=None):
        if self._chat_factory is not None: return self._chat_factory(model=model, sp=sp, tools=tools, ns=ns, hist=hist)
        from fastllm.chat import AsyncChat
        from fastllm.acomplete import split_vendor
        v, _ = split_vendor(model)
        return AsyncChat(model=model, sp=sp, tools=tools or None, hist=hist or None,
            ns=ns if ns is not None else {}, cache=(v == 'anthropic'))

    async def start_turn(self, dlg, prompt, tools=None, times=None):
        """Build the model input for prompt Message `prompt` of `dlg` with aihist's `build`, and start its stream. The `$`
        values come from one `eval` operation, and `times` gives Messages' run times by id. Returns `(chat, stream)`."""
        from fastllm.acomplete import split_vendor
        msgs = dlg.messages[:next(i for i, m in enumerate(dlg.messages) if m is prompt) + 1]
        names = get_exprs(ai_msgs(msgs))
        recs = await tools.op_json(dict(eval=names)) if tools is not None and names else {}
        sp, aim = render_sp(self.sp, split_vendor(self.model)[1]), self.aim_info
        *hist, last = response_input(build(dlg, prompt.id, aim, vars=recs, instructions=sp, times=times or {}))[0]
        if tools: tools.aim_info = aim
        schemas = await tools.schemas() if tools else None
        chat = self.make_chat(self.model, sp, tools=schemas, ns=tools.ns(schemas) if tools else {}, hist=hist)
        return chat, await chat(last.content, stream=True, think=self.think or None, max_steps=21)

    async def suggest(self, msgs, prefix, suffix=''):
        "One inline suggestion for the composer: the messages since the last prompt, the split input, the small model."
        msgs = ai_msgs(msgs)
        i = max((j + 1 for j, m in enumerate(msgs) if m.msg_type == 'prompt'), default=0)
        recent = '\n'.join(x for m in msgs[i:] if (x := hist_xml(m)))
        parts = [recent, f'<current-input>\n<prefix>{prefix}</prefix>']
        if suffix.strip(): parts.append(f'<suffix>{suffix}</suffix>')
        parts += ['</current-input>', 'Return only the suggestion text to insert immediately after the prefix.']
        chat = self.make_chat(self.suggest_model, SUGGEST_SP)
        rs = await chat('\n'.join(p for p in parts if p), stream=True)
        try: return ''.join([o.text or '' async for o in rs if isinstance(o, Text)]).lstrip('\n').rstrip()
        finally:
            if aclose := getattr(rs, 'aclose', None): await aclose()

import ast, html, os, re, sys, traceback
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Literal

from fastllm.chat import _trunc_param as trunc_param

from .tooling import ToolRegistry

_THINK_RE = re.compile(r"<thinking>\n.*?\n</thinking>\n*", flags=re.DOTALL)
_TOOL_PREFIX_RE = re.compile(r"^mcp__\w+__")

COMPLETION_THINK = "l"  # effort level for inline AI completions (Alt-.); intentionally fixed regardless of DEFAULT_THINK


@dataclass(frozen=True)
class BackendContext:
    cwd: str
    system_prompt: str = ""
    plugin_dirs: tuple[str, ...] = ()
    cli_path: str | None = None


@dataclass(frozen=True)
class PromptTurn:
    prompt: str
    full_prompt: str
    response: str
    history_line: int


@dataclass(frozen=True)
class StartupEvent:
    kind: Literal["code", "prompt"]
    line: int
    source: str | None = None
    prompt: str | None = None
    full_prompt: str | None = None
    response: str | None = None
    history_line: int | None = None


@dataclass(frozen=True)
class ConversationSeed:
    turns: tuple[PromptTurn, ...] = ()
    startup_events: tuple[StartupEvent, ...] = ()


@dataclass
class PreparedTurn:
    stream: AsyncIterator[object]
    _state: dict

    async def wait_provider_session_id(self): return self._state.get("provider_session_id")

    def set_provider_session_id(self, session_id):
        if session_id: self._state["provider_session_id"] = session_id


class TextResponse(str):
    @property
    def content(self): return str(self)


def _xml_text(text): return html.escape(text or "", quote=False)


def _blockquote(text): return "".join(f"> {line}\n" if line.strip() else ">\n" for line in text.splitlines()) if text else ""


def _fenced_block(text, info=""):
    text = text or ""
    fence = "~" * 3
    while fence in text: fence += "~"
    if text and not text.endswith("\n"): text += "\n"
    return f"{fence}{info}\n{text}{fence}\n"


def _is_note(source):
    try: tree = ast.parse(source)
    except SyntaxError: return False
    return (len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str))


def strip_thinking(text): return _THINK_RE.sub("", text or "").strip()


def thinking_to_blockquote(text):
    def _bq(m): return _blockquote(m.group(1).strip()) + "\n"
    return re.sub(r"<thinking>\n(.*?)\n</thinking>\n*", _bq, text or "", flags=re.DOTALL)


def replayable_assistant_text(text):
    text = text or ""
    return text if text.strip() else "<system>user interrupted</system>"


def tool_name(name): return _TOOL_PREFIX_RE.sub("", name or "")


def tool_call(name, args, mx=40):
    name = tool_name(name)
    if not args: return f"{name}()"
    return f"{name}({', '.join(f'{k}={trunc_param(v, mx=mx)}' for k,v in sorted(args.items()))})"


def compact_tool(name, args, result, is_error=False, max_len=100):
    call = tool_call(name, args)
    res = (result or "").strip().replace("\n", " ")
    if len(res) > max_len: res = res[:max_len-3] + "..."
    status = " [error]" if is_error else ""
    return f"\n\n🔧 {call}{status} => {res}\n\n" if res else f"\n\n🔧 {call}{status}\n\n"


def compact_cmd(command, output, exit_code, max_len=80):
    res = (output or "").strip().replace("\n", " ")
    if len(res) > max_len: res = res[:max_len-3] + "..."
    status = "" if exit_code in (None, 0) else f" [exit {exit_code}]"
    return f"\n\n🔧 {command}{status} => {res}\n\n" if res else f"\n\n🔧 {command}{status}\n\n"


def effort_level(level): return dict(l="low", m="medium", h="high").get(level, level or None)


def seed_to_flat_history(seed):
    hist = []
    for turn in seed.turns: hist += [turn.full_prompt, replayable_assistant_text(turn.response)]
    return hist


def seed_to_notebook_xml(seed):
    parts = ["<ipython-notebook>"]
    for o in seed.startup_events:
        if o.kind == "code":
            tag = "note" if _is_note(o.source or "") else "code"
            parts.append(f'<{tag} line="{int(o.line)}">{_xml_text(o.source or "")}</{tag}>')
        elif o.kind == "prompt":
            parts.append(f'<turn line="{int(o.history_line or 0)}"><user>{_xml_text(o.full_prompt or "")}</user>'
                f'<assistant>{_xml_text(o.response or "")}</assistant></turn>')
    parts.append("</ipython-notebook>")
    return "".join(parts)


class CommonStreamFormatter:
    def __init__(self):
        self.is_tty = False
        self.final_text = ""
        self.display_text = ""
        self._thinking_text = ""
        self._tool_text = ""
        self._live_commands = {}

    def _update_display(self):
        parts = []
        if self._thinking_text: parts.append(_blockquote(self._thinking_text).rstrip())
        if self.final_text: parts.append(self.final_text.rstrip())
        if self._tool_text: parts.append(self._tool_text.rstrip())
        live = "\n\n".join(self._live_command_text(o) for o in self._live_commands.values())
        if live: parts.append(live)
        self.display_text = "\n\n".join(o for o in parts if o)

    def _append_final(self, text):
        if text: self.final_text += text
        self._update_display()

    def _live_command_text(self, state):
        cmd = html.escape(state.get("command") or "command")
        text = f"⌛ <code>{cmd}</code>"
        if state.get("output"): text += "\n\n" + _fenced_block(state["output"], "text")
        return text.rstrip()

    def _format_event(self, event):
        if isinstance(event, str):
            self._append_final(event)
            return event
        if not isinstance(event, dict): return ""
        kind = event.get("kind")
        if kind == "thinking_start":
            self._thinking_text = ""
            self._update_display()
            return ""
        if kind == "thinking_delta":
            self._thinking_text += event.get("delta", "")
            self._update_display()
            return ""
        if kind == "thinking_end":
            stored = f"<thinking>\n{self._thinking_text}\n</thinking>\n\n" if self._thinking_text else ""
            self._thinking_text = ""
            self._append_final(stored)
            return "" if self.is_tty else stored
        if kind == "tool_start":
            self._tool_text = f"⌛ `{tool_call(event.get('name') or 'tool', event.get('input') or {})}`"
            self._update_display()
            return ""
        if kind == "tool_complete":
            self._tool_text = ""
            text = compact_tool(event.get("name") or "tool", event.get("input") or {}, event.get("content") or "",
                is_error=bool(event.get("is_error")), max_len=event.get("max_len", 100))
            self._append_final(text)
            return "" if self.is_tty else text
        if kind == "command_start":
            self._live_commands[event.get("id")] = dict(command=event.get("command"), cwd=event.get("cwd"), output="")
            self._update_display()
            return ""
        if kind == "command_delta":
            state = self._live_commands.setdefault(event.get("id"), dict(command=event.get("command"), cwd=event.get("cwd"), output=""))
            if event.get("command") and not state.get("command"): state["command"] = event["command"]
            state["output"] += event.get("delta", "")
            self._update_display()
            return ""
        if kind == "command_complete":
            self._live_commands.pop(event.get("id"), None)
            text = event.get("text") or compact_cmd(event.get("command") or "", event.get("output") or "", event.get("exit_code"),
                event.get("max_len", 80))
            self._append_final(text)
            return "" if self.is_tty else text
        text = event.get("text", "")
        if text: self._append_final(text)
        return text

    async def format_stream(self, stream):
        async for o in stream: yield self._format_event(o)


class BaseBackend:
    formatter_cls = CommonStreamFormatter

    def __init__(self, shell=None, cwd=None, system_prompt="", plugin_dirs=None, cli_path=None, tools=None):
        self.shell = shell
        self.ctx = BackendContext(cwd=str(Path(cwd or os.getcwd()).resolve()), system_prompt=system_prompt,
            plugin_dirs=tuple(str(Path(o).resolve()) for o in (plugin_dirs or [])), cli_path=cli_path)
        if tools is not None: self.tools = tools
        else:
            reg = getattr(shell, "_ipyai_tool_registry", None)
            self.tools = reg if reg is not None else ToolRegistry.from_ns(getattr(shell, "user_ns", {}))

    @property
    def ns(self): return getattr(self.tools, "ns", None)

    async def complete(self, prompt, *, model):
        turn = await self.prepare_turn(prompt=prompt, model=model, think=COMPLETION_THINK, provider_session_id=None, seed=ConversationSeed(),
            tool_mode="off", ephemeral=True)
        fmt = self.formatter_cls()
        if hasattr(fmt, "is_tty"): fmt.is_tty = True
        try: return TextResponse((await collect_text(fmt.format_stream(turn.stream))).strip())
        finally:
            if aclose := getattr(turn.stream, "aclose", None): await aclose()

    def prepared_turn(self, stream, provider_session_id=None, state=None):
        state = {} if state is None else state
        if provider_session_id is not None: state["provider_session_id"] = provider_session_id
        return PreparedTurn(stream=stream, _state=state)


async def collect_text(stream): return "".join([o async for o in stream if isinstance(o, str)])


def print_unexpected_error(label, exc, file=None):
    file = file or sys.stderr
    print(f"\n{label}: {exc}", file=file)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=file)

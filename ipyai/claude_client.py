"Claude backend that spawns `claude -p` per turn, using a synthetic session JSONL to carry prior context and a unix-socket MCP bridge for custom in-kernel tools."
import asyncio, json, os, re, shutil, unicodedata, uuid
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .backend_common import (BaseBackend, CommonStreamFormatter, ConversationSeed, compact_tool, effort_level,
    replayable_assistant_text, strip_thinking, tool_call, tool_name)
from .mcp_server import ToolSocketServer


BUILTIN_TOOLS = ["Bash", "Edit", "Read", "Skill", "WebFetch", "WebSearch", "Write"]
_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9]")
_MAX_SANITIZED = 200


def _iso(ts): return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stringify_content(content):
    if content is None: return ""
    if isinstance(content, str): return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text": parts.append(item.get("text", ""))
        else: parts.append(json.dumps(item, ensure_ascii=False, default=str))
    return "\n".join(o for o in parts if o)


def _simple_hash(s):
    h = 0
    for ch in s:
        h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
        if h >= 0x80000000: h -= 0x100000000
    h = abs(h)
    if h == 0: return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = []
    while h > 0:
        out.append(digits[h % 36])
        h //= 36
    return "".join(reversed(out))


def _sanitize_path(name):
    s = _SANITIZE_RE.sub("-", name)
    if len(s) <= _MAX_SANITIZED: return s
    return f"{s[:_MAX_SANITIZED]}-{_simple_hash(name)}"


def _claude_config_home():
    d = os.environ.get("CLAUDE_CONFIG_DIR")
    if d: return Path(unicodedata.normalize("NFC", d))
    return Path(unicodedata.normalize("NFC", str(Path.home()/".claude")))


def _project_dir(project_root): return _claude_config_home()/"projects"/_sanitize_path(str(project_root))


def write_synthetic_session(project_root, turns: Iterable, session_id=None):
    project_root = Path(project_root).resolve()
    session_id = session_id or str(uuid.uuid4())
    d = _project_dir(str(project_root))
    d.mkdir(parents=True, exist_ok=True)
    path = d/f"{session_id}.jsonl"
    turns = list(turns)
    now = datetime.now(timezone.utc) - timedelta(seconds=max(2, len(turns) * 2))
    parent_uuid = None
    lines = []
    for i,(prompt,response) in enumerate(turns):
        user_uuid,assistant_uuid = str(uuid.uuid4()), str(uuid.uuid4())
        lines.append(dict(type="user", uuid=user_uuid, parentUuid=parent_uuid, sessionId=session_id,
            timestamp=_iso(now + timedelta(seconds=i * 2)),
            cwd=str(project_root) if parent_uuid is None else None, message=dict(role="user", content=prompt)))
        lines.append(dict(type="assistant", uuid=assistant_uuid, parentUuid=user_uuid, sessionId=session_id,
            timestamp=_iso(now + timedelta(seconds=i * 2 + 1)),
            message=dict(role="assistant", content=[dict(type="text", text=strip_thinking(response))])))
        parent_uuid = assistant_uuid
    path.write_text("".join(json.dumps({k:v for k,v in o.items() if v is not None}, ensure_ascii=False, separators=(",", ":")) + "\n" for o in lines),
        encoding="utf-8")
    return dict(session_id=session_id, path=path)


def _jsonl_lines(path):
    try:
        with open(path, encoding="utf-8") as f: raw = [o for o in (line.strip() for line in f) if o]
    except OSError: return None
    out = []
    for line in raw:
        try: out.append(json.loads(line))
        except json.JSONDecodeError: return None
    return out


def _session_belongs_to_us(path, our_ids):
    lines = _jsonl_lines(path)
    if not lines: return False
    return any(o.get("sessionId") in our_ids for o in lines)


def _is_ai_title_stub(path):
    lines = _jsonl_lines(path)
    if not lines: return False
    return all(o.get("type") == "ai-title" for o in lines)


AsyncStreamFormatter = CommonStreamFormatter
_tool_name,_tool_call,_compact_tool = tool_name,tool_call,compact_tool


class ClaudeBackend(BaseBackend):
    formatter_cls = AsyncStreamFormatter

    def _cli_path(self): return self.ctx.cli_path or shutil.which("claude") or "claude"

    def _bridge_path(self): return shutil.which("ipyai-mcp-bridge") or "ipyai-mcp-bridge"

    def _mcp_config(self, sock_path):
        return dict(mcpServers=dict(ipy=dict(command=self._bridge_path(), args=[], env=dict(IPYAI_MCP_SOCK=sock_path))))

    def _cli_args(self, *, session_id, use_resume, model, think, allow_tools, sock_path, allowed_tool_names):
        args = [self._cli_path(), "-p", "--output-format", "stream-json", "--include-partial-messages", "--verbose",
            "--no-session-persistence", "--system-prompt", self.ctx.system_prompt or "", "--model", model, "--setting-sources", "user,project"]
        args += ["--resume", session_id] if use_resume else ["--session-id", session_id]
        if (e := effort_level(think)): args += ["--effort", e]
        for d in self.ctx.plugin_dirs: args += ["--plugin-dir", d]
        if allow_tools:
            allowed = [*BUILTIN_TOOLS, *allowed_tool_names]
            args += ["--tools", ",".join(BUILTIN_TOOLS), "--allowed-tools", *allowed]
            if sock_path: args += ["--mcp-config", json.dumps(self._mcp_config(sock_path))]
        else: args += ["--tools", ""]
        return args

    async def prepare_turn(self, *, prompt, model, think="l", provider_session_id=None, seed=None, tool_mode="on", ephemeral=False):
        seed = seed or ConversationSeed()
        state = {}
        turns = [(turn.full_prompt, replayable_assistant_text(turn.response)) for turn in seed.turns]
        proj_dir = _project_dir(str(Path(self.ctx.cwd).resolve()))
        existing_sessions = {p.name for p in proj_dir.glob("*.jsonl")} if proj_dir.exists() else set()
        if turns:
            info = write_synthetic_session(self.ctx.cwd, turns)
            session_id,use_resume = info["session_id"], True
        else: session_id,use_resume = str(uuid.uuid4()), False
        state["provider_session_id"] = session_id
        our_ids = {session_id}
        allow_tools = tool_mode != "off"
        sock_server = None
        tool_names = await self.tools.names() if allow_tools else []
        allowed_tool_names = await self.tools.claude_allowed_tool_names() if allow_tools else []
        if allow_tools and tool_names: sock_server = await ToolSocketServer(self.tools).start()
        sock_path = sock_server.sock_path if sock_server else ""

        async def _stream():
            proc = stderr_task = None
            stderr_buf = bytearray()

            async def _drain_stderr(stream):
                while True:
                    chunk = await stream.read(4096)
                    if not chunk: break
                    stderr_buf.extend(chunk)

            try:
                args = self._cli_args(session_id=session_id, use_resume=use_resume, model=model, think=think,
                    allow_tools=allow_tools, sock_path=sock_path, allowed_tool_names=allowed_tool_names)
                proc = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, cwd=self.ctx.cwd, start_new_session=True)
                stderr_task = asyncio.create_task(_drain_stderr(proc.stderr))
                try:
                    proc.stdin.write(prompt.encode())
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError): pass
                try: proc.stdin.close()
                except Exception: pass
                thinking_open = False
                tools = {}
                while True:
                    raw = await proc.stdout.readline()
                    if not raw: break
                    try: msg = json.loads(raw)
                    except json.JSONDecodeError: continue
                    typ = msg.get("type")
                    if typ == "system" and msg.get("subtype") == "init":
                        if (sid := msg.get("session_id")):
                            state["provider_session_id"] = sid
                            our_ids.add(sid)
                        continue
                    if typ == "stream_event":
                        event = msg.get("event") or {}
                        etype = event.get("type")
                        if etype == "content_block_start" and (event.get("content_block") or {}).get("type") == "thinking":
                            if not thinking_open:
                                thinking_open = True
                                yield dict(kind="thinking_start")
                        elif etype == "content_block_delta":
                            delta = event.get("delta") or {}
                            dtype = delta.get("type")
                            if dtype == "text_delta":
                                if (text := delta.get("text", "")): yield text
                            elif dtype == "thinking_delta":
                                if not thinking_open:
                                    thinking_open = True
                                    yield dict(kind="thinking_start")
                                yield dict(kind="thinking_delta", delta=delta.get("thinking", ""))
                        elif etype == "content_block_stop" and thinking_open:
                            thinking_open = False
                            yield dict(kind="thinking_end")
                        continue
                    if typ == "assistant":
                        message = msg.get("message") or {}
                        if (sid := msg.get("session_id")):
                            state["provider_session_id"] = sid
                            our_ids.add(sid)
                        for block in message.get("content") or []:
                            if block.get("type") == "tool_use":
                                tid = block.get("id")
                                tools[tid] = dict(name=block.get("name"), input=block.get("input") or {})
                                yield dict(kind="tool_start", id=tid, name=block.get("name"), input=block.get("input") or {})
                        continue
                    if typ == "user":
                        message = msg.get("message") or {}
                        blocks = message.get("content") if isinstance(message.get("content"), list) else []
                        for block in blocks:
                            if not isinstance(block, dict) or block.get("type") != "tool_result": continue
                            tid = block.get("tool_use_id")
                            meta = tools.get(tid, {})
                            yield dict(kind="tool_complete", id=tid, name=meta.get("name"), input=meta.get("input"),
                                content=_stringify_content(block.get("content")), is_error=bool(block.get("is_error")))
                        continue
                    if typ == "result":
                        if (sid := msg.get("session_id")):
                            state["provider_session_id"] = sid
                            our_ids.add(sid)
                        if msg.get("is_error"):
                            raise RuntimeError(msg.get("result") or msg.get("error") or stderr_buf.decode("utf-8", errors="replace") or "claude -p failed")
                        break
            finally:
                if proc is not None and proc.returncode is None:
                    try: proc.terminate()
                    except ProcessLookupError: pass
                if proc is not None:
                    try: await proc.wait()
                    except Exception: pass
                if stderr_task is not None:
                    stderr_task.cancel()
                    try: await stderr_task
                    except (asyncio.CancelledError, Exception): pass
                if sock_server is not None: await sock_server.stop()
                if proj_dir.exists():
                    for p in proj_dir.glob("*.jsonl"):
                        if p.name in existing_sessions: continue
                        if _session_belongs_to_us(p, our_ids) or _is_ai_title_stub(p):
                            try: p.unlink()
                            except FileNotFoundError: pass

        return self.prepared_turn(_stream(), provider_session_id=session_id, state=state)


class AsyncChat:
    def __init__(self, model, sp="", backend_factory=ClaudeBackend, **backend_kwargs):
        self.model,self.sp,self.backend_factory,self.backend_kwargs = model,sp,backend_factory,backend_kwargs

    async def __call__(self, prompt, think="l"):
        backend = self.backend_factory(system_prompt=self.sp, **self.backend_kwargs)
        return await backend.complete(prompt, model=self.model)

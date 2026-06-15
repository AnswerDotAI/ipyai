from types import SimpleNamespace

import pytest

from ipyai.backend_common import BaseBackend, COMPLETION_THINK, ConversationSeed
from ipyai.core import IPyAIController, LAST_RESPONSE, _eval_vars
from ipyai.tooling import ToolRegistry


async def test_tool_registry_exposes_provider_shapes(shell):
    def python(code: str):
        "Execute code"
        return code

    shell.user_ns["python"] = python
    reg = ToolRegistry.from_ns(shell.user_ns)

    assert await reg.names() == ["python"]
    assert await reg.claude_allowed_tool_names() == ["mcp__ipy__python"]
    tool = (await reg.codex_dynamic_tools())[0]
    assert tool["name"] == "python"
    assert tool["description"] == "Execute code"
    assert tool["inputSchema"]["type"] == "object"
    assert tool["inputSchema"]["required"] == ["code"]
    assert tool["inputSchema"]["properties"]["code"]["type"] == "string"


def test_controller_conversation_seed_is_typed(shell, test_db):
    ctrl = IPyAIController(shell=shell, db=test_db, session_number=1)
    shell.history_manager.add(1, "x = 1")
    ctrl.save_prompt("what", "<user-request>what</user-request>", "answer", 1)

    seed = ctrl.conversation_seed()

    assert seed.turns[0].prompt == "what"
    assert seed.turns[0].full_prompt == "<user-request>what</user-request>"
    assert seed.startup_events[0].kind == "code"
    assert seed.startup_events[1].kind == "prompt"


class DummyBackend(BaseBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    async def prepare_turn(self, **kwargs):
        self.calls.append(kwargs)
        async def _stream(): yield "hello"

        turn = self.prepared_turn(_stream())
        if kwargs.get("provider_session_id") is None and kwargs.get("tool_mode") != "off": turn.set_provider_session_id("sess_late")
        return turn


class ObjectFormatter:
    async def format_stream(self, stream):
        async for o in stream: yield o["text"] if isinstance(o, dict) else o


class ObjectStreamBackend(DummyBackend):
    formatter_cls = ObjectFormatter

    async def prepare_turn(self, **kwargs):
        self.calls.append(kwargs)
        async def _stream(): yield dict(text="hello")
        return self.prepared_turn(_stream())


class FailingBackend(DummyBackend):
    async def prepare_turn(self, **kwargs): raise RuntimeError("backend broke")


class FakeKeyBindings:
    def __init__(self): self.handlers = {}

    def add(self, *keys):
        def _decorator(fn):
            self.handlers[keys] = fn
            return fn
        return _decorator


class BrokenBridge:
    async def read_var(self, name): raise TimeoutError("kernel shell reply timeout")


class HistorySuggest:
    def get_suggestion(self, buffer, document):
        from prompt_toolkit.auto_suggest import Suggestion
        return Suggestion("_history")


async def test_base_backend_complete_enforces_tool_off_ephemeral_policy(shell):
    backend = DummyBackend(shell=shell)

    res = await backend.complete("hi", model="demo")

    assert str(res) == "hello"
    call = backend.calls[0]
    assert call["prompt"] == "hi"
    assert call["model"] == "demo"
    assert call["think"] == COMPLETION_THINK
    assert call["provider_session_id"] is None
    assert call["seed"] == ConversationSeed()
    assert call["tool_mode"] == "off"
    assert call["ephemeral"] is True


async def test_base_backend_complete_formats_provider_objects(shell):
    backend = ObjectStreamBackend(shell=shell)

    res = await backend.complete("hi", model="demo")

    assert str(res) == "hello"


async def test_prepared_turn_exposes_late_provider_session_id(shell):
    backend = DummyBackend(shell=shell)
    turn = await backend.prepare_turn(prompt="hi", model="demo", think="l", provider_session_id=None, seed=ConversationSeed())

    text = "".join([o async for o in turn.stream])

    assert text == "hello"
    assert await turn.wait_provider_session_id() == "sess_late"


async def test_core_run_prompt_passes_conversation_seed(shell, test_db, monkeypatch):
    backend = DummyBackend(shell=shell)
    ctrl = IPyAIController(shell=shell, backend_factory=lambda **kwargs: backend, db=test_db, session_number=1)
    ctrl.load()
    shell.history_manager.add(1, "x = 1")
    shell.execution_count = 2

    async def _fake_astream_to_stdout(stream, **kwargs): return "".join([o async for o in stream])
    monkeypatch.setattr("ipyai.core.astream_to_stdout", _fake_astream_to_stdout)

    await ctrl.run_prompt("hello")

    seed = backend.calls[0]["seed"]
    assert isinstance(seed, ConversationSeed)
    assert seed.startup_events[0].kind == "code"
    assert shell.user_ns[LAST_RESPONSE] == "hello"
    assert shell.execution_count == 3, f"execution_count should advance after a prompt (was 2): {shell.execution_count}"


async def test_core_run_prompt_prints_unexpected_backend_errors(shell, test_db, capsys):
    backend = FailingBackend(shell=shell)
    ctrl = IPyAIController(shell=shell, backend_factory=lambda **kwargs: backend, db=test_db, session_number=1)
    ctrl.load()

    with pytest.raises(RuntimeError, match="backend broke"): await ctrl.run_prompt("hello")

    err = capsys.readouterr().err
    assert "AI prompt failed" in err
    assert "RuntimeError: backend broke" in err


async def test_ai_suggest_prints_background_errors(shell, test_db, capsys):
    pt = SimpleNamespace(auto_suggest=None, key_bindings=FakeKeyBindings())
    shell.pt_cli = pt
    ctrl = IPyAIController(shell=shell, db=test_db, session_number=1)

    async def _fail(doc): raise RuntimeError("completion broke")
    ctrl._ai_complete = _fail
    ctrl._register_keybindings()

    tasks = []
    app = SimpleNamespace(create_background_task=tasks.append, invalidate=lambda: None)
    doc = SimpleNamespace(text="pri", text_before_cursor="pri", text_after_cursor="")
    buf = SimpleNamespace(document=doc, suggestion=None)
    event = SimpleNamespace(current_buffer=buf, app=app)

    pt.key_bindings.handlers[("escape", ".")](event)
    await tasks[0]

    err = capsys.readouterr().err
    assert "AI completion failed" in err
    assert "RuntimeError: completion broke" in err


async def test_ai_suggest_overrides_history_suggestion(shell, test_db):
    from prompt_toolkit.document import Document
    pt = SimpleNamespace(auto_suggest=HistorySuggest(), key_bindings=FakeKeyBindings())
    shell.pt_cli = pt
    ctrl = IPyAIController(shell=shell, db=test_db, session_number=1)

    async def _complete(doc): return "_ai"
    ctrl._ai_complete = _complete
    ctrl._register_keybindings()

    tasks = []
    app = SimpleNamespace(create_background_task=tasks.append, invalidate=lambda: None)
    doc = Document("pri")
    buf = SimpleNamespace(document=doc, suggestion=pt.auto_suggest.get_suggestion(SimpleNamespace(history=None), doc))
    event = SimpleNamespace(current_buffer=buf, app=app)

    assert buf.suggestion.text == "_history"
    pt.key_bindings.handlers[("escape", ".")](event)
    await tasks[0]

    assert buf.suggestion.text == "_ai"
    assert pt.auto_suggest.get_suggestion(buf, doc).text == "_ai"


async def test_variable_read_failures_are_printed(capsys):
    vals = await _eval_vars({"x"}, BrokenBridge())

    assert vals["x"] is not None
    err = capsys.readouterr().err
    assert "Variable reference failed" in err
    assert "TimeoutError: kernel shell reply timeout" in err


def test_refresh_prompt_prints_invalidate_errors(shell, test_db, capsys):
    class App:
        def invalidate(self): raise RuntimeError("redraw broke")

    shell.pt_cli = SimpleNamespace(app=App())
    ctrl = IPyAIController(shell=shell, db=test_db, session_number=1)

    ctrl._refresh_prompt()

    err = capsys.readouterr().err
    assert "Prompt redraw failed" in err
    assert "RuntimeError: redraw broke" in err

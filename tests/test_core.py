"Fast pure-unit tests: input transforms and session listing."
import json, os
from types import SimpleNamespace

from IPython.core.inputtransformer2 import TransformerManager

from ipyai.backends import BACKEND_CLAUDE_CLI, BACKEND_CODEX
from ipyai.core import SESSIONS_TABLE, _list_sessions, _resume_command, prompt_from_lines, transform_dots


def test_prompt_from_lines_and_transform_dots():
    lines = [".plan this work\\\n", "in two steps\n"]
    assert prompt_from_lines(lines) == "plan this work\nin two steps\n"
    code = "".join(transform_dots([".hello\n", "world\n"]))
    assert "run_cell_magic('ipyai'" in code


def test_cleanup_transform_works_with_ipython_transformer():
    tm = TransformerManager()
    tm.cleanup_transforms.insert(1, transform_dots)
    code = tm.transform_cell(".Ask a question\\\nwith a newline")
    assert code == "get_ipython().run_cell_magic('ipyai', '', 'Ask a question\\nwith a newline\\n')\n"


def test_list_sessions_filters_backend(test_db):
    cwd = os.getcwd()
    with test_db:
        test_db.execute(f"INSERT INTO {SESSIONS_TABLE} (session, remark) VALUES (?, ?)",
            (2, json.dumps(dict(cwd=cwd, backend=BACKEND_CLAUDE_CLI))))
        test_db.execute(f"INSERT INTO {SESSIONS_TABLE} (session, remark) VALUES (?, ?)",
            (3, json.dumps(dict(cwd=cwd, backend=BACKEND_CODEX))))
        test_db.execute(f"INSERT INTO {SESSIONS_TABLE} (session, remark) VALUES (?, ?)",
            (4, json.dumps(dict(cwd="/tmp/elsewhere", backend=BACKEND_CLAUDE_CLI))))

    rows = _list_sessions(test_db, cwd, BACKEND_CLAUDE_CLI)

    assert [row[0] for row in rows] == [2]


def test_resume_command_uses_existing_connection_file_when_attached(tmp_path, monkeypatch):
    "In --existing mode, the resume hint must point at the connection file, not at a bogus `-r session_id` that won't rebuild the attached state."
    import ipyai.core as core
    (tmp_path/"config.json").write_text('{"backend":"codex-api"}\n')
    monkeypatch.setattr(core, "CONFIG_PATH", tmp_path/"config.json")
    cf = "/tmp/kernel-1234.json"

    assert _resume_command(5, "codex-api") == "ipyai -r 5"
    assert _resume_command(5, "claude-api") == "ipyai -b claude-api -r 5"
    assert _resume_command(5, "codex-api", existing=cf) == f"ipyai --existing={cf}"
    assert _resume_command(5, "claude-api", existing=cf) == f"ipyai -b claude-api --existing={cf}"

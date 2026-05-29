# DEV

## Setup

Editable install:

```bash
pip install -e ipyai
```

Run tests:

```bash
cd ipyai
./tools/test.sh
```

The test harness keeps setup small: `tools/test.sh` redirects `XDG_CONFIG_HOME` to a repo-local temp dir so ipyai config writes stay out of a normal user config tree, then runs `pytest`. It does not redirect `CLAUDE_CONFIG_DIR`, because on macOS `claude -p` OAuth reads credentials from the login keychain keyed by the `userID` in `~/.claude.json`; redirecting the config dir breaks that lookup. Instead, `ClaudeBackend` sweeps any session jsonls it can identify as its own (or as known claude side-effect stubs) after each turn. The live Codex app-server test archives the threads it creates after the test so they do not show up in `codex resume`.

## File Map

- [ipyai/core.py](ipyai/core.py): client-side `IPyAIController`, prompt transforms, SQLite bookkeeping, notebook save/load, prompt mode, keybindings, Rich streaming display, backend selection
- [ipyai/backends.py](ipyai/backends.py): backend registry, canonical backend names, default models
- [ipyai/backend_common.py](ipyai/backend_common.py): shared backend context/base classes, typed conversation seed types, common stream formatter, replay helpers, and shared tool/command display helpers
- [ipyai/claude_client.py](ipyai/claude_client.py): Claude backend that spawns `claude -p` per turn, writes a synthetic session JSONL for context seeding, bridges custom tools through a unix socket + stdio MCP sidecar, and translates stream-json events into canonical backend events
- [ipyai/mcp_server.py](ipyai/mcp_server.py): in-kernel unix socket server that exposes the live `ToolRegistry` (list_tools, call_tool) to the MCP bridge subprocess
- [ipyai/mcp_bridge.py](ipyai/mcp_bridge.py): stdio MCP server subprocess entry point (`ipyai-mcp-bridge`) that `claude -p` spawns; forwards MCP tool calls over the unix socket
- [ipyai/api_client.py](ipyai/api_client.py): shared `_FastLLMBackend` plus two backends on top of it — `ClaudeAPIBackend` (Anthropic via `fastllm`) and `CodexAPIBackend` (Codex `responses` endpoint via `fastllm.AsyncChat`); this is the explicit exception to the common canonical-event formatter path and still uses fastllm's native formatter
- [ipyai/codex_client.py](ipyai/codex_client.py): Codex app-server backend, thread/session orchestration, and app-server event translation into canonical backend events
- [ipyai/tooling.py](ipyai/tooling.py): shared custom `ToolRegistry`, schema generation, and local tool calling helpers
- [ipyai/cli.py](ipyai/cli.py): `ipyai` console entry point
- [tests/conftest.py](tests/conftest.py): minimal shell/history harness with repo-local config paths
- [tests/test_backends.py](tests/test_backends.py): shared backend test helpers
- [tests/test_backend_claude_cli.py](tests/test_backend_claude_cli.py): Claude CLI end-to-end test
- [tests/test_backend_claude_api.py](tests/test_backend_claude_api.py): Claude API end-to-end test
- [tests/test_backend_codex.py](tests/test_backend_codex.py): Codex end-to-end test
- [tests/test_backend_codex_api.py](tests/test_backend_codex_api.py): Codex API end-to-end test
- [tests/test_mcp_server.py](tests/test_mcp_server.py): unit tests for the in-kernel tool socket server
- [tests/test_mcp_bridge.py](tests/test_mcp_bridge.py): end-to-end test that spawns `ipyai-mcp-bridge` and exercises it over real MCP stdio
- [tests/test_core.py](tests/test_core.py): small local guardrail tests for transforms and backend session filtering
- [samples/capture_sdk_shapes.py](samples/capture_sdk_shapes.py): legacy Claude Agent SDK capture script (kept for reference while porting to `claude -p --output-format=stream-json`)
- [samples/outputs/](samples/outputs/): committed normalized payload captures

## CLI Flag Plumbing

`ipyai` is a `ZMQTerminalIPythonApp` subclass. `IPyAIApp` defines the ipyai-specific flags directly with traitlets (`-b`, `-r`, `-l`, `-p`, `--resume-pick`, `--keep-alive`) while still inheriting standard `jupyter_console` flags. Bare `-r` is preprocessed to `--resume-pick`; `-r N` resumes the concrete ipyai session id.

## Current Architecture

### Prompt Flow

1. Input starting with `.` is rewritten into `%ipyai`.
2. `IPyAIController.run_prompt()` reconstructs recent code/output/note context from IPython history.
3. Variable refs like `$`name`` and shell refs like `!`cmd`` are injected above the prompt.
4. The selected backend streams the turn:
   - `core.py` first builds a typed `ConversationSeed`
   - each backend then `prepare_turn(...)`s using that seed
   - Claude CLI writes a synthetic session JSONL per turn, spawns `claude -p --resume`, and starts a unix-socket MCP bridge for custom tools
   - Claude API and Codex API both rebuild flat history from the typed seed through the shared `_FastLLMBackend`
   - Codex resumes or bootstraps an app-server thread from the typed seed
5. `astream_to_stdout()` renders the response through Rich in TTY mode and stores the final transcript text locally.

Completion policy is shared in `BaseBackend.complete()`:

- empty `ConversationSeed`
- `provider_session_id=None`
- `tool_mode="off"`
- `ephemeral=True`
- `think=COMPLETION_THINK` (fixed low effort for inline completions, independent of `DEFAULT_THINK`)

Backends can still override `complete()` if a provider genuinely requires it, but the default path is now the contract.

### State Model

There are two layers of state:

- IPython shell session state, stored in IPython's own SQLite DB
- backend conversation state, stored as provider session IDs or thread IDs when the backend supports them

`ipyai` uses:

- `claude_prompts` for AI prompt history
- `sessions.remark` JSON for `cwd`, `backend`, and `provider_session_id`

If prompt history exists locally but `provider_session_id` is missing, provider bootstrap is backend-specific:

- Claude CLI always writes a fresh synthetic transcript JSONL per turn and resumes from it, then deletes the file afterward (`--no-session-persistence` keeps claude from writing anything further)
- Claude API and Codex API use the typed flat-history seed directly
- Codex starts a new thread and sends the typed notebook-XML seed once

Notebook save/load is explicit only:

- `%ipyai save <filename>`
- `%ipyai load <filename>`
- `ipyai -l <filename>`

There is no implicit `startup.ipynb` behavior.

### Tools

The custom tool story is intentionally small:

- shared custom tools across all backends:
  `pyrun`, `bash`, `start_bgterm`, `write_stdin`, `close_bgterm`, `lnhashview_file`, `exhash_file`
- built-ins: `Bash`, `Edit`, `Read`, `Skill`, `WebFetch`, `WebSearch`, `Write`

`pyrun` does not call back into `InteractiveShell.run_cell*`. It delegates to `safepyrun`, looked up in `shell.user_ns`, matching the old `ipycodex` direct-call boundary and avoiding nested IPython cell execution.

Provider-specific tool exposure now fans out from the shared `ToolRegistry`:

- Claude CLI: unix-socket MCP bridge (`ipyai-mcp-bridge`) exposes the registry to `claude -p` via `--mcp-config`; allowed tool names use the `mcp__ipy__...` prefix
- Claude API and Codex API: OpenAI-style function schemas through `fastllm`
- Codex: app-server `dynamicTools`

The `ipyai` CLI loads `safepyrun` before `ipyai`, so normal terminal sessions get `pyrun` automatically. `ipyai` seeds the other custom tools into `shell.user_ns` directly.

### Skills

Skills are Claude-native:

- built-in `Skill` tool is enabled
- `--setting-sources user,project` is passed to `claude -p`
- optional plugin directories are discovered from `.claude/plugins` up the cwd parent chain and passed as repeated `--plugin-dir` flags

## Samples

The `samples/` directory holds committed stream-shape artifacts so event-wiring spelunking does not need to be repeated. The capture scripts there still import `claude_agent_sdk` and are kept only as historical reference; the live Claude backend no longer uses the SDK. To re-capture against `claude -p`, run it directly with `--output-format=stream-json --include-partial-messages --verbose` and save the output.

## Tests

The test suite is intentionally small and integration-heavy.

Current coverage focuses on:

- one real round-trip test for each backend
- notebook save/load followed by a real follow-up prompt
- backend-specific session metadata persistence
- prompt transform behavior
- backend session filtering in resume listings

## Notes

- `ipyai` resolves config paths via XDG.
- The repo-local test harness sets `XDG_CONFIG_HOME` so config writes stay out of a normal user config tree. `CLAUDE_CONFIG_DIR` is intentionally not redirected (keychain-based OAuth depends on `~/.claude.json`).

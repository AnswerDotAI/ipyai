# ipyai

`ipyai` is a terminal IPython app with four AI backends:

- Codex API (`codex-api`, default) — hits the Codex `responses` endpoint directly using your `~/.codex/auth.json` token
- Codex (`codex`) — local Codex app-server
- Claude CLI (`claude-cli`) — drives the `claude -p` CLI, so usage counts against your Claude subscription rather than API billing
- Claude API (`claude-api`)

It is aimed at terminal IPython, not notebook frontends.

## Install

```bash
pip install -e ipyai
```

`ipyai` uses `safepyrun` for live Python state. Backend requirements:

- `codex-api`: local `codex` login so `~/.codex/auth.json` holds a valid access token
- `codex`: local Codex app-server access
- `claude-cli`: local `claude` CLI install and Claude Code login (subscription auth)
- `claude-api`: Anthropic API access

## How to Use Prompts

There are several ways to send an AI prompt from ipyai:

**Dot prefix (`.`)** — In normal IPython mode, start any line with `.` to send it as a prompt. Everything after the dot is sent to the selected backend. Continuation lines (without a dot) are included too, so you can write multi-line prompts:

```python
.explain what this dataframe transform is doing
```

```python
.draft a plan for this notebook:
focus on state management
and failure cases
```

**Prompt mode** — When prompt mode is on, *every* line you type is sent as an AI prompt by default. To run normal Python code instead, prefix the line with `;`. Shell commands (`!`) and magics (`%`) still work as usual. There are three ways to enable prompt mode:

- **`Alt-p`** — toggle prompt mode on/off at any time from the terminal
- **`-p` flag** — start ipyai in prompt mode: `ipyai -p`
- **`prompt_mode` config** — set `"prompt_mode": true` in `config.json` to always start in prompt mode

You can also toggle prompt mode during a session with `%ipyai prompt`.

## CLI

```bash
ipyai
```

Flags:

```bash
ipyai -r                 # resume last session for the selected backend
ipyai -r 43              # resume session 43
ipyai -l session.ipynb   # load a saved notebook session at startup
ipyai -b claude-cli      # select backend: codex-api | codex | claude-cli | claude-api
ipyai -p                 # start in prompt mode
```

On exit, `ipyai` prints the session ID so you can resume later.

## Usage

`ipyai` is a normal IPython session — you can run Python code exactly as you would in plain IPython. On top of that, you can send prompts to the selected AI backend as described above. `%ipyai` / `%%ipyai` magics are also available.

Useful commands:

```python
%ipyai
%ipyai model sonnet
%ipyai completion_model haiku
%ipyai think m
%ipyai code_theme monokai
%ipyai log_exact true
%ipyai prompt
%ipyai save mysession
%ipyai load mysession
%ipyai sessions
%ipyai reset
```

## Context Model

For each AI prompt, `ipyai` sends:

- recent IPython code as `<code>`
- string-literal note cells as `<note>`
- recent outputs as `<output>`
- the current request as `<user-request>`
- referenced live variables as `<variable>`
- referenced shell command output as `<shell>`

Prompt history is stored in SQLite; for compatibility, the table is currently named `claude_prompts`. Session metadata is stored in IPython's `sessions.remark` JSON, including `cwd`, `backend`, and `provider_session_id`.

## Tools

`ipyai` exposes the same custom tools across all backends:

- `python`: run Python in the live IPython namespace
- `bash`: run an allowed shell command via `safecmd`
- `start_bgterm`: start a persistent shell session
- `write_stdin`: send input to a persistent shell session and read output
- `close_bgterm`: close a persistent shell session
- `lnhashview_file`: view hash-addressed file lines for verified edits
- `exhash_file`: apply verified hash-addressed edits to a file

When using the Claude CLI backend, it also enables these built-in Claude Code tools:

- `Bash`
- `Edit`
- `Read`
- `Skill`
- `WebFetch`
- `WebSearch`
- `Write`

Custom tools are exposed to `claude -p` through an in-process unix-socket MCP server plus a small stdio bridge subprocess (`ipyai-mcp-bridge`), so the subscription-driven CLI can still call live-kernel tools like `python`.

The `ipyai` CLI loads `safepyrun` before `ipyai`, so `python` is available by default in normal terminal use.
`bash`, `start_bgterm`, `write_stdin`, `close_bgterm`, `lnhashview_file`, and `exhash_file` are seeded into the user namespace by `ipyai`.

## Skills

When using the Claude CLI backend, `ipyai` enables the built-in `Skill` tool and passes `--setting-sources user,project` so `claude -p` loads your normal user- and project-level skills.

## Notebook Save/Load

`%ipyai save <filename>` writes a notebook snapshot. It stores:

- code cells
- note cells
- AI responses as markdown cells
- prompt metadata including both `prompt` and `full_prompt`

`%ipyai load <filename>` restores that notebook into a fresh session.

`ipyai -l <filename>` does the same during startup.

Backend restore is backend-specific:

- `claude-cli`: synthesizes a fresh Claude transcript JSONL each turn and `claude -p --resume`s from it
- `claude-api`: reuses the saved local prompt history directly on each turn
- `codex-api`: reuses the saved local prompt history directly on each turn (same flat-history flow as `claude-api`)
- `codex`: starts a fresh thread and sends the loaded notebook as XML context once

## Keyboard Shortcuts

- History ghost text: matching recent input appears as a grey suggestion; accept with Right, Ctrl-F, or Ctrl-E
- `Alt-.`: AI inline completion, overriding any visible history suggestion
- `Alt-p`: toggle prompt mode
- `Alt-Up/Down`: history navigation
- `F2`: open the current input in `$EDITOR` as a temporary Python file
- `Alt-Shift-W`: paste all Python code blocks from the last response
- `Alt-Shift-1` through `Alt-Shift-9`: paste the Nth Python code block
- `Alt-Shift-Up/Down`: cycle through extracted Python blocks

## Config

Config lives under `XDG_CONFIG_HOME/ipyai/`:

- `config.json`
- `sysp.txt`
- `exact-log.jsonl`

`config.json` supports:

```json
{
  "backend": "codex-api",
  "models": {
    "claude-cli":  {"model": "sonnet",                   "completion_model": "haiku",                   "think": "m"},
    "claude-api":  {"model": "claude-sonnet-4-6",        "completion_model": "claude-haiku-4-5-20251001","think": "m"},
    "codex":       {"model": "gpt-5.4",                  "completion_model": "gpt-5.4-mini",            "think": "m"},
    "codex-api":   {"model": "gpt-5.4",                  "completion_model": "gpt-5.4-mini",            "think": "m"}
  },
  "code_theme": "monokai",
  "log_exact": false,
  "prompt_mode": false
}
```

## Development

See [DEV.md](DEV.md).

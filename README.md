# ipyai

`ipyai` is terminal IPython with an AI assistant on the same transcript. Code cells, outputs, AI replies, and images print into the terminal's scrollback. You can scroll, search, and copy with your terminal or tmux as usual. [teleprint](https://github.com/answerdotai/teleprint) keeps a status bar and input line at the bottom of the screen.

## Install

```bash
pip install -e .
```

A local [rustygate](https://github.com/answerdotai/rustygate) must be running to serve the kernels. Launch `ipyai` to create a new kernel and shut it down on exit. Use `ipyai -k PREFIX` to attach to an existing gateway kernel without initializing it. Attached kernels remain running when ipyai exits.

## Models

Identify a model with a vendor-prefixed name, such as `codex/gpt-5.6-terra` (the default), `anthropic/claude-sonnet-4-6`, or `claude_code/claude-sonnet-4-6`. The prefix selects how ipyai accesses the model. For the same model name, `claude_code/` uses your Claude Code subscription and `anthropic/` uses an API key. Switch mid-session with `%ipyai model NAME`.

## Modes

Choose where to send input with one of three modes:

| Mode | Destination | Key | Input mark |
| --- | --- | --- | --- |
| `code` (default) | IPython kernel | `alt-c` | `»»» ` |
| `prompt` | AI assistant | `alt-p` | `››› ` |
| `shell` | Persistent shell | `alt-s` | `$$$ ` |

Clicking `[mode]` in the status bar cycles through the modes. An empty input shows the keys for the other two modes. Start in prompt mode with `ipyai -p` or `"prompt_mode": true` in config.

A prefix overrides the mode for one submission. Use `.` for an AI prompt, `;` for code, or `!` for a shell command. For example, this sends a prompt from code mode:

```python
.explain what this dataframe transform is doing
```

Each AI turn receives the session so far, including executed cells and their outputs, notes, shell commands, and previous prompts and replies.

Inside a prompt, `` $`name` `` includes a live variable's value. `` !`cmd` `` includes a shell command's output. These embedded commands run through the kernel, as `!` does in code. They do not use the persistent shell.

Lines starting with `%` go to the kernel in every mode. This includes `%ipyai ...` commands.

## The shell

Shell submissions run in your own bash or zsh with your rc loaded. rustygate hosts this persistent shell beside the kernel and closes it with the app. `cd`, exported variables, aliases, functions, and virtualenv activation persist across commands. The kernel's working directory follows the shell's.

Commands run on the terminal, including full-screen programs such as `vim` and `htop`. When a command finishes, ipyai adds a cleaned block of its output to the transcript and the AI's context.

The shell supports normal job control: `cmd &`, `ctrl-Z`, `jobs`, `fg`, and `bg`. Quitting while the shell is running warns once. Pressing `ctrl-D` again closes the shell and its jobs, like closing a terminal window. Typing `exit` ends the shell. The next shell command starts a new one.

Embedded forms such as `x = !ls` stay kernel-side with IPython's usual capture semantics.

## Folding and numbering

Large blocks fold to one summary line automatically. Click a block's gutter to toggle it. The newest foldable blocks have a digit in their gutter, such as `»4»` or `≡0≡`. Zero marks the newest block. Press `alt-digit` to toggle the corresponding block from the keyboard.

Tool calls in an AI turn fold by default. Use `alt-0` to show the latest one or `ctrl-O` to toggle the most recent block. Scrolling the wheel up inside tmux enters tmux copy-mode.

## Images

Images use kitty graphics with Unicode placeholders and remain visible through tmux and scrollback. Terminals without kitty graphics show a text placeholder. PNG and JPEG outputs are supported. Copies sent to the model are resized to at most 768² pixels. The session file and display keep the original images.

## Transcript mode

`ctrl-T` opens the transcript on the alt screen for browsing history that has scrolled away. The browsing keys are:

- Arrows and PageUp/PageDown move through the transcript. Shift-up/shift-down jump between your inputs, skipping outputs.
- Enter toggles a block open or closed.
- `/` and `?` search forward and backward. `n`/`N` move to the next and previous match. Matches are highlighted. Folded blocks expand when you select a match inside them.
- `g`/`G` jump to the first and last block.
- `y` copies the current block via OSC 52.
- `esc` or `ctrl-T` leaves.

`h` hides the current exchange from the AI, or shows it again. Hidden exchanges remain visible to you in dim text. They are excluded from the AI's context starting with the next turn. Their hidden state persists across save/load and resume.

`e` edits the current exchange. You can change a cell's source, a prompt's text, or the whole reply including tool calls and results. Enter saves the edit and Esc cancels it. The edited exchange becomes part of the AI's context. Nothing re-runs.

The view initially follows new output, like `less +F`, including a turn that is still running. Navigating stops the automatic scrolling. Press `G` to follow new output again.

Press `i` to type in the shared input line, or paste into it directly. Enter submits a nonempty input and returns to the live screen.

## The %ipyai magic

```python
%ipyai                        current settings and commands
%ipyai model NAME             set the turn model
%ipyai suggest_model NAME     set the inline-suggestion model
%ipyai think LEVEL            set think effort
%ipyai code_theme NAME        set the code highlight theme ('auto' redetects)
%ipyai prompt                 toggle prompt mode
%ipyai sessions               list past ipyai sessions for this directory
%ipyai reset                  start a fresh conversation (and a new resumable session file)
%ipyai save PATH              export the session dialog as a .ipynb
%ipyai load PATH              import a dialog .ipynb into the session
```

Omit the value from a setter to see the current setting. For example, `%ipyai model` shows the current model. Changes apply to the session only. They do not update `config.json`.

## Sessions

ipyai saves each session as a dialog `.ipynb` under `./.ipyai/sessions/`. This directory excludes itself from Git. The whole file is written on every event. History navigation and ghost suggestions use only this directory's sessions. Each mode recalls its own inputs: code cells, prompts, or shell commands.

Plain `ipyai` starts a new session. Use `ipyai -r` to resume one from the current directory. A single saved session resumes without a prompt. With several sessions, a picker lets you choose by digit, press Enter for the newest, or press `n` to start a new session. Use `ipyai -r PREFIX` to select by filename prefix. `ipyai --sessions` lists the saved sessions.

Resuming repaints the transcript without re-running anything. If the kernel identified in the session file is still alive, ipyai attaches to it.

For curated starter templates, use `%ipyai load` or `ipyai -l PATH` at startup. Loading re-runs a dialog's cells to rebuild kernel state without painting the transcript.

## Keys

- `Enter` submits complete Python input. Otherwise it inserts a continuation newline. `alt-enter` always inserts a newline.
- `Tab` completes, then cycles through the completion menu. `shift-Tab` cycles back in that menu. Outside the menu, `shift-Tab` inspects the call under the cursor.
- `alt-.` asks the suggest model for an inline completion
- `shift-alt-W` pastes all Python blocks from the last reply. `shift-alt-1` through `shift-alt-9` paste the numbered block. `shift-alt-up/down` cycle through the blocks.
- `up`/`down` (or `alt-up`/`alt-down`) navigate the current mode's history
- `F2` opens the input in `$EDITOR`. Save and quit to reload it. In `vim`, `:cq` abandons the edit.
- `ctrl-O` toggles the last block open or closed. `ctrl-T` opens transcript mode.
- `alt-r` recalls your last input for editing. See [Replacing a turn](#replacing-a-turn) below.
- `ctrl-C` cancels a running AI turn, else interrupts the kernel, else clears the input
- `ctrl-D` quits (warning first if the shell is running)
- `alt-c`/`alt-p`/`alt-s` pick code/prompt/shell mode. `alt-0`..`alt-9` toggle the block with that digit.

### Replacing a turn

`alt-r` recalls your last input, whether prompt, code, or shell. Submitting the edited input replaces that turn and removes everything after it. The AI answers using the corrected history. In transcript mode, `E` on any prompt starts a replacement from that point.

Esc cancels replacement mode. Submitting a different kind of input appends normally instead. For example, changing a recalled code cell into a prompt does not replace the original cell.

## Config

Configuration files are under `XDG_CONFIG_HOME/ipyai/`. Edit `sysp.txt` to change the system prompt. All keys in `config.json` are optional:

```json
{
  "model": "codex/gpt-5.6-terra",
  "suggest_model": "codex/gpt-5.6-luna",
  "think": "m",
  "code_theme": "auto",
  "prompt_mode": false
}
```

Every key is also a CLI flag for one launch. For example, `ipyai --model anthropic/claude-sonnet-4-6 --think h` selects a model and think effort. `--think` accepts `l`, `m`, `h`, or `x`. Run `ipyai --help` for the full list of flags.

Put imports and aliases for new sessions in an optional `startup.py` in the same directory. ipyai runs it when initializing each kernel it owns, with `__file__` bound. This follows clikernel's `~/.config/clikernel/startup.py` behaviour. An error in the file stops the launch and reports the filename. Attached kernels (`-k`) do not run it.

## Development

See [DEV.md](DEV.md).

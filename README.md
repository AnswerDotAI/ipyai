# ipyai

`ipyai` is terminal IPython with an AI assistant on the same transcript. It is built on [teleprint](https://github.com/answerdotai/teleprint): code cells, outputs, AI replies, and images all print through to the terminal's own scrollback, with a status bar and input line repainted at the bottom. History stays native: scroll, search, and copy with your terminal or tmux as usual.

## Install

```bash
pip install -e .
```

ipyai is a [rustygate](https://github.com/answerdotai/rustygate) client: a running local rustygate serves the kernels, like any Jupyter client setup. A plain launch creates a fresh kernel and shuts it down on exit; `ipyai -k PREFIX` attaches to an existing gateway kernel instead, taken as found and left running on exit.

## Models

Models live in one flat namespace: a vendor-prefixed string such as `codex/gpt-5.6-terra` (the default), `anthropic/claude-sonnet-4-6`, or `claude_code/claude-sonnet-4-6`. The prefix carries the transport as well as the vendor: `claude_code/` drives your Claude Code subscription while `anthropic/` uses an API key, for the same model name. Switch mid-session with `%ipyai model NAME`.

## Modes

ipyai has three modes, one per interpreter: `code` (the IPython kernel, the default), `prompt` (the AI), and `shell` (a persistent shell). `alt-c`/`alt-p`/`alt-s` pick a mode directly, and clicking the `[mode]` segment in the status bar cycles them. The composer mark shows where you are (`»»» `, `››› `, `$$$ `), and an empty input hints the keys for the other two modes. Start in prompt mode with `ipyai -p` or `"prompt_mode": true` in config.

A prefix overrides the mode for one submission, from anywhere: `.` sends a prompt, `;` runs code, a leading `!` runs shell. So in code mode:

```python
.explain what this dataframe transform is doing
```

Each AI turn sends the session so far: executed cells with their outputs, notes, shell commands, and earlier prompts and replies. Inside a prompt, `` $`name` `` includes a live variable's value and `` !`cmd` `` includes a shell command's output (run via the kernel, like an embedded `!` in code -- not the persistent shell). `%` lines go to the kernel from every mode, so `%ipyai ...` always works.

## The shell

Shell submissions run in one persistent shell, your own bash or zsh with your rc loaded, hosted by rustygate beside the kernel and closed with the app. `cd`, exported variables, aliases, functions, and virtualenv activation persist across commands, and the kernel's working directory follows the shell's. Commands run on the real terminal, so full-screen programs such as `vim` and `htop` work. When a command finishes, a cleaned block of its output enters the transcript and the AI's context.

Normal job control works, because the shell is real: `cmd &`, `ctrl-Z`, `jobs`, `fg`, `bg`. Quitting while the shell is running warns once; pressing `ctrl-D` again quits, closing the shell and its jobs like a terminal window. Typing `exit` ends the shell, and the next shell command starts a fresh one. Embedded forms such as `x = !ls` stay kernel-side with IPython's usual capture semantics.

## Folding and numbering

Big blocks fold to one summary line automatically, and everything visible stays live: click a block's gutter to toggle it. The newest foldable blocks also wear a digit in their gutter (`»4»`, `≡0≡`, with 0 the newest), and `alt-digit` toggles that block from the keyboard. Tool calls in an AI turn fold by default, so `alt-0` shows the latest one. `ctrl-O` toggles the most recent block. Wheel-up inside tmux enters tmux copy-mode for scrollback.

## Images

Images render through kitty graphics with Unicode placeholders, so they survive tmux and scrollback; terminals without kitty graphics get a text placeholder. PNG and JPEG outputs both work. Copies sent to the model are resized down to at most 768² pixels; the session file and the display keep the original.

## Transcript mode

`ctrl-T` opens the transcript on the alt screen for browsing history that has scrolled away. Arrows and PageUp/PageDown move, and shift-up/shift-down jump between the things you typed (skipping outputs); Enter toggles a block open or closed; `/` and `?` search forward and backward (matches highlighted, folded blocks expand on landing) with `n`/`N` for next and previous; `g`/`G` jump to the first and last block; `y` copies the current block via OSC 52; `h` hides the current exchange from the AI (or shows it again) -- hidden exchanges render dim, stay visible to you, drop out of the AI's context from the next turn, and remember their state across save/load and resume; `e` edits the current exchange (a cell's source, a prompt's text, or the whole reply -- tool calls and results included) with Enter writing back and Esc cancelling: the AI's memory changes, nothing re-runs. The view opens following the tail, `less +F` style, so a running turn streams into it; any navigation unpins, `G` re-pins. Typing (after `i`) or pasting composes into the shared input line, and Enter with content submits and returns to the live screen. `esc` or `ctrl-T` leaves.

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

Setters double as getters: `%ipyai model` with no value shows the current one. Settings are session-only; `config.json` is not written.

## Sessions

Each session is one dialog `.ipynb` under `./.ipyai/sessions/` (self-gitignored), written whole on every event, so the current directory owns its sessions: history navigation and ghost suggestions draw only on this directory's session files, and each mode recalls its own past (code cells in code mode, prompts in prompt mode, shell commands in shell mode). Plain `ipyai` starts a fresh session; resuming is explicit with `-r`. Bare `ipyai -r` picks from this directory's past sessions (one resumes silently, several open a picker: digits choose, `Enter` takes the newest, `n` starts fresh), and `ipyai -r PREFIX` resumes one by filename prefix (see `ipyai --sessions` for the list). Resume repaints the transcript without re-running anything, and warm-attaches the session's kernel when it is still alive (the kernel id is stamped in the file). `%ipyai load` does the reverse for curated starter templates: it silently re-runs a dialog's cells to rebuild kernel state, painting nothing (`ipyai -l PATH` at startup).

## Keys

- `Enter` submits when the input is complete Python, else inserts a continuation newline; `alt-enter` always inserts a newline
- `Tab` completes (then cycles the menu; `shift-Tab` cycles back); `shift-Tab` inspects the call under the cursor
- `alt-.` asks the suggest model for an inline completion
- `shift-alt-W` pastes all Python blocks from the last reply; `shift-alt-1` through `shift-alt-9` paste the Nth; `shift-alt-up/down` cycle through them
- `up`/`down` (or `alt-up`/`alt-down`) navigate the current mode's history
- `F2` opens the input in `$EDITOR`; save-quit reloads it, `vim`'s `:cq` abandons the edit
- `ctrl-O` toggles the last block open or closed; `ctrl-T` opens transcript mode
- `alt-r` recalls your last input (prompt, code, or shell) for editing, and submitting it *replaces* that turn -- the old version and everything after leave the record, and the AI answers the corrected history. In the transcript view, `E` on any prompt does the same from further back. Esc disarms; submitting as a different kind (rewriting a recalled cell into a prompt, say) appends normally instead.
- `ctrl-C` cancels a running AI turn, else interrupts the kernel, else clears the input
- `ctrl-D` quits (warning first if the shell is running)
- `alt-c`/`alt-p`/`alt-s` pick code/prompt/shell mode; `alt-0`..`alt-9` toggle the block wearing that digit

## Config

Config lives under `XDG_CONFIG_HOME/ipyai/`: `config.json` and `sysp.txt` (the system prompt, editable). `config.json` keys, all optional:

```json
{
  "model": "codex/gpt-5.6-terra",
  "suggest_model": "codex/gpt-5.6-luna",
  "think": "m",
  "code_theme": "auto",
  "prompt_mode": false
}
```

Every key is also a CLI flag for one launch: `ipyai --model anthropic/claude-sonnet-4-6 --think h`. Run `ipyai --help` for the full list; `--think` accepts `l`/`m`/`h`/`x`.

## Development

See [DEV.md](DEV.md).

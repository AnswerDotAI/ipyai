# ipyai

ipyai is a Rust terminal client for [Rustygate](https://github.com/AnswerDotAI/rustygate). Run Python code and ask questions in one notebook. Rustygate runs the kernel and the AI turns and saves the results in the `.ipynb` file.

## Run

In the fastws workspace, run `cargo develop` from this directory to install the native `ipyai` binary in the active environment. Use a running Rustygate with a Python kernelspec such as [ipymini](https://github.com/AnswerDotAI/ipymini). To start a local gateway, run `rustygate` in the directory where you want to keep your notebooks.

```bash
ipyai demo.ipynb
```

The path is relative to the gateway's files root. ipyai opens that notebook or creates it. Opening a running notebook shares its variables with the other clients. On exit ipyai saves through Rustygate and leaves reused kernels running. It stops its own kernel only when idle, with no other connected client or notebook terminal visible to your account. Other users' terminals stay open.

For a fresh session, run `ipyai` without a path. Rustygate picks an unused filename under `.ipyai/`. The kernel starts in your current directory for a loopback gateway, or the gateway's files root for a remote one.

Use `--gateway URL` or set `IPYAI_GATEWAY` to change the default `http://127.0.0.1:8787`. Pass `--token-file PATH` if the gateway requires authentication. For AI prompts against the default Solve.it server, start Rustygate with `AAI_USER_KEY` in its environment. Setting it only for ipyai won't change a running gateway. Code and notes don't need a model key.

## Use the notebook

Type Python at `»»»`. Prefix an input with `.` to ask the AI about the same notebook. Use a variable reference such as `` $`n` `` in a prompt to include its current value.

```text
»»» n = 6
»»» n * 7
««« 42
»»» .What is seven times $`n`?
```

Click a mode in the status line or use `Alt-C` (code), `Alt-P` (prompt), or `Alt-S` (shell). The prefixes `;`, `.`, and `!` switch modes for one submission. In the matching mode the prefix stays literal: `.gitignore` in prompt mode asks about `.gitignore`. Start in prompt mode with `ipyai -p`. Pass `--thinking true` or `--thinking false` to select the gateway's reasoning or standard model for this session's prompt turns.

Write `:note Remember this` to save a Markdown note without running it. A Python string literal on its own, including a triple-quoted string, also makes a note in code mode. `:code "hello"` explicitly executes the string. For a multiline note, type `:note` alone, then compose in note mode; `Alt-C` returns to code.

- `Enter` submits complete code. `Alt-Enter` inserts a newline. `Alt-Up` and `Alt-Down` recall inputs of the current kind. `Ctrl-A/E` move to the start or end of the draft, `Alt-B/F` move by word, `Ctrl-W` cuts the previous word, and `Ctrl-Y` yanks. `F2` opens the idle draft in `$EDITOR`.
- `Tab` opens or cycles the completion menu; `Enter` accepts a choice. `Shift-Tab` cycles backward, or inspects the code at the cursor when the menu is closed. `Alt-.` requests an inline completion.
- `Alt-Shift-W` inserts all runnable code fences from the last reply. `Alt-Shift-1` through `Alt-Shift-9` insert one fence by number. Rustygate identifies runnable blocks, including shell and kernel magic, and skips Markdown notes.
- Click a transcript gutter to fold or unfold that block. `Ctrl-O` toggles the latest foldable block. `Alt-0` through `Alt-9` toggle the numbered gutters.
- `Ctrl-T` opens the transcript browser. Click a block or use Up and Down to select it, the wheel or Page Up/Down to scroll, and `/` or `?` to search. `y` copies the block's source. Type or press `i` to compose with the transcript still visible; `Esc` returns to browsing. Use your terminal's mouse override (usually Shift-drag) for ordinary text selection.
- In the browser, `h` toggles whether the model skips that cell. On a prompt reply, `e` edits the answer; on an input, it edits the source. `E` revises and reruns a cell, discarding later cells. `Alt-R` recalls the latest runnable input. Press `Esc` to keep the recalled text as a new draft without replacing anything.
- A spinner beside the mode shortcuts means a cell or prompt is running. Press `Ctrl-C` to interrupt it. Interrupt a prompt before its first reply and ipyai puts it back in an empty composer. Rustygate keeps the interrupted turn in the shared notebook. At an idle composer `Ctrl-C` clears the draft, and `Ctrl-D` on an empty draft exits once, including when the persistent shell is open.

Switch to the gateway's persistent shell with `Alt-S` (`$$$`), or prefix a single command with `!`. Directory changes and exported variables survive across commands. The kernel follows changes to the shell's directory. ipyai records foreground command output, excluding alternate-screen drawings. Output a background job prints after the shell prompt returns is not saved. `x = !pwd` in code mode uses IPython's own capture. In Bash and zsh, shell mode supports job control and `!!` repeats the previous command.

After a prompt, `ctx` shows the last measured model input and the model's input limit, in tokens. It stays blank when the gateway can't report both, including when the last Responses request combined multiple provider calls. Errors from ipyai and save warnings stay in this terminal's transcript, not in the notebook.

You can also pipe one submission per line. The notebook still saves all cells; stdout contains results rather than the terminal UI.

```bash
printf '%s\n' 'x = 6' 'x * 7' | ipyai demo.ipynb
```

See [DEV.md](DEV.md) for the client protocol and tests.

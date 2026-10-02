# DEV

This file describes how ipyai is built, and what is deliberately not done yet. teleprint's README covers the terminal UI underneath it. `meta/ipyai-redesign-proposal.md` gives the reasoning behind the current structure.

## Architecture

ipyai is a rustygate client, like clikernel. rustygate runs all the time and hosts the kernels (ipymini by default) and the shell's terminal. ipyai starts and stops once per session. An unreachable gateway fails at startup with the command to run, and ipyai never starts one itself.

The session Dialog, from aidialog, holds all conversation content. `Controller` in `controller.py` is the only code that changes it, and it runs at most one foreground operation, an `Op`, at a time. `View` in `view.py` shows each Message as keyed teleprint blocks. `App` in `cli.py` owns the keys, input focus, the composer and the status line, and is the `ui` the controller calls. The controller imports no terminal code, so tests can drive it without a terminal.

### Messages and kernel output

The controller creates a Message when the user submits, before any work starts. A code cell runs as one `kc.reply` request whose msg_id is `{message id}.{token}`. `KernelSession` hands every inbound kernel message to `Controller.on_jmsg`, whatever request it belongs to. `on_jmsg` finds the Message named in the parent msg_id and writes the output into it with `Message.add_output`. Output that a thread prints after its cell has finished therefore still reaches that cell. Tool runs and helper calls use msg_ids that name no Message, so `on_jmsg` ignores their output. Comm messages go to the app's `%ipyai` handler whenever they arrive.

A kernel `input_request` opens its own one-line input in the tail. The prompt and the answer join the cell's output, except a password answer, which is never stored.

### AI turns

A prompt Message's output is the formatted reply text, which `dlg2hist` sends to the model on later turns. While a turn streams, the controller sets that output from `StreamAccum` on every stream event. When the turn completes, it sets it from `chat.full()`. The stored form is compact: tool results are cut to about 2,000 characters, and thinking is left out. While a turn streams, thinking shows as a `🧠` placeholder. The model still has full fidelity inside a tool loop, because fastllm keeps the whole turn in `chat.hist`.

`ai_msgs` selects what the AI sees: every Message except those hidden with `skipped`. `dlg2hist`, the scan for `$` and `!` references and inline suggestions all use it.

A turn stopped or failed before producing anything removes its prompt, and puts the text back in the composer. A turn stopped after producing output keeps the partial reply, followed by `aidialog.dialog.INTERRUPTED`. A turn that fails after producing output keeps it too, followed by a line naming the error.

### The view

`View.items(m)` gives the display items of one Message, each keyed `f'{m.id}:{part}'`. `View.mark(m)` queues a Message. Once per event-loop pass, the view compares each queued Message's items with the blocks it already shows. Equal items stay, changed ones are replaced in place, new ones go after their predecessor, and missing ones are dropped. Live output, resume, hide, edit and retry all reach the screen this way.

Reply text is split into top-level Markdown spans with `mdhtml.blocks`, and `mdhtml2term` renders each span. A tool call shows as one block, folded to its call line. Blocks that belong to no Message are notes, which show errors and status.

### Operations and input

An `Op` is one of a cell, a turn, a shell command, or a load's reruns. The app is busy while an `Op` runs, and Enter submits nothing until it ends. Ctrl-C calls `Controller.cancel`. For a cell it interrupts the kernel. For a turn or a load it cancels the task, and interrupts the kernel as well if the kernel is running their code. Ctrl-C is handled before any other key routing. Other keys go to the first of these that exists: a pending kernel input request, the session picker, the transcript view, the composer.

### Tools

`KernelTools` in `tools.py` is the one tool adapter. `py` is host-owned with a static schema, so attached kernels offer it too. It runs as a cell outside the user's history. Other tools are called with `EvalOps.eval`, which returns each result in its own request. A failing call does not stop the calls queued behind it. Tool names and schemas are read at the start of each turn.

### Shell

Shell submissions run in one persistent bash or zsh, a rustygate terminal that ipyai creates on first use and deletes at quit. The rc prints a private sentinel at every prompt: `ESC ] 7770;<exit code>;<pwd> BEL`. `GateShell.relay` passes bytes between the real terminal and the shell until that sentinel arrives. A `Framer` owned by the shell finds sentinels across websocket frames and across relays, and keeps the bytes that follow one. Each command runs inside `Compositor.borrow`, and a pyghostty mirror captures its cleaned output for the Message. A multi-line submission is sent as one `{ }` group, so the shell prompts once, after all of it. After each command the kernel changes to the shell's directory, best effort. A `%cd` in the kernel does not change the shell's directory.

Between commands, a drain task reads background output such as `[1] Done` notices. The controller records it as a `!# background output` Message before the next operation starts, and at quit.

F2 opens the composer text in `$EDITOR`, run as a local subprocess inside a borrow. Nothing is recorded.

### Sessions

Each session is one Dialog `.ipynb` under `./.ipyai/sessions/`. The controller saves it whole and atomically to the Dialog's `path_`: after each operation, after each tool round, and after each hide, edit or truncation. It never saves per token. The file's metadata records the kernel id, model and think level. History navigation and ghost suggestions read these files, separately for each composer mode.

`ipyai -r PREFIX` resumes a session file. Bare `-r` chooses one with the picker before any kernel starts, so a session whose kernel is still alive continues on it. Otherwise the session continues on a new kernel. `Controller.resume` then appends a note telling the AI that the namespace is empty, if code has run since the last such note. Resuming shows the transcript and runs nothing.

`%ipyai load` imports a dialog without displaying it. It reruns the dialog's code cells silently, under msg_ids that name no Message, so the AI keeps the file's stored outputs. Only errors show. `%ipyai reset` starts a new Dialog on the same kernel. Switching Dialogs never shuts a kernel down, and the app closes every kernel it owns at quit.

## Deliberately not done yet

- **Remote gateways.** All the machinery accepts a URL (`IPYAI_GATEWAY`), but remote use still needs token resolution shared with clikernel's `gateways.toml`. `#ai` and media references also resolve against the local filesystem. Local-only until designed.
- **`!` references through the persistent shell: rejected.** A `` !`cmd` `` reference runs through the kernel's own `getoutput`, with solveit's semantics. Only shell-mode submissions use the persistent shell.
- **`ranked_complete`.** Completion still uses `complete_request`. The kernel's `ranked_complete` is available when wanted.
- **Compaction.** It is deferred, and llmsurgery is its designated home. It will replace `ai_msgs` as the selection of what the AI sees. The ctx meter's measured `last_req_use` is its trigger signal.
- **Forks.** Each fork will keep its own Dialog and kernel. Switching between them will use `Controller.adopt`, which never shuts a kernel down.
- **Queueing and steering.** Tab will queue an input while the app is busy, and Enter will steer text into the running turn. fastllm's `after_tool_calls` callback can set the next prompt.

## Known gaps

- **Kernel death and restart.** ipyai does not replace a kernel that dies, and has no restart command. Requests to a dead kernel fail. jupywire's `on_dead` hook reports a death. Handling it would start a new kernel, run its setup, and add the new-kernel note.

# Developing ipyai

Run `cargo develop` in this repo to build and install the binary in the fastws environment. Cargo holds the version. The Python build backend packages the executable using maturin's `bindings = "bin"`; the client doesn't import Python. Python runs in Rustygate's kernel.

## A submission

1. [`Session::open`](src/gateway.rs) posts the notebook path and `dialog: {}` to `/api/kernels`, then opens its `/channels` WebSocket. A scratch session also passes `unique: true` and a separate `cwd`. A 201 response means this client started the kernel; a 200 means it joined an existing one.
2. `Session::save_input` adds or updates a message through `/api/messages?kid=…`. To run code or a prompt, `Session::execute` sends the saved ID as `execute_request.content.cell_id` on the shell channel. Sending the source with a cell ID in metadata would attribute output, but wouldn't tell Rustygate to execute the saved prompt. Notes need only the message operation.
3. Rustygate runs the saved cell, writes its output and broadcasts the change. ipyai batches reads of changed code outputs, including late output from an older execution. A prompt can display provisional `markdown` chunks if ipyai saw the turn start. After a mid-turn reconnect, it keeps the saved reply until the matching `cell_ops` commit instead of showing a suffix as the whole answer.

[`src/view.rs`](src/view.rs) applies gateway broadcasts and tracks output by cell and execution ID. [`src/terminal.rs`](src/terminal.rs) routes input and renders that view through teleprint. Don't write notebook state from either: Rustygate owns prompt context, model selection, tools and serialization. Its [protocol notes](https://github.com/AnswerDotAI/rustygate/blob/main/DEV.md) and [API routes](https://github.com/AnswerDotAI/rustygate/blob/main/docs/routes.md) describe the shared behavior.

If part of a block has scrolled offscreen before a width change, teleprint leaves the printed rows in native scrollback. If that block then grows, only the new text appears below it, as a plain continuation. `Ctrl-T` shows the whole formatted block at the current width.

Local diagnostics have no notebook cell ID. The transcript shows them beside the cells present when the notice arrived; they survive a notebook reload in the same session. Code tracebacks come from the saved output. The context meter reads the completed prompt's `execute_reply.metadata.context`; the client never counts tokens or queries a model catalog.

On exit, ipyai closes its shell and asks Rustygate to flush the notebook through the contents API. Unparented gateway stderr carries save warnings, including the location of an external conflict copy. Show those warnings even when the final flush succeeds. For a kernel it started, ipyai checks for active work, other clients and visible path-bound terminals. If it finds none, it requests Jupyter shutdown and waits for the process to stop. It avoids `DELETE /api/kernels/{id}`, which would also remove other users’ terminals. Rustygate can list the stopped process as `dead` until the path is reopened.

## Tests

Run `cargo test` for the Rust view tests and `pytest -q` for the Python acceptance tests. The acceptance suite requires the `dev` dependencies, `tmux`, and `ed`. The tests build this checkout's binary and start a disposable Rustygate. The assistant test uses a loopback model and a real kernel tool; it checks both terminal scrollback and the saved notebook. No model credentials are needed.

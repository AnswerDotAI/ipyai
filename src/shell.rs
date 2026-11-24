use anyhow::{Context, Result, bail, ensure};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::num::NonZeroU16;
use teleprint::tty::{RawEvent, Tty};
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;
use vt100::{Callbacks, Parser, Screen, capture::{BasicFormattedCaptureRange::Full, BasicFormattedCaptureState, RowContents, basic_formatted_to_plain}};

use crate::gateway::{Session, Socket};

// The gateway substitutes the rcfile path on its own machine. Start its shell,
// not the user's local executable. OSC 7 and 133 are the usual cwd/prompt marks.
const START: &str = r#"case "$SHELL" in *zsh) exec zsh -i;; *) exec bash --rcfile "$IPYAI_RC" -i;; esac"#;
const RC: &str = r#"
if [ -n "$ZSH_VERSION" ]; then
    export ZDOTDIR="$HOME"
    [ ! -r "$HOME/.zshrc" ] || . "$HOME/.zshrc"
else
    [ ! -r "$HOME/.bashrc" ] || . "$HOME/.bashrc"
fi
__ipyai_prompt() { local ec=$?; printf '\033]7;file://localhost%s\007\033]133;D;%d\007' "$PWD" "$ec"; __ipyai_ready=1; return "$ec"; }
__ipyai_start() { printf '\033]133;C\007'; }
if [ -n "$ZSH_VERSION" ]; then precmd_functions=(__ipyai_prompt); preexec_functions+=(__ipyai_start); RPROMPT=''; unsetopt promptsubst
else
    PROMPT_COMMAND=__ipyai_prompt; shopt -u promptvars
    __ipyai_ready=0
    trap 'if [[ $__ipyai_ready == 1 && $BASH_COMMAND != __ipyai_prompt ]]; then __ipyai_ready=0; __ipyai_start; fi' DEBUG
fi
PS1='$$$ '
"#;

#[derive(Default, PartialEq)]
enum Stage { #[default] Idle, Submitted, Running }

#[derive(Default)]
struct Capture { stage: Stage, at_prompt: bool, output: String, format: BasicFormattedCaptureState, finished: Option<(String, i32)>, cwd: String }

impl Capture {
    fn begin(&mut self, screen: &mut Screen, stage: Stage) {
        let (rows, cols) = screen.size();
        *screen = Parser::new(rows, cols, 0).screen().clone();
        self.output.clear();
        self.format = Default::default();
        self.at_prompt = false;
        self.stage = stage;
    }

    fn finish(&mut self, screen: &Screen) -> String {
        if !screen.alternate_screen() { screen.write_contents_formatted_basic(&mut self.output, Full(&mut self.format)).unwrap(); }
        let text: String = basic_formatted_to_plain(&self.output).collect();
        self.output.clear();
        self.format = Default::default();
        self.stage = Stage::Idle;
        text.trim_end_matches('\n').to_owned()
    }
}

impl Callbacks for Capture {
    fn unhandled_osc(&mut self, screen: &mut Screen, params: &[&[u8]]) {
        match params {
            [b"7", rest @ ..] => {
                let url = rest.join(&b';');
                if let Some(path) = url.strip_prefix(b"file://localhost") { self.cwd = String::from_utf8_lossy(path).into_owned(); }
            }
            [b"133", b"C"] if self.stage == Stage::Submitted => self.begin(screen, Stage::Running),
            [b"133", b"D", status] => {
                self.at_prompt = true;
                if self.stage != Stage::Idle && let Ok(ec) = str::from_utf8(status).unwrap_or("").parse() { self.finished = Some((self.finish(screen), ec)); }
            }
            _ => {}
        }
    }

    // Capture scrollback as plain rows are evicted, without retaining a second
    // terminal-sized grid per page. Full-screen programs draw on the other grid.
    fn on_scroll(&mut self, row: RowContents<'_>, alternate: bool) {
        if self.stage != Stage::Idle && !alternate { row.write_formatted_basic(&mut self.output, &mut self.format).unwrap(); }
    }
}

struct Pending { id: String, unsent: Option<String> }

pub struct Shell { pub name: String, ws: Socket, vt: Parser<Capture>, pending: Option<Pending> }

fn dim(n: u16) -> NonZeroU16 { NonZeroU16::new(n).unwrap_or(NonZeroU16::MIN) }

impl Shell {
    async fn open(s: &Session, cols: u16, rows: u16) -> Result<Self> {
        let name = format!("ipyai-{}", Uuid::new_v4().simple());
        let mut env = json!({"IPYAI_RC": "{rcfile}", "ZDOTDIR": "{rcdir}"});
        for key in ["TERM", "COLORTERM"] { if let Ok(v) = std::env::var(key) { env[key] = v.into(); } }
        let body = json!({"name": name, "cwd": s.cwd, "cols": cols, "rows": rows,
                         "argv": ["sh", "-c", START], "appendenv": env, "rc": RC});
        let ws = s.open_terminal(body).await?;
        Ok(Self { name, ws, vt: Parser::new_with_callbacks(dim(rows), dim(cols), 0, Capture::default()), pending: None })
    }

    async fn send(&mut self, frame: Value) -> Result<()> { self.ws.send(Message::Text(frame.to_string().into())).await?; Ok(()) }

    async fn resize(&mut self, w: u16, h: u16) -> Result<()> { self.vt.set_size(dim(h), dim(w)); self.send(json!(["set_size", h, w])).await }

    fn stdout(&mut self, data: &str, raw_shell: bool, tty: &mut Tty) -> Result<()> {
        let mut visible = String::new();
        // End segments at OSC terminators and DEC mode switches. vt100 keeps the
        // escape state across frames. Everything else goes to the real tty unaltered.
        for s in data.split_inclusive(['\x07', '\\', 'h', 'l']) {
            let submitted = self.vt.callbacks().stage != Stage::Idle;
            let alt = self.vt.screen().alternate_screen();
            self.vt.process(s.as_bytes());
            if raw_shell || submitted {
                visible.push_str(s);
                // A remote program leaving its alt screen must not write into our
                // notebook's primary screen. Restore the shell's own primary here.
                if alt && !self.vt.screen().alternate_screen() {
                    visible.push_str("\x1b[?1049h");
                    visible.push_str(&self.vt.screen().state_formatted());
                }
            }
        }
        if !visible.is_empty() { tty.write(&visible)?; }
        Ok(())
    }

    async fn relay(&mut self, tty: &mut Tty) -> Result<End> {
        let (w, h) = tty.size()?;
        self.resize(w, h).await?;
        let directly_attached = self.pending.is_none();
        let mut keys = [0; 8192];
        let mut utf8 = Vec::new();
        if self.pending.as_ref().is_none_or(|p| p.unsent.is_none()) { tty.alternate(true)?; tty.write(&self.vt.screen().state_formatted())?; }
        loop {
            if let Some((text, ec)) = self.vt.callbacks_mut().finished.take() && let Some(p) = self.pending.take() {
                return Ok(End::Recorded(p.id, text, ec));
            }
            if self.vt.callbacks().at_prompt && let Some(cmd) = self.pending.as_mut().and_then(|p| p.unsent.take()) {
                tty.alternate(true)?;
                let (screen, capture) = self.vt.screen_and_callbacks_mut();
                // A syntax error returns straight to the prompt without a command-start mark.
                capture.begin(screen, Stage::Submitted);
                let cmd = cmd.trim_end_matches('\n');
                let script = if cmd.contains('\n') { format!("{{\n{cmd}\n}}\r") } else { format!("{cmd}\r") };
                self.send(json!(["stdin", script])).await?;
            }
            tokio::select! {
                frame = self.ws.next() => {
                    let m = frame.context("terminal WebSocket ended without a disconnect")??;
                    if m.is_close() { return Ok(End::Exited(None)); }
                    let Message::Text(data) = m else { continue };
                    let f: Value = serde_json::from_str(&data)?;
                    match f[0].as_str() {
                        Some("stdout") => self.stdout(f[1].as_str().context("invalid terminal stdout")?, directly_attached, tty)?,
                        Some("disconnect") => return Ok(End::Exited(f[2].as_i64().map(|e| e as i32))),
                        Some("gap") => self.stdout(&format!("\r\n[terminal lost {} bytes]\r\n", f[1]), directly_attached, tty)?,
                        _ => {}
                    }
                }
                input = tty.events.raw(&mut keys) => match input? {
                    RawEvent::Resize(w, h) => self.resize(w, h).await?,
                    RawEvent::Input(0) => return Ok(End::Detached),
                    RawEvent::Input(n) => {
                        let split = keys[..n].iter().position(|b| *b == 0x1d);
                        let input = &keys[..split.unwrap_or(n)];
                        if directly_attached && !input.is_empty() { self.vt.callbacks_mut().at_prompt = false; }
                        utf8.extend_from_slice(input);
                        let complete = match str::from_utf8(&utf8) {
                            Err(e) if e.error_len().is_none() => e.valid_up_to(), _ => utf8.len()
                        };
                        if complete > 0 {
                            self.send(json!(["stdin", String::from_utf8_lossy(&utf8[..complete])])).await?;
                            utf8.drain(..complete);
                        }
                        if split.is_some() { return Ok(End::Detached); }
                    }
                }
            }
        }
    }
}

enum End { Detached, Exited(Option<i32>), Recorded(String, String, i32) }

pub async fn enter(s: &mut Session, tty: &mut Tty, cmd: Option<&str>, target: Option<String>) -> Result<()> {
    let (w, h) = tty.size()?;
    if let Some(sh) = &s.shell {
        ensure!(
            cmd.is_none() || (sh.pending.is_none() && sh.vt.callbacks().at_prompt),
            "terminal has unfinished input or a foreground command; use :shell to return to it"
        );
    } else { s.shell = Some(Shell::open(s, w, h).await?); }
    if let Some(cmd) = cmd {
        let id = s.save_input("shell", cmd, target.as_deref()).await?;
        s.shell.as_mut().unwrap().pending = Some(Pending { id, unsent: Some(cmd.into()) });
    }
    let sh = s.shell.as_mut().unwrap();
    let result = sh.relay(tty).await;
    tty.write("\x1b[0m\x1b[?1l\x1b>\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?25h")?;
    tty.alternate(false)?;
    tty.write("\x1b[?2004h")?;
    let (id, text, code, cwd) = match result {
        Ok(End::Detached) => return Ok(()),
        Ok(End::Recorded(id, text, ec)) => (Some(id), text, Some(ec), Some(sh.vt.callbacks().cwd.clone())),
        ended => {
            let mut sh = s.shell.take().unwrap();
            let close = s.close_terminal(&sh.name).await;
            let ec = match ended {
                Ok(End::Exited(ec)) => ec,
                Err(e) => {
                    close?;
                    bail!(e);
                }
                _ => unreachable!(),
            };
            close?;
            let (screen, capture) = sh.vt.screen_and_callbacks_mut();
            let output = if sh.pending.as_ref().is_some_and(|p| p.unsent.is_some()) { "[shell exited before running the command]".into() } else { capture.finish(screen) };
            (sh.pending.map(|p| p.id), output, ec, None)
        }
    };
    if let Some(id) = id { s.save_shell_output(&id, &text, code).await?; }
    if let Some(cwd) = cwd && !cwd.is_empty() && cwd != s.cwd { s.set_cwd(&cwd).await?; }
    Ok(())
}

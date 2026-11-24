use std::{borrow::Cow, collections::{HashSet, VecDeque}, future::Future, io::{self, IsTerminal}, pin::Pin, time::Duration};

use anyhow::{Context, Result, anyhow, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use teleprint::{Hit, Transcript, tty::{cursor_position, Event, KeyCode as K, KeyEvent, KeyEventKind, KeyModifiers as M, MouseButton, MouseEvent, MouseEventKind, Tty}};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::{browser::{Action, Browser, Focus}, display::{Display, Frame}, editor::Draft, gateway::Session, suggestions::Suggestion, view::{Cell, View}};

const HELP: &str =
    "Alt-C code  Alt-P prompt  Alt-S shell  .prompt  ;code  !shell  Ctrl-T transcript  Ctrl-C interrupt\n:note TEXT   :run / :skip / :delete / :edit / :retry ID   :quit";
const SPINNER: &[char] = &['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'];

struct Input { header: Value, prompt: String, password: bool, draft: Draft }
struct Query { id: String, source: String, pos: usize, enter: bool }
type Fim = Pin<Box<dyn Future<Output = Result<Option<String>>> + Send>>;
#[derive(PartialEq)]
enum Edit { Source, Reply, Retry }
struct Command { text: String, target: Option<String> }
enum ShellCmd { Open(Command), Waiting(String, Command), Ready(Command) }

enum Reply { Pending, Ok, Failed(Option<(String, String)>) }
struct Run { cell: String, id: String, reply: Reply, idle: bool, unanswered: Option<String> }

impl Run {
    async fn start(s: &mut Session, cell: String, unanswered: Option<String>) -> Result<Self> {
        let id = s.execute(&cell).await.with_context(|| format!("saved cell {cell}, but could not send its execute request"))?;
        Ok(Self { cell, id, reply: Reply::Pending, idle: false, unanswered })
    }

    fn unreported_error(&self, view: &View) -> Option<String> {
        let Reply::Failed(Some((name, value))) = &self.reply else { return None };
        let shown = view.get(&self.cell).is_some_and(|c| {
            c.outputs().iter().any(|o| o["output_type"] == "error" && o["ename"] == name.as_str() && o["evalue"] == value.as_str())
                || (c.kind() == "prompt" && name == "KeyboardInterrupt" && c.reply().contains(aihist::reply::INTERRUPTED))
        });
        (!shown).then(|| format!("{name}: {value}"))
    }
}

fn route<'a>(line: &'a str, mode: &'static str) -> (&'static str, Cow<'a, str>) {
    if mode == "note" { return (mode, line.into()); }
    let s = line.trim_start();
    match s.as_bytes().first() {
        Some(b'.') if mode != "prompt" => ("prompt", s[1..].into()),
        Some(b';') if mode != "code" => ("code", line.replacen(';', "", 1).into()),
        Some(b'!') if mode != "shell" && !s[1..].trim().is_empty() => ("shell", s[1..].into()),
        Some(b'%') => ("code", line.into()),
        _ => (mode, line.into())
    }
}

fn literal_note(source: &str) -> Option<String> {
    use rustpython_parser::{Parse, ast::{Constant, Expr, Stmt, Suite}};
    if !source.contains(['\'', '"']) { return None; }
    let mut stmts = Suite::parse(source, "<input>").ok()?;
    if stmts.len() != 1 { return None; }
    let Stmt::Expr(stmt) = stmts.pop()? else { return None };
    let Expr::Constant(expr) = *stmt.value else { return None };
    match expr.value { Constant::Str(s) => Some(s), _ => None }
}

fn push_span(out: &mut String, text: &str, style: &str, continuation: &str) {
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 { out.push_str(continuation); }
        if line.is_empty() { continue; }
        out.push_str(style);
        out.extend(line.chars().filter(|&c| c == '\t' || !c.is_control()));
        if !style.is_empty() { out.push_str("\x1b[0m"); }
    }
}

struct App {
    dirty_code: HashSet<String>, view: View, draft: Draft, mode: &'static str, busy: Option<Run>, shell: Option<ShellCmd>, edit: Option<(String, Edit)>,
    query: Option<Query>, queued: VecDeque<Event>, suggestion: Suggestion, fim: Option<Fim>, input: Option<Input>, history: Option<(usize, String)>,
    browse: Option<Browser>, fence_index: Option<(String, usize)>, context: Option<String>, spin: usize
}

fn text<'a>(v: &'a Value, field: &str) -> &'a str { v[field].as_str().unwrap_or_default() }
fn message_type(m: &Value) -> &str { m["header"]["msg_type"].as_str().or(m["msg_type"].as_str()).unwrap_or_default() }

impl App {
    fn set_mode(&mut self, mode: &'static str) { self.history = None; self.edit = None; self.mode = mode; }

    fn new(cells: &[Value]) -> Self {
        Self { view: View::new(cells), dirty_code: HashSet::new(), mode: "code", draft: Draft::default(), busy: None, shell: None, edit: None, query: None,
            input: None, history: None, browse: None, context: None, queued: VecDeque::new(), suggestion: Suggestion::None, fim: None,
            fence_index: None, spin: 0 }
    }

    async fn input(&mut self, s: &mut Session, kind: &str, content: &str, replace: Option<String>) -> Result<()> {
        if kind == "shell" {
            ensure!(replace.is_none() || !content.trim().is_empty(), "empty shell command");
            self.shell = Some(ShellCmd::Open(Command { text: content.into(), target: replace }));
        } else {
            let cell = s.save_input(kind, content, replace.as_deref()).await?;
            if matches!(kind, "code" | "prompt") { self.busy = Some(Run::start(s, cell, (kind == "prompt").then(|| content.into())).await?); }
        }
        Ok(())
    }

    async fn submit(&mut self, s: &mut Session, line: &str) -> Result<bool> {
        if self.edit.is_none() && line.starts_with(':') {
            let (first, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
            match first {
                ":quit" => return Ok(true),
                ":help" => self.view.notice(HELP),
                ":shell" => self.input(s, "shell", rest, None).await?,
                ":code" | ":prompt" | ":note" => {
                    let kind = match first { ":code" => "code", ":prompt" => "prompt", _ => "note" };
                    if rest.trim().is_empty() { self.set_mode(kind); } else { self.input(s, kind, rest, None).await?; }
                }
                ":run" | ":skip" | ":delete" | ":edit" | ":retry" => {
                    let rest = rest.trim();
                    let cell = if first == ":retry" && rest.is_empty() {
                        self.view.cells.iter().rev().find(|c| matches!(c.input_mode(), "code" | "prompt" | "shell")).context("no runnable input to recall")?
                    } else { self.resolve(rest)? };
                    let id = cell.id().to_owned();
                    match first {
                        ":run" => {
                            ensure!(matches!(cell.input_mode(), "code" | "prompt"), "only code or prompts can be run by ID");
                            self.busy = Some(Run::start(s, id, (cell.kind() == "prompt").then(|| cell.input_text())).await?);
                        }
                        ":skip" => s.toggle_skip(&id).await?,
                        ":delete" => s.delete(&id).await?,
                        _ => {
                            let rerun = first == ":retry";
                            ensure!(!rerun || matches!(cell.input_mode(), "code" | "prompt" | "shell"), "only code, prompts and shell commands can be retried");
                            let (source, kind) = (cell.input_text(), cell.input_mode());
                            self.draft.set(source);
                            self.mode = kind;
                            self.edit = Some((id, if rerun { Edit::Retry } else { Edit::Source }));
                        }
                    }
                }
                _ => bail!("unknown command {first}; :help lists commands"),
            }
        } else if let Some((id, edit)) = &self.edit && *edit != Edit::Retry {
            let fields = if *edit == Edit::Reply { json!({"output": line}) }
                else if self.mode == "shell" { json!({"content": format!("!{line}")}) } else { json!({"content": line}) };
            s.update(id, fields, false).await?;
            self.edit = None;
        } else if !line.trim().is_empty() {
            let (kind, content) = route(line, self.mode);
            let replace = self.edit.as_ref().filter(|_| kind == self.mode).map(|(id, _)| id.clone());
            if kind == "code" && let Some(note) = literal_note(&content) { self.input(s, "note", &note, replace).await?; }
            else if !content.trim().is_empty() { self.input(s, kind, &content, replace).await?; }
            self.edit = None;
        }
        Ok(false)
    }

    fn resolve(&self, id: &str) -> Result<&Cell> {
        if id.is_empty() { return self.view.cells.last().ok_or_else(|| anyhow!("the notebook is empty")); }
        if let Some(cell) = self.view.get(id) { return Ok(cell); }
        let mut found = self.view.cells.iter().filter(|c| c.id().starts_with(id));
        let cell = found.next().ok_or_else(|| anyhow!("no cell matches {id}"))?;
        ensure!(found.next().is_none(), "ambiguous cell ID {id}");
        Ok(cell)
    }

    fn history(&mut self, up: bool) {
        if !up && self.history.is_none() { return; }
        let mut seen = HashSet::new();
        let choices: Vec<_> = self.view.cells.iter().rev().filter(|c| c.input_mode() == self.mode).map(Cell::input_text).filter(|s| seen.insert(s.clone())).collect();
        let (n, old) = self.history.get_or_insert_with(|| (0, self.draft.text.clone()));
        *n = (if up { *n + 1 } else { n.saturating_sub(1) }).min(choices.len());
        self.draft.set(if *n == 0 { old.clone() } else { choices[*n - 1].clone() });
    }

    fn ghost(&self) -> Option<String> {
        if let Suggestion::Ghost(s) = &self.suggestion { return Some(s.clone()); }
        if !matches!(self.suggestion, Suggestion::None) || self.draft.text.is_empty() || self.draft.pos != self.draft.text.len() || self.edit.is_some() {
            return None;
        }
        self.view.cells.iter().rev().filter(|c| c.input_mode() == self.mode)
            .find_map(|c| c.input_text().strip_prefix(&self.draft.text)?.lines().next().filter(|s| !s.is_empty()).map(str::to_owned))
    }

    fn mode_bar(&self) -> (String, Vec<(std::ops::Range<usize>, &'static str)>) {
        let spin = if self.busy.is_some() { SPINNER[self.spin] } else { ' ' };
        let mut bar = format!("\x1b[33m{spin} \x1b[0;2m");
        let mut col = 2;
        let mut hits = Vec::new();
        for (mode, label) in [("code","Alt-C code"), ("prompt","Alt-P prompt"), ("shell","Alt-S shell")] {
            let end = col + label.len() + 2;
            hits.push((col..end, mode));
            bar += &if mode == self.mode { format!("\x1b[4m[{label}]\x1b[24m") } else { format!(" {label} ") };
            bar += "  ";
            col = end + 2;
        }
        if self.busy.is_none() && let Some(ctx) = &self.context { bar += ctx; }
        bar.truncate(bar.trim_end().len());
        bar += "\x1b[0m";
        (bar, hits)
    }

    fn mode_at_col(&self, col: usize) -> Option<&'static str> {
        self.mode_bar().1.into_iter().find(|(range, _)| range.contains(&col)).map(|(_, mode)| mode)
    }

    fn tail(&self, width: u16) -> (String, Option<usize>, Option<usize>) {
        if let Some(i) = &self.input {
            let pos = i.prompt.len() + if i.password { i.draft.cursor() * '•'.len_utf8() } else { i.draft.pos };
            let answer = if i.password { "•".repeat(i.draft.text.chars().count()) } else { i.draft.text.clone() };
            return (format!("{}{answer}", i.prompt), Some(pos), None);
        }
        let mut rows = self.suggestion.rows(width);
        let hint = if let Some((id, edit)) = &self.edit {
            Some(match edit {
                Edit::Retry => format!("retry {id} (discards later cells)"),
                Edit::Reply => format!("edit reply {id}"),
                Edit::Source => format!("edit {id}"),
            })
        } else if self.shell.is_some() { Some("opening shell; Ctrl-C cancels".into()) }
        else if self.mode == "note" { Some("[note]".into()) } else { None };
        if let Some(s) = hint { rows.extend(teleprint::wrap(&format!("\x1b[33m{s}\x1b[0m"), width)); }
        let status = rows.len();
        rows.push(teleprint::wrap(&self.mode_bar().0, width).into_iter().next().unwrap_or_default());
        let mut tail = rows.join("\n");
        let (glyph, color) = crate::display::input_mark(self.mode);
        tail += &format!("\n\x1b[{color}m{glyph}{glyph}{glyph} \x1b[0m");
        let cont = format!("\n\x1b[{color}m··· \x1b[0m");
        let spans = if self.mode == "code" && !self.draft.text.starts_with(':') && route(&self.draft.text, self.mode).0 == "code" {
            mdhtml2term::highlight_spans(&self.draft.text, "python")
        } else { vec![(String::new(), self.draft.text.as_str())] };

        // Split the original source spans, not the colored or tab-expanded display.
        let (mut remaining, mut right) = (self.draft.pos, String::new());
        for (style, s) in spans {
            let n = s.len().min(remaining);
            push_span(&mut tail, &s[..n], &style, &cont);
            push_span(&mut right, &s[n..], &style, &cont);
            remaining -= n;
        }
        let pos = tail.len();
        if let Some(ghost) = self.ghost() { push_span(&mut tail, &ghost, "\x1b[2m", &cont); }
        tail += &right;
        (tail, Some(pos), Some(status))
    }

    async fn send_draft(&mut self, s: &mut Session) -> Result<bool> {
        let line = std::mem::take(&mut self.draft.text);
        self.draft.pos = 0;
        self.history = None;
        let edit_in_place = matches!(self.edit, Some((_, Edit::Reply | Edit::Source)));
        let result = self.submit(s, &line).await;
        if result.is_err() { self.draft.set(line); }
        else if edit_in_place && let Some(b) = &mut self.browse { b.focus = Focus::Browse; }
        else { self.browse = None; }
        result
    }

    async fn refresh_code(&mut self, s: &Session) -> Result<bool> {
        let dirty = std::mem::take(&mut self.dirty_code);
        let ids: Vec<_> = dirty.iter().map(String::as_str).collect();
        Ok(self.view.code_outputs(s.code_outputs(&ids).await?))
    }

    async fn message(&mut self, s: &mut Session, m: &Value) -> Result<(bool, Option<Run>)> {
        if let Some(warning) = crate::gateway::notice(m) { self.view.notice(warning); return Ok((true, None)); }
        if let Some(id) = self.view.dirty_code(m) { self.dirty_code.insert(id.into()); }
        let mut redraw = self.view.apply(m);
        let parent = text(&m["parent_header"], "msg_id");
        let c = &m["content"];
        let own = self.busy.as_mut().filter(|r| r.id == parent);
        match (text(m, "channel"), message_type(m)) {
            ("iopub", "stream") if c["name"] == "markdown" && c["text"].as_str().is_some_and(|s| !s.is_empty()) => {
                if let Some(r) = own { r.unanswered = None; }
            }
            ("shell", "execute_reply") if matches!(&self.shell, Some(ShellCmd::Waiting(id, _)) if id == parent) => {
                if let Some(ShellCmd::Waiting(_, cmd)) = self.shell.take() {
                    match crate::gateway::reply_cwd(c) {
                        Ok(cwd) => {
                            s.cwd = cwd.into();
                            self.shell = Some(ShellCmd::Ready(cmd));
                        }
                        Err(e) => self.view.notice(format!("Could not open shell in kernel's directory: {e}")),
                    }
                }
                redraw = true;
            }
            ("stdin", "input_request") => {
                self.browse = None;
                self.input = Some(Input { header: m["header"].clone(), prompt: text(c, "prompt").into(), password: c["password"].as_bool() == Some(true),
                    draft: Draft::default() });
                redraw = true;
            }
            ("iopub", "execution_lost") => {
                self.view.reload(&s.cells().await?);
                self.shell = None;
                self.query = None;
                self.input = None;
                self.view.notice(format!("{}; request not replayed", text(c, "error")));
                if let Some(r) = &mut self.busy {
                    r.reply = Reply::Failed(None);
                    r.idle = true;
                }
                redraw = true;
            }
            ("iopub", "status") if c["execution_state"] == "idle" => { if let Some(r) = own { r.idle = true; } }
            ("iopub", "status") if c["execution_state"] == "dead" || c["execution_state"] == "restarting" => {
                self.view.notice(format!("Kernel {}", text(c, "execution_state")));
                self.shell = None;
                if let Some(r) = &mut self.busy {
                    r.reply = Reply::Failed(None);
                    r.idle = true;
                }
                self.query = None;
                self.input = None;
                redraw = true;
            }
            ("shell", "execute_reply") => {
                if let Some(r) = own {
                    r.unanswered = None;
                    r.reply = if c["status"] == "ok" { Reply::Ok } else {
                        Reply::Failed(c["ename"].as_str().filter(|name| !name.is_empty()).map(|name| (name.to_owned(), text(c, "evalue").into())))
                    };
                    if self.view.get(&r.cell).is_some_and(|cell| cell.kind() == "prompt") {
                        let ctx = &m["metadata"]["context"];
                        self.context = ctx["input_tokens"].as_u64().zip(ctx["max_input_tokens"].as_u64().filter(|&n| n != 0))
                            .map(|(used, max)| format!("ctx {used}/{max} ({}%)", u128::from(used) * 100 / u128::from(max)));
                    }
                }
            }
            ("shell", kind) if self.query.as_ref().is_some_and(|q| q.id == parent) => {
                let q = self.query.take().unwrap();
                if self.draft.text != q.source || self.draft.pos != q.pos { return Ok((redraw, None)); }
                match kind {
                    "is_complete_reply" if c["status"] == "incomplete" => {
                        self.draft.insert("\n");
                        if self.queued.is_empty() { self.draft.insert(text(c, "indent")); }
                    }
                    "is_complete_reply" => { self.send_draft(s).await?; }
                    "complete_reply" => self.suggestion = Suggestion::completions(c, &mut self.draft),
                    "inspect_reply" => self.suggestion = Suggestion::inspection(c),
                    _ => (),
                }
                redraw = true;
            }
            _ => (),
        }
        let done = if self.busy.as_ref().is_some_and(|r| !matches!(r.reply, Reply::Pending) && r.idle) { self.input = None; self.busy.take() } else { None };
        if done.is_some() && !self.dirty_code.is_empty() { self.refresh_code(s).await?; }
        if let Some(message) = done.as_ref().and_then(|r| r.unreported_error(&self.view)) { self.view.notice(message); }
        Ok((redraw || done.is_some(), done))
    }
}

fn edit_key(d: &mut Draft, k: KeyEvent) {
    match (k.modifiers, k.code) {
        (M::CONTROL, K::Char('a')) | (_, K::Home) => d.pos = 0,
        (M::CONTROL, K::Char('e')) | (_, K::End) => d.pos = d.text.len(),
        (M::CONTROL, K::Char('b')) | (_, K::Left) => d.left(),
        (M::CONTROL, K::Char('f')) | (_, K::Right) => d.right(),
        (M::ALT, K::Char('b' | 'f')) => d.pos = d.word(k.code == K::Char('b')),
        (M::CONTROL, K::Char('w')) | (M::ALT, K::Backspace) => d.cut(d.word(true), d.pos),
        (M::CONTROL, K::Char('u')) => d.cut(0, d.pos),
        (M::CONTROL, K::Char('k')) => d.cut(d.pos, d.text.len()),
        (M::CONTROL, K::Char('y')) => d.yank(),
        (_, K::Backspace) => d.backspace(),
        (M::CONTROL, K::Char('d')) | (_, K::Delete) => d.delete(),
        (m, K::Char(c)) if !m.intersects(M::CONTROL | M::ALT) => d.insert(c.encode_utf8(&mut [0; 4])),
        _ => (),
    }
}

impl App {
    async fn key(&mut self, s: &mut Session, tty: &mut Tty, display: &mut Display, frame: &Frame, k: KeyEvent) -> Result<bool> {
        if k.kind == KeyEventKind::Release { return Ok(false); }
        self.fim = None;
        if !matches!(k.code, K::Up | K::Down) { self.history = None; }
        if k.modifiers == M::CONTROL && k.code == K::Char('c') {
            self.shell = None;
            self.suggestion = Suggestion::None;
            self.queued.clear();
            if self.busy.is_some() || self.input.is_some() {
                s.interrupt().await?;
                if self.input.is_none() && self.draft.text.is_empty() && self.edit.is_none()
                    && let Some(prompt) = self.busy.as_mut().and_then(|r| r.unanswered.take()) {
                    self.mode = "prompt";
                    self.draft.set(prompt);
                }
                self.input = None;
            } else {
                self.draft = Draft::default();
                self.edit = None;
                self.query = None;
            }
            return Ok(false);
        }
        if let Some(i) = &mut self.input {
            if k.code == K::Enter {
                s.input_reply(&i.header, &i.draft.text).await?;
                self.input = None;
            } else { edit_key(&mut i.draft, k); }
            return Ok(false);
        }
        if self.query.as_ref().is_some_and(|q| q.enter) { self.queued.push_back(Event::Key(k)); return Ok(false); }
        let in_composer = self.browse.as_ref().is_none_or(|b| matches!(b.focus, Focus::Compose));
        if in_composer && matches!(self.suggestion, Suggestion::Menu { .. }) {
            match k.code {
                K::Tab | K::BackTab => { self.suggestion.cycle(k.code == K::BackTab, &mut self.draft); return Ok(false); }
                K::Enter => { self.suggestion = Suggestion::None; return Ok(false); } _ => ()
            }
        }
        if in_composer && (matches!(k.code, K::Right | K::End) || (k.modifiers == M::CONTROL && k.code == K::Char('e'))) && let Some(g) = self.ghost() {
            self.draft.insert(&g);
            self.suggestion = Suggestion::None;
            return Ok(false);
        }
        self.suggestion = Suggestion::None;
        if k.modifiers == M::CONTROL && k.code == K::Char('t') {
            self.browse = if self.browse.is_some() { None } else {
                let mut b = Browser::default();
                if self.edit.is_some() { b.focus = Focus::Compose; }
                Some(b)
            };
            return Ok(false);
        }
        if let Some(b) = &mut self.browse && in_composer && k.code == K::Esc {
            b.focus = Focus::Browse;
            if matches!(self.edit.take(), Some((_, Edit::Reply | Edit::Source))) { self.draft.set(String::new()); }
            self.query = None;
            return Ok(false);
        }
        if let Some(b) = &mut self.browse && !in_composer {
            let action = b.key(frame, k);
            let to_composer = matches!(action, Action::Type);
            match action {
                Action::Stay => (),
                Action::Leave => self.browse = None,
                Action::Type => b.focus = Focus::Compose,
                Action::Fold(key) => { display.toggle(&key); }
                Action::Expand(key) => { display.expand(&key); }
                Action::Skip(id) => s.toggle_skip(&id).await?,
                Action::Retry(id) if self.busy.is_none() => {
                    self.submit(s, &format!(":retry {id}")).await?;
                    self.browse.as_mut().unwrap().focus = Focus::Compose;
                }
                Action::Edit(id, output) if self.busy.is_none() => {
                    if let Some(c) = self.view.get(&id) {
                        self.mode = c.input_mode();
                        let reply = c.kind() == "prompt" && output;
                        self.draft.set(if reply { c.reply() } else { c.input_text() });
                        self.edit = Some((id, if reply { Edit::Reply } else { Edit::Source }));
                        b.focus = Focus::Compose;
                    }
                }
                Action::Copy(source) => tty.write(&format!("\x1b]52;c;{}\x07", STANDARD.encode(source)))?,
                _ => (),
            }
            if !to_composer { return Ok(false); }
        }
        match (k.modifiers, k.code) {
            (M::CONTROL, K::Char('d')) if self.draft.text.is_empty() => return Ok(true),
            (M::CONTROL, K::Char('o')) => { display.toggle_digit(0); }
            (M::ALT, K::Char(c)) if c.is_ascii_digit() => { display.toggle_digit(c as u8 - b'0'); }
            (M::ALT, K::Char('p')) => self.set_mode("prompt"),
            (M::ALT, K::Char('c')) => self.set_mode("code"),
            (M::ALT, K::Char('s')) => self.set_mode("shell"),
            (M::ALT, K::Char('r')) if self.busy.is_none() => { self.submit(s, ":retry").await?; }
            (_, K::F(2)) if self.busy.is_none() && self.shell.is_none() => {
                self.browse = None;
                edit_external(tty, &mut self.draft, self.mode)?;
            }
            (_, K::Esc) => {
                if let Some((_, edit)) = self.edit.take() && edit != Edit::Retry { self.draft.set(String::new()); }
                self.query = None;
            }
            (M::ALT, K::Enter) => self.draft.insert("\n"),
            (_, K::Enter) if self.busy.is_none() && self.query.is_none() && self.shell.is_none() => {
                let (kind, code) = route(&self.draft.text, self.mode);
                if kind == "code" && self.edit.is_none() && !self.draft.text.starts_with(':') && !code.trim().is_empty() {
                    let id = s.is_complete(&code).await?;
                    self.query = Some(Query { id, source: self.draft.text.clone(), pos: self.draft.pos, enter: true });
                } else { return self.send_draft(s).await; }
            }
            (_, K::Tab | K::BackTab) if self.busy.is_none() => {
                let id = if k.code == K::Tab { s.complete(&self.draft.text, self.draft.cursor()).await? }
                else { s.inspect(&self.draft.text, self.draft.cursor()).await? };
                self.query = Some(Query { id, source: self.draft.text.clone(), pos: self.draft.pos, enter: false });
            }
            (M::ALT, K::Char('.')) if self.busy.is_none() => {
                let (pfx, sfx) = self.draft.text.split_at(self.draft.pos);
                let id = self.edit.as_ref().map_or("", |(id, _)| id);
                self.fim = Some(Box::pin(s.fim(id, self.mode, pfx, sfx)));
            }
            (m, K::Char(c)) if m.contains(M::ALT) && (c == 'W' || (m.contains(M::SHIFT) && c.is_ascii_digit()) || "!@#$%^&*(".contains(c)) => {
                if let Some(cell) = self.view.cells.iter().rev().find(|c| c.kind() == "prompt") {
                    let fences = s.fences(cell.id()).await?;
                    let n = c.to_digit(10).map(|n| n as usize).or_else(|| "!@#$%^&*(".find(c).map(|n| n + 1));
                    if c == 'W' {
                        self.draft.insert(&fences.join("\n"));
                    } else if let Some(code) = n.and_then(|i| i.checked_sub(1)).and_then(|i| fences.get(i)) { self.draft.insert(code); }
                }
            }
            (m, K::Up | K::Down) if m.contains(M::ALT | M::SHIFT) => {
                if let Some(cell) = self.view.cells.iter().rev().find(|c| c.kind() == "prompt") {
                    let fences = s.fences(cell.id()).await?;
                    if !fences.is_empty() {
                        let n = fences.len();
                        let old = self.fence_index.as_ref().filter(|(id, _)| id == cell.id());
                        let i = old.map_or(if k.code == K::Up { 0 } else { n - 1 }, |(_, i)| if k.code == K::Up { (i + 1) % n } else { (i + n - 1) % n });
                        self.draft.set(fences[i].clone());
                        self.fence_index = Some((cell.id().to_string(), i));
                    }
                }
            }
            (m, K::Up | K::Down) => { let up = k.code == K::Up; if m == M::ALT || !self.draft.vertical(up) { self.history(up); } else { self.history = None; } }
            _ => {
                edit_key(&mut self.draft, k);
                self.query = None;
            }
        }
        Ok(false)
    }
}

fn edit_external(tty: &mut Tty, draft: &mut Draft, mode: &str) -> Result<()> {
    let suffix = match mode { "code" => ".py", "shell" => ".sh", _ => ".md" };
    let file = tempfile::Builder::new().suffix(suffix).tempfile()?;
    std::fs::write(file.path(), &draft.text)?;
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".into());
    // A line editor also writes to its terminal. Keep those rows away from the main transcript.
    tty.alternate(true)?;
    let status = tty.cooked(|| std::process::Command::new("sh").arg("-c").arg(format!("exec {editor} \"$1\"")).arg("ipyai-editor").arg(file.path()).status());
    tty.alternate(false)?;
    let status = status?;
    if status.success() {
        let mut source = std::fs::read_to_string(file.path())?;
        if source.ends_with('\n') { source.pop(); }
        draft.set(source);
    } else { bail!("{editor} exited with {status}"); }
    Ok(())
}

impl App {
    fn mouse(&mut self, m: MouseEvent, screen: &Transcript, display: &mut Display, frame: &Frame, status: Option<usize>, width: u16) -> Result<bool> {
        match m.kind {
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => {
                let up = m.kind == MouseEventKind::ScrollUp;
                if let Some(b) = &mut self.browse { b.scroll(frame, if up { -3 } else { 3 }); }
                else if !up || self.input.is_some() { return Ok(false); }
                else if let Some(pane) = std::env::var_os("TMUX_PANE") {
                    let r = std::process::Command::new("tmux").args(["copy-mode","-eu","-t"]).arg(pane).output()?;
                    ensure!(r.status.success(), "tmux copy-mode: {}", String::from_utf8_lossy(&r.stderr).trim());
                    return Ok(false);
                } else { self.browse = Some(Browser::default()); }
            }
            MouseEventKind::Down(MouseButton::Left) if self.input.is_none() && !self.query.as_ref().is_some_and(|q| q.enter) => {
                let (_, y) = cursor_position()?;
                match screen.hit(m.row as isize - y as isize) {
                    Some(Hit::Block(key)) if width > 4 && m.column < 3 => { return Ok(display.toggle(key)); }
                    Some(Hit::Tail(row)) if Some(row) == status => {
                        if let Some(mode) = self.mode_at_col(m.column.into()) {
                            self.set_mode(mode);
                            self.query = None;
                            self.suggestion = Suggestion::None;
                            self.fim = None;
                            if let Some(b) = &mut self.browse { b.focus = Focus::Compose; }
                        }
                    }
                    Some(Hit::Tail(row)) if self.browse.is_some() => {
                        let b = self.browse.as_mut().unwrap();
                        if let Some(key) = b.click(frame,row) { display.toggle(key); }
                    }
                    _ => return Ok(false),
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}

async fn interactive(s: &mut Session, mut app: App) -> Result<()> {
    let mut tty = Tty::open()?;
    tty.mouse(true)?;
    let (mut cols, mut rows) = tty.size()?;
    let mut display = Display::default();
    let mut frame = Frame::default();
    let mut main = Transcript::new(cols, rows);
    let mut browser = None;
    let mut status_row = None;
    let mut redraw = true;
    let mut ticked = false;
    let mut ticks = tokio::time::interval(Duration::from_millis(100));
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut next_refresh = tokio::time::Instant::now();
    loop {
        match app.shell.take() {
            Some(ShellCmd::Open(cmd)) if s.shell.is_none() => { app.shell = Some(ShellCmd::Waiting(s.request_cwd().await?, cmd)); }
            Some(ShellCmd::Open(cmd) | ShellCmd::Ready(cmd)) => {
                tty.mouse(false)?;
                if browser.is_some() { tty.alternate(false)?; app.browse = None; }
                if let Err(e) = crate::shell::enter(s, &mut tty, (!cmd.text.is_empty()).then_some(cmd.text.as_str()), cmd.target).await { app.view.notice(format!("{e:#}")); }
                tty.mouse(true)?;
                (cols, rows) = tty.size()?;
                redraw = true;
            }
            state => app.shell = state,
        }
        if redraw {
            if browser.is_some() != app.browse.is_some() {
                tty.alternate(app.browse.is_some())?;
                browser = app.browse.as_ref().map(|_| Transcript::new(cols, rows));
                display.invalidate_images();
            }
            frame = display.render(&app.view, cols, rows);
            if !frame.transmissions.is_empty() { tty.write(&std::mem::take(&mut frame.transmissions))?; }
        }
        if redraw || ticked {
            ticked = false;
            let (tail, cursor, status) = app.tail(cols);
            status_row = status;
            if let Some(b) = &mut app.browse {
                let active = browser.as_mut().unwrap();
                let (page, cursor) = b.page(&frame, cols, rows, &tail, cursor);
                status_row = status.map(|r| r + b.tail_start());
                tty.write(&active.sync_at(cols, rows, &[], &page, cursor))?;
            } else {
                tty.write(&main.sync_projected_at(cols, rows, frame.projection(), &tail, cursor))?;
            }
        }
        let queued = if app.query.as_ref().is_some_and(|q| q.enter) { None } else { app.queued.pop_front() };
        let event = if let Some(event) = queued { event } else {
            tokio::select! {
                _ = ticks.tick(), if app.busy.is_some() && app.input.is_none() => {
                    app.spin = (app.spin + 1) % SPINNER.len();
                    redraw = false;
                    ticked = true;
                    continue;
                }
                _ = tokio::time::sleep_until(next_refresh), if !app.dirty_code.is_empty() => {
                    redraw = app.refresh_code(s).await?;
                    next_refresh = tokio::time::Instant::now() + Duration::from_millis(16);
                    continue;
                }
                suggestion = async { app.fim.as_mut().unwrap().await }, if app.fim.is_some() => {
                    app.fim = None;
                    match suggestion { Ok(Some(s)) => app.suggestion = Suggestion::Ghost(s), Ok(None) => (), Err(e) => app.view.notice(e.to_string()) }
                    redraw = true;
                    continue;
                }
                msg = s.next() => {
                    match msg {
                        Ok(msg) => redraw = app.message(s, &msg).await?.0,
                        Err(err) => {
                            s.reconnect().await.map_err(|e| anyhow!("{err}; could not reconnect: {e}"))?;
                            app.view.reload(&s.cells().await?);
                            app.dirty_code.clear();
                            let model = s.model().await?;
                            app.busy = app.busy.take().and_then(|mut r| {
                                model["executions"].as_array()?.iter().find(|e| e["id"] == r.id)?;
                                // The shell reply may have fallen in the gap. This execution's idle still follows the save.
                                if matches!(r.reply, Reply::Pending) { r.reply = Reply::Ok; }
                                r.unanswered = None;
                                Some(r)
                            });
                            app.query = None; app.shell = None; app.input = None; app.queued.clear();
                            app.view.notice("Reconnected. Notebook reloaded; no request was replayed.");
                            redraw = true;
                        }
                    }
                    continue;
                }
                event = tty.events.next() => { let Some(event) = event.transpose()? else { break }; event }
            }
        };
        redraw = true;
        match event {
            Event::Key(k) => match app.key(s, &mut tty, &mut display, &frame, k).await {
                Ok(true) => break, Ok(false) => (), Err(e) => app.view.notice(format!("{e:#}"))
            },
            Event::Mouse(m) => match app.mouse(m, browser.as_ref().unwrap_or(&main), &mut display, &frame, status_row, cols) {
                Ok(dirty) => redraw = dirty, Err(e) => app.view.notice(format!("{e:#}"))
            },
            Event::Paste(text) => {
                let text = text.replace("\r\n", "\n").replace('\r', "\n");
                if let Some(i) = &mut app.input { i.draft.insert(&text); }
                else if app.query.as_ref().is_some_and(|q| q.enter) { app.queued.push_back(Event::Paste(text)); }
                else if let Some(Focus::Search(term)) = app.browse.as_mut().map(|b| &mut b.focus) { term.push_str(&text); }
                else {
                    if let Some(b) = &mut app.browse { b.focus = Focus::Compose; }
                    app.draft.insert(&text);
                    app.history = None;
                    app.query = None;
                }
                app.suggestion = Suggestion::None;
                app.fim = None;
            }
            Event::Resize(w, h) => (cols, rows) = (w, h),
            _ => redraw = false,
        }
    }
    tty.alternate(false)?;
    tty.write(&main.finish())?;
    Ok(())
}

async fn batch(s: &mut Session, mut app: App) -> Result<()> {
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut failed = false;
    loop {
        tokio::select! {
            line = lines.next_line(), if app.busy.is_none() => {
                let Some(line) = line? else { break };
                if app.submit(s, &line).await? { break; }
                ensure!(app.shell.is_none(), "an interactive shell needs a TTY; use :code !command for kernel-captured shell output");
            }
            msg = s.next() => {
                let msg = msg?;
                if let Some(done) = app.message(s, &msg).await?.1 {
                    failed |= matches!(done.reply, Reply::Failed(_));
                    if let Some(cell) = app.view.get(&done.cell) {
                        let out = if cell.kind() == "prompt" { cell.reply() } else { cell.output_text() };
                        if !out.is_empty() { println!("{}", teleprint::plain(out.trim_end_matches('\n'))); }
                    }
                }
            }
            _ = tokio::signal::ctrl_c() => { if app.busy.is_some() { s.interrupt().await?; } else { break; } }
        }
        for (_, message) in app.view.notices.drain(..) { eprintln!("{message}"); }
    }
    ensure!(!failed, "a notebook execution failed");
    Ok(())
}

pub async fn run(s: &mut Session, prompt: bool) -> Result<()> {
    let mut app = App::new(&s.cells().await?);
    if prompt { app.mode = "prompt"; }
    if io::stdin().is_terminal() && io::stdout().is_terminal() { interactive(s, app).await } else { batch(s, app).await }
}

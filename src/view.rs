//! A terminal's view of a bound notebook. Only the gateway changes the notebook.
use serde_json::Value;

use aihist::output::text_str as text;

fn saved_code(cell: &Value) -> bool { cell["cell_type"] == "code" && !aihist::cells::is_prompt(cell) }

fn revision() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
pub struct Cell {
    raw: Value,
    pub revision: usize,
    // Markdown chunks are a projection of a particular execution, not notebook outputs.
    prompt_run: Option<String>,
    draft: Option<String>,
    saw_add: bool,
}

impl Cell {
    fn new(raw: Value, saw_add: bool) -> Self { Self { raw, revision: revision(), prompt_run: None, draft: None, saw_add } }

    pub fn id(&self) -> &str { self.raw["id"].as_str().unwrap_or_default() }
    pub fn kind(&self) -> &'static str { aihist::cells::kind(&self.raw) }
    pub fn source(&self) -> String { aihist::cells::content(&self.raw) }
    pub fn input_mode(&self) -> &'static str { if self.kind() == "code" && self.raw["metadata"]["ipyai_shell"] == true { "shell" } else { self.kind() } }
    pub fn input_text(&self) -> String {
        let mut src = self.source();
        if self.input_mode() == "shell" && src.starts_with('!') { src.remove(0); }
        src
    }
    pub fn skipped(&self) -> bool { aihist::cells::truthy(&self.raw["metadata"]["skipped"]) }
    pub fn streaming(&self) -> bool { self.draft.is_some() }
    pub fn outputs(&self) -> &[Value] { self.raw["outputs"].as_array().map(Vec::as_slice).unwrap_or_default() }
    pub fn reply(&self) -> String {
        self.draft.clone().unwrap_or_else(|| self.outputs().iter().find_map(|o| o["data"].get("text/markdown")).map(text).unwrap_or_default())
    }
    pub fn output_text(&self) -> String {
        let (mut out, mut last_stream) = (String::new(), false);
        for o in self.outputs() {
            let stream = o["output_type"] == "stream";
            let s = match o["output_type"].as_str() {
                Some("stream") => text(&o["text"]),
                Some("error") => o["traceback"].as_array().filter(|a| !a.is_empty()).map(|a| a.iter().map(text).collect::<Vec<_>>().join("\n"))
                    .unwrap_or_else(|| format!("{}: {}", text(&o["ename"]), text(&o["evalue"]))),
                _ => {
                    let d = &o["data"];
                    if let Some(v) = d.get("text/markdown").or_else(|| d.get("text/plain")) { text(v) }
                    else if let Some(d) = d.as_object() { d.keys().next().map(|mime| format!("[{mime}]")).unwrap_or_default() }
                    else { String::new() }
                }
            };
            if s.is_empty() { continue; }
            if !(last_stream && stream) && !out.is_empty() && !out.ends_with('\n') { out.push('\n'); }
            out.push_str(&s);
            last_stream = stream;
        }
        out
    }

    fn update(&mut self, op: &Value, parent: Option<&str>) {
        self.revision = revision();
        // Updates are whole-cell snapshots. A source-only edit (or another execution's
        // commit during a retry) cannot replace the output this terminal is showing.
        let accept = op["output_changed"] == true && !(self.draft.is_some() && matches!((self.prompt_run.as_deref(), parent), (Some(a), Some(b)) if a != b));
        let mut old = std::mem::replace(&mut self.raw, op.clone());
        let fields = self.raw.as_object_mut().unwrap();
        fields.remove("op");
        fields.remove("output_changed");
        if accept {
            self.draft = None;
            if self.kind() != "prompt" { self.prompt_run = None; } else if let Some(p) = parent { self.prompt_run = Some(p.into()); }
        } else if let Some(v) = old.as_object_mut().and_then(|v| v.remove("outputs")) { fields.insert("outputs".into(), v); }
    }

    fn prompt_event(&mut self, kind: &str, c: &Value, p: &str) -> bool {
        if self.kind() != "prompt" { return false; }
        match kind {
            "execute_input" => {
                self.prompt_run = Some(p.into());
                self.draft = Some(String::new());
            }
            "stream" if c["name"] == "markdown" => {
                if self.saw_add && self.draft.is_none() && self.prompt_run.as_deref() != Some(p) {
                    // HTTP turns have no execute_input. A snapshot can be from halfway through the stream.
                    self.prompt_run = Some(p.into());
                    self.draft = Some(String::new());
                }
                if self.prompt_run.as_deref() != Some(p) { return false; }
                let Some(draft) = &mut self.draft else { return false; };
                draft.push_str(&text(&c["text"]));
            }
            "status" if c["execution_state"] == "idle" && self.prompt_run.as_deref() == Some(p) => { return self.draft.take().is_some(); }
            _ => return false,
        }
        true
    }
}

#[derive(Debug, Default)]
pub struct View { pub cells: Vec<Cell>, pub notices: Vec<(Option<String>, String)> }

impl View {
    pub fn new(cells: &[Value]) -> Self { let mut view = Self::default(); view.reload(cells); view }
    pub fn reload(&mut self, cells: &[Value]) { self.cells = cells.iter().cloned().map(|c| Cell::new(c, false)).collect(); }
    pub fn notice(&mut self, message: impl Into<String>) { self.notices.push((self.cells.last().map(|c| c.id().into()), message.into())); }
    pub fn get(&self, id: &str) -> Option<&Cell> { self.cells.iter().find(|c| c.id() == id) }
    pub fn dirty_code(&self, msg: &Value) -> Option<&str> {
        if msg["channel"] != "iopub" || !matches!(
            msg["header"]["msg_type"].as_str()?,
            "execute_input" | "stream" | "error" | "execute_result" | "display_data" | "update_display_data" | "clear_output"
        ) { return None; }
        let cell = self.get(msg["metadata"]["cell_id"].as_str()?)?;
        saved_code(&cell.raw).then(|| cell.id())
    }

    // Only code outputs come from this read. An older response can't replace a
    // source edit, revert a type change, move a cell, or bring back a deleted one.
    pub fn code_outputs(&mut self, cells: Vec<Value>) -> bool {
        let mut changed = false;
        for mut row in cells {
            if !saved_code(&row) || !row["metadata"].is_object() || !row["outputs"].is_array()
                || !row.get("execution_count").is_some_and(|n| n.is_null() || n.is_u64()) { continue; }
            let Some(i) = row["id"].as_str().and_then(|id| self.pos(id)) else { continue; };
            let cell = &mut self.cells[i];
            if !saved_code(&cell.raw) { continue; }
            if cell.raw["outputs"] != row["outputs"] || cell.raw["execution_count"] != row["execution_count"] {
                cell.raw["outputs"] = row["outputs"].take();
                cell.raw["execution_count"] = row["execution_count"].take();
                cell.revision = revision();
                changed = true;
            }
        }
        changed
    }

    fn pos(&self, id: &str) -> Option<usize> { self.cells.iter().position(|c| c.id() == id) }
    fn place(&self, op: &Value) -> usize {
        if let Some(p) = op["after"].as_str().and_then(|s| self.pos(s)) { return p + 1; }
        op["before"].as_str().and_then(|s| self.pos(s)).unwrap_or(self.cells.len())
    }
    fn op(&mut self, op: &Value, parent: Option<&str>) -> bool {
        match op["op"].as_str() {
            Some("add") => {
                let cell = Cell::new(op["cell"].clone(), true);
                if self.pos(cell.id()).is_some() { return false; } // The initial snapshot can already contain this queued add.
                let i = self.place(op);
                self.cells.insert(i, cell);
            }
            Some("update" | "delete") => {
                let Some(i) = op["id"].as_str().and_then(|s| self.pos(s)) else { return false; };
                if op["op"] == "delete" { self.cells.remove(i); } else { self.cells[i].update(op, parent); }
            }
            Some("move") => {
                let Some(ids) = op["ids"].as_array() else { return false; };
                let moved: Vec<_> = ids.iter().filter_map(Value::as_str).filter_map(|id| self.pos(id).map(|i| self.cells.remove(i))).collect();
                let i = self.place(op);
                self.cells.splice(i..i, moved);
            }
            _ => return false,
        }
        true
    }
    pub fn apply(&mut self, msg: &Value) -> bool {
        let kind = msg["header"]["msg_type"].as_str().unwrap_or_default();
        let parent = msg["parent_header"]["msg_id"].as_str();
        let c = &msg["content"];
        if msg["channel"] == "cells" && kind == "cell_ops" {
            let mut changed = false;
            if let Some(ops) = c["ops"].as_array() { for op in ops { changed |= self.op(op, parent); } }
            return changed;
        }
        if msg["channel"] != "iopub" { return false; }
        let Some(i) = msg["metadata"]["cell_id"].as_str().and_then(|id| self.pos(id)) else { return false; };
        let cell = &mut self.cells[i];
        let changed = parent.is_some_and(|p| cell.prompt_event(kind, c, p));
        if changed { cell.revision = revision(); }
        changed
    }
}

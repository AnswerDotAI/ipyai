//! Notebook content as keyed terminal blocks. Folding is local; source and outputs remain untouched.
use crate::view::{Cell, View};
use aihist::{output::text_str, reply};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use teleprint::{kitty, plain, wrap};

pub struct Block { pub key: String, pub cell: Option<String>, pub source: String, pub output: bool, pub text: String, pub foldable: bool }
#[derive(Default)]
pub struct Frame { pub blocks: Vec<Block>, pub transmissions: String }
impl Frame { pub fn projection(&self) -> impl Iterator<Item=(&str, &str, &str)> { self.blocks.iter().map(|b| (b.key.as_str(), b.source.as_str(), b.text.as_str())) } }

#[derive(Clone, Copy, PartialEq)]
enum Fold { Never, Tall, Tool }
struct Part { key: String, cell: String, source: String, output: bool, rows: Vec<String>, glyph: char, color: &'static str, fold: Fold, dim: bool }
pub fn input_mark(mode: &str) -> (char, &'static str) {
    match mode { "shell" => ('$', "1;33"), "prompt" => ('›', "1;35"), "note" => ('·', "1;36"), _ => ('»', "1;32") }
}

impl Part {
    fn new(c: &Cell, key: &str, source: String, rows: Vec<String>, kind: &str, fold: Fold) -> Self {
        let (glyph, color) = match kind {
            "reply" => ('‹', "35"), "tool" => ('≡', "33"), "error" => ('«', "31"), "out" => ('«', "94"), "image" => ('«', "35"), _ => input_mark(kind)
        };
        Self { key: format!("{}:{key}", c.id()), cell: c.id().into(), source, output: matches!(kind, "reply" | "tool" | "error" | "out" | "image"), rows, glyph,
            color, fold, dim: c.skipped() }
    }
}

struct Picture { img: kitty::Image, id: u32, width: u16, grid: String }

pub struct Display {
    cache: HashMap<(usize, u16), Vec<Part>>, toggles: HashMap<String, bool>, folds: HashMap<String, bool>, numbered: Vec<String>,
    pictures: HashMap<String, Option<Picture>>, kitty: bool, tmux: bool
}
impl Default for Display {
    fn default() -> Self {
        Self { cache: HashMap::new(), toggles: HashMap::new(), folds: HashMap::new(), numbered: vec![], pictures: HashMap::new(), kitty: kitty::supported(),
            tmux: std::env::var_os("TMUX").is_some() }
    }
}
impl Display {
    pub fn toggle(&mut self, key: &str) -> bool {
        let Some(v) = self.folds.get_mut(key) else { return false; };
        *v = !*v;
        self.toggles.insert(key.into(), *v);
        true
    }
    pub fn toggle_digit(&mut self, digit: u8) -> bool { self.numbered.get(usize::from(digit)).cloned().is_some_and(|key| self.toggle(&key)) }
    pub fn expand(&mut self, key: &str) -> bool { self.folds.get(key) == Some(&true) && self.toggle(key) }

    // Kitty stores image definitions separately on the primary and alternate screen.
    pub fn invalidate_images(&mut self) {
        if !self.kitty || self.pictures.is_empty() { return; }
        for p in self.pictures.values_mut().flatten() { p.width = 0; }
        self.cache.clear();
    }

    fn image(&mut self, payload: String, width: u16, tx: &mut String) -> (String, Vec<String>) {
        let picture = self.pictures.entry(payload.clone()).or_insert_with(|| {
            kitty::Image::decode(&payload).map(|img| Picture { img, id: (uuid::Uuid::new_v4().as_u128() as u32 & 0xffffff).max(1), width: 0, grid: String::new()
            })
        });
        let Some(p) = picture else { return ("[invalid image]".into(), wrap("[invalid image]", width)); };
        let (w, h) = p.img.size();
        let label = format!("[image {w}x{h}px]");
        if !self.kitty { return (label.clone(), wrap(&label, width)); }
        let cols = width.clamp(1, 40);
        if p.width != cols {
            let (transmit, grid) = p.img.render(p.id, cols, self.tmux);
            tx.push_str(&transmit);
            p.grid = grid;
            p.width = cols;
        }
        (label, wrap(&p.grid, width))
    }

    fn cell(&mut self, c: &Cell, width: u16, tx: &mut String) -> Vec<Part> {
        let (mode, source) = (c.input_mode(), c.input_text());
        let rows = match mode { "code" => mdhtml2term::highlight(&source, "python", width), "note" => mdhtml2term::render(&source, width), _ => wrap(&source, width) };
        let mut out = vec![Part::new(c, if mode == "prompt" { "ask" } else { "in" }, source, rows, mode, Fold::Never)];
        if mode == "prompt" {
            let response = c.reply();
            let sections = mdhtml2term::sections(&response, width);
            let last = sections.len().saturating_sub(1);
            for (i, section) in sections.into_iter().enumerate() {
                if section.code {
                    match reply::parse(&section.source).as_slice() {
                        [reply::Block::Tool(t)] => {
                            let call = tool_call(&t.name, &t.args);
                            let result = value_text(&t.result);
                            let mut rows = wrap(&format!("\x1b[1m{call}\x1b[0m"), width);
                            if !result.is_empty() { rows.extend(wrap(&format!("\x1b[2m{result}\x1b[0m"), width)); }
                            let src = if result.is_empty() { call } else { format!("{call}\n{result}") };
                            out.push(Part::new(c, &format!("tool:{}", t.id), src, rows, "tool", Fold::Tool));
                            continue;
                        }
                        [] => continue,
                        _ => (),
                    }
                }
                let src = section.source.trim_end_matches('\n');
                let pending = c.streaming() && i == last;
                let rows = if pending { wrap(src, width) } else { section.rows };
                if rows.is_empty() { continue; }
                let mut part = Part::new(c, &format!("r{i}"), src.into(), rows, "reply", if section.code || pending { Fold::Tall } else { Fold::Never });
                part.dim |= pending;
                out.push(part);
            }
            return out;
        }
        let outputs = c.outputs();
        let explicit: HashSet<_> = outputs.iter().filter(|o| o["output_type"] == "display_data").filter_map(image_payload).collect();
        let mut i = 0;
        while i < outputs.len() {
            let o = &outputs[i];
            let key = format!("out{i}");
            i += 1;
            let typ = o["output_type"].as_str().unwrap_or_default();
            let (kind, src, rows, fold) = match typ {
                "stream" => {
                    let mut s = text_str(&o["text"]);
                    while i < outputs.len() && outputs[i]["output_type"] == "stream" && outputs[i]["name"] == o["name"] {
                        s.push_str(&text_str(&outputs[i]["text"]));
                        i += 1;
                    }
                    let s = s.trim_end_matches('\n');
                    if s.is_empty() { continue; }
                    let rows = if o["name"] == "stderr" { wrap(&format!("\x1b[31m{s}\x1b[0m"), width) } else { wrap(s, width) };
                    ("out", plain(s), rows, Fold::Tall)
                }
                "error" => {
                    let s = o["traceback"].as_array().filter(|a| !a.is_empty()).map(|a| a.iter().map(text_str).collect::<Vec<_>>().join("\n"))
                        .unwrap_or_else(|| format!("{}: {}", text_str(&o["ename"]), text_str(&o["evalue"])));
                    ("error", plain(&s), wrap(&s, width), Fold::Never)
                }
                "display_data" | "execute_result" => {
                    let data = &o["data"];
                    if let Some(img) = image_payload(o) {
                        if typ == "execute_result" && explicit.contains(&img) { continue; }
                        let (src, rows) = self.image(img, width, tx);
                        ("image", src, rows, Fold::Never)
                    } else if let Some(md) = data.get("text/markdown") {
                        let s = text_str(md);
                        let rows = mdhtml2term::render(&s, width);
                        ("out", s, rows, Fold::Tall)
                    } else if let Some(text) = data.get("text/plain") {
                        let s = text_str(text);
                        ("out", plain(&s), wrap(&s, width), Fold::Tall)
                    } else if let Some(html) = data.get("text/html") {
                        let rows = mdhtml2term::render_html(&text_str(html), width);
                        ("out", plain(&rows.join("\n")), rows, Fold::Tall)
                    } else {
                        let Some(mime) = data.as_object().and_then(|d| d.keys().next()) else { continue };
                        let s = format!("[{mime}]");
                        let rows = wrap(&s, width);
                        ("out", s, rows, Fold::Never)
                    }
                }
                _ => continue,
            };
            if !rows.is_empty() { out.push(Part::new(c, &key, src, rows, kind, fold)); }
        }
        out
    }

    pub fn render(&mut self, view: &View, cols: u16, rows: u16) -> Frame {
        let mut frame = Frame::default();
        let width = if cols > 4 { cols - 4 } else { cols.max(1) };
        let threshold = usize::from((rows / 2).min(rows.saturating_sub(5)).max(3));
        let mut old = std::mem::take(&mut self.cache);
        for c in &view.cells {
            let key = (c.revision, width);
            let pieces = old.remove(&key).unwrap_or_else(|| self.cell(c, width, &mut frame.transmissions));
            self.cache.insert(key, pieces);
        }
        let parts: Vec<_> = view.cells.iter().flat_map(|c| &self.cache[&(c.revision, width)]).collect();
        self.folds = parts.iter().filter(|p| p.rows.len() > 1).map(|p| {
            let folded = self.toggles.get(&p.key).copied().unwrap_or(p.fold == Fold::Tool || p.fold == Fold::Tall && p.rows.len() > threshold);
            (p.key.clone(), folded)
        }).collect();
        self.numbered = parts.iter().rev().filter(|p| self.folds.contains_key(&p.key)).take(10).map(|p| p.key.clone()).collect();
        for p in parts {
            let n = p.rows.len();
            let selected = self.folds.get(&p.key);
            let mut rows = if selected == Some(&true) {
                let mut notice = format!(" … (+{} lines)", n - 1);
                if notice.chars().count() >= usize::from(width) { notice = "…".into(); }
                let avail = width.saturating_sub(notice.chars().count() as u16);
                let first = if avail > 0 { wrap(&p.rows[0], avail).into_iter().next().unwrap_or_default() } else { String::new() };
                vec![format!("{first}\x1b[2m{notice}\x1b[0m")]
            } else { p.rows.clone() };
            if rows.is_empty() { rows.push(String::new()); }
            let digit = self.numbered.iter().position(|k| k == &p.key).map(|n| char::from(b'0' + n as u8)).unwrap_or(p.glyph);
            for (i, row) in rows.iter_mut().enumerate() {
                let content = if p.dim { format!("\x1b[2m{}\x1b[0m", row.replace("\x1b[0m", "\x1b[0;2m")) } else { std::mem::take(row) };
                *row = if cols <= 4 { content }
                else if i == 0 { format!("\x1b[{}m{}{digit}{} \x1b[0m{content}", p.color, p.glyph, p.glyph) }
                else { format!("\x1b[2m··· \x1b[0m{content}") };
            }
            frame.blocks.push(Block { key: p.key.clone(), cell: Some(p.cell.clone()), source: p.source.clone(), output: p.output, text: rows.join("\n"),
                foldable: selected.is_some() });
        }
        for (i, (anchor, message)) in view.notices.iter().enumerate().rev() {
            let after = anchor.as_ref().and_then(|id| frame.blocks.iter().rposition(|b| b.cell.as_ref() == Some(id))).map_or(0, |n| n + 1);
            let text = wrap(&format!("\x1b[33m{message}\x1b[0m"), cols).join("\n");
            frame.blocks.insert(after, Block { key: format!("notice:{i}"), cell: None, source: message.clone(), output: false, text, foldable: false });
        }
        frame
    }
}

fn image_payload(o: &Value) -> Option<String> {
    let data = &o["data"];
    let mut s = text_str(data.get("image/png").or_else(|| data.get("image/jpeg"))?);
    s.retain(|c| !c.is_ascii_whitespace());
    Some(s)
}

fn value_text(v: &Value) -> String { match v { Value::Null => String::new(), Value::String(s) => s.clone(), _ => v.to_string() } }

fn tool_call(name: &str, args: &Value) -> String {
    let fields = args.as_object().into_iter().flatten().map(|(k, v)| {
        let s = value_text(v).replace('\n', "\\n");
        let mut value: String = s.chars().take(40).collect();
        if value.len() < s.len() { value.push('…'); }
        format!("{k}={value}")
    }).collect::<Vec<_>>();
    format!("{name}({})", fields.join(", "))
}

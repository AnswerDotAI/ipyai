use crate::display::{Block, Frame};
use regex::{Regex, RegexBuilder};
use std::fmt::Write;
use teleprint::tty::{KeyCode as K, KeyEvent, KeyModifiers as M};

pub enum Action { Stay, Leave, Type, Fold(String), Expand(String), Skip(String), Edit(String, bool), Retry(String), Copy(String) }

#[derive(Default)]
pub enum Focus { #[default] Browse, Compose, Search(String) }

#[derive(Default)]
pub struct Browser {
    key: Option<String>, // None follows new output.
    top: usize, visible: usize, total: usize, pub focus: Focus,
    pattern: Option<Regex>, backward: bool, reveal_match: bool, feedback: String
}

fn lines(frame: &Frame) -> impl Iterator<Item=(usize, &str)> {
    frame.blocks.iter().enumerate().flat_map(|(i,b)| b.text.split('\n').map(move |line| (i,line)))
}

fn pattern(query: &str) -> Option<Regex> {
    if query.is_empty() { return None; }
    let compile = |pat: &str| RegexBuilder::new(pat).case_insensitive(!query.chars().any(char::is_uppercase)).build();
    compile(query).or_else(|_| compile(&regex::escape(query))).ok()
}

fn without_gutter(plain: &str, gutter: bool) -> &str {
    if gutter { &plain[plain.char_indices().nth(4).map_or(plain.len(), |(i,_)| i)..] } else { plain }
}

// Offsets are in printable text, so a match can cross style changes inside a line.
fn marked(line: &str, selected: bool, pattern: Option<&Regex>, gutter: bool) -> String {
    if !selected && pattern.is_none() { return line.into(); }
    let plain = teleprint::plain(line);
    let body = plain.len() - without_gutter(&plain,gutter).len();
    let marker = selected.then_some(0..if gutter { body } else { plain.len() });
    let mut spans = marker.into_iter().chain(pattern.into_iter().flat_map(|re| re.find_iter(&plain[body..]))
        .filter(|m| !m.is_empty()).map(|m| body+m.start()..body+m.end())).peekable();
    let (mut out, mut pos) = (String::new(),0);
    for (style,run) in teleprint::runs(line) {
        let mut active = None;
        for c in run.chars() {
            while spans.peek().is_some_and(|s| s.end <= pos) { spans.next(); }
            let s = if spans.peek().is_some_and(|s| s.start <= pos) { style.invert() } else { style };
            if active != Some(s) { write!(out,"\x1b[0m{s}").unwrap(); active = Some(s); }
            out.push(c);
            pos += c.len_utf8();
        }
    }
    out += "\x1b[0m";
    out
}

impl Browser {
    fn index(&self, blocks: &[Block]) -> usize {
        self.key.as_ref().and_then(|key| blocks.iter().position(|b| &b.key == key)).unwrap_or_else(|| blocks.len().saturating_sub(1))
    }

    pub fn scroll(&mut self, frame: &Frame, rows: isize) {
        self.top = self.top.saturating_add_signed(rows).min(self.total.saturating_sub(self.visible));
        if let Some((i,_)) = lines(frame).nth(self.top) { self.key = Some(frame.blocks[i].key.clone()); }
    }

    pub fn click<'a>(&mut self, frame: &'a Frame, row: usize) -> Option<&'a str> {
        if row >= self.visible { return None; }
        let (i,_) = lines(frame).nth(self.top+row)?;
        let b = &frame.blocks[i];
        self.key = Some(b.key.clone());
        b.foldable.then_some(&b.key)
    }

    fn find(&mut self, blocks: &[Block], reverse: bool) -> Action {
        let Some(pattern) = &self.pattern else { return Action::Stay };
        let start = self.index(blocks);
        let n = blocks.len();
        for delta in 1..=n {
            let i = if reverse { (start + n - delta) % n } else { (start + delta) % n };
            if pattern.is_match(&blocks[i].source) {
                self.key = Some(blocks[i].key.clone());
                self.reveal_match = true;
                return Action::Expand(blocks[i].key.clone());
            }
        }
        self.feedback = "No match".into();
        Action::Stay
    }

    pub fn key(&mut self, frame: &Frame, k: KeyEvent) -> Action {
        let blocks = &frame.blocks;
        self.feedback.clear();
        if let Focus::Search(term) = &mut self.focus {
            match k.code {
                K::Esc => self.focus = Focus::Browse, K::Backspace => { term.pop(); } K::Char(c) => term.push(c),
                K::Enter => { self.pattern = pattern(term); self.focus = Focus::Browse; return self.find(blocks, self.backward); } _ => ()
            }
            return Action::Stay;
        }
        let i = self.index(blocks);
        if k.code == K::Esc || k.code == K::Char('q') { return Action::Leave; }
        if k.code == K::Char('i') { self.focus = Focus::Compose; return Action::Stay; }
        let Some(b) = blocks.get(i) else { return Action::Type };
        if k.modifiers.intersects(M::ALT | M::CONTROL) { return Action::Type; }
        match k.code {
            K::Enter | K::Char(' ') => return if b.foldable { Action::Fold(b.key.clone()) } else { Action::Stay },
            K::Char('h') => return b.cell.clone().map_or(Action::Stay,Action::Skip),
            K::Char('e') => return b.cell.clone().map_or(Action::Stay,|id|Action::Edit(id,b.output)),
            K::Char('E' | 'r') => return b.cell.clone().map_or(Action::Stay,Action::Retry),
            K::Char('y') => { self.feedback = "Copied selection".into(); return Action::Copy(b.source.clone()); }
            K::Up | K::Down if k.modifiers == M::SHIFT => {
                let input = |&j: &usize| blocks[j].cell.is_some() && !blocks[j].output;
                let idx = if k.code == K::Up { (0..i).rev().find(input) } else { (i+1..blocks.len()).find(input) };
                if let Some(idx) = idx { self.key = Some(blocks[idx].key.clone()); }
            }
            K::Up | K::Char('k') => self.key = Some(blocks[i.saturating_sub(1)].key.clone()),
            K::Down | K::Char('j') => self.key = Some(blocks[(i+1).min(blocks.len()-1)].key.clone()),
            K::Home | K::Char('g') => { self.key = Some(blocks[0].key.clone()); self.top = 0; }
            K::End | K::Char('G') => self.key = None,
            K::PageUp | K::PageDown => self.scroll(frame, self.visible.max(1) as isize * if k.code == K::PageUp { -1 } else { 1 }),
            K::Char('/' | '?') => { self.focus = Focus::Search(String::new()); self.backward = k.code == K::Char('?'); }
            K::Char('n' | 'N') => return self.find(blocks, self.backward ^ (k.code == K::Char('N'))),
            _ => return Action::Type,
        }
        Action::Stay
    }

    pub fn page(&mut self, frame: &Frame, width: u16, height: u16, tail: &str, cursor: Option<usize>) -> (String,Option<usize>) {
        let selected = self.index(&frame.blocks);
        let page = usize::from(height).saturating_sub(1 + teleprint::tail_height(tail,width,cursor));
        let lines: Vec<_> = lines(frame).collect();
        self.total = lines.len();
        self.visible = self.total.min(page);
        let first = lines.partition_point(|&(i,_)| i < selected);
        let end = lines.partition_point(|&(i,_)| i <= selected);
        if self.key.is_none() { self.top = self.total.saturating_sub(page); }
        else if std::mem::take(&mut self.reveal_match) {
            let gutter = width > 4 && frame.blocks[selected].cell.is_some();
            let rel = lines[first..end].iter().position(|(_,l)| self.pattern.as_ref().is_some_and(|p| p.is_match(without_gutter(&teleprint::plain(l),gutter)))).unwrap_or(0);
            self.top = first+rel;
        } else if self.top >= end { self.top = first; }
        else if first >= self.top + page { self.top = end.saturating_sub(page).min(first); }
        self.top = self.top.min(self.total.saturating_sub(page));
        let mut text = String::new();
        for (row, (i,line)) in lines.into_iter().enumerate().skip(self.top).take(page) {
            text += &marked(line,row == first,self.pattern.as_ref(),width > 4 && frame.blocks[i].cell.is_some());
            text.push('\n');
        }
        let footer = if let Focus::Search(query) = &self.focus { format!("{}{query}",if self.backward { '?' } else { '/' }) }
            else if !self.feedback.is_empty() { self.feedback.clone() }
            else if matches!(self.focus, Focus::Compose) { "Esc to browse".into() }
            else if let Some(b) = frame.blocks.get(selected) {
                if let Some(id) = &b.cell { format!("[{id}] {}  e edit  h skip  / find",if b.output { "output" } else { "input" }) }
                else { "notice  y copy  / find".into() }
            } else { "i compose  / find".into() };
        text += &teleprint::wrap(&footer,width).first().cloned().unwrap_or_default();
        let search_cursor = matches!(self.focus, Focus::Search(_)).then_some(text.len());
        text.push('\n');
        let cursor = search_cursor.or(cursor.map(|i| text.len()+i));
        text += tail;
        (text,cursor)
    }

    pub fn tail_start(&self) -> usize { self.visible+1 }
}

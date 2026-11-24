use unicode_segmentation::UnicodeSegmentation;

/// The insertion point is a byte offset on a grapheme boundary; Jupyter uses code points.
#[derive(Default)]
pub struct Draft { pub text: String, pub pos: usize, cut: String }

impl Draft {
    pub fn set(&mut self, text: String) {
        self.pos = text.len();
        self.text = text;
    }

    pub fn insert(&mut self, s: &str) {
        self.text.insert_str(self.pos, s);
        self.pos += s.len();
    }

    pub fn left(&mut self) { self.pos = self.text[..self.pos].grapheme_indices(true).next_back().map_or(0, |(i, _)| i); }

    pub fn right(&mut self) { if let Some(s) = self.text[self.pos..].graphemes(true).next() { self.pos += s.len(); } }

    pub fn word(&self, back: bool) -> usize {
        fn distance<'a>(it: impl Iterator<Item = &'a str>) -> usize {
            let mut it = it.peekable();
            let mut bytes = 0;
            for word in [false, true] {
                while let Some(g) = it.next_if(|g| g.starts_with(char::is_alphanumeric) == word) { bytes += g.len(); }
            }
            bytes
        }
        if back { self.pos - distance(self.text[..self.pos].graphemes(true).rev()) }
        else { self.pos + distance(self.text[self.pos..].graphemes(true)) }
    }

    pub fn cut(&mut self, a: usize, b: usize) {
        if a < b { self.cut = self.text.drain(a..b).collect(); self.pos = a; }
    }

    pub fn yank(&mut self) { self.text.insert_str(self.pos, &self.cut); self.pos += self.cut.len(); }

    pub fn backspace(&mut self) {
        let end = self.pos;
        self.left();
        self.text.drain(self.pos..end);
    }

    pub fn delete(&mut self) { if let Some(s) = self.text[self.pos..].graphemes(true).next() { self.text.drain(self.pos..self.pos + s.len()); } }

    fn home(&mut self) { self.pos = self.text[..self.pos].rfind('\n').map_or(0, |i| i + 1); }

    pub fn vertical(&mut self, up: bool) -> bool {
        let pos = self.pos;
        self.home();
        let column = self.text[self.pos..pos].graphemes(true).count();
        if up && self.pos > 0 {
            self.pos -= 1;
            self.home();
        } else if !up && let Some(newline) = self.text[pos..].find('\n') { self.pos = pos + newline + 1; }
        else { self.pos = pos; return false; }
        self.pos += self.text[self.pos..].split('\n').next().unwrap_or("").graphemes(true).take(column).map(str::len).sum::<usize>();
        true
    }

    pub fn cursor(&self) -> usize { self.text[..self.pos].chars().count() }

    pub fn complete(&mut self, start: usize, end: usize, word: &str) {
        let byte = |n| self.text.char_indices().nth(n).map_or(self.text.len(), |(i, _)| i);
        let (a, b) = (byte(start), byte(end));
        if a <= b {
            self.text.replace_range(a..b, word);
            self.pos = a + word.len();
        }
    }
}

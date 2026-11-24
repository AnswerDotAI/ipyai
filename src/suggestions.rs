use crate::editor::Draft;
use serde_json::Value;

#[derive(Default)]
pub enum Suggestion { #[default] None, Menu { choices: Vec<String>, selected: usize, start: usize }, Tip(String), Ghost(String) }

fn signature(help: &Value) -> Option<String> {
    let sig = help.get("signatures")?.get(help["activeSignature"].as_u64().unwrap_or(0) as usize)?;
    let label = sig["label"].as_str()?;
    let name = label.find('(').unwrap_or(label.len());
    let mut out = format!("\x1b[1m{}\x1b[22;2m", &label[..name]);
    let active = sig["activeParameter"].as_u64().or_else(|| help["activeParameter"].as_u64());
    let mut from = name;
    for (i, p) in sig["parameters"].as_array().into_iter().flatten().enumerate() {
        let Some(param) = p["label"].as_str() else { break; };
        let Some(at) = label[from..].find(param).map(|n| from + n) else { break; };
        out += &label[from..at];
        if active == Some(i as u64) { out += &format!("\x1b[22;1m{param}\x1b[22;2m"); }
        else { out += param; }
        from = at + param.len();
    }
    out += &label[from..];
    let doc = &sig["documentation"];
    if let Some(s) = doc.as_str().or_else(|| doc["value"].as_str()).filter(|s| !s.is_empty()) { out.push('\n'); out += s; }
    out += "\x1b[0m";
    Some(out)
}

impl Suggestion {
    pub fn completions(content: &Value, draft: &mut Draft) -> Self {
        let (Some(matches), Some(start), Some(end)) = (content["matches"].as_array(), content["cursor_start"].as_u64(), content["cursor_end"].as_u64()) else {
            return Self::None;
        };
        let choices: Vec<String> = matches.iter().filter_map(Value::as_str).map(str::to_owned).collect();
        if choices.is_empty() || start > end || end > draft.text.chars().count() as u64 { return Self::None; }
        draft.complete(start as usize, end as usize, &choices[0]);
        if choices.len() == 1 { Self::None } else { Self::Menu { choices, selected: 0, start: start as usize } }
    }

    pub fn inspection(content: &Value) -> Self {
        let data = &content["data"];
        Self::Tip(signature(&data["application/json"]).unwrap_or_else(|| data["text/plain"].as_str().unwrap_or("No documentation at cursor").into()))
    }

    pub fn cycle(&mut self, back: bool, draft: &mut Draft) {
        let Self::Menu { choices, selected, start } = self else { return };
        let n = choices.len();
        *selected = if back { (*selected + n - 1) % n } else { (*selected + 1) % n };
        draft.complete(*start, draft.cursor(), &choices[*selected]);
    }

    pub fn rows(&self, width: u16) -> Vec<String> {
        match self {
            Self::Tip(s) => {
                let mut rows = teleprint::wrap(s, width);
                if rows.len() > 7 { rows.truncate(6); rows.push("\x1b[2m…\x1b[0m".into()); }
                rows
            }
            Self::Menu { choices, selected, .. } => {
                let lo = selected.saturating_sub(4).min(choices.len().saturating_sub(8));
                let mut label = choices.iter().enumerate().skip(lo).take(8).map(|(i, s)| {
                    format!("\x1b[{}m{s}\x1b[0m", if i == *selected { 7 } else { 2 })
                }).collect::<Vec<_>>().join("  ");
                if choices.len() > 8 { label += &format!("  \x1b[2m({} matches)\x1b[0m", choices.len()); }
                teleprint::wrap(&label, width)
            }
            _ => vec![],
        }
    }
}

#[path = "../src/browser.rs"]
#[allow(dead_code)]
mod browser;
#[path = "../src/display.rs"]
#[allow(dead_code)] // The binary also uses this module for the interactive browser.
mod display;
#[path = "../src/view.rs"]
mod view;

use serde_json::{Value, json};
use teleprint::tty::KeyCode as K;
use view::View;

fn event(id: &str, run: &str, kind: &str, content: Value) -> Value {
    json!({"channel":"iopub", "header":{"msg_type":kind}, "parent_header":{"msg_id":run}, "metadata":{"cell_id":id}, "content":content})
}
fn ops(run: Option<&str>, changes: Vec<Value>) -> Value {
    json!({"channel":"cells", "header":{"msg_type":"cell_ops"}, "parent_header":{"msg_id":run}, "content":{"ops":changes}})
}

#[test]
fn one_shared_transcript() {
    let code = json!({"id":"c", "cell_type":"code", "source":"x", "metadata":{}, "outputs":[]});
    let mut prompt = json!({"id":"p", "cell_type":"code", "source":["%%prompt\n","Explain x"], "metadata":{"solveit_ai":1},
                            "outputs":[{"output_type":"display_data", "data":{"text/markdown":"previous reply"}}]});
    let delta = |run, s| event("p", run, "stream", json!({"name":"markdown","text":s}));
    let mut joined = View::new(std::slice::from_ref(&prompt));
    let mut earlier = prompt.clone();
    earlier["outputs"] = json!([]);
    joined.apply(&ops(None, vec![json!({"op":"add", "cell":earlier})])); // Socket event already included in the snapshot.
    joined.apply(&delta("retry", "only the suffix"));
    assert_eq!(joined.cells[0].reply(), "previous reply");
    let mut v = View::new(&[code]);
    v.apply(&ops(None, vec![json!({"op":"add", "cell":prompt, "output_changed":true})]));
    v.apply(&event("p", "first", "execute_input", json!({})));
    v.apply(&delta("first", "checking"));
    prompt["op"] = "update".into();
    prompt["metadata"]["skipped"] = 1.into();
    v.apply(&ops(None, vec![prompt.clone(), json!({"op":"move", "ids":["p"], "before":"c"})]));
    let p = &v.cells[0];
    assert!(p.skipped());
    assert_eq!((p.id(), p.source(), p.reply()), ("p", "Explain x".into(), "checking".into()));

    v.apply(&event("p", "retry", "execute_input", json!({})));
    v.apply(&delta("retry", "new"));
    prompt["output_changed"] = true.into();
    prompt["outputs"] = json!([{"output_type":"display_data","data":{"text/markdown":"saved reply"}}]);
    v.apply(&ops(Some("first"), vec![prompt.clone()]));
    v.apply(&delta("first", "obsolete"));
    assert_eq!(v.cells[0].reply(), "new");
    let committed = ops(Some("retry"), vec![prompt]);
    joined.apply(&committed);
    assert_eq!(joined.cells[0].reply(), "saved reply");
    joined.apply(&event("p", "later", "execute_input", json!({})));
    joined.apply(&delta("later", "from the beginning"));
    assert_eq!(joined.cells[0].reply(), "from the beginning");
    v.apply(&committed);
    v.apply(&delta("retry", "after commit"));
    assert_eq!(v.cells[0].reply(), "saved reply");

    let mut snapshot = json!({"id":"c", "cell_type":"code", "source":"stale source", "metadata":{}, "execution_count":2,
                              "outputs":[{"output_type":"stream","name":"stdout","text":"xyz"}]});
    assert!(v.code_outputs(vec![snapshot.clone()]));
    let rev = v.get("c").unwrap().revision;
    // The subscription can deliver a chunk already present in the HTTP snapshot.
    let chunk = event("c", "previous-run", "stream", json!({"name":"stdout","text":"z"}));
    assert_eq!(v.dirty_code(&chunk), Some("c"));
    assert!(!v.apply(&chunk) && !v.code_outputs(vec![snapshot.clone()]));
    let c = v.get("c").unwrap();
    assert_eq!((c.source(), c.output_text(), c.revision), ("x".into(), "xyz".into(), rev));

    v.apply(&ops(None, vec![json!({"op":"delete","id":"c"})]));
    let deleted = snapshot.clone();
    snapshot["id"] = "p".into(); // another fetched row has since become a prompt
    assert!(!v.code_outputs(vec![deleted, snapshot]));
    assert!(v.get("c").is_none());
    assert_eq!(v.get("p").unwrap().reply(), "saved reply");
    // A saved HTTP prompt has no execute_input. This subscription saw the cell added, so it has the complete stream.
    assert!(v.apply(&delta("http", "next preview")));
    assert_eq!(v.get("p").unwrap().reply(), "next preview");
}

#[test]
fn browsing_notebook_outputs_and_tool_results() {
    let png = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAE0lEQVR4nGPkkfvPAANMcBZeDgA6EwExjmZangAAAABJRU5ErkJggg==";
    let img = |typ| json!({"output_type":typ,"data":{"image/png":png,"text/plain":"<Figure>"}});
    let md = "**Held-out** score";
    let code = json!({"id":"c","cell_type":"code","source":"fig","outputs":[img("execute_result"),img("display_data"),img("display_data"),
        {"output_type":"display_data","data":{"text/markdown":md,"text/plain":"<Markdown object>"}},
        {"output_type":"execute_result","data":{"text/html":"<p>Dataset: <b>validation</b></p>"}}]});
    let reply = "See [@r]{ref=text}.[^n]\n\n```json {.tool}\n{\"id\":\"t\",\"name\":\"py\",\"args\":{\"code\":\"6*7\"},\"result\":\"42\"}\n```\n\n::: bibliography\nHoward and Ruder {#r year=2018}\n: ULMFiT.\n:::\n\n[^n]: Sources above.\n";
    let prompt = json!({"id":"p","cell_type":"code","metadata":{"solveit_ai":1},"source":"%%prompt\nExplain","outputs":[{"output_type":"display_data","data":{"text/markdown":reply}}]});
    let mut v = View::new(std::slice::from_ref(&code));
    v.notice("Saved a conflict copy");
    v.reload(&[code, prompt]);
    let mut d = display::Display::default();
    let f = d.render(&v, 70, 20);
    let i = f.blocks.iter().position(|b| b.cell.is_none() && b.source == "Saved a conflict copy").unwrap();
    assert_eq!((f.blocks[i-1].cell.as_deref(), f.blocks[i+1].cell.as_deref()), (Some("c"), Some("p")));
    assert_eq!(f.blocks.iter().filter(|b| b.cell.as_deref() == Some("c") && b.output && b.source == "[image 4x4px]").count(), 2);
    let text = f.blocks.iter().map(|b| teleprint::plain(&b.text)).collect::<Vec<_>>().join("\n");
    assert!(text.contains("Held-out score") && !text.contains("<Markdown object>"));
    assert!(text.contains("Dataset: validation"));
    assert!(text.contains("Howard and Ruder (2018)"));
    let mut browser = browser::Browser::default();
    for c in "/Held-out".chars() { browser.key(&f, K::Char(c).into()); }
    browser.key(&f, K::Enter.into());
    assert!(matches!(browser.key(&f, K::Char('y').into()), browser::Action::Copy(s) if s == md));
    let tool = |f: &display::Frame| teleprint::plain(&f.blocks.iter().find(|b| b.key == "p:tool:t").unwrap().text);
    assert!(!tool(&f).contains("42"));
    assert!(d.toggle_digit(0));
    assert!(tool(&d.render(&v, 70, 20)).contains("42"));
}

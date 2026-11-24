use anyhow::{Context, Result, bail, ensure};
use futures_util::{SinkExt, StreamExt};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use reqwest::{Client, RequestBuilder, Response, StatusCode, Url};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async, tungstenite::{Message, client::IntoClientRequest}};
use uuid::Uuid;

pub(crate) type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Deserialize)]
struct Notebook { id: String, path: String }

#[derive(Deserialize)]
struct Added { added_ids: [String; 1] }

#[derive(Deserialize)]
struct Cells { path: String, cells: Vec<Value> }

pub struct Session {
    http: Client, headers: HeaderMap, kernel_url: String, messages_url: String, socket: Socket, sid: String, owned: bool, allow_stdin: bool,
    thinking: Option<bool>, pub path: String, pub cwd: String, pub shell: Option<crate::shell::Shell>, base: String, channels_url: String
}

pub fn reply_cwd(content: &Value) -> Result<&str> {
    content["user_expressions"]["cwd"]["data"]["application/json"]["cwd"].as_str().context("kernel did not return its working directory")
}

pub fn notice(msg: &Value) -> Option<String> {
    if msg["channel"] != "iopub" || msg["header"]["msg_type"] != "stream" || msg["content"]["name"] != "stderr"
        || msg["parent_header"]["msg_id"].as_str().is_some_and(|id| !id.is_empty()) { return None; }
    Some(teleprint::plain(msg["content"]["text"].as_str()?.trim_end()))
}

async fn checked(res: Response) -> Result<Response> {
    let status = res.status();
    if !status.is_success() {
        let body = res.json::<Value>().await.unwrap_or_default();
        bail!("Rustygate returned {status}: {}", body["message"].as_str().unwrap_or("no gateway error detail"));
    }
    Ok(res)
}

async fn request(req: RequestBuilder) -> Result<Response> { checked(req.send().await?).await }

async fn remove(req: RequestBuilder) -> Result<()> {
    let res = req.send().await?;
    if res.status() != StatusCode::NOT_FOUND { checked(res).await?; }
    Ok(())
}

fn auth_headers(file: Option<&Path>) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    if let Some(file) = file {
        let contents = std::fs::read_to_string(file).context("reading gateway token file")?;
        let token = contents.trim_end_matches(['\r', '\n']);
        ensure!(!token.is_empty(), "gateway token file is empty");
        let mut auth = HeaderValue::from_str(&format!("Bearer {token}")).context("invalid gateway token")?;
        auth.set_sensitive(true);
        headers.insert(AUTHORIZATION, auth);
    }
    Ok(headers)
}

fn buffered_envelope(frame: &[u8]) -> Result<Value> {
    // Jupyter puts the JSON envelope first, then optional raw buffers (e.g. widget state).
    let mut words = frame.chunks_exact(4).map(|b| u32::from_be_bytes(b.try_into().unwrap()) as usize);
    let count = words.next().filter(|n| *n > 0).context("missing notebook frame parts")?;
    let start = words.next().context("missing notebook frame offset")?;
    let end = if count == 1 { frame.len() } else { words.next().context("missing notebook buffer offset")? };
    ensure!(count < frame.len() / 4 && start == 4 * (count + 1), "invalid notebook frame header");
    Ok(serde_json::from_slice(frame.get(start..end).context("invalid notebook frame offsets")?)?)
}

async fn connect_socket(endpoint: &str, headers: &HeaderMap) -> Result<Socket> {
    let mut url = Url::parse(endpoint)?;
    let scheme = if url.scheme() == "https" { "wss" } else { "ws" };
    url.set_scheme(scheme).map_err(|()| anyhow::anyhow!("invalid gateway URL"))?;
    let mut req = url.as_str().into_client_request()?;
    req.headers_mut().extend(headers.clone());
    Ok(connect_async(req).await.context("opening gateway WebSocket")?.0)
}

impl Session {
    pub async fn open(gateway: &Url, path: Option<&str>, token_file: Option<&Path>, allow_stdin: bool, thinking: Option<bool>) -> Result<Self> {
        ensure!(
            matches!(gateway.scheme(), "http" | "https") && gateway.has_host() && gateway.username().is_empty() && gateway.password().is_none()
                && gateway.query().is_none() && gateway.fragment().is_none(),
            "use an HTTP(S) gateway URL without credentials, query or fragment; pass credentials with --token-file"
        );
        let headers = auth_headers(token_file)?;
        let http = Client::builder().default_headers(headers.clone()).redirect(reqwest::redirect::Policy::none()).build()?;
        let base = gateway.as_str().trim_end_matches('/').to_owned();
        let api = format!("{base}/api");
        let mut body = json!({"path": path.unwrap_or(".ipyai/session.ipynb"), "dialog": {}});
        if path.is_none() {
            body["unique"] = true.into();
            let host = gateway.host_str().unwrap_or_default().trim_matches(['[', ']']);
            let local = host == "localhost" || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
            body["cwd"] = if local { std::env::current_dir()?.to_str().context("working directory is not UTF-8")?.into() } else { ".".into() };
        }
        let res = request(http.post(format!("{api}/kernels")).json(&body)).await.context("opening notebook on Rustygate")?;
        let owned = res.status() == StatusCode::CREATED;
        let Notebook { id, path } = res.json().await?;
        let kernel_url = format!("{api}/kernels/{id}");
        let messages_url = format!("{api}/messages?kid={id}");
        let sid = Uuid::new_v4().to_string();
        let channels_url = format!("{kernel_url}/channels?session_id={sid}");
        let socket = connect_socket(&channels_url, &headers).await.with_context(|| {
            if owned { format!("could not connect to the kernel channels; new kernel {id} for {path} was left running") }
            else { format!("could not attach to kernel {id} for {path}") }
        })?;
        Ok(Self { http, headers, base, kernel_url, messages_url, sid, socket, channels_url, owned, allow_stdin, thinking, path, cwd: String::new(), shell: None
        })
    }

    async fn read_cells(&self, query: &[(&str, &str)]) -> Result<Cells> {
        Ok(request(self.http.get(format!("{}/cells", self.kernel_url)).query(query)).await?.json().await?)
    }

    pub async fn cells(&mut self) -> Result<Vec<Value>> {
        let doc = self.read_cells(&[]).await?;
        self.path = doc.path;
        Ok(doc.cells)
    }

    pub async fn code_outputs(&self, ids: &[&str]) -> Result<Vec<Value>> {
        if ids.is_empty() { return Ok(vec![]); }
        Ok(self.read_cells(&[("ids", &ids.join(",")), ("fields", "id,cell_type,metadata,outputs,execution_count")]).await?.cells)
    }

    async fn kernel_model(&self) -> Result<Option<Value>> {
        let res = self.http.get(&self.kernel_url).send().await?;
        if res.status() == StatusCode::NOT_FOUND { return Ok(None); }
        Ok(Some(checked(res).await?.json().await?))
    }

    pub async fn model(&self) -> Result<Value> { self.kernel_model().await?.context("kernel no longer exists") }

    pub async fn fences(&self, id: &str) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct Fences { msgs: Vec<Part> }
        #[derive(Deserialize)]
        struct Part { msg_type: String, content: String }
        let req = self.http.get(&self.messages_url).query(&[("id", id), ("view", "fences")]);
        let cells: Vec<Fences> = request(req).await?.json().await?;
        Ok(cells.into_iter().flat_map(|c| c.msgs).filter(|m| m.msg_type == "code").map(|m| m.content).collect())
    }

    pub async fn open_terminal(&self, body: Value) -> Result<Socket> {
        #[derive(Deserialize)]
        struct Terminal { name: String }
        let name = body["name"].as_str().context("terminal needs a new name")?;
        ensure!(body["path"].is_null(), "a shared notebook terminal cannot be owned by this session");
        let terminal: Terminal = request(self.http.post(format!("{}/api/terminals", self.base)).json(&body)).await?.json().await?;
        ensure!(name == terminal.name, "gateway returned a different terminal");
        let connection = connect_socket(&format!("{}/terminals/websocket/{name}", self.base), &self.headers).await;
        if connection.is_err() { self.close_terminal(name).await?; }
        connection
    }

    pub async fn close_terminal(&self, name: &str) -> Result<()> { remove(self.http.delete(format!("{}/api/terminals/{name}", self.base))).await }

    pub async fn save_shell_output(&self, id: &str, output: &str, exit: Option<i32>) -> Result<()> {
        let mut outputs = vec![];
        if !output.is_empty() { outputs.push(json!({"output_type": "stream", "name": "stdout", "text": output})); }
        if let Some(code) = exit.filter(|&c| c != 0) { outputs.push(json!({"output_type": "stream", "name": "stderr", "text": format!("[exit {code}]\n")})); }
        self.update(id, json!({"output": outputs}), false).await
    }

    pub async fn request_cwd(&mut self) -> Result<String> {
        let expr = "__import__('IPython.display', fromlist=['JSON']).JSON({'cwd': __import__('os').getcwd()})";
        let content = json!({"code": "", "silent": true, "store_history": false, "allow_stdin": false, "user_expressions": {"cwd": expr}});
        self.send("shell", "execute_request", content, &json!({})).await
    }

    pub async fn set_cwd(&mut self, pwd: &str) -> Result<()> {
        if self.cwd == pwd { return Ok(()); }
        let code = format!("__import__('os').chdir({})", serde_json::to_string(pwd)?);
        self.send("shell", "execute_request", json!({"code": code, "silent": true, "store_history": false, "allow_stdin": false}), &json!({})).await?;
        self.cwd = pwd.into();
        Ok(())
    }

    async fn command(&self, body: Value) -> Result<Response> { request(self.http.post(&self.messages_url).json(&body)).await }

    pub async fn save_input(&self, kind: &str, content: &str, replace: Option<&str>) -> Result<String> {
        let mut msg = if kind == "shell" { json!({"msg_type":"code", "content":format!("!{content}"), "mergemeta":{"ipyai_shell":true}}) }
            else { json!({"msg_type":kind, "content":content}) };
        if let Some(id) = replace {
            msg["output"] = "".into();
            self.update(id, msg, true).await?;
            Ok(id.into())
        } else {
            let Added { added_ids: [id] } = self.command(json!({"op":"add", "position":"end", "msgs":[msg]})).await?.json().await?;
            Ok(id)
        }
    }

    pub async fn update(&self, id: &str, fields: Value, truncate_after: bool) -> Result<()> {
        self.command(json!({"op": "update", "msgs": [{"id": id, "msg": fields}], "truncate_after": truncate_after})).await?;
        Ok(())
    }

    pub async fn toggle_skip(&self, id: &str) -> Result<()> { self.command(json!({"op": "toggle", "ids": [id], "key": "skipped"})).await?; Ok(()) }

    pub async fn delete(&self, id: &str) -> Result<()> { self.command(json!({"op": "delete", "ids": [id]})).await?; Ok(()) }

    async fn send(&mut self, channel: &str, kind: &str, content: Value, parent: &Value) -> Result<String> {
        let msg_id = Uuid::new_v4().to_string();
        let metadata = match (content.get("cell_id"), self.thinking) {
            (Some(_), Some(v)) if kind == "execute_request" => json!({"use_thinking": v}), _ => json!({})
        };
        let msg = json!({"channel": channel, "content": content, "metadata": metadata, "parent_header": parent, "header": {
            "msg_type": kind, "msg_id": msg_id, "session": self.sid, "username": "ipyai", "version": "5.3", "date": jiff::Timestamp::now().to_string()}});
        self.socket.send(Message::Text(msg.to_string().into())).await?;
        Ok(msg_id)
    }

    pub async fn execute(&mut self, id: &str) -> Result<String> {
        let content =
            json!({"cell_id": id, "silent": false, "store_history": true, "allow_stdin": self.allow_stdin, "stop_on_error": true, "user_expressions": {}});
        self.send("shell", "execute_request", content, &json!({})).await
    }

    pub async fn input_reply(&mut self, header: &Value, text: &str) -> Result<()> {
        self.send("stdin", "input_reply", json!({"value": text}), header).await?;
        Ok(())
    }

    pub async fn is_complete(&mut self, code: &str) -> Result<String> { self.send("shell", "is_complete_request", json!({"code": code}), &json!({})).await }

    pub async fn complete(&mut self, code: &str, cursor: usize) -> Result<String> {
        self.send("shell", "complete_request", json!({"code": code, "cursor_pos": cursor}), &json!({})).await
    }

    pub async fn inspect(&mut self, code: &str, cursor: usize) -> Result<String> {
        self.send("shell", "inspect_request", json!({"code": code, "cursor_pos": cursor, "detail_level": 0}), &json!({})).await
    }

    pub fn fim(&self, cell: &str, kind: &str, pfx: &str, sfx: &str) -> impl Future<Output = Result<Option<String>>> + Send + use<> {
        #[derive(Deserialize)]
        struct Suggestions { completions: Vec<Suggestion> }
        #[derive(Deserialize)]
        struct Suggestion { text: String }
        let body = json!({"cell": cell, "mtype": kind, "pfx": pfx, "sfx": sfx, "max_tokens": 7000});
        let req = self.http.post(format!("{}/fim", self.kernel_url)).json(&body);
        async move {
            let suggestions: Suggestions = request(req).await?.json().await?;
            Ok(suggestions.completions.into_iter().next().map(|c| c.text))
        }
    }

    pub async fn reconnect(&mut self) -> Result<()> { self.socket = connect_socket(&self.channels_url, &self.headers).await?; Ok(()) }

    pub async fn next(&mut self) -> Result<Value> {
        while let Some(frame) = self.socket.next().await {
            match frame? {
                Message::Text(text) => return Ok(serde_json::from_str(&text)?), Message::Binary(frame) => return buffered_envelope(&frame),
                Message::Close(_) => break, _ => ()
            }
        }
        bail!("notebook channels closed")
    }

    pub async fn interrupt(&self) -> Result<()> { request(self.http.post(format!("{}/interrupt", self.kernel_url))).await?; Ok(()) }

    async fn finish_kernel(&mut self) -> Result<()> {
        let Some(model) = self.kernel_model().await? else { return Ok(()); };
        let path = model["path"].as_str().context("kernel has no bound notebook")?;
        // This metadata-only read flushes the notebook using Rustygate's own writer.
        let mut file = Url::parse(&format!("{}/api/contents", self.base))?;
        file.path_segments_mut().expect("HTTP URL").extend(path.split('/'));
        request(self.http.get(file)).await.context("saving the notebook")?;
        if !self.owned || model["connections"] != 1 || model["execution_state"] != "idle" || !model["executions"].as_array().is_some_and(Vec::is_empty) {
            return Ok(());
        }
        let terms: Vec<Value> = request(self.http.get(format!("{}/api/terminals", self.base)).query(&[("path", path)])).await?.json().await?;
        if !terms.is_empty() { return Ok(()); }
        // Gateway DELETE also destroys other profiles' notebook terminals. Stop only the kernel process.
        self.send("control", "shutdown_request", json!({"restart": false}), &json!({})).await?;
        while self.kernel_model().await?.is_some_and(|m| m["execution_state"] != "dead") { tokio::time::sleep(Duration::from_millis(100)).await; }
        Ok(())
    }

    pub async fn close(mut self) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(6), async {
            let term = match self.shell.take() { Some(sh) => self.close_terminal(&sh.name).await, None => Ok(()) };
            let result = term.and(self.finish_kernel().await);
            // The final contents flush can itself announce a saved conflict copy.
            if self.socket.close(None).await.is_ok() {
                while let Ok(msg) = self.next().await { if let Some(warning) = notice(&msg) { eprintln!("{warning}"); } }
            }
            result
        }).await.with_context(|| format!("timed out closing {}; the kernel was not force-deleted", self.path))?
    }
}

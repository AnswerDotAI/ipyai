mod browser;
mod display;
mod editor;
mod gateway;
mod shell;
mod suggestions;
mod terminal;
mod view;

use anyhow::Result;
use clap::Parser;
use std::io::IsTerminal;

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Open this notebook on Rustygate, or create a fresh session notebook if omitted.
    notebook: Option<String>,
    /// Start in prompt mode instead of code mode.
    #[arg(short, long)]
    prompt: bool,
    /// Select the reasoning or standard model for this session's prompt turns only.
    #[arg(long, value_name = "BOOL")]
    thinking: Option<bool>,
    /// Rustygate HTTP endpoint.
    #[arg(long, env = "IPYAI_GATEWAY", default_value = "http://127.0.0.1:8787")]
    gateway: reqwest::Url,
    /// File containing the bearer token for this gateway.
    #[arg(long)]
    token_file: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let mut session = gateway::Session::open(&args.gateway, args.notebook.as_deref(), args.token_file.as_deref(), interactive, args.thinking).await?;
    let result = terminal::run(&mut session, args.prompt).await;
    result.and(session.close().await)
}

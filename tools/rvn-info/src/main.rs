mod cli;

use crate::cli::{Cli, InfoArgs};
use clap::Parser;
use log::debug;
use rvn_core_parser::{load_model, types::Value};
use rvn_globals::{GlobalOpts, get_globals, init_globals};

pub async fn run_info_cmd(cmd: InfoArgs, _globals: &GlobalOpts) -> anyhow::Result<()> {
    debug!("rvn-info::header");
    debug!("rvn-info file: {:#?}", &cmd.file);

    let gguf = load_model(&cmd.file)?;
    if cmd.header {
        println!("Header: {:?}", gguf.header);
    }
    if cmd.metadata {
        println!("Metadata:");
        for (k, v) in gguf.metadata() {
            if let Some(key) = k.strip_prefix("tokenizer") {
                debug!("[VERBOSE] tokenizer metadata: {} => {:?}", key, v);
                continue;
            }

            match v {
                Value::String(s) => println!("  {}: \"{}\"", k, s),
                Value::Uint32(n) => println!("  {}: {}", k, n),
                Value::Float32(f) => println!("  {}: {:.6}", k, f),
                _ => println!("  {}: {:?}", k, v),
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let _ = env_logger::try_init();

    init_globals(cli.g);
    run_info_cmd(cli.args, get_globals()).await
}

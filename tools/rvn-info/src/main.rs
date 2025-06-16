mod cli;
mod info;

use crate::cli::{Cli, InfoArgs};
use clap::Parser;
use log::debug;
use rvn_core_parser::{InfoDump, load_model, types::Value};
use rvn_globals::{GlobalOpts, OutputFormat, get_globals, init_globals};

pub async fn run_info_cmd(cmd: InfoArgs, globals: &GlobalOpts) -> anyhow::Result<()> {
    debug!("rvn-info::header");
    debug!("rvn-info file: {:#?}", &cmd.file);

    let gguf = load_model(&cmd.file)?;

    // pretty
    if globals.format == OutputFormat::Pretty {
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
        if cmd.tensors {
            for (name, t) in gguf.iter() {
                println!("{name}: shape={:?}, dtype={:?}", t.shape, t.kind);
            }
        }

        return Ok(()); //early return, logic is clean
    }

    // json
    let dump = InfoDump {
        header: cmd.header.then_some(&gguf.header),
        //metadata: cmd.metadata.then(|| gguf.metadata()),
        metadata: cmd.metadata.then(|| {
            gguf.metadata()
                .iter()
                .filter(|(k, _)| !k.starts_with("tokenizer"))
                .map(|(k, v)| (k.as_str(), v))
                .collect()
        }),
        tensors: cmd
            .tensors
            .then(|| gguf.iter().map(|(n, t)| (n.as_str(), t)).collect()),
    };

    match globals.format {
        OutputFormat::Json => serde_json::to_writer_pretty(std::io::stdout(), &dump)?,
        OutputFormat::Yaml => serde_yaml::to_writer(std::io::stdout(), &dump)?,
        OutputFormat::Pretty => unreachable!(), // handled above
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

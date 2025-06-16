mod cli;

use crate::cli::{Cli, DiffArgs};
use clap::Parser;
use log::debug;
use rvn_core_diff::{diff_header, diff_metadata, diff_tensors};
use rvn_core_parser::load_model;
use rvn_core_tensor::DiffDump;
use rvn_globals::{GlobalOpts, OutputFormat, get_globals, init_globals};

pub fn print_pretty(d: &DiffDump) {
    // header
    if let Some(header) = &d.header {
        println!("header report");
        for (field, (a, b)) in header {
            if a == b {
                println!("  {field}: {a}");
            } else {
                println!("  {field}: {} -> {}", a, b);
            }
        }
        println!();
    }

    // metadata
    if let Some(meta) = &d.metadata {
        println!("metadata report");

        for (k, v) in &meta.added {
            println!("  + {k}: {v:?}");
        }
        for (k, v) in &meta.removed {
            println!("  - {k}: {v:?}");
        }
        for (k, (va, vb)) in &meta.changed {
            println!("  ~ {k}: {va:?} -> {vb:?}");
        }
        println!();
    }

    // tensors
    if let Some(t) = &d.tensors {
        println!("tensors report");

        for (name, tensor) in &t.added {
            println!(
                "  + {name}: shape={:?}, dtype={:?}",
                tensor.shape, tensor.kind
            );
        }
        for (name, tensor) in &t.removed {
            println!(
                "  - {name}: shape={:?}, dtype={:?}",
                tensor.shape, tensor.kind
            );
        }
        for (name, (ta, tb)) in &t.changed {
            println!("  ~ {name}:");
            println!("      lhs: shape={:?}, dtype={:?}", ta.shape, ta.kind);
            println!("      rhs: shape={:?}, dtype={:?}", tb.shape, tb.kind);
        }
        println!();
    }

    if d.header.is_none() && d.metadata.is_none() && d.tensors.is_none() {
        println!("No differences 🎉");
    }
}

pub async fn run_diff_cmd(cmd: DiffArgs, globals: &GlobalOpts) -> anyhow::Result<()> {
    debug!(
        "rvn-diff file_a file_b -> {:#?} {:#?}",
        &cmd.file_a, &cmd.file_b
    );

    let a = load_model(&cmd.file_a)?;
    let b = load_model(&cmd.file_b)?;

    // ---------- structured (json / yaml) ----------
    let dump = DiffDump {
        header: cmd
            .wants_header()
            .then(|| diff_header(&a.header, &b.header))
            .flatten(),

        metadata: cmd
            .wants_metadata()
            .then(|| diff_metadata(a.metadata(), b.metadata()))
            .flatten(),

        tensors: cmd
            .wants_tensors()
            .then(|| {
                diff_tensors(
                    a.iter().map(|(n, t)| (n.as_str(), t)),
                    b.iter().map(|(n, t)| (n.as_str(), t)),
                )
            })
            .flatten(),
    };

    match globals.format {
        OutputFormat::Pretty => print_pretty(&dump), // handcrafted pretty view
        OutputFormat::Json => serde_json::to_writer_pretty(std::io::stdout(), &dump)?,
        OutputFormat::Yaml => serde_yaml::to_writer(std::io::stdout(), &dump)?,
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let _ = env_logger::try_init();

    init_globals(cli.g);
    run_diff_cmd(cli.args, get_globals()).await
}

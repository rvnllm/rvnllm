mod cli;

use crate::cli::{Cli, DiffArgs};
use clap::Parser;
use log::debug;
use rvn_core_parser::{DiffDump, Header, MetaDiff, Tensor, TensorDiff, load_model, types::Value};
use rvn_globals::{GlobalOpts, OutputFormat, get_globals, init_globals};
use std::collections::HashMap;

fn diff_header<'a>(
    a: &'a Header,
    b: &'a Header,
) -> Option<Vec<(&'static str, (&'a u64, &'a u64))>> {
    let mut v = Vec::new();
    if a.tensor_count != b.tensor_count {
        v.push(("tensor_count", (&a.tensor_count, &b.tensor_count)));
    }
    if a.metadata_kv_count != b.metadata_kv_count {
        v.push((
            "metadata_kv_count",
            (&a.metadata_kv_count, &b.metadata_kv_count),
        ));
    }
    (!v.is_empty()).then_some(v)
}

fn diff_metadata<'a>(
    a: &'a HashMap<String, Value>,
    b: &'a HashMap<String, Value>,
) -> Option<MetaDiff<'a>> {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    for (k, va) in a {
        if k.starts_with("tokenizer") {
            continue;
        } // keep the noise out
        match b.get(k) {
            None => removed.push((k.as_str(), va)),
            Some(vb) if vb != va => changed.push((k.as_str(), (va, vb))),
            _ => {}
        }
    }
    for (k, vb) in b {
        if k.starts_with("tokenizer") {
            continue;
        }
        if !a.contains_key(k) {
            added.push((k.as_str(), vb));
        }
    }
    (!added.is_empty() || !removed.is_empty() || !changed.is_empty()).then_some(MetaDiff {
        added,
        removed,
        changed,
    })
}

fn diff_tensors<'a>(
    a: impl Iterator<Item = (&'a str, &'a Tensor)>,
    b: impl Iterator<Item = (&'a str, &'a Tensor)>,
) -> Option<TensorDiff<'a>> {
    use std::collections::HashMap;

    let map_a: HashMap<_, _> = a.collect();
    let map_b: HashMap<_, _> = b.collect();

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    for (k, ta) in &map_a {
        match map_b.get(k) {
            None => removed.push((*k, *ta)),
            Some(tb) if ta.shape != tb.shape || ta.kind != tb.kind => {
                changed.push((*k, (*ta, *tb)))
            }
            _ => {}
        }
    }
    for (k, tb) in &map_b {
        if !map_a.contains_key(k) {
            added.push((*k, *tb));
        }
    }
    (!added.is_empty() || !removed.is_empty() || !changed.is_empty()).then_some(TensorDiff {
        added,
        removed,
        changed,
    })
}

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

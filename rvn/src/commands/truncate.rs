// Truncate command implementation
//
// Purpose:
// This command creates a minimal version of a GGUF file by removing most tensors,
// keeping only a small subset for fast unit testing, benchmarks, or inspection.
//
// Use case:
// - Generates GGUF test fixtures under 1MB
// - Great for verifying tensor logic, parser correctness, CLI tooling

// Steps:
//    Load model (this is done through load_model -> mmaped so ultra fast
//
//    Keep tensor names:
//        token_embd.weight
//        blk.0.* through blk.N.* (where N = layers - 1)
//        output.* (optional)
//    Create new GGUF header, metadata, and tensors.
//    Rewrite new offsets.
//    Save to output.

use parsers::tensor::{load_model, Tensor};
use parsers::types::Value;
use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use std::io::{Write, Seek, BufWriter};
use anyhow::{Result, Context};
use log::debug;
use ctensor::tensor_view::TensorView;



fn write_gguf_file<W: Write + Seek> 
(
    writer: &mut W, 
    metadata: &HashMap<String, Value> , 
    tensors: &[(String, Tensor)], 
    tensor_blob: &[u8]
) -> anyhow::Result<()>
{
    // 1. Write GGUF header (magic, versions, etc)
    writer.write_all(b"GGUF");
    writer.write_all(buf)
    //
    // 2. Write metadata (key/value pairs)
    // 3. Write tensor descriptors (names, shapes, dtype, offsets)
    // 4. Append raw tensor data
    todo!("write GGUF header + tensors for minimal file");

}


pub fn run_truncate<P: AsRef<Path>>(
    path: P, 
    output: P, 
    layers: usize,
    verbose: bool) -> Result<()> {
    
    debug!("[run_truncate]");

    let gguf = load_model(&path)?;

    let mut selected = vec![];

    for (name, tensor) in gguf.tensors.iter() {
        //debug!("{:#?}", name);
        if name.starts_with("token_emb") || 
           name.starts_with("output") ||
           (name.starts_with("blk.") && {
               if let Some(layer_num) = name.split('.').nth(1) {
                   layer_num.parse::<usize>().unwrap_or(999) < layers
               } else {
                   false
               }
           }) {
               selected.push((name.clone(), tensor.clone())); // not cloning the whole tensor just
        }
    }

    if verbose {
        println!("Keeping {} tensors out of {}", selected.len(), gguf.tensors.len());
    }

    /* DO NOT UNCOMMENT, this is here as a help
    #[derive(Debug)]
        pub struct Tensor {
        pub name: String,
        pub kind: u32,
        pub offset: u64,
        pub size: u64,
        pub shape: Vec<u64>,
    }*/

    // realign offsets
    let mut new_blob = Vec::new();
    let mut new_tensors = Vec::new();

    for (name, tensor) in selected {
        let view: TensorView = tensor.view(&gguf.raw_bytes())?;
        let offset = new_blob.len() as u64;
        new_blob.extend_from_slice(view.data);
        let new_tensor = Tensor {
            name: name.clone(),
            kind: tensor.kind,
            offset,
            size: view.data.len() as u64,
            shape: tensor.shape.clone(),
        };
        new_tensors.push((name, new_tensor));
    }

    let mut out = BufWriter::new(File::create(&output).context("creating output file")?);
    write_gguf_file(&mut out, &gguf.metadata, &new_tensors, &new_blob)?;

    Ok(())
}

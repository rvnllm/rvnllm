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
use std::io::SeekFrom;



/**
What’s supported: (nothing fancy, only for unit testing the other CLI commands
    Basic GGUF V2 headers
    Metadata (only strings for now)
    Tensors with shapes, offsets, size, and kinds
    Direct blob data

TODO: 
    advanced metadata types 
    quantized tensor formats (coming right up AFTER quant FFI implementation.
    add padding/alignment (e.g., 32 or 64 bytes if needed for speed/memory alignment)
    extend metadata support: integers, floats, arrays
    add a flag for --fake-data (zero-fill) to avoid carrying 100MB blobs in tests
    write fuzz tests for malformed tensors (offset/size mismatch)
 */

fn write_gguf_file<W: Write + Seek> (
    writer: &mut W, 
    metadata: &HashMap<String, Value> , 
    tensors: &[(String, Tensor)], 
    blob: &[u8]
) -> anyhow::Result<()>
{
    // 1. Write GGUF header (magic, versions, etc)
    writer.write_all(b"GGUF")?;               // Magic
    writer.write_all(&[2, 0, 0, 0])?;         // Version (V2)

    let metadata_len = metadata.len() as u64;
    let tensor_count = tensors.len() as u64;

    writer.write_all(&metadata_len.to_le_bytes())?;
    writer.write_all(&tensor_count.to_le_bytes())?;

    // 2. Write metadata (key/value pairs)
    for (key, _) in metadata {
        let key_bytes = key.as_bytes();
        writer.write_all(&(key_bytes.len() as u64).to_le_bytes())?;
        writer.write_all(key_bytes)?;

        // Only supporting basic values for now, good enough for building GGUF files for unit
        // testing
        //match val {
        //    Value::String(s) => {
        //        writer.write_all(&[0x01])?;
        //        let str_bytes = s.as_bytes();
        //        writer.write_all(&(str_bytes.len() as u64).to_le_bytes())?;
        //        writer.write_all(str_bytes)?;
        //    },
        //    _ => unimplemented!("Other metadata types not yet supported"),
        writer.write_all(&[0x01])?; // Type tag (0x01 = u32?)
        writer.write_all(&0u32.to_le_bytes())?;
        //}
    }

    // 3. Write tensor descriptors (names, shapes, dtype, offsets)
    for (name, tensor) in tensors {
        let name_bytes = name.as_bytes();
        writer.write_all(&(name_bytes.len() as u64).to_le_bytes())?;
        writer.write_all(name_bytes)?;

        writer.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;
        for &dim in &tensor.shape {
            writer.write_all(&(dim as u64).to_le_bytes())?;
        }

        writer.write_all(&tensor.kind.to_le_bytes())?;
        writer.write_all(&tensor.offset.to_le_bytes())?;
        writer.write_all(&tensor.size.to_le_bytes())?;
    }

    // 4. Append raw tensor data
    writer.seek(SeekFrom::Start(tensors[0].1.offset))?;
    writer.write_all(blob)?;

    Ok(())
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

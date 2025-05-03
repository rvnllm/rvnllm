//use anyhow::Result;
use std::path::Path;
use parsers::tensor::load_model;

pub fn run_list<P: AsRef<Path>>(path: P) -> anyhow::Result<()> 
{
    let gguf = load_model(&path)?;
    println!("Tensor count: {}", gguf.tensors.len());
    for (name, tensor) in &gguf.tensors {
        println!("{} => shape: {:?}", name, tensor.shape);
    }
    Ok(())
}

use std::path::Path;
use anyhow::Result;

pub fn load_model<P: AsRef<Path>>(path: P) -> Result<()>{
    println!("loading the model");

    Ok(())
}

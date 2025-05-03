//! 
//!
//! Used by: `decode-test`, `debug`, etc.
//! Related files: `tensor.rs`, `gguf.rs`

// mods
mod commands;
mod cli;

// use
use commands::dispatch::dispatch;
use clap::Parser;
use anyhow::{Result, anyhow};
use crate::cli::RvnCli;

fn main() -> Result<()>
{
    env_logger::init();

    let cli = RvnCli::parse();
    dispatch(cli)


}

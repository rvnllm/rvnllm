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
use anyhow::Result;
use crate::cli::RvnCli;

fn main() -> Result<()>
{
    env_logger::init();
    // CLI login
    let cli = RvnCli::parse();
    dispatch(cli)

    // <> logic comes here

}

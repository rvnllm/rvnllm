use clap::{Args, Parser};
use rvn_globals::GlobalOpts;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "rvn-diff",
    version = env!("CARGO_PKG_VERSION"),
    about = "Local-first LLM engine & tooling."
)]
pub struct Cli {
    #[command(flatten)]
    pub g: GlobalOpts,

    #[command(flatten)]
    pub args: DiffArgs,
    // Later: Dump, Validate, Convert, etc.
}

#[derive(Args, Debug)]
pub struct DiffArgs {
    pub file_a: PathBuf,
    pub file_b: PathBuf,

    #[clap(long)]
    pub header: bool,

    #[clap(long)]
    pub metadata: bool,

    #[clap(long)]
    pub tensors: bool,
}
impl DiffArgs {
    pub fn wants_header(&self) -> bool {
        self.header || self.none_specified()
    }
    pub fn wants_metadata(&self) -> bool {
        self.metadata || self.none_specified()
    }
    pub fn wants_tensors(&self) -> bool {
        self.tensors || self.none_specified()
    }

    fn none_specified(&self) -> bool {
        !(self.header || self.metadata || self.tensors)
    }
}

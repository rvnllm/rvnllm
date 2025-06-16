use clap::{Args, Parser};
use rvn_globals::GlobalOpts;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "rvn-info",
    version = env!("CARGO_PKG_VERSION"),
    about = "Local-first LLM engine & tooling."
)]
pub struct Cli {
    #[command(flatten)]
    pub g: GlobalOpts,

    #[command(flatten)]
    pub args: InfoArgs,
    // Later: Dump, Validate, Convert, etc.
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Path to the model (GGUF / safetensors / ONNX)
    #[arg(short = 'f', long = "file", value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
    pub file: PathBuf,

    /// Show header section
    #[arg(long)]
    pub header: bool,

    /// Show model key-value pairs
    #[arg(long)]
    pub metadata: bool,

    /// Inspect one or more named tensors
    #[arg(long)]
    pub tensors: bool,
    //  Load vocabulary from gguf file
    //  #[arg(long)]
    //  pub vocab: bool,
}
impl InfoArgs {
    fn none_specified(&self) -> bool {
        !(self.header || self.metadata || self.tensors) // || self.vocab)
    }
    pub fn wants_header(&self) -> bool {
        self.header || self.none_specified()
    }
    pub fn wants_metadata(&self) -> bool {
        self.metadata || self.none_specified()
    }
    pub fn wants_tensors(&self) -> bool {
        self.tensors || self.none_specified()
    }
    // todo: grab the vocab from metadata?
    //    pub fn wants_vocab(&self) -> bool {
    //      self.vocab || self.none_specified()
    //}
}

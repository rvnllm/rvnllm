use clap::{Parser, Subcommand, Args, ValueEnum};
use std::collections::HashMap;
use std::io::Cursor;
use std::path::PathBuf;
use anyhow::{Result, bail};
use memmap2::Mmap;
use once_cell::sync::Lazy;
use byteorder::{LittleEndian, ReadBytesExt};

#[derive(Parser)]
#[command(
    name = "rvnllm",
    version = env!("CARGO_PKG_VERSION"),
    about = "High-performance GGUF loader. Load, inspect, validate, and run models fast.",
    after_help = r#"
EXAMPLES:
  rvnllm info header --file llama2.gguf
  rvnllm list tensors --file llama2.gguf
  rvnllm dump tensor --file llama2.gguf --name blk.0.attn_q.weight --format f32
  rvnllm forward --file llama2.gguf --input "The raven is" --device cuda
  rvnllm validate --file llama2.gguf --profile paranoid
"#
)]
pub struct RvnCli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Info(InfoCommand),

    List {
        #[arg(short, long)]
        file: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    Dump {
        #[arg(short, long)]
        file: String,
        #[arg(short, long)]
        name: String,
        #[arg(long, default_value = "shape")]
        format: DumpFormat,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    Forward(ForwardArgs),

    Diff {
        #[arg(short = 'a', long)]
        file_a: String,
        #[arg(short = 'b', long)]
        file_b: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    Analyze {
        #[arg(short, long)]
        file: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    Profile {
        #[arg(short, long)]
        file: String,
        #[arg(long, default_value = "cpu")]
        device: Device,
        #[arg(long, default_value = "32")]
        tokens: usize,
        #[arg(long, default_value = "none")]
        cache_mode: CacheMode,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    Validate {
        #[arg(short, long)]
        file: String,
        #[arg(long, default_value = "llama")]
        profile: ValidationProfile,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    Watch {
        #[arg(short, long)]
        file: String,
        #[arg(long, default_value = "llama")]
        profile: WatchProfile,
        #[arg(long)]
        dummy_forward: bool,
        #[arg(long)]
        check_tokenizer: bool,
        #[arg(long)]
        check_entropy: bool,
        #[arg(long)]
        scan_triggers: bool,
        #[arg(long)]
        verbose: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    WatchPerf {
        #[arg(short, long)]
        file: String,
        #[arg(short, long)]
        input: String,
        #[arg(long, value_delimiter = ',')]
        metrics: Vec<PerfMetric>,
        #[arg(long)]
        preset: Option<PerfPreset>,
        #[arg(long, default_value = "cpu")]
        device: Device,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Args)]
pub struct InfoCommand {
    #[arg(short, long)]
    pub file: String,
    #[arg(long)]
    pub header: bool,
    #[arg(long)]
    pub metadata: bool,
    #[arg(long)]
    pub tensor: Option<String>,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Args)]
pub struct ForwardArgs {
    #[arg(short, long)]
    pub file: String,
    #[arg(short, long)]
    pub input: String,
    #[arg(long, default_value = "cpu")]
    pub device: Device,
    #[arg(long, default_value = "none")]
    pub cache_mode: CacheMode,
    #[arg(long)]
    pub quantize: bool,
    #[arg(long)]
    pub stream: bool,
    #[arg(long)]
    pub personality: Option<String>,
    #[arg(long, default_value = "text")]
    pub output_format: OutputFormat,
    #[arg(long)]
    pub dump_activations: bool,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum Device {
    Cpu,
    Cuda,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum DumpFormat {
    Shape,
    F32,
    Raw,
    Json,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum OutputFormat {
    Text,
    Json,
    Logits,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum CacheMode {
    None,
    Kv,
    Full,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum ValidationProfile {
    Llama,
    Strict,
    Paranoid,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum WatchProfile {
    Llama,
    Strict,
    Paranoid,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum PerfMetric {
    Time,
    Cache,
    Kv,
    Attention,
    Logits,
    Heatmap,
    Memory,
    Entropy,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum PerfPreset {
    Minimal,
    Deep,
    Debug,
}


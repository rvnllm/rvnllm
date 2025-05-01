use clap::{Parser, Subcommand, Args, ValueEnum};

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
    /// View model metadata, headers, tensor info
    Info(InfoCommand),

    /// List available tensors
    List {
        #[arg(short, long)]
        file: String,
    },

    /// Dump the content of a tensor
    Dump {
        #[arg(short, long)]
        file: String,
        #[arg(short, long)]
        name: String,
        #[arg(long, default_value = "shape")]
        format: DumpFormat,
    },

    /// Run a forward pass with optional options
    Forward(ForwardArgs),

    /// Compare tensors between two models
    Diff {
        #[arg(short = 'a', long)]
        file_a: String,
        #[arg(short = 'b', long)]
        file_b: String,
    },

    /// Analyze structure and estimate usage
    Analyze {
        #[arg(short, long)]
        file: String,
    },

    /// Profile model performance
    Profile {
        #[arg(short, long)]
        file: String,
        #[arg(long, default_value = "cpu")]
        device: Device,
        #[arg(long, default_value = "32")]
        tokens: usize,
        #[arg(long, default_value = "none")]
        cache_mode: CacheMode,
    },

    /// Validate GGUF file integrity and structure
    Validate {
        #[arg(short, long)]
        file: String,

        /// Validation profile: choose how strict to be
        #[arg(long, default_value = "llama")]
        profile: ValidationProfile,
    },
    /// Inspect and audit model integrity, shape, and suspicious patterns
    Watch {
        #[arg(short, long)]
        file: String,

        /// Validation profile: how deep to scan
        #[arg(long, default_value = "llama")]
        profile: WatchProfile,

        /// Enable single-token dummy forward pass
        #[arg(long)]
        dummy_forward: bool,

        /// Enable tokenizer/token ID mapping inspection
        #[arg(long)]
        check_tokenizer: bool,

        /// Enable entropy & output distribution analysis
        #[arg(long)]
        check_entropy: bool,

        /// Enable scan for known suspicious trigger phrases
        #[arg(long)]
        scan_triggers: bool,

        /// Dump suspicious findings even if no hard errors
        #[arg(long)]
        verbose: bool,
    },

    WatchPerf {
        #[arg(short, long)]
        file: String,

        #[arg(short, long)]
        input: String,

        #[arg(long, value_delimiter = ',')]
        metrics: Vec<PerfMetric>,

        #[arg(long, default_value = "cpu")]
        device: Device,
    }
    
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
pub enum CacheMode {
    None,
    Kv,
    Full,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum ValidationProfile {
    /// Basic GGUF format validation (header, tensors, metadata)
    Llama,

    /// Also checks expected tensor shapes and presence
    Strict,

    /// Includes quantization sanity, offset boundaries, entropy heuristics
    Paranoid,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum WatchProfile {
    /// Basic GGUF format validation (header, tensors, metadata)
    Llama,

    /// Expected tensor structure, shape sanity, dtype checks
    Strict,

    /// Entropy, offset boundaries, forward simulation, tokenizer analysis
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

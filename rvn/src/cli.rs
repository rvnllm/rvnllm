use clap::{Parser, Subcommand, Args, ValueEnum};
use std::path::PathBuf;


#[derive(Parser)]
#[command(
    name = "rvnllm",
    version = env!("CARGO_PKG_VERSION"),
    about = "High-performance GGUF loader. Load, inspect, validate, and run models fast.",
    after_help = r#"
USAGE:
    rvnllm <COMMAND> [OPTIONS]

COMMANDS:
    info             View model metadata, headers, or tensor info
    dump             Dump tensor contents (supports multiple formats)
    forward          Execute full forward pass (WIP)
    forward-simple   Run attention-only forward pass for inspection
    decode-test      Decode a tensor and check for anomalies
    diff             Compare two models' tensor sets
    validate         Run structural integrity checks on GGUF files
    analyze          Analyze tensor structures and usage heuristics
    profile          Measure model performance (CPU/CUDA)
    watch            Inspect and audit model for suspicious patterns
    watch-perf       Run forward pass and collect performance metrics

OPTIONS:
    -h, --help        Show this help message
    -V, --version     Show version info

EXAMPLES:
    rvnllm info --file llama2.gguf --header
    rvnllm list --file llama2.gguf
    rvnllm decode-test --file llama2.gguf --name blk.0.attn_q.weight
    rvnllm forward-simple --file llama2.gguf --q ... --k ... --v ...
    "#
)]
pub struct RvnCli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {

    Info(InfoCommand),

    #[command(about = "List all tensor names and shapes")]
    List {
        #[arg(short, long)]
        file: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    #[command(about = "Dump tensor contents (supports multiple formats)")]
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

    #[command(about = "Run attention-only forward pass for inspection (Note: qunatization under development only f32 for now.")]
    ForwardSimple {
        #[arg(short, long)]
        file: String,
        #[arg(long)]
        q: String,
        #[arg(long)]
        k: String,
        #[arg(long)]
        v: String,
    },

    Forward(ForwardArgs),

    #[command(about = "Compare two models' tensor sets")]
    Diff {
        #[arg(short = 'a', long)]
        file_a: String,
        #[arg(short = 'b', long)]
        file_b: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    #[command(about = "Decode a tensor and check for anomalies")]
    DecodeTest {
        #[arg(short, long)]
        file: String,
        #[arg(short, long)]
        name: String,
        #[arg(long, default_value_t = false)]
        verbose: bool,
        #[arg(long, default_value_t = false)]
        json: bool,
        #[arg(long, default_value_t = false)]
        fail_on_anomaly: bool,
    },

    #[command(about = "Full internal structure is dumped, a ParsedGGUF structure with header, metadata and tensors")]
    Debug {
        #[arg(short, long)]
        file: String,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long)]
        output: Option<String>,
        #[arg(long)]
        compat: bool, // <-- Add this
    },   

    #[command(about = "Analyze tensor structures and usage heuristics. (Note: under development)")]
    Analyze {
        #[arg(short, long)]
        file: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    #[command(about = "Analyze tensor structures and usage heuristics. (Note: under development)")]
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


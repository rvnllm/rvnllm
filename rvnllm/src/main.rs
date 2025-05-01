mod cli_commands;

use std::fmt::format;
use std::{collections::HashMap, fs::File, hash::Hash};
use std::path::Path;
use ctensor::tensor_view::{TensorView, TensorDType};
use memmap2::Mmap;
use once_cell::sync::{Lazy, OnceCell};
use anyhow::{anyhow, bail, Context, Result};
use cli_commands::{RvnCli, Command, ValidationProfile};
use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use log::debug;
use std::io::{Cursor, Read};


#[macro_export]
macro_rules! check_debug_dev_sanity {
    () => {
        if std::env::var("WOKE_UP").unwrap_or_default() == "true"
            && std::env::var("COFFEE").unwrap_or_default() == "0"
        {
            panic!("Debugging without caffeine detected. Please abort mission.");
        }
    };
}


// todo: move this outta here just here for quick prototyping
/// Typed representation of metadata values
#[derive(Debug, Clone)]
pub enum Value {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

/// Discriminator codes per GGUF spec
#[derive(Debug, Clone, Copy)]
pub enum MetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

// this is here to avoid idiotic magic numbers proliferating the codebase. So nope use this 
// usage
// let version_num = cursor.read_u32::<LittleEndian>()?;
// let version = GgufVersion::try_from(version_num)?;
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GgufVersion {
    V2 = 2,
    V3 = 3,
}
impl TryFrom<u32> for GgufVersion {
    type Error = anyhow::Error;

    fn try_from(v: u32) -> Result<Self> {
        match v {
            2 => Ok(GgufVersion::V2),
            3 => Ok(GgufVersion::V3),
            _ => bail!("Unsupported GGUF version: {}", v),
        }
    }
}
impl From<GgufVersion> for u32 {
    fn from(ver: GgufVersion) -> u32 {
        ver as u32
    }
}
/*
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F32, //0
    F16,  //1
    Q4_0, //2
    I8, 
    U8,
}
*/
fn get_format(kind: u32) -> Option<&'static Box<dyn TensorFormat>> {
    TENSOR_REGISTRY.get().and_then(|m| m.get(&kind))
}


/**
 * ParsedGGUF supporting structures */
pub struct Tensor {
    pub name: String,
    pub kind: u32,
    pub offset: u64,
    pub size: u64,
    pub shape: Vec<u64>,
}
impl Tensor {
    pub fn view<'a>(&self, blob: &'a [u8]) -> Result<TensorView<'a>> {
        let end = self.offset + self.size;
        if end as usize > blob.len() {
            bail!("tensor '{}' slice out of bounds", self.name);
        }

        // get the format from the registry
        let fmt = get_format(self.kind)
            .ok_or_else(|| anyhow!("unknown tensor kind {}", self.kind))?;
        // decode the tensor 
        // TODO: tensor lib!!!
        let shape_usize: Vec<usize> = self.shape.iter().map(|&d| d as usize).collect();
        fmt.decode(&blob[self.offset as usize..end as usize], &shape_usize)
    }
}





pub struct Header {
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}
impl Default for Header {
    fn default() -> Self {
        Header {
            tensor_count: 0,
            metadata_kv_count: 0,
        }
    }
}


/**
 * TENSOR_REGITRY supporting structs
 */
pub trait TensorFormat: Send + Sync {
    fn id(&self) -> u32;
    fn name(&self) -> &'static str;
    fn decode<'a>(&self, bytes: &'a [u8], shape: &[usize]) -> Result<TensorView<'a>>;
}

/**enum ggml_type: uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    ...
*/
struct F32Format;
impl TensorFormat for F32Format {
    fn id(&self) -> u32 { 0 }
    fn name(&self) -> &'static str{ "F32" }
    fn decode<'a>(&self, bytes: &'a [u8], shape: &[usize]) -> Result<TensorView<'a>> {
        // TODO: proper quantisied code CPU or CUDA --->>> call kernel
        Ok(TensorView { data: bytes, shape: shape.to_vec(), dtype: TensorDType::F32 })
    }
}

struct Q4_0Format;
impl TensorFormat for Q4_0Format {
    fn id(&self) -> u32 { 2 }
    fn name(&self) -> &'static str{ "Q4_0" }
    fn decode<'a>(&self, bytes: &'a [u8], shape: &[usize]) -> Result<TensorView<'a>> {
        // TODO: proper quantisied code CPU or CUDA --->>> call kernel
        Ok(TensorView { data: bytes, shape: shape.to_vec(), dtype: TensorDType::Q4_0 })
    }
}
//... implementation for the other tensor formats


/* -------------------------------------------------------------------------------------- */
/**
 * parsed GGUF structure holding information about the header, metadata and 
 * the tensors, the tensors are zero copy, only offsets are stored into the mmap
 * <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md */
pub struct ParsedGGUF {
    // public 
    pub header: Header,
    pub metadata: HashMap<String, Vec<u8>>,
    pub tensors: HashMap<String, Tensor>,
    // private
    mmap: Mmap,        // ->>> not exposed private field
}
impl ParsedGGUF {
    pub fn metadata(&self) -> &HashMap<String, Vec<u8>> { &self.metadata }
    pub fn tensor(&self, name: &str) -> Option<&Tensor> { self.tensors.get(name) }
    pub fn tensor_view(&self, name: &str) -> Result<TensorView<'_>> {
        self.tensor(name).ok_or_else(|| anyhow!("no tensor {}", name))?  
            .view(&self.mmap[..])
    }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Tensor)> { self.tensors.iter() }
}

static PARSER_REGISTRY: Lazy<Vec<Box<dyn GgufParser>>> = Lazy::new(|| {
    vec![
        Box::new(ParserV2),
        Box::new(ParserV3),
        // ..... fture parsers
    ]
});
// Look up the right parser for this on-disk GGUF version.
fn parser_for(version: GgufVersion) -> Option<&'static Box<dyn GgufParser>> {
    PARSER_REGISTRY
        .iter()
        .find(|parser| parser.version() == version)
}

struct GgufBody {
    pub header: Header,
    pub metadata: HashMap<String, Value>, // now typed values
    pub tensors: HashMap<String, Tensor>,
}

trait GgufParser: Send + Sync {
    fn version(&self) -> GgufVersion;
    fn parse<'b>(&self, cur: &mut Cursor<&'b [u8]>)-> Result<GgufBody>;
}
struct ParserV3;
impl GgufParser for ParserV3 {
    fn version(&self) -> GgufVersion { GgufVersion::V3 } // todo: move this an enum, no hardcoded magic symbols
    fn parse<'b>(&self, cursor: &mut Cursor<&'b [u8]>) -> Result<GgufBody> {
        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        debug!("tensor_count: {:#?}", tensor_count);

        Ok(
            GgufBody {
                header: Header {
                tensor_count: 0,
                metadata_kv_count: 0,
            },
            metadata: HashMap::new(),
            tensors: HashMap::new(),
        })
    }
}

struct ParserV2;
impl GgufParser for ParserV2 {
    fn version(&self) -> GgufVersion { 
        GgufVersion::V2
    } // todo: move this an enum, no hardcoded magic symbols
    fn parse<'b>(&self, cursor: &mut Cursor<&'b [u8]>) -> Result<GgufBody> {
        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        debug!("tensor_count: {:#?}", tensor_count);

        Ok(
            GgufBody {
                header: Header {
                tensor_count: 0,
                metadata_kv_count: 0,
            },
            metadata: HashMap::new(),
            tensors: HashMap::new(),
        })
    }
}




/* ---------------------------------------------------------------------------------- */


/* ---------------------------------------------------------------------------------- */

static TENSOR_REGISTRY: Lazy<OnceCell<HashMap<u32, Box<dyn TensorFormat>>>> = Lazy::new(OnceCell::default);
fn register_builtin_formats() {
//    TENSOR_REGISTRY.get
    // todo: this is quite fragile here, the HashMap accepts 2 as key it is also the numerical code
    // for the tensor type aka  GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_Q4_2 = 4 ...
    // create an enum and when inserting use it the new(F32Format(....))
    TENSOR_REGISTRY.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert(0, Box::new(F32Format) as Box<dyn TensorFormat>);
        m.insert(2, Box::new(Q4_0Format) as Box<dyn TensorFormat>);
        m
    });
}

pub fn load_model<P: AsRef<Path>>(path: P) -> anyhow::Result<ParsedGGUF>// Result<ParsedGGUF, Box<dyn std::error::Error>>
{
    register_builtin_formats();
    
    // debug block, need to factor this one of here
    debug!("[DEBUG] fn: {:#?}", "load_model");
    if let Some(map) = TENSOR_REGISTRY.get() {
    debug!("[DEBUG] Registered tensor formats:");
    for (k, v) in map {
        debug!("[DEBUG]  Kind: {} -> {}", k, v.name());
    }
    } else {
        debug!("[DEBUG] TENSOR_REGISTRY is not initialized.");
    }
    
    // 1) open & mmap
    let file = File::open(&path)
        .with_context(|| format!("cannot open '{}'", path.as_ref().display()))?;
    
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("cannot mmap '{}'", path.as_ref().display()))?;
    
    debug!("MMAP created: len = {}", mmap.len());

    // 2) magic & version
    let mut cursor = Cursor::new(&mmap[..]);
    debug!("Cursor position: {}", cursor.position());
    debug!("Cursor peek: {:?}", &mmap[0..16.min(mmap.len())]); // show header bytes
    const MAGIC: u32 = 0x4655_4747;  //gguf
    let magic = cursor.read_u32::<LittleEndian>()?;
    if magic != MAGIC { bail!("[ERROR] invalid GGUF magic 0x{magic:08X}"); }  // todo: check magic command for the validity checks
    let version_num = cursor.read_u32::<LittleEndian>()?;
    let version = GgufVersion::try_from(version_num)?;

    debug!("Version: {:#?}", version);

    // 3) parse body
    debug!("Looking up parser for version {:#?}", version);
    for p in &*PARSER_REGISTRY {
        debug!("Registered parser version: {:#?}", p.version());
    }

    let parser = parser_for(version).ok_or_else(|| {
        debug!("No parser found for version {:#?}", version);
        anyhow!("unsupported GGUF version {:#?}", version)
    })?;


    debug!("Parser: {:#?}", parser.version());

    // unchanged up to parsing
    let mut body = parser.parse(&mut cursor)?;


    let parsed = ParsedGGUF {
        header: Header::default(),
        metadata: HashMap::new(),
        tensors: HashMap::new(),
        mmap,
    };

    Ok(parsed)
}

fn main() -> anyhow::Result<()> 
{
    env_logger::init();
    
    println!("main");
    check_debug_dev_sanity!();

    //let cli = RvnCli::parse();
    
    let path = std::env::args()
        .nth(1)
        .expect("usage: rvnllm <path/to/model.gguf>");
    println!("[DEBUG] path: {:#?}", path);
    
    //let _file = File::open("../model/llama-2-13b-ensemble-v5.Q6_K.gguf")?;
    /*let (tensor_map, mmap) =*/ load_model(&path);


    Ok(())
}

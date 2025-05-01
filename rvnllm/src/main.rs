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
use std::fmt::Display;


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

#[derive(Debug)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    Count = 19,
}
impl Display for GGMLType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GGMLType::F32 => write!(f, "F32"),
            GGMLType::F16 => write!(f, "F16"),
            GGMLType::Q4_0 => write!(f, "Q4_0"),
            GGMLType::Q4_1 => write!(f, "Q4_1"),
            GGMLType::Q5_0 => write!(f, "Q5_0"),
            GGMLType::Q5_1 => write!(f, "Q5_1"),
            GGMLType::Q8_0 => write!(f, "Q8_0"),
            GGMLType::Q8_1 => write!(f, "Q8_1"),
            GGMLType::Q2K => write!(f, "Q2K"),
            GGMLType::Q3K => write!(f, "Q3K"),
            GGMLType::Q4K => write!(f, "Q4K"),
            GGMLType::Q5K => write!(f, "Q5K"),
            GGMLType::Q6K => write!(f, "Q6K"),
            GGMLType::Q8K => write!(f, "Q8K"),
            GGMLType::I8 => write!(f, "I8"),
            GGMLType::I16 => write!(f, "I16"),
            GGMLType::I32 => write!(f, "I32"),
            GGMLType::Count => write!(f, "Count"),
        }
    }
}

impl TryFrom<u32> for GGMLType {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> std::prelude::v1::Result<Self, Self::Error> {
        Ok(match value {
            0 => GGMLType::F32,
            1 => GGMLType::F16,
            2 => GGMLType::Q4_0,
            3 => GGMLType::Q4_1,
            6 => GGMLType::Q5_0,
            7 => GGMLType::Q5_1,
            8 => GGMLType::Q8_0,
            9 => GGMLType::Q8_1,
            10 => GGMLType::Q2K,
            11 => GGMLType::Q3K,
            12 => GGMLType::Q4K,
            13 => GGMLType::Q5K,
            14 => GGMLType::Q6K,
            15 => GGMLType::Q8K,
            16 => GGMLType::I8,
            17 => GGMLType::I16,
            18 => GGMLType::I32,
            19 => GGMLType::Count,
            _ => return Err(anyhow!("invalid GGML type")),
        })
    }
}



impl TryFrom<u32> for MetadataValueType {
    type Error = anyhow::Error;
    fn try_from(v: u32) -> Result<Self> {
        match v {
            0 => Ok(MetadataValueType::Uint8),
            1 => Ok(MetadataValueType::Int8),
            2 => Ok(MetadataValueType::Uint16),
            3 => Ok(MetadataValueType::Int16),
            4 => Ok(MetadataValueType::Uint32),
            5 => Ok(MetadataValueType::Int32),
            6 => Ok(MetadataValueType::Float32),
            7 => Ok(MetadataValueType::Bool),
            8 => Ok(MetadataValueType::String),
            9 => Ok(MetadataValueType::Array),
            10 => Ok(MetadataValueType::Uint64),
            11 => Ok(MetadataValueType::Int64),
            12 => Ok(MetadataValueType::Float64),
            other => Err(anyhow!("wrong metadata type: {}", other)),
        }
    }
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
#[derive(Debug)]
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

#[derive(Debug)]
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
#[derive(Debug)]
pub struct ParsedGGUF {
    // public 
    pub header: Header,
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, Tensor>,
    // private
    mmap: Mmap,        // ->>> not exposed private field
}
impl ParsedGGUF {
    pub fn metadata(&self) -> &HashMap<String, Value> { &self.metadata }
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

/// Read a length-prefixed UTF-8 string (u64 length)
fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
    let len = cursor.read_u64::<LittleEndian>()?;
    let mut buf = vec![0u8; len as usize];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

/// Read a metadata value (primitive or recursively for arrays)
fn read_value(cursor: &mut Cursor<&[u8]>) -> Result<Value> {
    let val_type: MetadataValueType = cursor.read_u32::<LittleEndian>()?.try_into()?;
    //    let val = read_metadata_val(&mut cursor);
    let val = match val_type {
        MetadataValueType::Uint8 => Value::Uint8(cursor.read_u8()?),
        MetadataValueType::Int8 => Value::Int8(cursor.read_i8()?),
        MetadataValueType::Uint16 => Value::Uint16(cursor.read_u16::<LittleEndian>()?),
        MetadataValueType::Int16 => Value::Int16(cursor.read_i16::<LittleEndian>()?),
        MetadataValueType::Uint32 => Value::Uint32(cursor.read_u32::<LittleEndian>()?),
        MetadataValueType::Int32 => Value::Int32(cursor.read_i32::<LittleEndian>()?),
        MetadataValueType::Float32 => Value::Float32(cursor.read_f32::<LittleEndian>()?),
        MetadataValueType::Bool => Value::Bool(cursor.read_u8()? == 1),
        MetadataValueType::String => Value::String(read_string(cursor)?),
        MetadataValueType::Array => {
            // Recursive handling!
            let inner_elem_type: MetadataValueType = cursor.read_u32::<LittleEndian>()?.try_into()?;
            let inner_array_len = cursor.read_u64::<LittleEndian>()?;
            //  println!("[DEBUG] inner_array_len: {:#?}", inner_array_len);
            let mut inner_arr = Vec::new();
            for _ in 0..inner_array_len {
                // Recursive call: match on inner_elem_type again
                let inner_item = match inner_elem_type {
                    MetadataValueType::Uint8 => Value::Uint8(cursor.read_u8()?),
                    MetadataValueType::Int8 => Value::Int8(cursor.read_i8()?),
                    MetadataValueType::Uint16 => Value::Uint16(cursor.read_u16::<LittleEndian>()?),
                    MetadataValueType::Int16 => Value::Int16(cursor.read_i16::<LittleEndian>()?),
                    MetadataValueType::Uint32 => Value::Uint32(cursor.read_u32::<LittleEndian>()?),
                    MetadataValueType::Int32 => Value::Int32(cursor.read_i32::<LittleEndian>()?),
                    MetadataValueType::Float32 => Value::Float32(cursor.read_f32::<LittleEndian>()?),
                    MetadataValueType::Bool => Value::Bool(cursor.read_u8()? != 0),
                    MetadataValueType::String => Value::String(read_string(cursor)?),
                    MetadataValueType::Uint64 => Value::Uint64(cursor.read_u64::<LittleEndian>()?),
                    MetadataValueType::Int64 => Value::Int64(cursor.read_i64::<LittleEndian>()?),
                    MetadataValueType::Float64 => Value::Float64(cursor.read_f64::<LittleEndian>()?),
                    _ => return Err(anyhow!("[ERROR] unsupported nested array element")),
                };    
                inner_arr.push(inner_item);
            }
            Value::Array(inner_arr)
        },
        MetadataValueType::Uint64 => Value::Uint64(cursor.read_u64::<LittleEndian>()?),
        MetadataValueType::Int64 => Value::Int64(cursor.read_i64::<LittleEndian>()?),
        MetadataValueType::Float64 => Value::Float64(cursor.read_f64::<LittleEndian>()?),
    };
    

    Ok(val)
}

struct ParserV2;
impl GgufParser for ParserV2 {
    fn version(&self) -> GgufVersion { 
        GgufVersion::V2
    } // todo: move this an enum, no hardcoded magic symbols
    fn parse<'b>(&self, cursor: &mut Cursor<&'b [u8]>) -> Result<GgufBody> {
        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        debug!("tensor_count: {:#?}", tensor_count);
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()?;
        debug!("metadata_kv_count: {:#?}", metadata_kv_count);
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(cursor)?;
            let val = read_value(cursor)?;
            //debug!("key: {:#?} {:#?}", key, val);
            metadata.insert(key, val);
        }

        println!("metadata: {:#?}", metadata.len());

        // parse V2 tensor descriptors into `tensors`
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for _ in 0..tensor_count {
            // 1) name
            let name = read_string(cursor)?;
            // 2) dims and shape
            let dims = cursor.read_u32::<LittleEndian>()? as usize;
            let mut shape = Vec::with_capacity(dims);
            for _ in 0..dims {
                shape.push(cursor.read_u64::<LittleEndian>()?);
            }
            // 3) kind, offset
            let kind = cursor.read_u32::<LittleEndian>()?;
            let offset = cursor.read_u64::<LittleEndian>()?;
            // 4) infer type size via GGMLType or block_size logic
            let block_size = match kind {
                k if k < 2 => 1,
                k if k < 10 => 32,
                _ => 256,
            };
            let ggml_type: GGMLType = kind.try_into()?;
            let type_size = match ggml_type {
                GGMLType::F32 => 4,
                GGMLType::F16 => 2,
                GGMLType::Q4_0 => 2 + block_size/2,
                GGMLType::Q4_1 => 2 + 2 + block_size/2,
                GGMLType::Q5_0 => 2 + 4 + block_size/2,
                GGMLType::Q5_1 => 2 + 2 + 4 + block_size/2,
                GGMLType::Q8_0 => 2 + block_size,
                GGMLType::Q8_1 => 4 + 4 + block_size,
                GGMLType::Q2K => block_size/16 + block_size/4 + 2 + 2,
                GGMLType::Q3K => block_size/8 + block_size/4 + 12 + 2,
                GGMLType::Q4K => 2 + 2 + 12 + block_size/2,
                GGMLType::Q5K => 2 + 2 + 12 + block_size/8 + block_size/2,
                GGMLType::Q6K => block_size/2 + block_size/4 + block_size/16 + 2,
                _ => return Err(anyhow!("unsupported GGMLType {:?}", ggml_type)),
            };
            let parameters: u64 = shape.iter().product();
            let size = parameters * type_size as u64 / block_size as u64;
            let tensor = Tensor { name: name.clone(), kind, offset, size, shape };
            debug!("[DEBUG] tensors hell yeah: {:#?}", tensor);

            tensors.insert(name, tensor);
        }


        Ok(GgufBody {
            header: Header { tensor_count, metadata_kv_count },
            metadata,
            tensors,
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


    Ok(ParsedGGUF { header: body.header, metadata: body.metadata, tensors: body.tensors, mmap })

    //let parsed = ParsedGGUF {
    //    header: Header::default(),
    //    metadata: HashMap::new(),
    //    tensors: HashMap::new(),
    //    mmap,
    //};

    //Ok(parsed)
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
    let gguf  = load_model(&path);
    debug!("[DEBUG] gguf: {:#?}", gguf);

    Ok(())
}

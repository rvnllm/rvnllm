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
use crate::cli_commands::DumpFormat;
use serde_json::json;
use clap::CommandFactory;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use std::io::{BufWriter, Write};
 use rayon::iter::IntoParallelRefIterator;

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

struct Q40Format;
impl TensorFormat for Q40Format {
    fn id(&self) -> u32 { 2 }
    fn name(&self) -> &'static str{ "Q4_0" }
    fn decode<'a>(&self, bytes: &'a [u8], shape: &[usize]) -> Result<TensorView<'a>> {
        // TODO: proper quantisied code CPU or CUDA --->>> call kernel
        Ok(TensorView { data: bytes, shape: shape.to_vec(), dtype: TensorDType::Q4_0 })
    }
}


/// Decode a Q6K tensor into f32 on the CPU
pub fn decode_q6k_cpu(raw: &[u8], shape: &[usize]) -> Vec<f32> {
    debug!("decode_q6k_cpu");
    let n = shape.iter().product::<usize>();
    let mut out = Vec::with_capacity(n);

    // assume header: 4B scale + 4B bias
    // data: 6 bits per element ⇒ 32 elems ⇒ 192 bits ⇒ 24 bytes
    const BLOCK_ELEMS: usize = 32;
    const HEADER_BYTES: usize = 8;
    const DATA_BYTES:   usize = (BLOCK_ELEMS * 6 + 7) / 8;
    const BLOCK_BYTES:  usize = HEADER_BYTES + DATA_BYTES;

    for block in raw.chunks_exact(BLOCK_BYTES) {
        let scale   = f32::from_le_bytes(block[0..4].try_into().unwrap());
        let bias    = f32::from_le_bytes(block[4..8].try_into().unwrap());
        let data    = &block[8..8+DATA_BYTES];
        let mut bitpos = 0;

        for _ in 0..BLOCK_ELEMS {
            // locate the 6 bits
            let byte_idx = bitpos / 8;
            let bit_off  = bitpos % 8;
            let mut v = (data[byte_idx] >> bit_off) as u32;
            if bit_off > 2 {
                // crosses into next byte
                let next = data[byte_idx + 1] as u32;
                v |= next << (8 - bit_off);
            }
            v &= 0x3F; // mask lower 6 bits
            out.push(v as f32 * scale + bias);
            bitpos += 6;
            if out.len() == n { break; }
        }
        if out.len() == n { break; }
    }
    out
}


//struct Q4KSFormat;
//impl TensorFormat for Q4KSFormat {
//    fn id(&self) -> u32 { 14 }
//    fn name(&self) -> &'static str{ "Q4_K_S" }
//    fn decode<'a>(&self, bytes: &'a [u8], shape: &[usize]) -> Result<TensorView<'a>> {
//        let decoded = decode_q6k_cpu(bytes, &shape.iter().map(|&d| d as u64).collect::<Vec<_>>());
//        let ptr = decoded.as_ptr();
//        let len = decoded.len();
//        std::mem::forget(decoded); // We’re leaking for zero-copy view
//        Ok(TensorView {
//            data: unsafe { std::slice::from_raw_parts(ptr as *const u8, len * 4) },
//            shape: shape.to_vec(),
//            dtype: TensorDType::F32,
//        })
//
//    }
//}
//Fixes the shape parameter type to &[u64], matching TensorFormat trait’s signature, so no longer have to cast or collect into a Vec<u64> at each call.
//Eliminates the manual leak (mem::forget) by returning an owned byte buffer wrapped in a Cow::Owned. 
//Decode into a Vec<f32>,
// Cast it to bytes via bytemuck::cast_slice into a fresh Vec<u8>,
//Wrap that in Cow::Owned,
//And hand it back in TensorView.data.
//Because TensorView.data now owns its bytes (when needed), Rust will automatically free them when the view goes out of scope—no manual forgetting required.

//use std::borrow::Cow;
pub fn decode_q4ks_cpu(raw: &[u8], shape: &[usize]) -> Vec<f32> {
    let n = shape.iter().product::<usize>();
    let mut out = Vec::with_capacity(n);

    let values_per_block = 64;
    let scale_size = 4 * 2;
    let zp_size = 1;
    let q_bytes = 32;
    let block_size = scale_size + zp_size + q_bytes;

    for block in raw.chunks_exact(block_size) {

        let scale_0 = f32::from_le_bytes(block[0..4].try_into().unwrap());
        let scale_1 = f32::from_le_bytes(block[4..8].try_into().unwrap());
        debug!("scale_0: {}, scale_1: {}", scale_0, scale_1);
        
        let zero_point = block[8] as i32;
        let qdata = &block[9..];

        for (i, &byte) in qdata.iter().enumerate() {
            let lo = byte & 0x0F;
            let hi = byte >> 4;

            let scale = if (2 * i) < values_per_block / 2 { scale_0 } else { scale_1 };

            let val0 = ((lo as i32 - zero_point) as f32) * scale;
            let val1 = ((hi as i32 - zero_point) as f32) * scale;

            out.push(val0);
            out.push(val1);
            if out.len() >= n {
                return out;
            }
        }
    }

    out
}

struct Q4KSFormat;
impl<'a> TensorFormat for Q4KSFormat {
  fn id(&self) -> u32 { 14 }
  fn name(&self) -> &'static str { "Q4_K_S" }

  fn decode<'b>(
    &self,
    bytes: &'b [u8],
    shape: &[usize]
  ) -> Result<TensorView<'b>> {
    // 1) decode to floats on the CPU
    debug!("[DEBUG] Q4KSFormat");
    debug!("raw block: {:?}", &bytes[0..41]);
    let f32vec = decode_q4ks_cpu(bytes, shape);

    // 2) reinterpret as bytes
    let raw_bytes = bytemuck::cast_slice::<f32, u8>(&f32vec).to_vec();

    // 3) wrap in a Cow so the view owns its data
    //let data = Cow::Owned(raw_bytes);
    ////TODO: FIX THIS but cascades eveywhere
    let leaked_slice: &'static [u8] = Box::leak(raw_bytes.into_boxed_slice());
    Ok(TensorView {
        data:   leaked_slice,
        shape:  shape.iter().map(|&d| d as usize).collect(),
        dtype:  TensorDType::F32,
    })

  }
}


struct Q3KMFormat;
impl TensorFormat for Q3KMFormat {
    fn id(&self) -> u32 { 12 }
    fn name(&self) -> &'static str{ "Q3_K_M" }
    fn decode<'a>(&self, bytes: &'a [u8], shape: &[usize]) -> Result<TensorView<'a>> {
        // TODO: proper quantisied code CPU or CUDA --->>> call kernel
        Ok(TensorView { data: bytes, shape: shape.to_vec(), dtype: TensorDType::Q3_K_M })
    }
}

pub fn decode_q2k_cpu(raw: &[u8], shape: &[usize]) -> Vec<f32> {
    debug!("[DEBUG] decode_q2k_cpu");
    let n = shape.iter().product::<usize>();
    let mut out = Vec::with_capacity(n);
    let block_bytes = 4 + 4 + (32/8); // 4B scale + 4B zp + 4B data
    for block in raw.chunks_exact(block_bytes) {
        let scale    = f32::from_le_bytes(block[0..4].try_into().unwrap());
        let zero_pt  = u32::from_le_bytes(block[4..8].try_into().unwrap()) as f32;
        let data     = &block[8..12];
        // unpack 8 values per byte
        for (i, &byte) in data.iter().enumerate() {
            for bit in 0..8 {
                let v2 = (byte >> (bit*2)) & 0x03;
                // dequant: (v2 - zero_pt) * scale
                out.push((v2 as f32 - zero_pt) * scale);
                if out.len() == n { break; }
            }
            if out.len() == n { break; }
        }
    }
    out
}

/*struct Q2KFormat;
impl TensorFormat for Q2KFormat {
    fn id(&self) -> u32 { 10 }
    fn name(&self) -> &'static str{ "Q2_K" }
    fn decode<'a>(&self, bytes: &'a [u8], shape: &[usize]) -> Result<TensorView<'a>> {
        let decoded = decode_q2k_cpu(bytes, &shape.iter().map(|&d| d as u64).collect::<Vec<_>>());
        let ptr = decoded.as_ptr();
        let len = decoded.len();
        std::mem::forget(decoded); // We’re leaking for zero-copy view
        Ok(TensorView {
            data: unsafe { std::slice::from_raw_parts(ptr as *const u8, len * 4) },
            shape: shape.to_vec(),
            dtype: TensorDType::F32,
        })
    }
}*/
//Fixes the shape‐type mismatch
//Drops the mem::forget leak
//Keeps zero‐copy semantics from the user’s perspective (they still get a [u8] slice), but now it’s safely owned and freed when the view is dropped.
struct Q2KFormat;
impl TensorFormat for Q2KFormat {
    fn id(&self) -> u32 { 10 }
    fn name(&self) -> &'static str { "Q2_K" }

    fn decode<'a>(
        &self,
        bytes: &'a [u8],
        shape: &[usize],
    ) -> Result<TensorView<'a>> {
        // 1) Decode to Vec<f32>
        let f32vec = decode_q2k_cpu(bytes, shape);

        // 2) Cast f32 buffer into u8 bytes
        let raw_bytes = bytemuck::cast_slice::<f32, u8>(&f32vec).to_vec();

        // 3) Leak the Vec<u8> so we can hand back &'static [u8]
        let leaked_slice: &'static [u8] =
            Box::leak(raw_bytes.into_boxed_slice());

        // 4) Return the view as usual
        Ok(TensorView {
            data:   leaked_slice,
            shape:  shape.iter().map(|&d| d as usize).collect(),
            dtype:  TensorDType::F32,
        })
    }
}

struct Q6KFormat;
impl TensorFormat for Q6KFormat {
    fn id(&self) -> u32 { 18 }
    fn name(&self) -> &'static str{ "Q2_K" }
    fn decode<'a>(&self, bytes: &'a [u8], shape: &[usize]) -> Result<TensorView<'a>> {
        // TODO: proper quantisied code CPU or CUDA --->>> call kernel
        Ok(TensorView { data: bytes, shape: shape.to_vec(), dtype: TensorDType::Q6_K })
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
    pub fn raw_bytes(&self) -> &[u8] {
        &self.mmap[..]
    }
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

////
pub fn parse_metadata_common(cursor: &mut Cursor<&[u8]>, metadata_kv_count: u64) -> Result<HashMap<String, Value>> {
    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_string(cursor)?;
        let val = read_value(cursor)?;
        //debug!("key: {:#?} {:#?}", key, val);
        metadata.insert(key, val);
    }
    Ok(metadata)
}

pub fn parse_tensors_common(cursor: &mut Cursor<&[u8]>, tensor_count: u64) -> Result<HashMap<String, Tensor>> {
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
    Ok(tensors)
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
        debug!("tensor_count: {}", tensor_count);
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()?;
        debug!("metadata_kv_count: {:#?}", metadata_kv_count);

        let metadata = parse_metadata_common(cursor, metadata_kv_count)?;
        let tensors = parse_tensors_common(cursor,tensor_count)?;
 
        Ok(GgufBody {
            header: Header {
                tensor_count,
                metadata_kv_count,
            },
            metadata,
            tensors,
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
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()?;
        debug!("metadata_kv_count: {:#?}", metadata_kv_count);

        let metadata = parse_metadata_common(cursor, metadata_kv_count)?;
        let tensors = parse_tensors_common(cursor,tensor_count)?;
    
        Ok(GgufBody {
            header: Header { tensor_count, metadata_kv_count },
            metadata,
            tensors,
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
        m.insert(0,     Box::new(F32Format) as Box<dyn TensorFormat>);
        m.insert(2,     Box::new(Q40Format) as Box<dyn TensorFormat>);
        m.insert(10,    Box::new(Q2KFormat) as Box<dyn TensorFormat>);
        m.insert(12,    Box::new(Q3KMFormat) as Box<dyn TensorFormat>);
        m.insert(14,    Box::new(Q4KSFormat) as Box<dyn TensorFormat>);
        m.insert(18,    Box::new(Q6KFormat) as Box<dyn TensorFormat>);
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

pub fn collect_tensors_by_suffix<'a>(
    tensors: &'a HashMap<String, Tensor>,
    suffix: &str,
) -> Vec<(&'a String, &'a Tensor)> {
    tensors
        .iter()
        .filter(|(k, _)| k.ends_with(suffix))
        .collect()
}

pub fn resolve_tensor_slice<'a>(view: &'a TensorView) -> Result<&'a [u8]> {
    use TensorDType::*;
    match view.dtype {
        F32 => Ok(bytemuck::cast_slice(view.as_f32_slice()?)),
        Q4_0 => view.as_q4_0_slice(),
        Q4_1 => view.as_q4_1_slice(),
        Q5_0 => view.as_q5_0_slice(),
        Q5_1 => view.as_q5_1_slice(),
        Q8_0 => view.as_q8_0_slice(),
        Q8_1 => view.as_q8_1_slice(),
        Q2_K => view.as_q2_k_slice(),
        Q3_K_S => view.as_q3_k_s_slice(),
        Q3_K_M => view.as_q3_k_m_slice(),
        Q3_K_L => view.as_q3_k_l_slice(),
        Q4_K_S => view.as_q4_k_s_slice(),
        Q4_K_M => view.as_q4_k_m_slice(),
        Q5_K_S => view.as_q5_k_s_slice(),
        Q5_K_M => view.as_q5_k_m_slice(),
        Q6_K => view.as_q6_k_slice(),
        _ => bail!("Unsupported dtype for dispatch: {:?}", view.dtype),
    }
}

fn main() -> anyhow::Result<()> 
{
    env_logger::init();
    println!("main");
    check_debug_dev_sanity!();


    let cli = RvnCli::parse();

    match cli.command {
        Command::List { file, .. } => {
            let gguf = load_model(&file)?;
            println!("Tensor count: {}", gguf.tensors.len());
            for (name, tensor) in &gguf.tensors {
                println!("{} => shape: {:?}", name, tensor.shape);
            }
        }

        Command::ForwardSimple { file, q, k, v } => {
            let gguf = load_model(&file)?;
            let qv = gguf.tensor_view(&q)?.as_f32_slice()?.to_vec();
            let kv = gguf.tensor_view(&k)?.as_f32_slice()?.to_vec();
            let vv = gguf.tensor_view(&v)?.as_f32_slice()?.to_vec();

            let d_k = qv.len();
            let n_tokens = kv.len() / d_k;
            let mut scores = vec![0.0f32; n_tokens];
            for i in 0..n_tokens {
                let start = i * d_k;
                let end = start + d_k;
                scores[i] = qv.iter()
                    .zip(&kv[start..end])
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
            }
            let scale = 1.0 / (d_k as f32).sqrt();
            for s in &mut scores { *s *= scale; }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter_mut().map(|s| {*s = (*s - max).exp(); *s}).sum();
            for s in &mut scores { *s /= sum; }

            let d_v = vv.len() / n_tokens;
            let mut output = vec![0.0f32; d_v];
            for i in 0..n_tokens {
                for j in 0..d_v {
                    output[j] += scores[i] * vv[i * d_v + j];
                }
            }
            println!("Attention output: {:?}", &output[..output.len().min(10)]);
        }
        
        Command::Info(cmd) => {
            let gguf = load_model(&cmd.file)?;
            if cmd.header {
                println!("Header: {:?}", gguf.header);
            }
            if cmd.metadata {
                println!("Metadata:");
                for (k, v) in gguf.metadata() {
                    if let Some(key) = k.strip_prefix("tokenizer") {
                        debug!("[VERBOSE] tokenizer metadata: {} => {:?}", key, v);
                        continue;
                    }

                    match v {
                        Value::String(s) => println!("  {}: \"{}\"", k, s),
                        Value::Uint32(n) => println!("  {}: {}", k, n),
                        Value::Float32(f) => println!("  {}: {:.6}", k, f),
                        _ => println!("  {}: {:?}", k, v),
                    }
                }

            }
            if let Some(name) = cmd.tensor {
                if let Some(tensor) = gguf.tensor(&name) {
                    println!("Tensor '{}': shape = {:?}, kind = {}", name, tensor.shape, tensor.kind);
                } else {
                    eprintln!("Tensor '{}' not found", name);
                }
            }
        }

        Command::Dump { file, name, format, .. } => {
            let gguf = load_model(&file)?;
            let view = gguf.tensor_view(&name)?;
            match format {
                DumpFormat::Shape => {
                    println!("Shape: {:?}", view.shape);
                }
                DumpFormat::F32 => {
                    let f32s = view.as_f32_slice()?;
                    println!("{:?}", &f32s[..f32s.len().min(10)]);
                }
                DumpFormat::Raw => {
                    println!("Raw bytes: {:?}", &view.data[..view.data.len().min(10)]);
                }
                DumpFormat::Json => {
                    println!("{{ \"shape\": {:?} }}", view.shape);
                }
            }
        }

        Command::Analyze { file, .. } => {
            let gguf = load_model(&file)?;
            println!("Tensors: {}", gguf.tensors.len());
            let total_bytes: u64 = gguf.tensors.values().map(|t| t.size).sum();
            println!("Total tensor size: {} bytes", total_bytes);
        }

        Command::Validate { file, profile, .. } => {
            let gguf = load_model(&file)?;
            println!("Validating '{}' with profile {:?}" , file, profile);
            // Placeholder: real validation logic
            println!("OK (stub) — no issues found.");
        }

        // Placeholder: remaining commands to be implemented
        Command::Forward(_) => {
            println!("[TODO] Forward pass not implemented yet");
        }

        Command::Diff { file_a, file_b, .. } => {
            println!("[TODO] Diff '{}' vs '{}' not implemented yet", file_a, file_b);
        }

        Command::Profile { file, .. } => {
            println!("[TODO] Profile not implemented yet for file: {}", file);
        }

        Command::Watch { file, .. } => {
            println!("[TODO] Watch not implemented yet for file: {}", file);
        }

        Command::WatchPerf { file, .. } => {
            println!("[TODO] WatchPerf not implemented yet for file: {}", file);
        }
 
        Command::Debug { file, threads, output } => {
            if let Some(n) = threads {
                let _ = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build_global();
            }
            let gguf = load_model(&file)?;
            let mut entries: Vec<_> = gguf.iter().collect();
            entries.sort_by_key(|(_, t)| t.offset);

            let mut writer: Box<dyn Write + Send> = match output {
                Some(path) => Box::new(BufWriter::new(File::create(path)?)),
                None => Box::new(std::io::stdout()),
            };

            writeln!(writer, "==[ GGUF Dump ]==")?;

            let results: Vec<String> = entries.par_iter().map(|(name, tensor)| {
                let mut buf = String::new();
                use std::fmt::Write as _;
                writeln!(buf, "  [{}]:", name).ok();
                writeln!(buf, "    kind: {}", tensor.kind).ok();
                writeln!(buf, "    offset: {}", tensor.offset).ok();
                writeln!(buf, "    size: {}", tensor.size).ok();
                writeln!(buf, "    shape: {:?}", tensor.shape).ok();

                if tensor.size < 1024 * 4 {
                    if let Ok(view) = tensor.view(gguf.raw_bytes()) {
                        if view.dtype == TensorDType::F32 {
                            if let Ok(slice) = view.as_f32_slice() {
                                let preview: Vec<_> = slice.iter().take(10).cloned().collect();
                                writeln!(buf, "    preview: {:?}", preview).ok();
                            }
                        }
                    }
                }
                buf
            }).collect();

            for line in results {
                writer.write_all(line.as_bytes())?;
            }
            writer.flush()?;
        }
        Command::DecodeTest { file, name, verbose, json, fail_on_anomaly } => {
            let gguf = load_model(&file)?;
            let view = gguf.tensor_view(&name)?;
            let f32s = view.as_f32_slice()?;

            let (nan_count, inf_count, zeros, min, max, sum) = f32s.iter().fold(
                (0, 0, 0, f32::MAX, f32::MIN, 0.0f64),
                |(n, i, z, min, max, sum), &val| {
                    (
                        n + val.is_nan() as usize,
                        i + (val.is_infinite() as usize),
                        z + (val == 0.0) as usize,
                        if val.is_nan() { min } else { val.min(min) },
                        if val.is_nan() { max } else { val.max(max) },
                        if val.is_finite() { sum + val as f64 } else { sum }
                    )
                },
            );

            let mean = sum / f32s.len() as f64;

            if json {
                let out = json!({
                    "tensor": name,
                    "shape": view.shape,
                    "dtype": format!("{:?}", view.dtype),
                    "nan": nan_count,
                    "inf": inf_count,
                    "zeros": zeros,
                    "min": min,
                    "max": max,
                    "mean": mean,
                    "preview": &f32s[..f32s.len().min(10)]
                });
                println!("{}", out);
            } else {
                println!("Tensor '{}': shape: {:?}, dtype: {:?}", name, view.shape, view.dtype);
                println!("Element count: {}", view.num_elements());
                println!("Byte length: {} (expected {})", view.data.len(), view.expected_byte_len());

                if view.expected_byte_len() != view.data.len() {
                    println!("[WARN] Byte length mismatch");
                }

                if nan_count > 0 || inf_count > 0 {
                    println!("[ERROR] Tensor contains {} NaN and {} Inf values", nan_count, inf_count);
                } else {
                    println!("[OK] No NaN or Inf values found");
                }

                println!("Zero values: {} ({:.2}%)", zeros, (zeros as f64 / f32s.len() as f64) * 100.0);
                println!("Min: {:.6}, Max: {:.6}, Mean: {:.6}", min, max, mean);
                println!("Preview: {:?}", &f32s[..f32s.len().min(if verbose { 20 } else { 10 })]);
            }

            if fail_on_anomaly && (nan_count > 0 || inf_count > 0) {
                std::process::exit(1);
            }

        }
    }

    Ok(())
}

//let cli = RvnCli::parse();
    
//    let path = std::env::args()
  //      .nth(1)
    //    .expect("usage: rvnllm <path/to/model.gguf>");
//    println!("[DEBUG] path: {:#?}", path);
    
    //let _file = File::open("../model/llama-2-13b-ensemble-v5.Q6_K.gguf")?;
  //  let gguf  = load_model(&path)?;

    //pub struct ParsedGGUF {
    // public 
    //    pub header: Header,
    //    pub metadata: HashMap<String, Value>,
    //    pub tensors: HashMap<String, Tensor>,
        // private
    //    mmap: Mmap,        // ->>> not exposed private field
    //}
    
//    for (key, tensor) in &gguf.tensors {
  //      println!("{} => shape: {:?}", key, tensor.shape);
   // }
  //  let q_weights = collect_tensors_by_suffix(&gguf.tensors, "attn_q.weight");
    //let ffns = collect_tensors_by_suffix(&gguf.tensors, "ffn_norm.weight");

    //for (name, tensor) in q_weights {
    //    println!("Found Q: {} => {:?}", name, tensor.shape);
    //}
 //   let q_view = gguf.tensor_view("blk.0.attn_q.weight")?;
   // let k_view = gguf.tensor_view("blk.0.attn_k.weight")?;
   // let v_view = gguf.tensor_view("blk.0.attn_v.weight")?;
    
  //  let q_view_slice = resolve_tensor_slice(&q_view)?;

    //let d_model = q_view.shape[0];
//    let input = vec![1.0f32; d_model]; // mock input, can be random or fixed
   // let input = vec![1.0f32 / (d_model as f32).sqrt(); d_model];
  /*  let input_view = TensorView {
        data: unsafe {
            std::slice::from_raw_parts(input.as_ptr() as *const u8, d_model * 4)
        },
        shape: vec![1, d_model],
        dtype: TensorDType::F32,
    };
    let d_k = q_view.shape[1];
    let d_v = v_view.shape[1];

    let mut q_proj = vec![0.0f32; d_k];
    let mut k_proj = vec![0.0f32; d_k];
    let mut v_proj = vec![0.0f32; d_v];

    matmul(&input_view, &q_view, &mut q_proj)?;
    matmul(&input_view, &k_view, &mut k_proj)?;
    matmul(&input_view, &v_view, &mut v_proj)?;
    
    let q_proj_view = TensorView {
        data: unsafe { std::slice::from_raw_parts(q_proj.as_ptr() as *const u8, q_proj.len() * 4) },
        shape: vec![1, d_k],
        dtype: TensorDType::F32,
    };
    let k_proj_view = TensorView {
        data: unsafe { std::slice::from_raw_parts(k_proj.as_ptr() as *const u8, k_proj.len() * 4) },
        shape: vec![1, d_k],
        dtype: TensorDType::F32,
    };
    let v_proj_view = TensorView {
        data: unsafe { std::slice::from_raw_parts(v_proj.as_ptr() as *const u8, v_proj.len() * 4) },
        shape: vec![1, d_v],
        dtype: TensorDType::F32,
    };
    let v_data = v_proj_view.as_f32_slice()?;
    debug!("v vector: {:?}", &v_data[..10]);
    
    let mut output = vec![0.0f32; d_v];
    attention_forward(&q_proj_view, &k_proj_view, &v_proj_view, &mut output)?;

    //let f32s: &[f32] = bytemuck::cast_slice(q_view_slice);


    println!("output {:#?}", &output[..10]);
    

    //println!("[DEBUG] gguf: {:#?}", gguf);*/



// recap
// Tensor data length mismnatch issue
// The Suspects
// Wrong number of decoded values (out.len())
// → maybe the block decoder isn't covering the entire tensor -> implement checks here

// Wrong expected shape
// → shape is [n, d], but decoder outputs fewer than n*d -> implement checks

// Cast bug in bytemuck or manual slice logic -> get rid off it, extra layer of unecessary
// complexity
// → handing off too few or too many bytes

// Improper block size assumptions
// → maybe Q4_K_S uses 128 or 256-value blocks, not 64?

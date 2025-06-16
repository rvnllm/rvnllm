pub mod types;

use crate::types::{GGMLType, MetadataValueType};
use anyhow::{Context, Result, anyhow, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use log::debug;
use memmap2::Mmap;
use rvn_core_tensor::{Header, Tensor, TensorKind};
use rvn_globals::types::Value;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

#[derive(Serialize)]
pub struct InfoDump<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<&'a Header>,

    //    #[serde(skip_serializing_if = "Option::is_none")]
    //  pub metadata:  Option<&'a HashMap<String, Value>>,
    // (key, value) pairs that survived the filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<(&'a str, &'a Value)>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensors: Option<Vec<(&'a str, &'a Tensor)>>, // nicer key type
}

#[derive(Default, Debug, Serialize)]
pub struct ParsedGGUF {
    // public
    pub header: Header,
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, Tensor>,
}
impl ParsedGGUF {
    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.metadata
    }
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Tensor)> {
        self.tensors.iter()
    }
}
// Optionally, give yourself a shorter alias:
impl ParsedGGUF {
    /// Temporary stand-in while parsing logic is under construction.
    pub fn mock() -> Self {
        Self::default()
    }
}

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

pub struct GgufBody {
    pub header: Header,
    pub metadata: HashMap<String, Value>, // now typed values
    pub tensors: HashMap<String, Tensor>,
}

// ----------------------- PARSER STUFF
static V2: ParserV2 = ParserV2;
static V3: ParserV3 = ParserV3;

static PARSER_REGISTRY: &[&dyn GgufParser] = &[&V2, &V3];

fn parser_for(version: GgufVersion) -> Option<&'static dyn GgufParser> {
    PARSER_REGISTRY
        .iter()
        .copied() // Option<&&dyn _> → Option<&dyn _>
        .find(|p| p.version() == version)
}

trait GgufParser: Send + Sync {
    fn version(&self) -> GgufVersion;
    fn parse(&self, cur: &mut Cursor<&[u8]>) -> Result<GgufBody>;
}
struct ParserV3;
impl GgufParser for ParserV3 {
    fn version(&self) -> GgufVersion {
        GgufVersion::V3
    } // todo: move this an enum, no hardcoded magic symbols

    fn parse(&self, cursor: &mut Cursor<&[u8]>) -> Result<GgufBody> {
        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        debug!("tensor_count: {}", tensor_count);
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()?;
        debug!("metadata_kv_count: {:#?}", metadata_kv_count);

        let metadata = parse_metadata_common(cursor, metadata_kv_count)?;
        let tensors = parse_tensors_common(cursor, tensor_count)?;

        Ok(GgufBody {
            header: Header {
                version: 3,
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
    fn parse(&self, cursor: &mut Cursor<&[u8]>) -> Result<GgufBody> {
        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        debug!("tensor_count: {:#?}", tensor_count);
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()?;
        debug!("metadata_kv_count: {:#?}", metadata_kv_count);

        let metadata = parse_metadata_common(cursor, metadata_kv_count)?;
        let tensors = parse_tensors_common(cursor, tensor_count)?;

        Ok(GgufBody {
            header: Header {
                version: 2,
                tensor_count,
                metadata_kv_count,
            },
            metadata,
            tensors,
        })
    }
}

/// Read length-prefixed UTF-8 string (u64 length)
fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
    let len = cursor.read_u64::<LittleEndian>()?;
    let mut buf = vec![0u8; len as usize];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

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
            let inner_elem_type: MetadataValueType =
                cursor.read_u32::<LittleEndian>()?.try_into()?;
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
                    MetadataValueType::Float32 => {
                        Value::Float32(cursor.read_f32::<LittleEndian>()?)
                    }
                    MetadataValueType::Bool => Value::Bool(cursor.read_u8()? != 0),
                    MetadataValueType::String => Value::String(read_string(cursor)?),
                    MetadataValueType::Uint64 => Value::Uint64(cursor.read_u64::<LittleEndian>()?),
                    MetadataValueType::Int64 => Value::Int64(cursor.read_i64::<LittleEndian>()?),
                    MetadataValueType::Float64 => {
                        Value::Float64(cursor.read_f64::<LittleEndian>()?)
                    }
                    _ => return Err(anyhow!("[ERROR] unsupported nested array element")),
                };
                inner_arr.push(inner_item);
            }
            Value::Array(inner_arr)
        }
        MetadataValueType::Uint64 => Value::Uint64(cursor.read_u64::<LittleEndian>()?),
        MetadataValueType::Int64 => Value::Int64(cursor.read_i64::<LittleEndian>()?),
        MetadataValueType::Float64 => Value::Float64(cursor.read_f64::<LittleEndian>()?),
    };

    Ok(val)
}

pub fn parse_metadata_common(
    cursor: &mut Cursor<&[u8]>,
    metadata_kv_count: u64,
) -> Result<HashMap<String, Value>> {
    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_string(cursor)?;
        let val = read_value(cursor)?;
        //debug!("key: {:#?} {:#?}", key, val);
        metadata.insert(key, val);
    }
    Ok(metadata)
}

pub fn parse_tensors_common(
    cursor: &mut Cursor<&[u8]>,
    tensor_count: u64,
) -> Result<HashMap<String, Tensor>> {
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
        let kind_raw = cursor.read_u32::<LittleEndian>()?;
        let kind = TensorKind::try_from(kind_raw)?;

        let _offset = cursor.read_u64::<LittleEndian>()?;

        // 4) infer type size via GGMLType or block_size logic
        let block_size = match kind_raw {
            k if k < 2 => 1,
            k if k < 10 => 32,
            _ => 256,
        };
        let ggml_type: GGMLType = kind_raw.try_into()?;
        let type_size = match ggml_type {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::Q4_0 => 2 + block_size / 2,
            GGMLType::Q4_1 => 2 + 2 + block_size / 2,
            GGMLType::Q5_0 => 2 + 4 + block_size / 2,
            GGMLType::Q5_1 => 2 + 2 + 4 + block_size / 2,
            GGMLType::Q8_0 => 2 + block_size,
            GGMLType::Q8_1 => 4 + 4 + block_size,
            GGMLType::Q2K => block_size / 16 + block_size / 4 + 2 + 2,
            GGMLType::Q3K => block_size / 8 + block_size / 4 + 12 + 2,
            GGMLType::Q4K => 2 + 2 + 12 + block_size / 2,
            GGMLType::Q5K => 2 + 2 + 12 + block_size / 8 + block_size / 2,
            GGMLType::Q6K => block_size / 2 + block_size / 4 + block_size / 16 + 2,
            _ => return Err(anyhow!("unsupported GGMLType {:?}", ggml_type)),
        };
        let parameters: u64 = shape.iter().product();
        let size = parameters * type_size as u64 / block_size as u64;
        let tensor = Tensor {
            name: name.clone(),
            kind,
            size,
            shape,
        };
        debug!("[DEBUG] tensors hell yeah: {:#?}", tensor);

        tensors.insert(name, tensor);
    }
    Ok(tensors)
}

pub fn load_model<P: AsRef<Path>>(path: P) -> anyhow::Result<ParsedGGUF> {
    debug!("[DEBUG] Loading model {:#?}", path.as_ref().display());

    // 1) open & mmap
    let file =
        File::open(&path).with_context(|| format!("cannot open '{}'", path.as_ref().display()))?;

    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("cannot mmap '{}'", path.as_ref().display()))?;

    debug!("MMAP created: len = {}", mmap.len());

    // 2) magic & version
    let mut cursor = Cursor::new(&mmap[..]);
    debug!("Cursor position: {}", cursor.position());
    debug!("Cursor peek: {:?}", &mmap[0..16.min(mmap.len())]); // show header bytes
    const MAGIC: u32 = 0x4655_4747; //gguf
    let magic = cursor.read_u32::<LittleEndian>()?;
    if magic != MAGIC {
        bail!("[ERROR] invalid GGUF magic 0x{magic:08X}");
    } // todo: check magic command for the validity checks
    let version_num = cursor.read_u32::<LittleEndian>()?;
    let version = GgufVersion::try_from(version_num)?;

    debug!("Version: {:#?}", version);

    // 3) parse body
    debug!("Looking up parser for version {:#?}", version);
    for p in PARSER_REGISTRY.iter() {
        // `.iter()` yields `&&dyn`; `.copied()` drops one `&`
        debug!("Registered parser version: {:?}", p.version());
    }

    let parser = parser_for(version).ok_or_else(|| {
        debug!("No parser found for version {:#?}", version);
        anyhow!("unsupported GGUF version {:#?}", version)
    })?;

    debug!("Parser: {:#?}", parser.version());
    // unchanged up to parsing
    let body = parser.parse(&mut cursor)?;

    Ok(ParsedGGUF {
        header: body.header,
        metadata: body.metadata,
        tensors: body.tensors,
    })
}

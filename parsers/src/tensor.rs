use log::info;
use std::collections::HashMap;
use anyhow::{bail, anyhow, Result};
use memmap2::Mmap;
use std::path::Path;
use once_cell::sync::{Lazy, OnceCell};
use crate::types::{Value, MetadataValueType} ; 
use ctensor::tensor_view::{TensorView, TensorDType};
use log::debug;


/****************************
 * Tensor Formats -> aka what quantization format are we talking 
 * about Q4 Q3 etc
 * The trait defines the structure implementations are the various
 * types
 * TensorFormat knows how to decode itself
 * later maybe add device specific stuff, right now only F32 is supported
 * Q40 is working
 * Todo: Full qunatization support. Sprint 2 day 4-5 
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

/*****************************
 * Tensor registry
 */
static TENSOR_REGISTRY: Lazy<OnceCell<HashMap<u32, Box<dyn TensorFormat>>>> = Lazy::new(OnceCell::default);
fn register_tensor_registry_formats() {
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

fn get_format(kind: u32) -> Option<&'static Box<dyn TensorFormat>> {
    TENSOR_REGISTRY.get().and_then(|m| m.get(&kind))
}

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

/* -------------------------------------------------------------------------------------- */
/**
 * parsed GGUF structure holding information about the header, metadata and 
 * the tensors, the tensors are zero copy, only offsets are stored into the mmap
 * <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md */
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

pub fn load_model<P: AsRef<Path>>(path: P) -> anyhow::Result<ParsedGGUF>
{
    info!("[DEBUg] Loading model {:#?}", path.as_ref().display());

    register_tensor_registry_formats();

    let parsed = ParsedGGUF {
        header: Header::default(),
        metadata: HashMap::new(),
        tensors: HashMap::new(),
        mmap,
    };
    
    Ok(parsed)
}

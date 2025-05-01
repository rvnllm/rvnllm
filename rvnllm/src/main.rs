use std::{collections::HashMap, fs::File, hash::Hash};
use std::path::Path;
use ctensor::tensor_view::{TensorView, TensorDType};
use memmap2::Mmap;
use once_cell::sync::{Lazy, OnceCell};
use anyhow::{anyhow, bail, Context, Result};


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

/**
 * ParsedGGUF supporting structures */
pub struct Tensor {
    pub name: String,
    pub kind: u32,
    pub offset: u64,
    pub size: u64,
    pub shape: Vec<u64>,
}

pub struct Header {
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
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

struct Q4_0;
impl TensorFormat for Q4_0 {
    fn id(&self) -> u32 { 2 }
    fn name(&self) -> &'static str{ "Q4_0" }
    fn decode<'a>(&self, bytes: &'a [u8], shape: &[usize]) -> Result<TensorView<'a>> {
        // TODO: proper quantisied code CPU or CUDA --->>> call kernel
        Ok(TensorView { data: bytes, shape: shape.to_vec(), dtype: TensorDType::Q4_0 })
    }
}
//... implementation for the other tensor formats


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


static TENSOR_REGISTRY: Lazy<OnceCell<HashMap<u32, Box<dyn TensorFormat>>>> = Lazy::new(OnceCell::default);
fn register_builtin_formats() {
//    TENSOR_REGISTRY.get
}

pub fn load_model<P: AsRef<Path>>(path: P) //-> Result<ParsedGGUF> 
{
    register_builtin_formats();
    //Ok(())
}

fn main() -> anyhow::Result<()> 
{
    println!("main");
    check_debug_dev_sanity!();
    
    let path = std::env::args()
        .nth(1)
        .expect("usage: rvnllm <path/to/model.gguf>");
    
    //let _file = File::open("../model/llama-2-13b-ensemble-v5.Q6_K.gguf")?;
    /*let (tensor_map, mmap) =*/ load_model(&path);


    Ok(())
}

use smallvec::SmallVec;

#[derive(Debug)]
pub struct Tensor {
    pub name: String,
    pub kind: u32,
    pub size: u64,
    pub shape: Vec<u64>,
}

pub type ShapeBuf = SmallVec<[u64; 6]>;

#[derive(Debug, Default)]
pub struct Header {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

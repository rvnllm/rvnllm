use serde::Serialize;
use smallvec::SmallVec;

pub type ShapeBuf = SmallVec<[u64; 6]>;

#[derive(Debug, Default, Serialize)]
pub struct Header {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

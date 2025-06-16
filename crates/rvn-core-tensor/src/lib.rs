use rvn_globals::types::Value;
use serde::Serialize;
use smallvec::SmallVec;
use std::fmt;

pub type ShapeBuf = SmallVec<[u64; 6]>;

#[derive(Debug, Default, Serialize)]
pub struct Header {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

#[derive(Serialize)]
pub struct DiffDump<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<Vec<(&'static str, (&'a u64, &'a u64))>>, // (field,(a,b))

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MetaDiff<'a>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensors: Option<TensorDiff<'a>>,
}

#[derive(Serialize)]
pub struct MetaDiff<'a> {
    pub added: Vec<(&'a str, &'a Value)>,
    pub removed: Vec<(&'a str, &'a Value)>,
    pub changed: Vec<(&'a str, (&'a Value, &'a Value))>,
}

#[derive(Serialize)]
pub struct TensorDiff<'a> {
    pub added: Vec<(&'a str, &'a Tensor)>,
    pub removed: Vec<(&'a str, &'a Tensor)>,
    // Only shape / dtype mismatch; value-level diff would melt CPUs
    pub changed: Vec<(&'a str, (&'a Tensor, &'a Tensor))>,
}

#[derive(Debug, Serialize)]
pub struct Tensor {
    pub name: String,
    pub kind: TensorKind,
    pub size: u64,
    pub shape: Vec<u64>,
}

#[repr(u32)] // keeps the on-disk value the same
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Serialize)]
pub enum TensorKind {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    Q4_0 = 3,
    // add more as needed
}
impl TryFrom<u32> for TensorKind {
    type Error = anyhow::Error;

    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(TensorKind::F32),
            1 => Ok(TensorKind::F16),
            2 => Ok(TensorKind::I8),
            3 => Ok(TensorKind::Q4_0),
            other => anyhow::bail!("unknown tensor kind {}", other),
        }
    }
}
impl fmt::Display for TensorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TensorKind::F32 => "F32",
            TensorKind::F16 => "F16",
            TensorKind::I8 => "I8",
            TensorKind::Q4_0 => "Q4_0",
        };
        f.write_str(s)
    }
}

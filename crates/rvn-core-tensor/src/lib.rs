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

#[repr(u32)] // Keeps values aligned with on-disk GGUF format
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Serialize)]
pub enum TensorKind {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    Q4_0 = 3,
    Q4_1 = 4,
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
    I16 = 16,
    I32 = 17,
}
impl TryFrom<u32> for TensorKind {
    type Error = anyhow::Error;

    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(TensorKind::F32),
            1 => Ok(TensorKind::F16),
            2 => Ok(TensorKind::I8),
            3 => Ok(TensorKind::Q4_0),
            4 => Ok(TensorKind::Q4_1),
            6 => Ok(TensorKind::Q5_0),
            7 => Ok(TensorKind::Q5_1),
            8 => Ok(TensorKind::Q8_0),
            9 => Ok(TensorKind::Q8_1),
            10 => Ok(TensorKind::Q2K),
            11 => Ok(TensorKind::Q3K),
            12 => Ok(TensorKind::Q4K),
            13 => Ok(TensorKind::Q5K),
            14 => Ok(TensorKind::Q6K),
            15 => Ok(TensorKind::Q8K),
            16 => Ok(TensorKind::I16),
            17 => Ok(TensorKind::I32),
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
            TensorKind::Q4_1 => "Q4_1",
            TensorKind::Q5_0 => "Q5_0",
            TensorKind::Q5_1 => "Q5_1",
            TensorKind::Q8_0 => "Q8_0",
            TensorKind::Q8_1 => "Q8_1",
            TensorKind::Q2K => "Q2K",
            TensorKind::Q3K => "Q3K",
            TensorKind::Q4K => "Q4K",
            TensorKind::Q5K => "Q5K",
            TensorKind::Q6K => "Q6K",
            TensorKind::Q8K => "Q8K",
            TensorKind::I16 => "I16",
            TensorKind::I32 => "I32",
        };
        f.write_str(s)
    }
}

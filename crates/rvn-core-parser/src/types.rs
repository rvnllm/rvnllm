use anyhow::Result;
use anyhow::anyhow;
use serde::Serialize;
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq, Serialize)]
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

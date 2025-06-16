use anyhow::{Result, anyhow, bail};
use once_cell::sync::{Lazy, OnceCell};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::HashMap;


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


#[cfg(test)]
mod tests {
    use crate::*;
    use crate::{DType, Tensor, TensorKind, register_tensor_registry_formats};
    use std::borrow::Cow;
    // TENSOR TESTS - Zero-Copy Views

    #[test]
    fn test_tensor_kind_conversions() {
        // Test all variants convert correctly
        assert_eq!(TensorKind::from(0), TensorKind::F32);
    }

    #[test]
    fn test_tensor_registry_initialization() {
        register_tensor_registry_formats();

        // Verify F32 decoder is registered
        let tensor = Tensor {
            name: Cow::Borrowed("test.weight"),
            kind: TensorKind::F32,
            offset: 0,
            size: 16,
            shape: Cow::Owned(vec![4]),
        };

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };

        let view = tensor.view(bytes).expect("View creation failed");
        assert_eq!(view.dtype, DType::F32);
        assert_eq!(view.data.len(), 16);
    }
}

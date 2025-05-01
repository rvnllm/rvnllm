#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub kind: u32,
    pub offset: u64,
    pub size: u64,
    // shape is the number of elements in each dimension
    pub shape: Vec<u64>,
}
impl Tensor {
    pub fn view<'a>(&self, mmap: &'a [u8]) -> anyhow::Result<TensorView<'a>> {
        let start = self.offset as usize;
        let end = start + self.size as usize;
        if end > mmap.len() {
            anyhow::bail!("[ERROR][Tensor] Tensor out of bounds: {} > {}", end, mmap.len());
        }

        let dtype = match self.kind {
            0 => TensorDType::F32,
            1 => TensorDType::F16,
            2 => TensorDType::Q4_0,
            16 => TensorDType::I8,
            _ => anyhow::bail!("[ERROR][Tensor] Unsupported tensor dtype kind {}", self.kind),
        };

        Ok(TensorView {
            data: &mmap[start..end],
            shape: self.shape.iter().map(|&d| d as usize).collect(),
            dtype,
        })
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F32, //0
    F16,  //1
    Q4_0, //2
    I8, // 16
}


// Tensor View -> a view into the mmap
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    pub data: &'a [u8],         // Slice into mmap
    pub shape: Vec<usize>,      // Tensor dimensions
    pub dtype: TensorDType,     // How to interpret bytes
}

impl<'a> TensorView<'a> {
    pub fn elements_size(&self) -> usize {
        match self.dtype {
            TensorDType::F32 => 0,
            TensorDType::F16 => 1,
            TensorDType::Q4_0 => 2,
            TensorDType::I8 => 16,
        }
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn expected_byte_len(&self) -> usize {
        self.num_elements() * self.elements_size()
    }

    pub fn as_f32_slice(&self) -> anyhow::Result<&'a [f32]> {
        if self.dtype != TensorDType::F32 {
            anyhow::bail!("[ERROR][TensorView] Tensor is not f32");
        }
        if self.expected_byte_len() as usize != self.data.len() {
            anyhow::bail!("[ERROR][TensorView] Tensor data length mismatch");
        }
        let ptr = self.data.as_ptr() as *const f32;
        unsafe {
            Ok(std::slice::from_raw_parts(ptr, self.num_elements() as usize))
        }
    }

    pub fn as_i8_slice(&self) -> anyhow::Result<&'a [i8]> {
        if self.dtype != TensorDType::I8 {
            anyhow::bail!("[ERROR][TensorView] Tensor is not i8");
        }
        if self.expected_byte_len() != self.data.len() {
            anyhow::bail!("[ERROR][TensorView] Tensor data length mismatch");
        }
        let ptr = self.data.as_ptr() as *const i8;
        unsafe {
            Ok(std::slice::from_raw_parts(ptr, self.num_elements() as usize))
        }
    }

}


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
    I8, // 1
    // Packed quant types—all addressable in 1 byte.
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3KS,
    Q3KM,
    Q3KL,
    Q4KS,
    Q4KM,
    Q5KS,
    Q5KM,
    Q6K
}


// Tensor View -> a view into the mmap
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    pub data: &'a [u8],         // Slice into mmap
    pub shape: Vec<usize>,      // Tensor dimensions
    pub dtype: TensorDType,     // How to interpret bytes
}

pub trait AsRawQuant {
    fn as_bytes(&self) -> anyhow::Result<&[u8]>;
}

macro_rules! impl_quant_view {
    ($fn_name:ident, $variant:ident) => {
        pub fn $fn_name(&self) -> anyhow::Result<&[u8]> {
            if self.dtype != TensorDType::$variant {
                anyhow::bail!("[TensorView] not {}", stringify!($variant));
            }
            if self.expected_byte_len() != self.data.len() {
                println!("[DEBUG] expected: {:#?}  data: {:#?}", self.expected_byte_len(), self.data.len());
                anyhow::bail!("[TensorView] length mismatch for {}", stringify!($variant));
            }
            Ok(self.data)
        }
    };
}

impl<'a> TensorView<'a> {
    pub fn elements_size(&self) -> usize {
        match self.dtype {
            TensorDType::F32    => 4,
            TensorDType::F16    => 2,
            TensorDType::I8     => 1,

            // Packed quant types—all addressable in 1 byte.
            TensorDType::Q4_0   => 1,
            TensorDType::Q4_1   => 1,
            TensorDType::Q5_0   => 1,
            TensorDType::Q5_1   => 1,
            TensorDType::Q8_0   => 1,
            TensorDType::Q8_1   => 1, 
            TensorDType::Q2K   => 1,
            TensorDType::Q3KS => 1, 
            TensorDType::Q3KM => 1, 
            TensorDType::Q3KL => 1, 
            TensorDType::Q4KS => 1, 
            TensorDType::Q4KM => 1, 
            TensorDType::Q5KS => 1, 
            TensorDType::Q5KM => 1, 
            TensorDType::Q6K   => 1,
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

    impl_quant_view!(as_q4_0_slice, Q4_0);
    impl_quant_view!(as_q4_1_slice, Q4_1);
    impl_quant_view!(as_q5_0_slice, Q5_0);
    impl_quant_view!(as_q5_1_slice, Q5_1);
    impl_quant_view!(as_q8_0_slice, Q8_0);
    impl_quant_view!(as_q8_1_slice, Q8_1);
    impl_quant_view!(as_q2_k_slice, Q2K);
    impl_quant_view!(as_q3_k_s_slice, Q3KS);
    impl_quant_view!(as_q3_k_m_slice, Q3KM);
    impl_quant_view!(as_q3_k_l_slice, Q3KL);
    impl_quant_view!(as_q4_k_s_slice, Q4KS);
    impl_quant_view!(as_q4_k_m_slice, Q4KM);
    impl_quant_view!(as_q5_k_s_slice, Q5KS);
    impl_quant_view!(as_q5_k_m_slice, Q5KM);
    impl_quant_view!(as_q6_k_slice, Q6K);
}


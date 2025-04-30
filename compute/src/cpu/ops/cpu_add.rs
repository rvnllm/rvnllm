use anyhow;
use ctensor::tensor_view::{TensorDType, TensorView};

// Takes two tensors a and b (both f32)
// Fills out with out[i] = a[i] + b[i]
// No allocations inside
// CPU version first
// Clean memory breathing
pub fn add<'a>(
    a: &'a TensorView<'a>,
    b: &'a TensorView<'a>,
    out: &'a mut [f32],
) -> anyhow::Result<()> {

    if a.dtype != TensorDType::F32 || b.dtype != TensorDType::F32 {
        anyhow::bail!("[ERROR][cpu::add] only f32 add supported");
    }
    if a.shape != b.shape {
        anyhow::bail!("[ERROR][cpu::add] shape mismatch in add, cannot add different shapes");
    }
    if out.len() != a.num_elements() {
        anyhow::bail!("[ERROR][cpu::add] ouput buffer does not match");
    }

    let a_f32 = a.as_f32_slice()?;
    let b_f32 = b.as_f32_slice()?;

    for i in 0..out.len() {
        out[i] = a_f32[i] + b_f32[i];
    }

    Ok(())
}


use ctensor::tensor_view::{TensorView, TensorDType};
use anyhow;

pub fn softmax_arr_f32(input: &[f32], output: &mut [f32]) -> anyhow::Result<()> {
    if input.len() != output.len() {
        anyhow::bail!("[ERROR][cpu][softmax_arr_f32] input and output lengths mismatch");
    }

    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Calculate exp(x - max) for each element
    let mut sum = 0.0;
    for (i, &x) in input.iter().enumerate() {
        output[i] = (x - max_val).exp();
        sum += output[i];
    }

    // Normalize
    for v in output.iter_mut() {
        *v /= sum;
    }

    Ok(())
}

/*
 * Softmax CPU naive implementation. 1st iteration
 */
pub fn softmax<'a>(
    a: &'a TensorView<'a>,
    out: &'a mut [f32],
) -> anyhow::Result<()> {
    
    if a.dtype != TensorDType::F32 {
        anyhow::bail!("[ERROR][cpu][sotmax_tv_f32] only f32 supported");
    }

    if a.shape.len() != 1 {
        anyhow::bail!("[ERROR][cpu][softmax_tv_f32] only 1D tensors supported");
    }

    if a.num_elements() != out.len() {
        anyhow::bail!("[ERROR][cpu][softmax_tv_f32] output length mismathc");
    }

    let input = a.as_f32_slice()?;
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0;
    for (i, &v) in input.iter().enumerate() {
        out[i] = (v - max_val).exp();
        sum += out[i];
    }

    for v in out.iter_mut() {
        *v /= sum;
    }

    Ok(())
}

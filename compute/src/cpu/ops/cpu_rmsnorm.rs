// y = x / sqrt(mean(x²)) * weight
// Where:

//    x = input vector
//    weight = learnable scale tensor (no bias)
//    mean(x²) = mean of squares of input values
//    Normalize by sqrt(mean(x²))

// Normalize tensor with Root Mean Square.
// Needed right before attention and MLP.

use ctensor::tensor_view::{TensorView, TensorDType};


/**  
 input: TensorView with raw data
 weight: TensorView with scale factors (same shape)
 output: mutable buffer for result
 eps: tiny float like 1e-6 to avoid division by zero

a clean stabilization op
Breathes the neuron into a normalized state
Prevents exploding or vanishing activations
Makes training + inference resilient

Step	Meaning
v * v	Get energy/magnitude of each value
sum()	Total energy
/ len	Average energy (mean square)
sqrt()	Normalize magnitude — root mean square
+ eps	Prevent div-by-zero or nan (stabilizer)
x / rms	Standardize size of each value
* weight	Learnable per-dimension scaling
*
* RAVEN ->
* Embedded
Hardware-adaptive
Transparent
Breathing
Elegant
Unstoppable
 */
macro_rules! assert_no_alias {
    ($a:expr, $b:expr, $msg:expr) => {
        debug_assert_ne!(
            ($a).as_ptr(),
            ($b).as_ptr(),
            concat!("Aliased buffers: ", $msg)
        );
    };
}

pub fn rmsnorm<'a>(
    input: &'a TensorView<'a>,
    weight: &'a TensorView<'a>,
    output: &'a mut [f32],
    eps: f32,
) -> anyhow::Result<()> {
    if input.dtype != TensorDType::F32 || weight.dtype != TensorDType::F32 {
        anyhow::bail!("[ERROR][cpu] RMSNorm: only f32 supported");
    }

    if weight.num_elements() != weight.num_elements() || input.num_elements() != output.len() {
        anyhow::bail!("[ERROR][cpu] RMSNorm: size mismatch");
    }
    
    // 1. get the input vector !! no copy slice out a tensor sitting in a mmap
    let input_slice = input.as_f32_slice()?;
    let weight_slice = weight.as_f32_slice()?;
    assert_no_alias!(input_slice, weight_slice, "rmsnorm");
    //debug_assert_ne!(
      //  input_slice.as_ptr(),
        //weight_slice.as_ptr(),
       // "Weight tensor is aliasing the input buffer!"
    //);


    let input_slice_len = input_slice.len() as f32;

    // 2. compute mean (x^2)
    let mean_square = input_slice
        .iter()
        .map(|&v| v * v)
        .sum::<f32>() / input_slice_len;

    // 3. normalize
    // note: eps to avoid div by 0
    let rms = (mean_square + eps).sqrt();
    

//    output.iter_mut() // iterate over output !! same length
  //      .zip(input_slice.iter()     // zip with input_slice iterator
    //    .zip(weight_slice.iter())   // zip input_slice with weight_slice??? or iter_mut -- maybe
                                    // doesnt matter just zip three things together


    for (out, (&x, &w)) in output.iter_mut().zip(input_slice.iter().zip(weight_slice.iter())) {
        *out = (x / rms) * w;
    }

    Ok(())
}

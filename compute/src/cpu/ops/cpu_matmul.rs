
//[TensorView (A)]   [TensorView (B)]
//      │                  │
//      └── multiply ───────┘
//              │
//         [TensorView (Result)]
// CPU Ops = Clean, safe, composable
// GPU Ops = Same API, different backend
// TensorView = Unified memory surface
//
// This way, when we switch to CUDA later,
// All ops just reroute underneath — no high-level code breakage.
use ctensor::tensor_view::{TensorView, TensorDType};

// TensorView that is tied to the same memory as your input TensorViews (a and b).
pub fn matmul<'a>(
    a: &'a TensorView<'a>,
    b: &'a TensorView<'a>,
    out: &'a mut [f32],
) -> anyhow::Result<()> {
    if a.dtype != TensorDType::F32 || b.dtype != TensorDType::F32 {
        anyhow::bail!("Only f32 matmul supported");
    }

    let (m, k1) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    if k1 != k2 {
        anyhow::bail!("Shape mismatch: {} != {}", k1, k2);
    }

    let a_f32 = a.as_f32_slice()?;
    let b_f32 = b.as_f32_slice()?;

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..k1 {
                sum += a_f32[i * k1 + k] * b_f32[k * n + j];
            }
            out[i * n + j] = sum;
        }
    }

    Ok(())
}

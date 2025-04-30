use ctensor::tensor_view::{TensorView, TensorDType};
use compute::cpu::ops::cpu_attention::attention_forward;


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_forward_simple() {
        // Dummy q, k, v
        let q_data = vec![1.0f32, 0.0]; // [1, 2]
        let k_data = vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ]; // [3, 2]
        let v_data = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]; // [3, 2]

        let q = TensorView {
            data: unsafe { std::slice::from_raw_parts(q_data.as_ptr() as *const u8, q_data.len() * 4) },
            shape: vec![1, 2],
            dtype: TensorDType::F32,
        };
        let k = TensorView {
            data: unsafe { std::slice::from_raw_parts(k_data.as_ptr() as *const u8, k_data.len() * 4) },
            shape: vec![3, 2],
            dtype: TensorDType::F32,
        };
        let v = TensorView {
            data: unsafe { std::slice::from_raw_parts(v_data.as_ptr() as *const u8, v_data.len() * 4) },
            shape: vec![3, 2],
            dtype: TensorDType::F32,
        };

        let mut out = vec![0.0f32; 2]; // output will be [1, 2]

        attention_forward(&q, &k, &v, &mut out).expect("attention forward failed");

        println!("[TEST] Attention output: {:?}", out);

        // Quick sanity check: output must not be zero
        assert!(out.iter().any(|&v| v != 0.0));
    }
}

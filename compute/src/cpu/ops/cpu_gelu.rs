use ctensor::tensor_view::{TensorView, TensorDType};


pub fn _gelu(x: f32) -> f32 {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
}
pub fn gelu(input: &TensorView, output: &mut [f32]) -> anyhow::Result<()> {
    let input_slice = input.as_f32_slice()?;
    if input_slice.len() != output.len() {
        anyhow::bail!("[ERROR][cpu_gelu] GELU: size mismatch");
    }

    for (o, &x) in output.iter_mut().zip(input_slice.iter()) {
        *o = _gelu(x);
    }

    Ok(())
}



#[test]
fn test_gelu_simple() {
    println!("[TEST] test_gelu_simple");

    let input_data = vec![-1.0f32, 0.0, 1.0];
    let expected = vec![
        _gelu(-1.0),
        _gelu(0.0),
        _gelu(1.0),
    ];

    // Build TensorView
    let input_tensor = TensorView {
        data: unsafe {
            std::slice::from_raw_parts(input_data.as_ptr() as *const u8, input_data.len() * 4)
        },
        shape: vec![3],
        dtype: TensorDType::F32,
    };

    let mut output = vec![0.0f32; 3];
    gelu(&input_tensor, &mut output).unwrap();

    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5);
    }
}



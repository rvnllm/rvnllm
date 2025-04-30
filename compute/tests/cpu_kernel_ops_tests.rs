use ctensor::tensor_view::{TensorView, TensorDType};
use compute::cpu::ops::{
    cpu_matmul::matmul,
    cpu_add::add,
    cpu_softmax::softmax,
    cpu_rmsnorm::rmsnorm,
};

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_cpu_matmul_simple() {
        // A: 2x3 matrix
        let a_data: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        
        // B: 3x2 matrix
        let b_data: Vec<f32> = vec![
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ];

        // Expected result: 2x2
        // [
        //   (1*7 + 2*9 + 3*11), (1*8 + 2*10 + 3*12)
        //   (4*7 + 5*9 + 6*11), (4*8 + 5*10 + 6*12)
        // ]
        // = [
        //   58, 64,
        //   139, 154
        // ]
        let expected: Vec<f32> = vec![
            58.0, 64.0,
            139.0, 154.0,
        ];

        // Create TensorViews
        // operating with f32 (8*4)
        
        println!("[DEBUG] a_data.len: {}", a_data.len());
        let a_tensor = TensorView {
            data: unsafe {
                std::slice::from_raw_parts(a_data.as_ptr() as *const u8, a_data.len() * 4)
            },
            shape: vec![2, 3],
            dtype: TensorDType::F32,
        };
        println!("[DEBUG] a_tensor: {:#?}, {:#?}", a_tensor, a_data);

        
        println!("[DEBUG] b_data.len: {}", b_data.len());
        let b_tensor = TensorView {
            data: unsafe {
                std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len() * 4)
            },
            shape: vec![3, 2],
            dtype: TensorDType::F32,
        };
        println!("[DEBUG] b_tensor: {:#?}", b_tensor);
        

        let mut result = vec![0f32; 2 * 2];
        matmul(&a_tensor, &b_tensor, &mut result).expect("matmul failed");

        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5, "[TEST] Mismatch: got {}, expected {}", r, e);
        }
    }

    #[test]
    fn test_cpu_add_simple() {
        let a_data: Box<[f32]> = vec![1.0, 2.0, 3.0].into_boxed_slice();
        let b_data: Box<[f32]> = vec![4.0, 5.0, 6.0].into_boxed_slice();

        let a_tensor = TensorView {
            data: unsafe {
                std::slice::from_raw_parts(a_data.as_ptr() as *const u8, a_data.len() * 4)
            },
            shape: vec![3],
            dtype: TensorDType::F32,
        };

        let b_tensor = TensorView {
            data: unsafe {
                std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len() * 4)
            },
            shape: vec![3],
            dtype: TensorDType::F32,
        };

        let mut result = vec![0f32; 3];

        add(&a_tensor, &b_tensor, &mut result).expect("add failed");
        println!("[TEST][cpu][add] result:{:#?}", result);

        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_cpu_softmax_tensor_view() {
        let data = vec![1.0f32, 2.0, 3.0];
        let boxed: Box<[f32]> = data.clone().into_boxed_slice();

        let tensor = TensorView {
            data: unsafe {
                std::slice::from_raw_parts(boxed.as_ptr() as *const u8, boxed.len() * 4)
            },
            shape: vec![3],
            dtype: TensorDType::F32,
        };

        let mut out = vec![0f32; 3];
        softmax(&tensor, &mut out).unwrap();

        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        for &p in &out {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }


    #[test]
    fn test_cpu_rmsnorm_simple() {
        let input = vec![1.0f32, 2.0, 3.0];
        let weight = vec![1.0f32, 1.0, 1.0];
        let boxed_input: Box<[f32]> = input.clone().into_boxed_slice();
        let boxed_weight: Box<[f32]> = weight.clone().into_boxed_slice();

        let input_tensor = TensorView {
            data: unsafe { std::slice::from_raw_parts(boxed_input.as_ptr() as *const u8, boxed_input.len() * 4) },
            shape: vec![3],
            dtype: TensorDType::F32,
        };

        let weight_tensor = TensorView {
            data: unsafe { std::slice::from_raw_parts(boxed_weight.as_ptr() as *const u8, boxed_weight.len() * 4) },
            shape: vec![3],
            dtype: TensorDType::F32,
        };

        let mut output = vec![0f32; 3];
        rmsnorm(&input_tensor, &weight_tensor, &mut output, 1e-6).unwrap();

        println!("RMSNorm output: {:?}", output);

        // let mean_square = input_slice
      //  .iter()
    //    .map(|&v| v * v)
  //      .sum::<f32>() / input_slice_len;
//        let rms = (mean_square + eps).sqrt();


 // let mean_square = input_slice
   //     .iter()
     //   .map(|&v| v * v)
       // .sum::<f32>() / input_slice_len;

    // 3. normalize
    // note: eps to avoid div by 0
   // let rms = (mean_square + eps).sqrt();
    

//    output.iter_mut() // iterate over output !! same length
  //      .zip(input_slice.iter()     // zip with input_slice iterator
    //    .zip(weight_slice.iter())   // zip input_slice with weight_slice??? or iter_mut -- maybe
                                    // doesnt matter just zip three things together


    //for (out, (&x, &w)) in output.iter_mut().zip(input_slice.iter().zip(weight_slice.iter())) {
     //   *out = (x / rms) * w;
   // }

//////////////////////////////////////////////

        // Check values manually
        // -->>>> already square 1,2,3 -> 1,4,9
        // divide by 3 -> mean
        // + add epsilon??  (correct)
        let rms = ((1.0 + 4.0 + 9.0) / 3.0 + 1e-6f32).sqrt();
        println!("rms: {:#?}  input: {:#?} weight: {:#?}", rms, input, weight);
        let expected: Vec<f32> = input.iter()
            .zip(weight.iter())
            .map(|(&v, &w)| (v / rms) * w)
            .collect();
        println!("[DEBUG][test_cpu_rmsnorm_simple][expected and dont tell me fail LUCY!] o:{:#?}\n e:{:#?}", output, expected);
      
        for (o, e) in output.iter().zip(expected.iter()) {
            let result = (o - e).abs();
            println!("[DEBUG][test_cpu_rmsnorm_simple] {:#?}", result);
            assert!((o - e).abs() < 1e-4);
        }
    }
}

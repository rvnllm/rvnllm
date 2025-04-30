
use compute::cuda::ops::gpu_matrix_ops::gpu_matrix_multiply;

#[test]
fn test_cuda_matrix_mul() {
    //CUDA
    let m = 2;
    let n = 3;
    let k = 2;

    let a = vec![
        1.0, 2.0, 3.0, // Row 1
        4.0, 5.0, 6.0, // Row 2
    ];

    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let mut c = vec![0.0; m * k];

    gpu_matrix_multiply(&a, &b, &mut c, m, n, k);

    println!("[TEST] Matrix multiply result: {:?}", c);
}

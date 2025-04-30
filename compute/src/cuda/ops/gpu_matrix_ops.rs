#[link(name = "llrs_compute", kind = "static")]
unsafe extern "C" {
    fn launch_matrix_multiply(a: *const f64, b: *const f64, c: *mut f64, m: i32, n: i32, k: i32);
}

pub fn gpu_matrix_multiply(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    unsafe {
        launch_matrix_multiply(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            m as i32,
            n as i32,
            k as i32,
        );
    }
}


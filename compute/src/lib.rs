pub mod cuda {
    pub mod ops {
        pub mod gpu_matrix_ops;
    }
}

pub mod cpu {
    pub mod ops {
        pub mod cpu_matmul;
        pub mod cpu_add;
        pub mod cpu_softmax;
        pub mod cpu_rmsnorm;
        pub mod cpu_gelu;
        pub mod cpu_attention;
    }
}

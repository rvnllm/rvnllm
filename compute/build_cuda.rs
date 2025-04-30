use std::{env, path::PathBuf, process::Command};

pub fn build() {
    println!("[BUILD] cuda");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let cuda_sources = ["src/cuda/kernel/matrix_mul.cu"];

    let mut objects = vec![];

    let is_release = std::env::var("PROFILE").unwrap() == "release";
    let mut nvcc_flags = vec!["-c", "--compiler-options", "-fPIC"];
    if is_release {
        nvcc_flags.push("-O3");
    } else {
        nvcc_flags.push("-G"); // for cuda debug
    }

    for src in &cuda_sources {
        let src_path = manifest_dir.join(src);
        let obj_path = out_dir.join(
            PathBuf::from(src)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string()
                + ".o",
        );

        let status = Command::new("nvcc")
            .args([
                "-c",
                src_path.to_str().unwrap(),
                "-o",
                obj_path.to_str().unwrap(),
                "--compiler-options",
                "-fPIC",
            ])
            .args(&nvcc_flags) // <--- USE IT
            .status()
            .expect("Failed to compile CUDA");

        assert!(status.success(), "CUDA compilation failed");

        objects.push(obj_path);
    }

    let mut build = cc::Build::new();
    build.cpp(true);
    for obj in objects {
        build.object(obj);
    }
    build.compile("llrs_compute");

    for src in cuda_sources {
        println!("cargo:rerun-if-changed={}", src);
    }
}

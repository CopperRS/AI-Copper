
use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cpp/lib.cpp");
    println!("cargo:rerun-if-changed=cpp/CMakeLists.txt");

    let cmake_build_dir = Path::new("cpp").join("build");

    if cmake_build_dir.exists() {
        std::fs::remove_dir_all(&cmake_build_dir).expect("Failed to clean cmake build directory");
    }
    std::fs::create_dir_all(&cmake_build_dir).unwrap();

    let mut cmake_config = Command::new("cmake");
    cmake_config
        .arg("-S")
        .arg("cpp")
        .arg("-B")
        .arg(&cmake_build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release");

    if cfg!(target_os = "windows") {
        cmake_config.arg("-A").arg("x64");
        let torch_path = "C:\\libtorch";
        cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", torch_path));
        cmake_config.env("LIBTORCH", torch_path);
    } else {
        let torch_path = env::var("LIBTORCH").expect("LIBTORCH environment variable not set");
        cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", torch_path));
    }

    let config_status = cmake_config.status().expect("Failed to run cmake configuration");
    if !config_status.success() {
        panic!("CMake configuration failed");
    }

    let build_status = Command::new("cmake")
        .arg("--build")
        .arg(&cmake_build_dir)
        .status()
        .expect("Failed to run cmake build");
    if !build_status.success() {
        panic!("CMake build failed");
    }

    let build_dir_abs = cmake_build_dir.canonicalize().unwrap();

    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native={}", build_dir_abs.join("Debug").display());
        println!("cargo:rustc-link-lib=static=ai_copper");

        println!("cargo:rustc-link-search=native=C:\\libtorch\\lib");
        println!("cargo:rustc-link-lib=dylib=c10");
        println!("cargo:rustc-link-lib=dylib=torch");
        println!("cargo:rustc-link-lib=dylib=torch_cpu");

    } else {
        println!("cargo:rustc-link-search=native={}", build_dir_abs.display());
        println!("cargo:rustc-link-lib=dylib=ai_copper");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir_abs.display());

        println!("cargo:rustc-link-search=native=/home/moonx02/Documentos/libtorch/lib");
        println!("cargo:rustc-link-lib=dylib=c10");
        println!("cargo:rustc-link-lib=dylib=torch");
        println!("cargo:rustc-link-lib=dylib=torch_cpu");

        println!("cargo:rustc-env=LD_LIBRARY_PATH={}", build_dir_abs.display());
    }
}

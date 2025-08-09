use std::env;
use std::fs;
use std::path::{Path, PathBuf};  
use std::process::Command;

fn copy_dlls_to_target(dll_source_dirs: &[&str], target_dir: &Path) {
    for dir in dll_source_dirs {
        let source_dir = Path::new(dir);
        if source_dir.exists() {
            for entry in fs::read_dir(source_dir).expect("Failed to read DLL source directory") {
                let entry = entry.expect("Failed to get entry");
                let path = entry.path();
                if path.extension().map(|ext| ext.to_ascii_lowercase() == "dll").unwrap_or(false) {
                    let file_name = path.file_name().unwrap();
                    let dest = target_dir.join(file_name);
                    println!("Copying {:?} to {:?}", path, dest);
                    fs::copy(&path, &dest).expect("Failed to copy DLL");
                }
            }
        }
    }
}

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
        .arg("-DCMAKE_BUILD_TYPE=Debug"); 

    if cfg!(target_os = "windows") {
        cmake_config.arg("-A").arg("x64");
        let torch_path = "C:\\libtorch";
        let tensorflow_path = "C:\\libtensorflow";
        cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", torch_path));
        cmake_config.env("LIBTORCH", torch_path);
        cmake_config.env("TENSORFLOW_ROOT", tensorflow_path); 
    } else {
        let torch_path = env::var("LIBTORCH").expect("LIBTORCH environment variable not set");
        let tensorflow_path = env::var("TENSORFLOW_ROOT").expect("TENSORFLOW_ROOT environment variable not set");
        cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", torch_path));
        cmake_config.env("TENSORFLOW_ROOT", tensorflow_path);
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

        println!("cargo:rustc-link-search=native=C:\\libtensorflow\\lib");
        println!("cargo:rustc-link-lib=static=tensorflow"); 
        println!("cargo:rustc-link-lib=dylib=tensorflow");

       
        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
        let mut target_dir = PathBuf::from(&out_dir);
        for _ in 0..3 {
            target_dir = target_dir.parent().unwrap().to_path_buf();
        }

        let dll_dirs = [
            "C:\\libtorch\\bin",
            "C:\\libtensorflow\\lib",
        ];
        copy_dlls_to_target(&dll_dirs, &target_dir);

    } else {
        println!("cargo:rustc-link-search=native={}", build_dir_abs.display());
        println!("cargo:rustc-link-lib=dylib=ai_copper");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir_abs.display());

        println!("cargo:rustc-link-search=native=/home/moonx02/Documentos/libtorch/lib");
        println!("cargo:rustc-link-lib=dylib=c10");
        println!("cargo:rustc-link-lib=dylib=torch");
        println!("cargo:rustc-link-lib=dylib=torch_cpu");

        println!("cargo:rustc-link-search=native=/home/moonx02/Documentos/libtensor/lib");
        println!("cargo:rustc-link-lib=dylib=tensorflow");
        println!("cargo:rustc-env=LD_LIBRARY_PATH={}", build_dir_abs.display());
    }
}

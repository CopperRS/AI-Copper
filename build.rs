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
                if path.extension().map(|ext| ext == "dll").unwrap_or(false) {
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
        // Tentar remover o diretório com retry
        let max_attempts = 5;
        let mut attempt = 0;
        while attempt < max_attempts {
            match std::fs::remove_dir_all(&cmake_build_dir) {
                Ok(()) => break,
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    attempt += 1;
                    if attempt == max_attempts {
                        panic!("Failed to clean cmake build directory after {} attempts: {}", max_attempts, e);
                    }
                    std::thread::sleep(std::time::Duration::from_secs(1)); // Aguarda 1 segundo
                }
                Err(e) => panic!("Failed to clean cmake build directory: {}", e),
            }
        }
    }
    std::fs::create_dir_all(&cmake_build_dir).unwrap();

    let torch_path = env::var("LIBTORCH").expect("LIBTORCH environment variable not set.");
    let tensorflow_path = env::var("TENSORFLOW_ROOT").expect("TENSORFLOW_ROOT environment variable not set.");

    // Criar um valor persistente para Command
    let mut cmd = Command::new("cmake");
    let cmake_config = cmd
        .arg("-S")
        .arg("cpp")
        .arg("-B")
        .arg(&cmake_build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Debug")
        .arg(format!("-DCMAKE_PREFIX_PATH={}", torch_path))
        .env("LIBTORCH", &torch_path)
        .env("TENSORFLOW_ROOT", &tensorflow_path);

    if cfg!(target_os = "windows") {
        cmake_config.arg("-A").arg("x64");
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
        println!("cargo:rustc-link-lib=dylib=ai_copper");

        let torch_lib_dir = Path::new(&torch_path).join("lib");
        println!("cargo:rustc-link-search=native={}", torch_lib_dir.display());
        if torch_lib_dir.exists() {
            for entry in fs::read_dir(&torch_lib_dir).expect("Failed to read libtorch/lib directory") {
                let entry = entry.expect("Failed to get entry");
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "lib" {
                        if let Some(_file_name) = path.file_name() {
                            let file_name = _file_name.to_string_lossy();
                            if file_name.contains("ittnotify") {
                                continue;
                            }
                            if let Some(file_stem) = path.file_stem() {
                                let lib_name = file_stem.to_string_lossy();
                                let lib_name = lib_name.strip_prefix("lib").unwrap_or(&lib_name);
                                println!("cargo:rustc-link-lib=dylib={}", lib_name);
                            }
                        }
                    }
                }
            }
        }

        let tf_lib_dir = Path::new(&tensorflow_path).join("lib");
        println!("cargo:rustc-link-search=native={}", tf_lib_dir.display());
        if tf_lib_dir.exists() {
            for entry in fs::read_dir(&tf_lib_dir).expect("Failed to read tensorflow/lib directory") {
                let entry = entry.expect("Failed to get entry");
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "lib" {
                        if let Some(_file_name) = path.file_name() {
                            let _file_name = _file_name.to_string_lossy();
                            if let Some(file_stem) = path.file_stem() {
                                let lib_name = file_stem.to_string_lossy();
                                let lib_name = lib_name.strip_prefix("lib").unwrap_or(&lib_name);
                                println!("cargo:rustc-link-lib=dylib={}", lib_name);
                            }
                        }
                    }
                }
            }
        }

        println!("cargo:rustc-link-lib=dylib=protobuf-lite");

        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
        let mut target_dir = PathBuf::from(&out_dir);
        for _ in 0..3 {
            target_dir = target_dir.parent().unwrap().to_path_buf();
        }

        let dll_dirs = [
            Path::new(&torch_path).join("bin").to_string_lossy().to_string(),
            Path::new(&tensorflow_path).join("bin").to_string_lossy().to_string(),
        ];
        let dll_dirs_ref: Vec<&str> = dll_dirs.iter().map(|s| s.as_str()).collect();
        copy_dlls_to_target(&dll_dirs_ref, &target_dir);
    } else { // Lógica para Linux
        println!("cargo:rustc-link-search=native={}", build_dir_abs.display());
        println!("cargo:rustc-link-lib=dylib=ai_copper");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir_abs.display());

        let torch_lib_dir = Path::new(&torch_path).join("lib");
        println!("cargo:rustc-link-search=native={}", torch_lib_dir.display());
        if torch_lib_dir.exists() {
            for entry in fs::read_dir(&torch_lib_dir).expect("Failed to read libtorch/lib directory") {
                let entry = entry.expect("Failed to get entry");
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "so" {
                        if let Some(file_stem) = path.file_stem() {
                            let lib_name = file_stem.to_string_lossy();
                            let lib_name = lib_name.strip_prefix("lib").unwrap_or(&lib_name);
                            println!("cargo:rustc-link-lib=dylib={}", lib_name);
                        }
                    }
                }
            }
        }

        let tf_lib_dir = Path::new(&tensorflow_path).join("lib");
        println!("cargo:rustc-link-search=native={}", tf_lib_dir.display());
        if tf_lib_dir.exists() {
            for entry in fs::read_dir(&tf_lib_dir).expect("Failed to read tensorflow/lib directory") {
                let entry = entry.expect("Failed to get entry");
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "so" {
                        if let Some(file_stem) = path.file_stem() {
                            let lib_name = file_stem.to_string_lossy();
                            let lib_name = lib_name.strip_prefix("lib").unwrap_or(&lib_name);
                            println!("cargo:rustc-link-lib=dylib={}", lib_name);
                        }
                    }
                }
            }
        }

        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
        let mut target_dir = PathBuf::from(&out_dir);
        for _ in 0..3 {
            target_dir = target_dir.parent().unwrap().to_path_buf();
        }
        let so_dirs = [
            Path::new(&torch_path).join("lib").to_string_lossy().to_string(),
            Path::new(&tensorflow_path).join("lib").to_string_lossy().to_string(),
        ];
        let so_dirs_ref: Vec<&str> = so_dirs.iter().map(|s| s.as_str()).collect();
        copy_dlls_to_target(&so_dirs_ref, &target_dir);
        println!("cargo:rustc-env=LD_LIBRARY_PATH={}", build_dir_abs.display());
    }
}